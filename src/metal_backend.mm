// Metal backend: implements the game's 62 OpenGL functions on Metal.
//
// All state here is touched only by the game's render thread, except for the
// window setup, which runs on the main thread before rendering starts.

#include "backend.h"
#include "brf.h"
#include "material_names.h"
#include "shader_xlate.h"

#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <QuartzCore/CABase.h>
#import <ImageIO/ImageIO.h>
#import <QuartzCore/CAMetalLayer.h>

#include <atomic>
#include <chrono>
#include <dirent.h>
#include <unistd.h>
#include <mach-o/dyld.h>
#include <cmath>
#include <csignal>
#include <cstdarg>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

enum {
    MAX_ATTRIBS = 16,
    MAX_UNITS = 32,
    MAX_VERTEX_SLOTS = 4,
    VERTEX_SLOT_BASE = 10,    // Metal buffer index of the first vertex buffer; 0 is the uniform block
    CONSTANT_SLOT = 14,       // buffer supplying attributes the game left disabled
    UNIFORM_CHUNK = 4 << 20,
};

// ---- objects -----------------------------------------------------------------

struct Buffer {
    id<MTLBuffer> mtl;
    void *scale_of = nullptr; // the MTLBuffer texture_per_metre was worked out for
    float texture_per_metre = 0;
    void *wind_weights = nullptr; // the MTLBuffer whose vertices have been given heights for the wind (see weigh_foliage)
};

struct Texture {
    id<MTLTexture> mtl;
    uint8_t material = 0; // what the texture shows, when its file was recognised: see MaterialClass
    float horse_eye[3] = {0, 0, 0}; // where a horse's eye is in this texture and how large, in the shader's coordinates; size 0 for none
    bool woman = false;       // a woman's face, whose eyes are elsewhere in the texture than a man's (see k_eye_function)
    bool radial_wind = false; // foliage that hangs from a crown (a palm's leaves): the wind takes hold by distance from the middle, not by height
    bool replaced = false; // holds a texture pack's image instead of the game's (see texture_packs.inc)
    float hair_colour[4] = {0, 0, 0, 0}; // the long hair's own colour, as painted (0 to 1); w 1 when a hair map came with one
    id<MTLTexture> hair;   // a pack's hair map for this texture (<name>_hair), see k_fur_function in shader_xlate.cpp
    id<MTLTexture> pbr;    // a pack's metalness and roughness map for this texture, see k_xl_pbr_sampler
    // How the game asked for level 0, kept so that a frame capture can ask again (see frame_capture.inc).
    GLint gl_internal = 0;
    GLenum gl_format = 0, gl_type = 0;
    int gl_w = 0, gl_h = 0;
    MTLPixelFormat format = MTLPixelFormatInvalid;
    int top_level = -1; // highest mip level that has content
    double scale = 1;   // storage is this many times the size the game asked for (see s_scale and s_render_scale)
    id<MTLTexture> upscaled; // the scene in this texture enlarged to the screen's size, for upscaled_frame only (see apply_upscale)
    uint64_t upscaled_frame = 0;
    bool screen_sized = false; // a render target the size of the screen: scaled, and multisampled if that is on
    id<MTLTexture> ms;         // multisampled companion that is rendered into; `mtl` receives the resolve
    id<MTLTexture> srgb_view;  // the same storage read through an sRGB format, made on demand
    // sampler parameters, with OpenGL's defaults
    GLenum min_filter = GL_NEAREST_MIPMAP_LINEAR, mag_filter = GL_LINEAR, wrap_s = GL_REPEAT, wrap_t = GL_REPEAT;
    float anisotropy = 1;
    int max_level = 1000;
    id<MTLSamplerState> sampler; // nil when the parameters changed since it was built
};

struct Renderbuffer {
    id<MTLTexture> mtl; // multisampled itself when it belongs to a multisampled target
    double scale = 1;
    GLenum gl_internal = 0; // as the game asked for it, for frame captures
    int gl_w = 0, gl_h = 0;
};

struct Framebuffer {
    GLuint color_texture = 0;
    GLuint depth_renderbuffer = 0, depth_texture = 0;
};

struct Shader {
    GLenum type = 0;
    std::string source;
};

struct PipelineKey {
    uint8_t format[MAX_ATTRIBS];  // MTLVertexFormat, 0 when the shader does not read the attribute
    uint8_t slot[MAX_ATTRIBS];
    uint16_t offset[MAX_ATTRIBS];
    uint16_t stride[MAX_VERTEX_SLOTS];
    uint16_t color_format, depth_format;
    uint16_t blend_src, blend_dst, blend_eq;
    uint8_t blend;
    uint8_t samples;
    uint8_t pad[2];
    bool operator==(const PipelineKey &o) const { return !memcmp(this, &o, sizeof *this); }
};
struct PipelineKeyHash {
    size_t operator()(const PipelineKey &k) const { return (size_t)wb_hash(&k, sizeof k); }
};

struct UniformLocation {
    std::string name;
    const XlUniform *uniform[2] = {nullptr, nullptr}; // per stage: vertex, fragment
    int texture_index[2] = {-1, -1};                  // set for samplers
    bool srgb = false;                                // sampler: bind an sRGB view (XlStage::srgb_samplers)
    bool is_sun_dir = false, is_sun_color = false;    // vSunDir and vSunColor, which contact shadows read
    bool is_sky = false;                              // the wbm_sky sampler, which gets the sky atlas
    bool is_hair_map = false;                         // the wbm_hair_map sampler, which gets the bound texture's hair map
    bool is_fur_map = false;                          // the wbm_fur_map sampler, which gets the same map
    bool is_pbr_map = false;                          // the wbm_pbr_map sampler, which gets the bound texture's metalness map
    bool is_dyn_map = false;                          // the wbm_dyn_map sampler, which gets the per-frame shadow map
    bool is_sun_matrix = false;                       // matSunViewProj, the engine's shadow-space matrix
    GLint unit = 0;                                   // sampler: the texture unit the game assigned
};

// One stage's MSL being compiled by Metal in the background. Programs that
// share a shader share the compile.
struct StageCompile {
    dispatch_group_t group;
    id<MTLFunction> function;
    std::string error;
};

struct Program {
    std::vector<std::shared_ptr<Shader>> shaders;
    std::map<std::string, int> attribs;
    bool linked = false;
    XlProgram xl;
    std::shared_ptr<StageCompile> compile[2];
    id<MTLFunction> function[2]; // set from compile[] the first time the program draws
    bool functions_ready = false;
    std::vector<uint8_t> uniform_data[2];
    // Where each stage's uniform block was last copied for the GPU. It is
    // copied again only after a value changed or a new frame began.
    bool uniform_dirty[2] = {true, true};
    id<MTLBuffer> pushed_buffer[2];
    NSUInteger pushed_offset[2] = {0, 0};
    uint64_t pushed_frame[2] = {0, 0};
    std::vector<UniformLocation> locations;
    std::unordered_map<std::string, GLint> location_by_name;
    int yflip_offset = -1;   // vertex stage, see k_xl_yflip_uniform
    int samples_offset = -1; // fragment stage, see k_xl_samples_uniform
    int headroom_offset = -1; // fragment stage, see k_xl_headroom_uniform
    int grade_offset = -1;    // fragment stage, wbm_grade
    int limit_offset = -1;    // fragment stage, see k_xl_limit_uniform
    int dyn_window_offset = -1, dyn_near_offset = -1, dyn_bias_offset = -1;
    int amb_offset[2] = {-1, -1}; // per stage, see k_xl_ambient_uniform
    int material_offset = -1;     // fragment stage, see k_xl_material_uniform
    int pbr_offset = -1;          // fragment stage, see k_xl_pbr_sampler
    int skin_offset[2] = {-1, -1}; // per stage, see k_xl_skin_uniform
    int wind_offset = -1;          // vertex stage, see k_xl_wind_uniform
    int hair_colour_offset = -1; // fragment stage, the same
    int horse_eye_offset = -1; // fragment stage, the same
    int shell_offset[2] = {-1, -1}; // per stage: wbm_shell, see k_fur_function
    int drift_offset = -1;          // vertex stage: wbm_drift, see drift_of
    int fur_offset = -1;      // fragment stage, see k_fur_function in shader_xlate.cpp
    int gaze_offset = -1;     // fragment stage, see k_eye_function in shader_xlate.cpp
    int leaf_sun_offset = -1; // vertex stage, see k_leaf_function in shader_xlate.cpp // fragment stage, see k_xl_dyn_sampler
    // For the per-frame shadow map (see record_caster). The engine marks a
    // character's ground shadow, which that map replaces, with one blur constant.
    const XlUniform *u_wvp = nullptr, *u_world = nullptr, *u_bones = nullptr;
    int alpha_texture_location = -1;
    bool is_blob_shadow = false;
    uint32_t input_mask = 0; // attribute locations the vertex shader reads
    std::unordered_map<PipelineKey, id<MTLRenderPipelineState>, PipelineKeyHash> pipelines;
};

struct Attrib {
    bool enabled = false;
    GLuint buffer = 0;
    GLint size = 4;
    GLenum type = GL_FLOAT;
    GLboolean normalized = GL_FALSE;
    GLsizei stride = 0;
    uintptr_t offset = 0;
};

// ---- device and frame ----------------------------------------------------------

id<MTLDevice> s_device;
id<MTLCommandQueue> s_queue;
CAMetalLayer *s_layer;
id<MTLTexture> s_screen_depth;

id<CAMetalDrawable> s_drawable;
id<MTLCommandBuffer> s_cmd;
id<MTLRenderCommandEncoder> s_pass;
id<MTLTexture> s_pass_color, s_pass_depth; // attachments of the open pass
bool s_pass_screen;
double s_pass_scale = 1;
int s_pass_samples = 1;

// Multisampling (WBM_MSAA). Screen-sized render targets, and the screen, are
// rendered into multisampled textures that are resolved when each pass ends.
// Being multisampled is a property of the target, not of the pass, so a
// target's contents always live in the same place.
int s_msaa = 1;
id<MTLTexture> s_screen_ms;
uint64_t s_scene_frame; // the last frame in which the engine drew into its HDR scene buffer

// The game renders at the window's size in points. On a Retina display that is
// half the panel's resolution, so the screen and every screen-sized render
// target are allocated s_scale times larger and viewports are scaled to match.
int s_scale = 1;
// WBM_UPSCALE (for example 0.67): the scene is drawn into targets that share
// of the screen's size each way, and MetalFX enlarges the finished picture to
// the screen's own before the engine's last pass puts it there (see
// apply_upscale). Every screen-sized target but the screen itself is then
// s_render_scale times the game's size, not s_scale times.
double s_render_scale = 1;
bool s_upscale;
float s_sharpen; // WBM_SHARPEN, 0 to 1: see apply_sharpen
int s_screen_w, s_screen_h; // the size the game believes the screen has

bool is_screen_sized(int w, int h)
{
    return w == s_screen_w && h == s_screen_h;
}

double scale_for(int w, int h)
{
    return is_screen_sized(w, h) ? s_render_scale : 1;
}

// Uniform data is copied into chunks that live until the GPU has used them.
std::mutex s_chunk_mu;
std::vector<id<MTLBuffer>> s_free_chunks;
NSMutableArray<id<MTLBuffer>> *s_frame_chunks;
size_t s_chunk_used;

// ---- OpenGL state ----------------------------------------------------------------

std::unordered_map<GLuint, Buffer> s_buffers;
std::unordered_map<GLuint, Texture> s_textures;
std::unordered_map<GLuint, Renderbuffer> s_renderbuffers;
std::unordered_map<GLuint, Framebuffer> s_framebuffers;
std::unordered_map<GLuint, std::shared_ptr<Shader>> s_shaders;
std::unordered_map<GLuint, Program> s_programs;
GLuint s_next_object; // shaders and programs share one id space, as in OpenGL
GLuint s_next_texture, s_next_buffer, s_next_framebuffer, s_next_renderbuffer;

GLuint s_fbo, s_rbo, s_array_buffer, s_element_buffer, s_program, s_unit;
GLuint s_bound_texture[MAX_UNITS];
Attrib s_attribs[MAX_ATTRIBS];

MTLClearColor s_clear_color = MTLClearColorMake(0, 0, 0, 0);
struct {
    GLint x, y;
    GLsizei w, h;
} s_viewport;
bool s_depth_test, s_depth_mask = true, s_cull, s_blend;
GLenum s_depth_func = GL_LESS, s_cull_face = GL_BACK, s_polygon_mode = GL_FILL;
GLenum s_blend_src = GL_ONE, s_blend_dst = GL_ZERO, s_blend_eq = GL_FUNC_ADD;

XlOptions s_shader_options; // set once in mtl_attach_window, from the environment
bool s_reverse_depth;       // mirrors s_shader_options.reverse_depth

// What the open render pass already has set, so that a draw only issues the
// Metal calls whose values changed since the previous draw (WBM_STATE_CACHE=0
// turns the skipping off). Cleared whenever a pass starts.
struct PassState {
    void *pipeline, *depth_state;
    double viewport[4];
    int winding, cull, fill; // -1 while unknown
    void *texture[2][16], *sampler[2][16];
    void *buffer[2][31];
    NSUInteger offset[2][31];
};
PassState s_set;
bool s_state_cache = true;
uint64_t s_frame_serial = 1; // counts presented frames; uniform storage is per frame
GLuint s_last_key_program;   // the previous draw's pipeline, reused while its key repeats
PipelineKey s_last_key;
id<MTLRenderPipelineState> s_last_pipeline;
struct {
    uint64_t made, skipped, uniform_copies, uniform_reuses;
} s_call_stats;

// The pass most recently rendered into, remembered after it has ended so that
// work can be appended to its target (see apply_ambient_occlusion).
struct LastPass {
    id<MTLTexture> color, ms_color, depth;
    bool scene = false;
    int samples = 1, draws = 0;
    uint64_t occluded_in_frame = 0;
};
LastPass s_last;

// Extended-range output (WBM_HDR=1). The drawable is 16-bit float in extended
// sRGB, so the game's gamma-encoded colours mean what they did, and values
// above 1 are brighter than display white. How far above the screen can go
// right now (it depends on its brightness setting) is polled on the main
// thread and handed to the final shader.
bool s_hdr;
float s_hdr_peak = 4.0f; // WBM_HDR_PEAK: the most headroom to use
float s_grade[4];        // WBM_CONTRAST, WBM_SATURATION, WBM_WARMTH: each about -1 to 1, 0 for no change
std::atomic<float> s_headroom{1.0f};
NSWindow *s_window;
float s_lod_bias; // WBM_LOD_BIAS: negative values pick sharper mip levels (macOS 26 and later)

// Ambient occlusion (WBM_AO=1), applied to the engine's HDR scene buffer when
// the engine moves on from it to post-processing.
bool s_ao;
float s_ao_radius = 0.6f, s_ao_strength = 0.5f; // metres; WBM_AO_RADIUS, WBM_AO_STRENGTH
bool s_ao_full; // WBM_AO_FULL: compute at the depth buffer's full resolution instead of half
// The scene camera, worked out afresh each frame from the matrices of a draw
// into the scene buffer (see observe_camera). clip z = a * w + b; p00 and p11
// are the projection's x and y scales; view holds the camera's axes in world
// space, the third being the direction of increasing w.
struct Camera {
    double a = 0, b = 0, p00 = 1, p11 = 1, view[3][3] = {};
    double forward_sign = 1; // +1 when the shaders' view space has its third axis pointing forwards, -1 when back
    uint64_t projection_frame = 0, view_frame = 0;
    // World space to clip space, for this frame and the one before (motion blur).
    double view_proj[16] = {}, last_view_proj[16] = {};
    uint64_t view_proj_frame = 0, last_view_proj_frame = 0;
};
Camera s_camera;
bool s_pass_scene, s_pass_prepass, s_pass_shadow_map;

// The per-frame shadow map (WBM_DYNAMIC_SHADOWS=1): see record_caster and render_dynamic_shadows.
bool s_dyn_shadows;
float s_dyn_range = 60.0f;     // WBM_DYN_RANGE: half the width of the window, in metres
float s_dyn_bias_m = 0.05f;    // WBM_DYN_BIAS: metres
float s_dyn_near_range = 12.0f; // WBM_DYN_NEAR: half the width of the fine window around the camera, in metres; 0 for none
float s_dyn_near[4];           // the fine window, as the shaders' wbm_dyn_near
float s_sun_matrix[16];        // matSunViewProj as last set for the scene
bool s_sun_matrix_valid;
float s_dyn_window[4];         // what shaders get as wbm_dyn; z = 0 until a map exists
float s_dyn_bias;
uint64_t s_dyn_frame;          // the frame the map was last drawn in
id<MTLTexture> s_dyn_map, s_dyn_depth;
id<MTLSamplerState> s_dyn_sampler;
struct {
    uint64_t frames, casters, scenery, overlays, late;
} s_dyn_stats;
// Screen-space contact shadows (WBM_CONTACT_SHADOWS=1), computed in the same
// pass as ambient occlusion. They need the sun: its direction and colour as
// last given to a shader while the scene was being drawn.
bool s_contact_shadows;
float s_contact_strength = 0.5f, s_contact_length = 0.4f; // WBM_CONTACT_STRENGTH; WBM_CONTACT_LENGTH in metres
// Light shafts from the sun (WBM_LIGHT_SHAFTS=1), run after occlusion on the same scene buffer.
bool s_light_shafts;
float s_shaft_strength = 0.7f; // WBM_SHAFT_STRENGTH
// The sky atlas that PBR reflections sample (see k_xl_sky_sampler), redrawn
// from the engine's sky dome each frame it is seen.
id<MTLTexture> s_sky_atlas;
id<MTLSamplerState> s_sky_sampler;
uint64_t s_sky_frame;
float s_sun_dir[4], s_sun_color[4];
bool s_sun_valid;
void apply_ambient_occlusion();

// Ambient light from the captured sky (WBM_SKY_AMBIENT=1), see k_xl_ambient_uniform.
bool s_sky_ambient;
float s_sky_ambient_strength = 0.6f; // WBM_SKY_AMBIENT_STRENGTH: how far the sky's colour tints the ambient light
float s_bounce_strength = 0.5f;      // WBM_BOUNCE: sunlight bounced off the ground, as a share of the ideal amount
float s_amb[24];                     // the shaders' wbm_amb
std::mutex s_amb_mu;
float s_amb_faces[6][3]; // mean colour of each sky face as last read back, gamma-encoded; guarded by s_amb_mu
bool s_amb_faces_valid;  // guarded by s_amb_mu
uint64_t s_amb_stats_frames;

// Material classes (WBM_MATERIALS=1), see k_xl_material_uniform. Textures are
// recognised by their content, matched against the files in the game's
// texture folders, and classed by words in the file's name.
// metalness factor, roughness factor, highlight strength, 1 + metalness of grey parts (0: unknown)
const float k_material_params[MAT_COUNT][4] = {
    {1.0f, 1.0f, 1.0f, 0.0f},   // unknown: the shader's own estimate stands
    {1.5f, 0.6f, 1.2f, 1.95f},  // metal: its grey parts are metal, and smoother
    {0.0f, 3.0f, 0.35f, 1.0f},  // cloth: never metal, matt
    {0.0f, 1.4f, 0.8f, 1.8f},   // leather: a broad, soft sheen; helmets and buckles share its textures
    {0.0f, 1.6f, 0.7f, 1.0f},   // skin
    {0.0f, 2.2f, 0.5f, 1.5f},   // wood: nails, bands and bosses on it may be metal
    {0.0f, 2.5f, 0.4f, 1.0f},   // stone and ground
    {0.0f, 1.2f, 0.9f, 1.0f},   // a horse's coat
    {0.0f, 1.6f, 0.7f, 1.0f},   // hair and beards
    {0.0f, 2.0f, 0.5f, 1.0f},   // foliage
};
// Where the eye is painted on the game's plain horses, which come in three
// layouts (measured on horse_a, hunting_horse and arabian_horse_a). Armoured
// horses have theirs behind a chamfron and are left alone.
void horse_eye_for(const std::string &name, float *out)
{
    static const struct {
        const char *name;
        float x, y, size;
    } table[] = {{"horse_a", 0.081f, 0.156f, 0.011f},       {"horse_b", 0.081f, 0.156f, 0.011f},       {"hunting_horse", 0.069f, 0.133f, 0.011f},
                 {"steppe_horse", 0.069f, 0.133f, 0.011f},  {"saddle_horse", 0.069f, 0.133f, 0.011f},  {"courser", 0.069f, 0.133f, 0.011f},
                 {"sumpter_horse", 0.069f, 0.133f, 0.011f}, {"arabian_horse_a", 0.643f, 0.327f, 0.012f}, {"arabian_horse_b", 0.643f, 0.327f, 0.012f}};
    out[0] = out[1] = out[2] = 0;
    for (auto &row : table)
        if (name == row.name) out[0] = row.x, out[1] = 1.0f - row.y, out[2] = row.size; // the meshes' coordinates run upwards
}

// Reads the colour that texture_tool wrote beside a hair map (<name>_hair.txt).
void read_hair_colour(const std::string &map_path, float *out)
{
    out[0] = out[1] = out[2] = out[3] = 0;
    size_t dot = map_path.rfind('.');
    if (dot == std::string::npos) return;
    if (FILE *f = fopen((map_path.substr(0, dot) + ".txt").c_str(), "r")) {
        int c[3];
        if (fscanf(f, "%d %d %d", &c[0], &c[1], &c[2]) == 3)
            for (int k = 0; k < 3; k++) out[k] = (float)c[k] / 255.0f, out[3] = 1;
        fclose(f);
    }
}

bool is_womans_face(const std::string &name) { return name.find("woman") != std::string::npos || name.find("female") != std::string::npos || name.find("lady") != std::string::npos; }
bool s_materials;
bool s_debug_skin = getenv("WBM_DEBUG_SKIN") != nullptr;
float s_eye[4];            // the camera's place in the world, from render_dynamic_shadows
uint64_t s_eye_frame;
float s_foliage_backlight; // WBM_FOLIAGE: how much sunlight comes through leaves, 0 for none
float s_wind_strength;     // WBM_WIND, 0 for none
double s_wind_time;        // the same for every draw of a frame
// Shells of fur (WBM_FUR_SHELLS): how many layers a coat near the camera is drawn in, 0 for none.
int s_fur_shells;
float s_texture_per_metre = 0.3f; // of the mesh being drawn: how much of its texture a metre of its surface covers
// WBM_HAIR_STRANDS: mane and tail as strands of their own, in place of layers
// (see build_strands). WBM_TAIL_LENGTH and WBM_MANE_LENGTH: how long the
// longest are, metres. WBM_HAIR_STRANDS_DENSITY: strands to the square metre.
bool s_hair_strands;
float s_pretend_speed; // metres a second a previewed model is taken to be going forwards, though it stays where it is (see drift_of)
// WBM_PEOPLE_HAIR: people's hair and beards as strands over the hair mesh, for
// characters within WBM_PEOPLE_HAIR_RANGE metres, at most WBM_PEOPLE_HAIR_MOST
// meshes a frame; WBM_PEOPLE_HAIR_LENGTH: the longest strand, metres;
// WBM_PEOPLE_HAIR_DENSITY: strands to the square metre of mesh.
bool s_people_hair;
float s_people_hair_range = 5.0f, s_people_hair_length = 0.06f, s_people_hair_density = 60000.0f;
int s_people_hair_most = 12, s_people_hair_drawn;
uint64_t s_people_hair_frame;
float s_tail_length = 0.8f, s_mane_length = 0.18f, s_strand_density = 30000.0f;
float s_hair_length = 0.07f; // WBM_HAIR_LENGTH: how far mane and tail stand off the body, metres; 0 for no layers of their own
float s_fur_length = 0.012f, s_fur_range = 9.0f, s_fur_density = 1400.0f; // WBM_FUR_LENGTH, metres; WBM_FUR_RANGE, metres; hairs per metre
float s_shell_now[4];  // the shaders' wbm_shell for the draw being prepared
uint64_t s_shell_draws;
float s_fur_strength; // WBM_FUR: the coat of horses, 0 for none
float s_gaze_strength = 1.0f; // WBM_EYES_MOVE: how far the eyes wander; 0 keeps them still
float s_skin_backlight = 0.6f; // WBM_SSS_BACKLIGHT: sunlight through thin skin; 0 for none
// The game's texture files, keyed by content: filled once at start, read-only afterwards.
struct KnownTexture {
    uint8_t cls;     // MaterialClass
    bool flipped;    // this is the file's content turned upside down, as the game uploads most textures
    uint32_t name;   // index into s_texture_names
};
std::unordered_map<uint64_t, KnownTexture> s_material_by_content;
std::vector<std::string> s_texture_names; // lower case, without the extension
std::vector<std::string> s_texture_paths; // where each was found
struct {
    std::atomic<int> uploads{0}, matched{0}, recognised{0}, strangers{0};
    int files = 0, classed[MAT_COUNT] = {}, dumped = 0;
} s_material_stats;

// A key for one level of a block-compressed texture, from 32 blocks in each of
// 8 rows of blocks spread over the image. (The start of the data alone will
// not do: many textures begin with the same empty margin.) rows[i] points to
// the whole of row key_row(i).
//
// alpha: 0 for DXT1, 3 for DXT3, 5 for DXT5. The alpha half of a block is
// left out of the key except for DXT5's two alpha values: the game's way of
// turning the alpha indices upside down does not always give what turning the
// image does (measured: a few bits differ in some blocks), so they cannot be
// predicted from the file.
enum { KEY_ROWS = 8, KEY_COLUMNS = 32 };
size_t key_row(uint32_t height, int i) { return (size_t)((height + 3) / 4) * (size_t)(2 * i + 1) / (2 * KEY_ROWS); }

uint64_t content_key_rows(uint32_t width, uint32_t height, int alpha, const uint8_t *const *rows)
{
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](const uint8_t *p, size_t n) {
        for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 1099511628211ull;
    };
    mix((const uint8_t *)&width, 4);
    mix((const uint8_t *)&height, 4);
    size_t block = alpha ? 16 : 8, across = (width + 3) / 4;
    for (int i = 0; i < KEY_ROWS; i++) {
        for (int j = 0; j < KEY_COLUMNS; j++) {
            const uint8_t *b = rows[i] + across * (size_t)(2 * j + 1) / (2 * KEY_COLUMNS) * block;
            if (!alpha) {
                mix(b, 8);
            } else {
                if (alpha == 5) mix(b, 2);
                mix(b + 8, 8);
            }
        }
    }
    return h;
}

// The same for a level held whole in memory; 0 if it is shorter than its size requires.
uint64_t content_key(uint32_t width, uint32_t height, const void *data, size_t size, int alpha)
{
    size_t row_bytes = (size_t)((width + 3) / 4) * (alpha ? 16 : 8);
    if (size < row_bytes * ((height + 3) / 4)) return 0;
    const uint8_t *rows[KEY_ROWS];
    for (int i = 0; i < KEY_ROWS; i++) rows[i] = (const uint8_t *)data + key_row(height, i) * row_bytes;
    return content_key_rows(width, height, alpha, rows);
}

// Reads the headers of the DDS files in one folder and keys their first few
// mip levels by content (the game may leave out the largest, depending on its
// texture detail setting).
void scan_texture_folder(const std::string &dir)
{
    DIR *d = opendir(dir.c_str());
    if (!d) return;
    while (struct dirent *e = readdir(d)) {
        std::string name = e->d_name;
        if (name.size() < 5 || strcasecmp(name.c_str() + name.size() - 4, ".dds")) continue;
        int cls = classify_texture_name(name.substr(0, name.size() - 4));
        s_material_stats.files++;
        uint32_t name_index = (uint32_t)s_texture_names.size();
        s_texture_names.push_back(name.substr(0, name.size() - 4));
        s_texture_paths.push_back(dir + "/" + name);
        for (char &c : s_texture_names.back()) c = (char)tolower((unsigned char)c);
        FILE *f = fopen((dir + "/" + name).c_str(), "rb");
        if (!f) continue;
        uint8_t header[128];
        uint32_t height, width, mips, fourcc;
        if (fread(header, 1, sizeof header, f) == sizeof header && !memcmp(header, "DDS ", 4)) {
            memcpy(&height, header + 12, 4);
            memcpy(&width, header + 16, 4);
            memcpy(&mips, header + 28, 4);
            memcpy(&fourcc, header + 84, 4);
            size_t block = !memcmp(&fourcc, "DXT1", 4) ? 8 : (!memcmp(&fourcc, "DXT3", 4) || !memcmp(&fourcc, "DXT5", 4)) ? 16 : 0;
            int alpha = !memcmp(&fourcc, "DXT3", 4) ? 3 : !memcmp(&fourcc, "DXT5", 4) ? 5 : 0;
            long at = sizeof header;
            for (uint32_t level = 0; block && level < std::min<uint32_t>(std::max<uint32_t>(mips, 1), 4) && width && height; level++) {
                size_t size = (size_t)((width + 3) / 4) * ((height + 3) / 4) * block;
                size_t row_bytes = (size_t)((width + 3) / 4) * block, rows = (size_t)((height + 3) / 4);
                // Once as the file has it, and once upside down: the game turns most
                // textures over before handing them to OpenGL, whose rows run bottom
                // to top. Upside down, row r is the file's row from the other end,
                // with the four rows inside each of its blocks reversed too.
                for (int flipped = 0; flipped < 2; flipped++) {
                    std::vector<uint8_t> data[KEY_ROWS];
                    const uint8_t *pointers[KEY_ROWS];
                    bool ok = true;
                    for (int k = 0; k < KEY_ROWS && ok; k++) {
                        size_t r = key_row(height, k);
                        data[k].resize(row_bytes);
                        ok = !fseek(f, at + (long)((flipped ? rows - 1 - r : r) * row_bytes), SEEK_SET) && fread(data[k].data(), 1, row_bytes, f) == row_bytes;
                        for (size_t b = 0; ok && flipped && b + block <= row_bytes; b += block) {
                            uint8_t *q = data[k].data() + b;
                            if (alpha == 3) { // four rows of explicit alpha, two bytes each
                                std::swap(q[0], q[6]), std::swap(q[1], q[7]), std::swap(q[2], q[4]), std::swap(q[3], q[5]);
                            } else if (alpha == 5) { // sixteen 3-bit alpha indices after the two alpha values
                                uint64_t bits = 0;
                                for (int n = 0; n < 6; n++) bits |= (uint64_t)q[2 + n] << (8 * n);
                                uint64_t turned = ((bits & 0xFFF) << 36) | ((bits >> 12 & 0xFFF) << 24) | ((bits >> 24 & 0xFFF) << 12) | (bits >> 36 & 0xFFF);
                                for (int n = 0; n < 6; n++) q[2 + n] = (uint8_t)(turned >> (8 * n));
                            }
                            uint8_t *colour = q + (block == 16 ? 8 : 0); // two colours, then four rows of indices, a byte each
                            std::swap(colour[4], colour[7]), std::swap(colour[5], colour[6]);
                        }
                        pointers[k] = data[k].data();
                    }
                    // Unknown ones go in too, to tell them from strangers.
                    if (ok) s_material_by_content[content_key_rows(width, height, alpha, pointers)] = {(uint8_t)cls, flipped != 0, name_index};
                }
                at += (long)size;
                width = std::max<uint32_t>(width / 2, 1);
                height = std::max<uint32_t>(height / 2, 1);
            }
            if (cls != MAT_UNKNOWN) s_material_stats.classed[cls]++;
        }
        fclose(f);
    }
    closedir(d);
}

void scan_game_textures()
{
    // The game's folder: four levels above its executable when this is the
    // game; for the frame player, WBM_GAME or where Steam puts it.
    char path[4096];
    uint32_t size = sizeof path;
    std::string root;
    if (const char *game = getenv("WBM_GAME")) {
        root = game;
    } else if (!_NSGetExecutablePath(path, &size) && strstr(path, "/Mount and Blade.app/Contents/MacOS/")) {
        root = path; // <game>/Mount and Blade.app/Contents/MacOS/Mount and Blade
        for (int up = 0; up < 4; up++) {
            size_t slash = root.rfind('/');
            if (slash == std::string::npos) return;
            root.resize(slash);
        }
    } else if (const char *home = getenv("HOME")) {
        root = std::string(home) + "/Library/Application Support/Steam/steamapps/common/MountBlade Warband";
    }
    scan_texture_folder(root + "/Textures");
    scan_texture_folder(root + "/TexturesHD");
    std::string module;
    if (const char *home = getenv("HOME")) {
        if (FILE *f = fopen((std::string(home) + "/Library/Application Support/MBWarband/last_module_warband").c_str(), "r")) {
            char name[512];
            size_t n = fread(name, 1, sizeof name - 1, f);
            fclose(f);
            module.assign(name, n);
            while (!module.empty() && (module.back() == '\n' || module.back() == '\r' || module.back() == ' ')) module.pop_back();
        }
    }
    if (!module.empty()) scan_texture_folder(root + "/Modules/" + module + "/Textures");
}
// Motion blur from the camera's own movement (WBM_MOTION_BLUR=1): see apply_motion_blur.
bool s_motion_blur;
float s_motion_blur_strength = 1.0f; // WBM_MOTION_BLUR_STRENGTH: exposure time in units of 1/48 s
void apply_motion_blur();
// Anti-aliasing after the fact (WBM_FXAA=1): see apply_fxaa.
bool s_replaying;  // a captured frame is being played back (frame_capture.inc)
bool s_previewing; // the frame player is showing a model (model_preview.inc)
bool s_preview_keeps_scene;
bool s_preview_in_scene; // the model has been drawn into the scene buffer this frame: what follows may be post-processed
bool s_fxaa;
void apply_fxaa();
id<MTLTexture> apply_upscale() API_AVAILABLE(macos(13.0));
void apply_sharpen(id<MTLTexture> picture);
// Frame-rate limits (WBM_MAX_FPS, and WBM_MENU_FPS for frames without a 3D scene); 0 for none.
double s_max_fps, s_menu_fps;

void reset_pass_state()
{
    memset(&s_set, 0, sizeof s_set);
    s_set.winding = s_set.cull = s_set.fill = -1;
    s_set.viewport[2] = -1;
}

// True when the value is already set in this pass and the call can be skipped.
bool already(bool same)
{
    if (same && s_state_cache) return s_call_stats.skipped++, true;
    return s_call_stats.made++, false;
}

std::unordered_map<std::string, std::shared_ptr<StageCompile>> s_stage_compiles; // by MSL source
struct {
    int links, cache_hits, stage_compiles;
    double translate_ms, wait_ms;
} s_shader_stats;

std::unordered_map<uint64_t, id<MTLSamplerState>> s_samplers;
id<MTLDepthStencilState> s_depth_states[2][8][2]; // [test][func - GL_NEVER][write]
id<MTLDepthStencilState> s_depth_off;
id<MTLTexture> s_white;
id<MTLBuffer> s_constant_attrib, s_quad_indices;
int s_quad_capacity;
bool s_debug_clear;

// Reports a problem once per distinct message, so a per-draw failure cannot flood the log.
void complain(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void complain(const char *fmt, ...)
{
    static std::map<std::string, int> seen;
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (seen[buf]++ == 0) wb_note("  !! metal: %s", buf);
}

template <class Map> auto find(Map &map, GLuint id) -> decltype(&map.begin()->second)
{
    auto it = map.find(id);
    return it == map.end() ? nullptr : &it->second;
}

void gen_ids(GLuint &counter, GLsizei n, GLuint *out)
{
    for (GLsizei i = 0; i < n; i++) out[i] = ++counter;
}

// ---- command buffer and passes -----------------------------------------------------

id<MTLCommandBuffer> command_buffer()
{
    if (!s_cmd) s_cmd = [s_queue commandBuffer];
    return s_cmd;
}

void end_pass()
{
    if (!s_pass) return;
    [s_pass endEncoding];
    s_pass = nil;
    s_pass_color = s_pass_depth = nil;
}

id<MTLTexture> new_render_texture(MTLPixelFormat format, NSUInteger width, NSUInteger height, int samples)
{
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:width height:height mipmapped:NO];
    d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead; // ambient occlusion reads the depth buffer
    d.storageMode = MTLStorageModePrivate;
    if (samples > 1) {
        d.textureType = MTLTextureType2DMultisample;
        d.sampleCount = (NSUInteger)samples;
    }
    return [s_device newTextureWithDescriptor:d];
}

id<MTLTexture> new_depth_texture(NSUInteger width, NSUInteger height, int samples)
{
    return new_render_texture(MTLPixelFormatDepth32Float, width, height, samples);
}

struct Target {
    id<MTLTexture> color, depth; // color is what later passes sample; with multisampling it receives the resolve
    id<MTLTexture> ms_color;     // what is rendered into when multisampled
    bool screen = false;
    bool scene = false; // the engine's HDR scene buffer: screen-sized, RGBA16F, with depth
    bool prepass = false; // the engine's depth pre-pass: screen-sized, R16F, with depth
    bool shadow_map = false; // the engine's own shadow map of the scenery: large, square, R32F
    double scale = 1;
    int samples = 1;
};

// The attachments the bound framebuffer currently describes. Returns false
// when there is nothing to draw into.
bool resolve_target(Target &t)
{
    if (s_fbo == 0) {
        if (!s_drawable) {
            if (!s_layer) return false;
            s_drawable = [s_layer nextDrawable];
            if (!s_drawable) return false;
        }
        t.color = s_drawable.texture;
        // When the engine renders the 3D scene into its own HDR buffer (this frame
        // or the last), all the screen receives is one full-screen image and the
        // UI. Multisampling that gains nothing and costs a full-screen resolve,
        // so the screen is then single-sampled.
        int samples = (s_scene_frame + 1 >= s_frame_serial) ? 1 : s_msaa;
        if (!s_screen_depth || s_screen_depth.width != t.color.width || s_screen_depth.height != t.color.height ||
            (int)s_screen_depth.sampleCount != samples) {
            s_screen_depth = new_depth_texture(t.color.width, t.color.height, samples);
            s_screen_ms = samples > 1 ? new_render_texture(t.color.pixelFormat, t.color.width, t.color.height, samples) : nil;
        }
        t.depth = s_screen_depth;
        t.ms_color = s_screen_ms;
        t.samples = samples;
        t.screen = true;
        t.scale = s_scale;
        return true;
    }
    Framebuffer *fb = find(s_framebuffers, s_fbo);
    if (!fb) return false;
    if (Texture *tex = find(s_textures, fb->color_texture)) {
        t.color = tex->mtl;
        t.scale = tex->scale;
        if (tex->mtl && tex->screen_sized && s_msaa > 1) {
            if (!tex->ms) tex->ms = new_render_texture(tex->mtl.pixelFormat, tex->mtl.width, tex->mtl.height, s_msaa);
            t.ms_color = tex->ms;
            t.samples = s_msaa;
        }
    }
    if (Renderbuffer *rb = find(s_renderbuffers, fb->depth_renderbuffer)) {
        t.depth = rb->mtl;
        if (!t.color) t.scale = rb->scale;
    } else if (Texture *tex = find(s_textures, fb->depth_texture)) {
        t.depth = tex->mtl;
        if (!t.color) t.scale = tex->scale;
    }
    if (!t.color && t.depth) t.samples = (int)t.depth.sampleCount;
    if (Texture *tex = find(s_textures, fb->color_texture))
        t.scene = tex->screen_sized && tex->format == MTLPixelFormatRGBA16Float && t.depth != nil;
    if (Texture *tex = find(s_textures, fb->color_texture))
        t.prepass = tex->screen_sized && tex->format == MTLPixelFormatR16Float && t.depth != nil;
    if (Texture *tex = find(s_textures, fb->color_texture))
        t.shadow_map = !tex->screen_sized && tex->format == MTLPixelFormatR32Float && tex->mtl.width >= 2048 && tex->mtl.width == tex->mtl.height;
    if (t.color && t.depth && (int)t.depth.sampleCount != t.samples) {
        complain("framebuffer mixes a %d-sample colour target with a %d-sample depth buffer; depth is ignored", t.samples,
                 (int)t.depth.sampleCount);
        t.depth = nil;
    }
    return t.color || t.depth;
}

// Opens a render pass on the target. clear_mask selects what the pass clears
// as it starts; everything else is loaded.
void start_pass(const Target &t, GLbitfield clear_mask)
{
    end_pass();
    // The engine has drawn the scene and is moving to another target: this is
    // where post-processing starts, and the point to darken the scene first.
    // (A scene is told from other uses of its buffer by the number of draws; a previewed model is a scene of few.)
    if ((s_ao || s_contact_shadows || s_light_shafts || s_motion_blur || s_fxaa || s_upscale || s_sharpen > 0) && s_last.scene && s_last.color != t.color && (s_last.draws > 100 || s_preview_in_scene) &&
        s_last.occluded_in_frame != s_frame_serial) {
        s_last.occluded_in_frame = s_frame_serial;
        if (s_ao || s_contact_shadows || s_light_shafts) apply_ambient_occlusion();
        if (s_motion_blur) apply_motion_blur();
        if (s_fxaa) apply_fxaa(); // last: edges as they will be shown
        id<MTLTexture> shown = nil;
        if (@available(macOS 13.0, *)) shown = s_upscale ? apply_upscale() : nil;
        if (s_sharpen > 0) apply_sharpen(shown ? shown : s_last.color);
    }
    MTLRenderPassDescriptor *d = [MTLRenderPassDescriptor renderPassDescriptor];
    if (t.color) {
        bool clear = clear_mask & GL_COLOR_BUFFER_BIT;
        d.colorAttachments[0].loadAction = clear ? MTLLoadActionClear : MTLLoadActionLoad;
        if (t.ms_color) { // keep the samples for later passes on this target, and resolve for anything that reads it
            d.colorAttachments[0].texture = t.ms_color;
            d.colorAttachments[0].resolveTexture = t.color;
            d.colorAttachments[0].storeAction = MTLStoreActionStoreAndMultisampleResolve;
        } else {
            d.colorAttachments[0].texture = t.color;
            d.colorAttachments[0].storeAction = MTLStoreActionStore;
        }
        d.colorAttachments[0].clearColor = s_debug_clear && t.screen ? MTLClearColorMake(0.10, 0.30, 0.60, 1.0) : s_clear_color;
    }
    if (t.depth) {
        bool clear = (clear_mask & GL_DEPTH_BUFFER_BIT) && s_depth_mask;
        d.depthAttachment.texture = t.depth;
        d.depthAttachment.loadAction = clear ? MTLLoadActionClear : MTLLoadActionLoad;
        d.depthAttachment.storeAction = MTLStoreActionStore;
        d.depthAttachment.clearDepth = s_reverse_depth ? 0.0 : 1.0; // the far plane
    }
    s_pass = [command_buffer() renderCommandEncoderWithDescriptor:d];
    reset_pass_state();
    s_pass_color = t.color;
    s_pass_depth = t.depth;
    s_pass_screen = t.screen;
    s_pass_scale = t.scale;
    s_pass_samples = t.samples;
    if (t.scene) s_scene_frame = s_frame_serial;
    s_pass_scene = t.scene;
    s_pass_prepass = t.prepass;
    s_pass_shadow_map = t.shadow_map;
    if (s_last.color != t.color) s_last.draws = 0; // a clear restarts the pass on the same target; keep counting
    s_last.color = t.color;
    s_last.ms_color = t.ms_color;
    s_last.depth = t.depth;
    s_last.scene = t.scene;
    s_last.samples = t.samples;
}

bool ensure_pass()
{
    Target t;
    if (!resolve_target(t)) return false;
    if (!s_pass || s_pass_color != t.color || s_pass_depth != t.depth) start_pass(t, 0);
    return true;
}

// ---- uniform storage -----------------------------------------------------------------

// Copies bytes into this frame's uniform storage and returns where they went.
id<MTLBuffer> push_uniforms(const std::vector<uint8_t> &data, NSUInteger *offset)
{
    size_t need = (data.size() + 255) & ~(size_t)255;
    if (!s_frame_chunks) s_frame_chunks = [NSMutableArray array];
    if (s_frame_chunks.count == 0 || s_chunk_used + need > UNIFORM_CHUNK) {
        id<MTLBuffer> chunk;
        {
            std::lock_guard<std::mutex> lock(s_chunk_mu);
            if (!s_free_chunks.empty()) {
                chunk = s_free_chunks.back();
                s_free_chunks.pop_back();
            }
        }
        if (!chunk) chunk = [s_device newBufferWithLength:UNIFORM_CHUNK options:MTLResourceStorageModeShared];
        [s_frame_chunks addObject:chunk];
        s_chunk_used = 0;
    }
    id<MTLBuffer> chunk = s_frame_chunks.lastObject;
    memcpy((uint8_t *)chunk.contents + s_chunk_used, data.data(), data.size());
    *offset = s_chunk_used;
    s_chunk_used += need;
    return chunk;
}

// ---- textures and samplers ---------------------------------------------------------------

struct FormatInfo {
    MTLPixelFormat format = MTLPixelFormatInvalid;
    int bytes_per_pixel = 0;      // of the data the game supplies; 0 for compressed
    int block_bytes = 0;          // compressed formats
    bool expand_rgb = false;      // 24-bit RGB data has to be widened to RGBA
    MTLTextureSwizzleChannels swizzle = MTLTextureSwizzleChannelsDefault;
};

FormatInfo pick_format(GLint internalformat, GLenum format, GLenum type)
{
    FormatInfo f;
    const MTLTextureSwizzleChannels luminance = {MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleOne};
    switch (internalformat) { // sized formats, used for render targets
    case GL_RG16F: f.format = MTLPixelFormatRG16Float; f.bytes_per_pixel = 4; return f;
    case GL_RGBA16F: f.format = MTLPixelFormatRGBA16Float; f.bytes_per_pixel = 8; return f;
    case GL_RGBA32F: f.format = MTLPixelFormatRGBA32Float; f.bytes_per_pixel = 16; return f;
    case GL_R8: f.format = MTLPixelFormatR8Unorm; f.bytes_per_pixel = 1; return f;
    case GL_R16F: f.format = MTLPixelFormatR16Float; f.bytes_per_pixel = 2; return f;
    case GL_R32F: f.format = MTLPixelFormatR32Float; f.bytes_per_pixel = 4; return f;
    case GL_DEPTH_COMPONENT:
    case GL_DEPTH_COMPONENT16:
    case GL_DEPTH_COMPONENT24:
    case GL_DEPTH_COMPONENT32: f.format = MTLPixelFormatDepth32Float; f.bytes_per_pixel = 4; return f;
    }
    if (type == GL_UNSIGNED_BYTE) {
        switch (format) {
        case GL_BGRA: f.format = MTLPixelFormatBGRA8Unorm; f.bytes_per_pixel = 4; return f;
        case GL_RGBA: f.format = MTLPixelFormatRGBA8Unorm; f.bytes_per_pixel = 4; return f;
        case GL_RGB: f.format = MTLPixelFormatRGBA8Unorm; f.bytes_per_pixel = 3; f.expand_rgb = true; return f;
        case GL_RED: f.format = MTLPixelFormatR8Unorm; f.bytes_per_pixel = 1; return f;
        case GL_ALPHA: f.format = MTLPixelFormatA8Unorm; f.bytes_per_pixel = 1; return f;
        case GL_LUMINANCE: f.format = MTLPixelFormatR8Unorm; f.bytes_per_pixel = 1; f.swizzle = luminance; return f;
        case GL_LUMINANCE_ALPHA:
            f.format = MTLPixelFormatRG8Unorm;
            f.bytes_per_pixel = 2;
            f.swizzle = {MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleGreen};
            return f;
        }
    }
    if (type == GL_UNSIGNED_SHORT && format == GL_LUMINANCE) {
        f.format = MTLPixelFormatR16Unorm;
        f.bytes_per_pixel = 2;
        f.swizzle = luminance;
    }
    return f;
}

FormatInfo pick_compressed_format(GLenum internalformat)
{
    FormatInfo f;
    switch (internalformat) {
    case GL_COMPRESSED_RGB_S3TC_DXT1_EXT:
    case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT: f.format = MTLPixelFormatBC1_RGBA; f.block_bytes = 8; break;
    case GL_COMPRESSED_RGBA_S3TC_DXT3_EXT: f.format = MTLPixelFormatBC2_RGBA; f.block_bytes = 16; break;
    case GL_COMPRESSED_RGBA_S3TC_DXT5_EXT: f.format = MTLPixelFormatBC3_RGBA; f.block_bytes = 16; break;
    }
    return f;
}

int mip_count(int w, int h)
{
    int n = 1;
    while (w > 1 || h > 1) {
        w = std::max(1, w / 2);
        h = std::max(1, h / 2);
        n++;
    }
    return n;
}

// Makes sure the texture has storage whose level `level` is w x h in the given
// format, recreating it when the game respecifies it differently.
bool ensure_storage(Texture &t, const FormatInfo &f, int level, int w, int h)
{
    if (t.mtl && t.format == f.format && level < (int)t.mtl.mipmapLevelCount &&
        std::max(1, (int)t.mtl.width >> level) == w && std::max(1, (int)t.mtl.height >> level) == h)
        return true;
    if (t.mtl && level != 0) {
        complain("mip level %d (%dx%d) does not fit its %dx%d texture", level, w, h, (int)t.mtl.width, (int)t.mtl.height);
        return false;
    }
    int base_w = w << level, base_h = h << level;
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f.format
                                                                                 width:base_w
                                                                                height:base_h
                                                                             mipmapped:NO];
    d.mipmapLevelCount = mip_count(base_w, base_h);
    // Any uncompressed texture may later be attached to a framebuffer, except
    // the luminance ones: Metal does not allow rendering to a swizzled texture.
    const MTLTextureSwizzleChannels none = MTLTextureSwizzleChannelsDefault;
    bool swizzled = memcmp(&f.swizzle, &none, sizeof none) != 0;
    d.usage = (f.block_bytes || swizzled) ? MTLTextureUsageShaderRead : MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    if (s_shader_options.srgb_textures && !swizzled) d.usage |= MTLTextureUsagePixelFormatView;
    d.swizzle = f.swizzle;
    t.mtl = [s_device newTextureWithDescriptor:d];
    t.format = f.format;
    t.top_level = -1;
    t.sampler = nil;
    t.ms = nil;
    t.srgb_view = nil;
    return t.mtl != nil;
}

Texture *bound_texture()
{
    return s_unit < MAX_UNITS ? find(s_textures, s_bound_texture[s_unit]) : nullptr;
}

MTLSamplerAddressMode address_mode(GLenum wrap)
{
    switch (wrap) {
    case GL_REPEAT: return MTLSamplerAddressModeRepeat;
    case GL_MIRRORED_REPEAT: return MTLSamplerAddressModeMirrorRepeat;
    case GL_CLAMP_TO_BORDER: return MTLSamplerAddressModeClampToBorderColor;
    default: return MTLSamplerAddressModeClampToEdge;
    }
}

// The texture seen through the sRGB twin of its format, or nil if it has none.
id<MTLTexture> srgb_view_for(Texture &t)
{
    if (t.srgb_view) return t.srgb_view;
    MTLPixelFormat format;
    switch (t.format) {
    case MTLPixelFormatBC1_RGBA: format = MTLPixelFormatBC1_RGBA_sRGB; break;
    case MTLPixelFormatBC2_RGBA: format = MTLPixelFormatBC2_RGBA_sRGB; break;
    case MTLPixelFormatBC3_RGBA: format = MTLPixelFormatBC3_RGBA_sRGB; break;
    case MTLPixelFormatRGBA8Unorm: format = MTLPixelFormatRGBA8Unorm_sRGB; break;
    case MTLPixelFormatBGRA8Unorm: format = MTLPixelFormatBGRA8Unorm_sRGB; break;
    default: return nil;
    }
    if (!(t.mtl.usage & MTLTextureUsagePixelFormatView)) return nil;
    return t.srgb_view = [t.mtl newTextureViewWithPixelFormat:format];
}

id<MTLSamplerState> sampler_for(Texture &t)
{
    if (t.sampler) return t.sampler;
    bool mipmapped = t.min_filter >= GL_NEAREST_MIPMAP_NEAREST && t.min_filter <= GL_LINEAR_MIPMAP_LINEAR;
    int top = mipmapped ? std::max(0, std::min(t.max_level, t.top_level)) : 0;
    int aniso = (int)std::max(1.0f, std::min(16.0f, t.anisotropy));
    uint64_t key = (uint64_t)t.min_filter | (uint64_t)t.mag_filter << 16 | (uint64_t)(t.wrap_s & 0xff) << 32 |
                   (uint64_t)(t.wrap_s >> 8 & 0xf) << 40 | (uint64_t)(t.wrap_t & 0xff) << 44 | (uint64_t)(t.wrap_t >> 8 & 0xf) << 52 |
                   (uint64_t)top << 56 | (uint64_t)(aniso > 1) << 63;
    id<MTLSamplerState> __strong &cached = s_samplers[key];
    if (!cached) {
        MTLSamplerDescriptor *d = [MTLSamplerDescriptor new];
        bool min_linear = t.min_filter == GL_LINEAR || t.min_filter == GL_LINEAR_MIPMAP_NEAREST || t.min_filter == GL_LINEAR_MIPMAP_LINEAR;
        d.minFilter = min_linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
        d.magFilter = t.mag_filter == GL_LINEAR ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
        d.mipFilter = !mipmapped ? MTLSamplerMipFilterNotMipmapped
                                 : (t.min_filter == GL_NEAREST_MIPMAP_LINEAR || t.min_filter == GL_LINEAR_MIPMAP_LINEAR)
                                       ? MTLSamplerMipFilterLinear
                                       : MTLSamplerMipFilterNearest;
        d.sAddressMode = address_mode(t.wrap_s);
        d.tAddressMode = address_mode(t.wrap_t);
        d.borderColor = MTLSamplerBorderColorTransparentBlack;
        d.lodMaxClamp = top;
        d.maxAnisotropy = min_linear ? aniso : 1;
        if (s_lod_bias != 0) {
            if (@available(macOS 26.0, *)) d.lodBias = s_lod_bias;
        }
        cached = [s_device newSamplerStateWithDescriptor:d];
    }
    return t.sampler = cached;
}

// ---- programs -----------------------------------------------------------------------------

void link_program(GLuint program_id, Program &p)
{
    p.linked = false;
    p.pipelines.clear();
    p.locations.clear();
    p.location_by_name.clear();
    p.yflip_offset = p.samples_offset = p.grade_offset = p.headroom_offset = p.limit_offset = p.dyn_window_offset = p.dyn_near_offset = p.dyn_bias_offset = -1;
    p.amb_offset[0] = p.amb_offset[1] = p.material_offset = p.pbr_offset = p.skin_offset[0] = p.skin_offset[1] = p.wind_offset = p.leaf_sun_offset = p.gaze_offset = p.fur_offset = p.hair_colour_offset = p.horse_eye_offset = p.shell_offset[0] = p.shell_offset[1] = p.drift_offset = -1;
    p.u_wvp = p.u_world = p.u_bones = nullptr;
    p.alpha_texture_location = -1;
    s_last_key_program = 0; // the pipeline remembered from the previous draw may be this program's
    std::string vs, fs;
    for (auto &sh : p.shaders) (sh->type == GL_VERTEX_SHADER ? vs : fs) = sh->source;
    bool from_cache = false;
    CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
    p.xl = xl_translate_cached(vs, fs, p.attribs, s_shader_options, &from_cache);
    s_shader_stats.translate_ms += (CFAbsoluteTimeGetCurrent() - started) * 1000;
    s_shader_stats.links++;
    s_shader_stats.cache_hits += from_cache;
    if (!p.xl.ok) {
        wb_note("  !! metal: program %u did not translate:\n%s", program_id, p.xl.log.c_str());
        return;
    }
    const XlStage *stages[2] = {&p.xl.vs, &p.xl.fs};
    p.functions_ready = false;
    for (int s = 0; s < 2; s++) {
        // Metal compiles the source in the background; the program waits for it
        // only when it first draws (see finish_compile). The engine links every
        // program at start-up and uses a few of them right away.
        std::shared_ptr<StageCompile> &compile = s_stage_compiles[stages[s]->msl];
        if (!compile) {
            compile = std::make_shared<StageCompile>();
            compile->group = dispatch_group_create();
            dispatch_group_enter(compile->group);
            std::shared_ptr<StageCompile> result = compile;
            NSString *entry = @(stages[s]->entry.c_str());
            [s_device newLibraryWithSource:@(stages[s]->msl.c_str())
                                   options:nil
                         completionHandler:^(id<MTLLibrary> library, NSError *error) {
                           if (library) result->function = [library newFunctionWithName:entry];
                           if (!result->function) result->error = error ? error.localizedDescription.UTF8String : "entry point not found";
                           dispatch_group_leave(result->group);
                         }];
            s_shader_stats.stage_compiles++;
        }
        p.compile[s] = compile;
        p.uniform_data[s].assign((size_t)stages[s]->uniform_buffer_size, 0);

        auto location = [&](const std::string &name) -> UniformLocation & {
            auto it = p.location_by_name.find(name);
            if (it == p.location_by_name.end()) {
                it = p.location_by_name.emplace(name, (GLint)p.locations.size()).first;
                p.locations.emplace_back();
                p.locations.back().name = name;
                p.locations.back().is_sun_dir = name == "vSunDir";
                p.locations.back().is_sun_color = name == "vSunColor";
                p.locations.back().is_sky = name == k_xl_sky_sampler;
                p.locations.back().is_dyn_map = name == k_xl_dyn_sampler;
                p.locations.back().is_pbr_map = name == k_xl_pbr_sampler;
                p.locations.back().is_fur_map = name == "wbm_fur_map";
                p.locations.back().is_hair_map = name == "wbm_hair_map";
                p.locations.back().is_sun_matrix = name == "matSunViewProj";
            }
            return p.locations[(size_t)it->second];
        };
        for (const XlUniform &u : stages[s]->uniforms) {
            if (u.name == k_xl_yflip_uniform) p.yflip_offset = u.offset;
            else if (u.name == k_xl_samples_uniform) p.samples_offset = u.offset;
            else if (u.name == k_xl_headroom_uniform) p.headroom_offset = u.offset;
            else if (u.name == "wbm_grade" && s == 1) p.grade_offset = u.offset;
            else if (u.name == k_xl_limit_uniform) p.limit_offset = u.offset;
            else if (u.name == k_xl_dyn_window_uniform) p.dyn_window_offset = u.offset;
            else if (u.name == k_xl_dyn_near_uniform) p.dyn_near_offset = u.offset;
            else if (u.name == k_xl_dyn_bias_uniform) p.dyn_bias_offset = u.offset;
            else if (u.name == k_xl_ambient_uniform) p.amb_offset[s] = u.offset;
            else if (u.name == k_xl_material_uniform) p.material_offset = u.offset;
            else if (u.name == k_xl_pbr_uniform) p.pbr_offset = u.offset;
            else if (u.name == k_xl_skin_uniform) p.skin_offset[s] = u.offset;
            else if (u.name == k_xl_wind_uniform) p.wind_offset = u.offset;
            else if (u.name == "wbm_leaf_sun") p.leaf_sun_offset = u.offset;
            else if (u.name == "wbm_gaze") p.gaze_offset = u.offset;
            else if (u.name == "wbm_fur") p.fur_offset = u.offset;
            else if (u.name == "wbm_horse_eye") p.horse_eye_offset = u.offset;
            else if (u.name == "wbm_hair_colour") p.hair_colour_offset = u.offset;
            else if (u.name == "wbm_shell") p.shell_offset[s] = u.offset;
            else if (u.name == "wbm_drift" && s == 0) p.drift_offset = u.offset;
            else location(u.name).uniform[s] = &u;
        }
        for (const XlSampler &smp : stages[s]->samplers) location(smp.name).texture_index[s] = smp.texture_index;
        for (const std::string &name : stages[s]->srgb_samplers) location(name).srgb = true;
        p.uniform_dirty[s] = true;
        p.pushed_frame[s] = 0;
    }
    auto vertex_uniform = [&](const char *name) -> const XlUniform * {
        auto it = p.location_by_name.find(name);
        return it == p.location_by_name.end() ? nullptr : p.locations[(size_t)it->second].uniform[0];
    };
    p.u_wvp = vertex_uniform("matWorldViewProj");
    p.u_world = vertex_uniform("matWorld");
    p.u_bones = vertex_uniform("matWorldArray");
    auto diffuse = p.location_by_name.find("diffuse_texture");
    if (diffuse != p.location_by_name.end() && p.locations[(size_t)diffuse->second].texture_index[1] >= 0) p.alpha_texture_location = diffuse->second;
    p.is_blob_shadow = fs.find("0.0234375") != std::string::npos;
    p.input_mask = 0;
    for (int loc : p.xl.vs.input_locations)
        if (loc >= 0 && loc < MAX_ATTRIBS) p.input_mask |= 1u << loc;
    p.linked = true;
}

// Waits for the program's two shaders to finish compiling. Returns false, and
// retires the program, if either failed.
bool finish_compile(Program &p)
{
    CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
    for (int s = 0; s < 2; s++) {
        StageCompile &c = *p.compile[s];
        dispatch_group_wait(c.group, DISPATCH_TIME_FOREVER);
        if (!c.function) {
            complain("program %u %s shader did not compile: %s", s_program, s ? "fragment" : "vertex", c.error.c_str());
            p.linked = false;
            return false;
        }
        p.function[s] = c.function;
    }
    s_shader_stats.wait_ms += (CFAbsoluteTimeGetCurrent() - started) * 1000;
    return p.functions_ready = true;
}

// Writes `count` values of `components` 32-bit words each into the uniform's
// place in both stages' buffers, converting between int and float as needed.
void set_uniform(GLint loc, const void *data, int count, int components, bool data_is_int)
{
    Program *p = find(s_programs, s_program);
    if (!p || loc < 0 || (size_t)loc >= p->locations.size()) return;
    UniformLocation &l = p->locations[(size_t)loc];
    if ((l.texture_index[0] >= 0 || l.texture_index[1] >= 0) && data_is_int) l.unit = *(const GLint *)data;
    if (s_dyn_shadows && l.is_sun_matrix && components == 16 && count >= 1) {
        memcpy(s_sun_matrix, data, sizeof s_sun_matrix); // the same for every program in a scene
        s_sun_matrix_valid = true;
    }
    if ((s_contact_shadows || s_light_shafts || s_sky_ambient || s_foliage_backlight > 0) && (l.is_sun_dir || l.is_sun_color) && components == 4 && count >= 1) {
        // The UI and other passes set these too; only the values in use while the
        // scene buffer is bound describe the scene's sun.
        Target t;
        if (s_fbo != 0 && resolve_target(t) && t.scene) {
            if (l.is_sun_dir) {
                memcpy(s_sun_dir, data, sizeof s_sun_dir);
                s_sun_valid = true;
            } else {
                memcpy(s_sun_color, data, sizeof s_sun_color);
            }
        }
    }
    for (int s = 0; s < 2; s++) {
        const XlUniform *u = l.uniform[s];
        if (!u) continue;
        int words = std::min(components, u->components * u->columns);
        int n = std::min(count, std::max(1, u->array_size));
        int stride = u->array_size ? u->array_stride : words * 4;
        uint8_t *dst = p->uniform_data[s].data() + u->offset;
        const uint32_t *src = (const uint32_t *)data;
        for (int i = 0; i < n; i++, dst += stride, src += components) {
            uint32_t converted;
            const void *value = src;
            size_t bytes = (size_t)words * 4;
            if (u->is_int != data_is_int) {
                if (data_is_int) {
                    float v = (float)*(const GLint *)src;
                    memcpy(&converted, &v, 4);
                } else {
                    GLint v = (GLint) * (const GLfloat *)src;
                    memcpy(&converted, &v, 4);
                }
                value = &converted;
                bytes = 4;
            }
            if (memcmp(dst, value, bytes) != 0) { // the engine re-sends unchanged values for every draw
                memcpy(dst, value, bytes);
                p->uniform_dirty[s] = true;
            }
        }
    }
}

// ---- pipeline and fixed-function state ------------------------------------------------------

MTLVertexFormat vertex_format(const Attrib &a)
{
    if (a.type == GL_FLOAT && a.size >= 1 && a.size <= 4) return (MTLVertexFormat)(MTLVertexFormatFloat + a.size - 1);
    if (a.type == GL_UNSIGNED_BYTE && a.size == 4) return a.normalized ? MTLVertexFormatUChar4Normalized : MTLVertexFormatUChar4;
    if (a.type == GL_SHORT && a.size == 2) return a.normalized ? MTLVertexFormatShort2Normalized : MTLVertexFormatShort2;
    if (a.type == GL_SHORT && a.size == 4) return a.normalized ? MTLVertexFormatShort4Normalized : MTLVertexFormatShort4;
    return MTLVertexFormatInvalid;
}

MTLBlendFactor blend_factor(GLenum f)
{
    switch (f) {
    case GL_ZERO: return MTLBlendFactorZero;
    case GL_ONE: return MTLBlendFactorOne;
    case GL_SRC_COLOR: return MTLBlendFactorSourceColor;
    case GL_ONE_MINUS_SRC_COLOR: return MTLBlendFactorOneMinusSourceColor;
    case GL_SRC_ALPHA: return MTLBlendFactorSourceAlpha;
    case GL_ONE_MINUS_SRC_ALPHA: return MTLBlendFactorOneMinusSourceAlpha;
    case GL_DST_ALPHA: return MTLBlendFactorDestinationAlpha;
    case GL_ONE_MINUS_DST_ALPHA: return MTLBlendFactorOneMinusDestinationAlpha;
    case GL_DST_COLOR: return MTLBlendFactorDestinationColor;
    case GL_ONE_MINUS_DST_COLOR: return MTLBlendFactorOneMinusDestinationColor;
    case GL_SRC_ALPHA_SATURATE: return MTLBlendFactorSourceAlphaSaturated;
    default: complain("unsupported blend factor %s", E_blend(f)); return MTLBlendFactorOne;
    }
}

MTLBlendOperation blend_operation(GLenum eq)
{
    switch (eq) {
    case GL_FUNC_SUBTRACT: return MTLBlendOperationSubtract;
    case GL_FUNC_REVERSE_SUBTRACT: return MTLBlendOperationReverseSubtract;
    case GL_MIN: return MTLBlendOperationMin;
    case GL_MAX: return MTLBlendOperationMax;
    default: return MTLBlendOperationAdd;
    }
}

struct VertexBinding {
    GLuint buffer[MAX_VERTEX_SLOTS];
    uintptr_t base[MAX_VERTEX_SLOTS];
    int slots = 0;
};
VertexBinding s_draw_binding; // of the draw most recently prepared

// Describes the current vertex arrays for the attributes the program reads.
// Attributes that share a buffer and stride share a Metal buffer slot.
bool build_vertex_layout(const Program &p, PipelineKey &key, VertexBinding &binding)
{
    for (int i = 0; i < MAX_ATTRIBS; i++) {
        if (!(p.input_mask & (1u << i))) continue;
        const Attrib &a = s_attribs[i];
        if (!a.enabled) { // OpenGL supplies a constant; so does the constant buffer
            key.format[i] = MTLVertexFormatFloat4;
            key.slot[i] = CONSTANT_SLOT;
            continue;
        }
        MTLVertexFormat format = vertex_format(a);
        if (format == MTLVertexFormatInvalid) {
            complain("unsupported vertex attribute: size %d, type %s", a.size, E(a.type));
            return false;
        }
        if (!a.buffer) {
            complain("vertex attribute %d comes from client memory, which is not supported", i);
            return false;
        }
        int slot = 0;
        while (slot < binding.slots && !(binding.buffer[slot] == a.buffer && key.stride[slot] == a.stride)) slot++;
        if (slot == binding.slots) {
            if (slot == MAX_VERTEX_SLOTS) {
                complain("more than %d vertex buffers in one draw", MAX_VERTEX_SLOTS);
                return false;
            }
            binding.buffer[slot] = a.buffer;
            binding.base[slot] = a.offset;
            key.stride[slot] = (uint16_t)a.stride;
            binding.slots++;
        }
        binding.base[slot] = std::min(binding.base[slot], a.offset);
        key.format[i] = (uint8_t)format;
        key.slot[i] = (uint8_t)slot;
    }
    for (int i = 0; i < MAX_ATTRIBS; i++)
        if (key.format[i] && key.slot[i] != CONSTANT_SLOT) key.offset[i] = (uint16_t)(s_attribs[i].offset - binding.base[key.slot[i]]);
    return true;
}

id<MTLRenderPipelineState> pipeline_for(Program &p, const PipelineKey &key)
{
    auto it = p.pipelines.find(key);
    if (it != p.pipelines.end()) return it->second;

    MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = p.function[0];
    d.fragmentFunction = p.function[1];
    MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
    bool uses_constant = false;
    for (int i = 0; i < MAX_ATTRIBS; i++) {
        if (!key.format[i]) continue;
        vd.attributes[i].format = (MTLVertexFormat)key.format[i];
        vd.attributes[i].offset = key.offset[i];
        vd.attributes[i].bufferIndex = key.slot[i] == CONSTANT_SLOT ? CONSTANT_SLOT : VERTEX_SLOT_BASE + key.slot[i];
        uses_constant |= key.slot[i] == CONSTANT_SLOT;
    }
    for (int s = 0; s < MAX_VERTEX_SLOTS; s++) {
        if (!key.stride[s]) continue;
        vd.layouts[VERTEX_SLOT_BASE + s].stride = key.stride[s];
        vd.layouts[VERTEX_SLOT_BASE + s].stepFunction = MTLVertexStepFunctionPerVertex;
    }
    if (uses_constant) {
        vd.layouts[CONSTANT_SLOT].stride = 16;
        vd.layouts[CONSTANT_SLOT].stepFunction = MTLVertexStepFunctionConstant;
        vd.layouts[CONSTANT_SLOT].stepRate = 0;
    }
    d.vertexDescriptor = vd;
    MTLRenderPipelineColorAttachmentDescriptor *c = d.colorAttachments[0];
    c.pixelFormat = (MTLPixelFormat)key.color_format;
    if (key.blend) {
        c.blendingEnabled = YES;
        c.sourceRGBBlendFactor = c.sourceAlphaBlendFactor = blend_factor(key.blend_src);
        c.destinationRGBBlendFactor = c.destinationAlphaBlendFactor = blend_factor(key.blend_dst);
        c.rgbBlendOperation = c.alphaBlendOperation = blend_operation(key.blend_eq);
    }
    d.depthAttachmentPixelFormat = (MTLPixelFormat)key.depth_format;
    d.rasterSampleCount = key.samples;

    NSError *error = nil;
    id<MTLRenderPipelineState> state = [s_device newRenderPipelineStateWithDescriptor:d error:&error];
    if (!state) complain("pipeline for program %u failed: %s", s_program, error.localizedDescription.UTF8String);
    p.pipelines[key] = state; // nil is cached too, so a failure is reported and attempted once
    return state;
}

id<MTLDepthStencilState> depth_state(bool has_depth)
{
    bool test = s_depth_test && has_depth;
    if (!test) { // OpenGL neither tests nor writes depth while the test is disabled
        if (!s_depth_off) s_depth_off = [s_device newDepthStencilStateWithDescriptor:[MTLDepthStencilDescriptor new]];
        return s_depth_off;
    }
    unsigned func = (s_depth_func - GL_NEVER) & 7;
    id<MTLDepthStencilState> __strong &state = s_depth_states[1][func][s_depth_mask];
    if (!state) {
        static const MTLCompareFunction compare[8] = {MTLCompareFunctionNever,        MTLCompareFunctionLess,
                                                      MTLCompareFunctionEqual,        MTLCompareFunctionLessEqual,
                                                      MTLCompareFunctionGreater,      MTLCompareFunctionNotEqual,
                                                      MTLCompareFunctionGreaterEqual, MTLCompareFunctionAlways};
        MTLDepthStencilDescriptor *d = [MTLDepthStencilDescriptor new];
        // With reversed depth nearer is larger, so the ordering comparisons swap.
        static const unsigned reversed[8] = {0, 4, 2, 6, 1, 5, 3, 7};
        d.depthCompareFunction = compare[s_reverse_depth ? reversed[func] : func];
        d.depthWriteEnabled = s_depth_mask;
        state = [s_device newDepthStencilStateWithDescriptor:d];
    }
    return state;
}

// ---- ambient occlusion ---------------------------------------------------------------

// Three full-screen passes written directly in Metal's shading language.
//
// ao_linearize, at half resolution: turn the depth buffer into distances. The
// depth buffer is multisampled and at Retina size; reading it dozens of times
// per pixel is what made the first version of this halve the frame rate.
//
// ao_compute, at half resolution: for each pixel, rebuild its position in view
// space, estimate the surface normal from neighbouring distances, and look at
// 16 nearby points on a rotated spiral. Points that rise
// above the surface's tangent plane within the radius occlude it; the closer,
// the more. This is the "Alchemy" estimator, a horizon-based method that needs
// only depth.
//
// ao_blur, at half resolution: blur the result over 3x3 texels, ignoring
// texels at a different depth so that occlusion does not leak across
// silhouettes.
//
// ao_apply, at full resolution: multiply the scene's colour by it.
const char k_ao_source[] = R"(
#include <metal_stdlib>
using namespace metal;

struct Params {
    float a, b;        // clip z = a * w + b, from the camera's projection
    float p00, p11;    // projection scale in x and y
    float2 depth_size; // of the depth buffer, in pixels
    float2 ao_size;    // of the half-resolution textures
    float radius, strength, reversed, fade_start;
    float4 sun;        // xyz: direction to the sun in view space; w: distance within which contact shadows are left out, or 0
    float shadow_strength, shadow_length;
    float2 shaft_size; // of the light-shaft textures, in pixels
    float4 shaft;      // xy: the sun's position on screen (0..1), z: how visible it is, w: strength
};

#if WBM_MSAA
typedef depth2d_ms<float> DepthTexture;
#else
typedef depth2d<float> DepthTexture;
#endif

// Distance in front of the camera for a pixel of the depth buffer; a very
// large value for the sky.
//
// With multisampling a pixel holds several depths. Along cut-out edges (hair,
// leaves, grass) they differ from sample to sample, and taking any single one
// makes thin shapes flicker in and out of the occlusion. The farthest is used:
// a pixel only counts as covered by a thin shape once all of it is.
static float view_z(DepthTexture t, int2 c, Params p)
{
    c = clamp(c, int2(0), int2(p.depth_size) - 1);
#if WBM_MSAA
    float d = t.read(uint2(c), 0);
    for (uint i = 1; i < t.get_num_samples(); i++) {
        float e = t.read(uint2(c), i);
        d = p.reversed > 0.5 ? min(d, e) : max(d, e);
    }
#else
    float d = t.read(uint2(c), 0);
#endif
    bool sky = p.reversed > 0.5 ? d <= 0.0 : d >= 1.0;
    if (sky) return 1.0e6;
    float ndc = p.reversed > 0.5 ? 1.0 - 2.0 * d : 2.0 * d - 1.0;
    return p.b / (ndc - p.a);
}

struct VertexOut {
    float4 position [[position]];
};

vertex VertexOut ao_vertex(uint id [[vertex_id]])
{
    float2 corner = float2((id << 1) & 2, id & 2); // one triangle that covers the screen
    VertexOut out;
    out.position = float4(corner * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

// Pass 1, at the occlusion resolution (half the depth buffer's, or all of it
// with WBM_AO_FULL): distances, so that the passes after it read a small
// single-sample texture instead of the multisampled depth buffer.
fragment float4 ao_linearize(VertexOut in [[stage_in]], DepthTexture depth [[texture(0)]], constant Params &p [[buffer(0)]])
{
    return float4(view_z(depth, int2(in.position.xy * p.depth_size / p.ao_size), p));
}

static float load_z(texture2d<float> z, int2 c, Params p)
{
    return z.read(uint2(clamp(c, int2(0), int2(p.ao_size) - 1))).x;
}

static float3 view_position(texture2d<float> z, int2 c, Params p)
{
    float w = load_z(z, c, p);
    float2 ndc = (float2(c) + 0.5) / p.ao_size * 2.0 - 1.0;
    return float3(ndc.x * w / p.p00, ndc.y * w / p.p11, w);
}

// Pass 2, half resolution: occlusion.
fragment float4 ao_compute(VertexOut in [[stage_in]], texture2d<float> z [[texture(0)]], constant Params &p [[buffer(0)]])
{
    int2 c = int2(in.position.xy);
    float3 P = view_position(z, c, p);
    if (P.z > p.fade_start * 2.0) return float4(1.0); // far away, or sky

    // The normal, from whichever neighbour on each axis is closer in depth, so
    // that it is not taken across an edge.
    float3 right = view_position(z, c + int2(1, 0), p) - P, left = P - view_position(z, c - int2(1, 0), p);
    float3 down = view_position(z, c + int2(0, 1), p) - P, up = P - view_position(z, c - int2(0, 1), p);
    float3 dx = abs(right.z) < abs(left.z) ? right : left;
    float3 dy = abs(down.z) < abs(up.z) ? down : up;
    float3 n = normalize(cross(dy, dx));
    if (dot(n, P) > 0.0) n = -n; // towards the camera

    float wanted_px = p.radius * abs(p.p00) / P.z * 0.5 * p.ao_size.x;
    float radius_px = clamp(wanted_px, 2.0, 96.0 * p.ao_size.x / p.depth_size.x);
    // Close to the camera the radius does not fit on screen and is cut short.
    // The distance it then really covers is what the weights below go by, and
    // the bias with it; kept at the full radius, a face seen close up had its
    // eye sockets and the sides of its nose blackened as if they were pits.
    float radius = p.radius * min(1.0, radius_px / max(wanted_px, 0.001));
    float noise = fract(52.9829189 * fract(dot(in.position.xy, float2(0.06711056, 0.00583715))));
    float bias = (0.02 + 0.004 * P.z) * radius / p.radius;
    float sum = 0.0;
    const int directions = 4, steps = 4;
    for (int i = 0; i < directions; i++) {
        float angle = (float(i) + noise) * (6.2831853 / float(directions));
        float2 dir = float2(cos(angle), sin(angle));
        for (int s = 0; s < steps; s++) {
            float t = (float(s) + fract(noise * 7.0 + 0.37 * float(i)) + 0.2) / float(steps);
            float3 v = view_position(z, c + int2(dir * radius_px * t), p) - P;
            float dd = dot(v, v);
            sum += max(0.0, dot(v, n) - bias) / (dd + 0.0001 * radius * radius) * saturate(1.0 - dd / (radius * radius));
        }
    }
    float occlusion = sum * radius * 2.0 / float(directions * steps);
    float ao = saturate(1.0 - p.strength * occlusion);

    // Contact shadows: walk a short way from the surface towards the sun and see
    // whether something on screen is in the way. This catches what the shadow
    // map is too coarse for: feet on the ground, gear against a body, grass.
    // Only surfaces facing the sun are tested, since the others are unlit
    // already, and a hit counts only if the blocker is thin enough to plausibly
    // be in front of the ray instead of merely nearer to the camera.
    if (p.shadow_strength > 0.0) {
        float3 L = normalize(p.sun.xyz);
        float facing = dot(n, L);
        if (facing > 0.05) {
            float3 origin = P + n * (0.015 + 0.004 * P.z);
            float hit = 0.0;
            const int shadow_steps = 10;
            for (int s = 0; s < shadow_steps; s++) {
                float t = (float(s) + noise) / float(shadow_steps) * p.shadow_length;
                float3 S = origin + L * t;
                if (S.z <= 0.05) break;
                float2 ndc = float2(S.x * p.p00, S.y * p.p11) / S.z;
                int2 sc = int2((ndc * 0.5 + 0.5) * p.ao_size);
                if (any(sc < int2(0)) || any(sc >= int2(p.ao_size))) break;
                float behind = S.z - z.read(uint2(sc)).x;
                if (behind > 0.02 + 0.003 * S.z && behind < 0.25 + 0.01 * S.z) {
                    hit = 1.0 - 0.5 * t / p.shadow_length;
                    break;
                }
            }
            // Where the per-frame shadow map's fine window reaches, things already
            // cast exact shadows; this estimate next to them reads as a second one.
            float beyond = p.sun.w > 0.0 ? smoothstep(p.sun.w * 0.6, p.sun.w, P.z) : 1.0;
            ao *= 1.0 - p.shadow_strength * hit * smoothstep(0.05, 0.3, facing) * beyond;
        }
    }
    return float4(mix(ao, 1.0, smoothstep(p.fade_start, p.fade_start * 2.0, P.z)));
}

// Pass 3, half resolution: blur over 3x3 texels, skipping those at a different
// distance so that occlusion does not leak across silhouettes.
fragment float4 ao_blur(VertexOut in [[stage_in]], texture2d<float> ao [[texture(0)]], texture2d<float> z [[texture(1)]],
                        constant Params &p [[buffer(0)]])
{
    int2 c = int2(in.position.xy);
    float zc = load_z(z, c, p);
    float total = 0.0, weight = 0.0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            int2 t = clamp(c + int2(i, j), int2(0), int2(p.ao_size) - 1);
            float w = exp(-abs(load_z(z, t, p) - zc) / (0.03 * zc + 0.02)) / (1.0 + abs(float(i)) + abs(float(j)));
            total += ao.read(uint2(t)).x * w;
            weight += w;
        }
    }
    return float4(weight > 0.0001 ? total / weight : 1.0);
}

// Pass 4, full resolution, blended onto the scene: one filtered sample. All the
// per-pixel work was done at half resolution; this pass touches four times as
// many pixels and has to stay cheap.
fragment float4 ao_apply(VertexOut in [[stage_in]], texture2d<float> ao [[texture(0)]], constant Params &p [[buffer(0)]])
{
    constexpr sampler smooth(filter::linear, address::clamp_to_edge);
    float a = ao.sample(smooth, in.position.xy / p.depth_size).x;
    return float4(a, a, a, 1.0);
}

// Light shafts ("god rays"). Sky near the sun is bright; anything in front of
// it is not. Smearing that bright-sky mask along lines towards the sun's
// position on screen makes streaks of light pass the edges of whatever blocks
// it, which is how shafts through trees and past towers look.
//
// shaft_mask, quarter resolution: the scene's colour where the pixel is sky,
// weighted towards the sun. shaft_blur: 40 steps from each pixel towards the
// sun, fading with distance, jittered per pixel. shaft_apply: added to the scene.
fragment float4 shaft_mask(VertexOut in [[stage_in]], texture2d<float> scene [[texture(0)]], texture2d<float> z [[texture(1)]],
                           constant Params &p [[buffer(0)]])
{
    constexpr sampler smooth(filter::linear, address::clamp_to_edge);
    float2 uv = in.position.xy / p.shaft_size;
    float sky = z.sample(smooth, uv).x > 1.0e5 ? 1.0 : 0.0;
    float2 ndc = uv * 2.0 - 1.0;
    float3 dir = normalize(float3(ndc.x / p.p00, ndc.y / p.p11, 1.0));
    float near_sun = pow(saturate(dot(dir, normalize(p.sun.xyz))), 24.0);
    return float4(scene.sample(smooth, uv).rgb * sky * (0.1 + 0.9 * near_sun), 1.0);
}

fragment float4 shaft_blur(VertexOut in [[stage_in]], texture2d<float> mask [[texture(0)]], constant Params &p [[buffer(0)]])
{
    constexpr sampler smooth(filter::linear, address::clamp_to_edge);
    const int samples = 40;
    float2 uv = in.position.xy / p.shaft_size;
    float2 step = (p.shaft.xy - uv) * (0.85 / float(samples));
    float noise = fract(52.9829189 * fract(dot(in.position.xy, float2(0.06711056, 0.00583715))));
    float2 q = uv + step * noise;
    float3 sum = float3(0.0);
    float weight = 1.0;
    for (int i = 0; i < samples; i++) {
        sum += mask.sample(smooth, q).rgb * weight;
        weight *= 0.965;
        q += step;
    }
    return float4(sum * (p.shaft.z * p.shaft.w / float(samples)), 1.0);
}

fragment float4 shaft_apply(VertexOut in [[stage_in]], texture2d<float> shafts [[texture(0)]], constant Params &p [[buffer(0)]])
{
    constexpr sampler smooth(filter::linear, address::clamp_to_edge);
    return float4(shafts.sample(smooth, in.position.xy / p.depth_size).rgb, 0.0);
}
)";

struct AOParams { // must match Params in k_ao_source
    float a, b, p00, p11;
    float depth_size[2], ao_size[2];
    float radius, strength, reversed, fade_start;
    float sun[4];
    float shadow_strength, shadow_length, shaft_size[2];
    float shaft[4];
};

id<MTLLibrary> s_ao_library;
id<MTLRenderPipelineState> s_ao_linearize_pipeline, s_ao_compute_pipeline, s_ao_blur_pipeline;
id<MTLTexture> s_ao_z_texture, s_ao_blurred_texture;
id<MTLRenderPipelineState> s_shaft_mask_pipeline, s_shaft_blur_pipeline, s_shaft_apply_pipeline;
id<MTLTexture> s_shaft_mask_texture, s_shaft_texture;
std::map<std::pair<int, int>, id<MTLRenderPipelineState>> s_ao_apply_pipelines; // by colour format and sample count
id<MTLTexture> s_ao_texture;

bool ao_setup()
{
    if (s_ao_library) return true;
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.preprocessorMacros = @{@"WBM_MSAA" : @(s_msaa > 1 ? 1 : 0)};
    NSError *error = nil;
    s_ao_library = [s_device newLibraryWithSource:@(k_ao_source) options:options error:&error];
    if (!s_ao_library) {
        complain("ambient occlusion shaders did not compile: %s", error.localizedDescription.UTF8String);
        return s_ao = s_contact_shadows = s_light_shafts = false;
    }
    MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = [s_ao_library newFunctionWithName:@"ao_vertex"];
    d.fragmentFunction = [s_ao_library newFunctionWithName:@"ao_linearize"];
    d.colorAttachments[0].pixelFormat = MTLPixelFormatR32Float;
    s_ao_linearize_pipeline = [s_device newRenderPipelineStateWithDescriptor:d error:&error];
    d.fragmentFunction = [s_ao_library newFunctionWithName:@"ao_compute"];
    d.colorAttachments[0].pixelFormat = MTLPixelFormatR8Unorm;
    s_ao_compute_pipeline = [s_device newRenderPipelineStateWithDescriptor:d error:&error];
    d.fragmentFunction = [s_ao_library newFunctionWithName:@"ao_blur"];
    s_ao_blur_pipeline = [s_device newRenderPipelineStateWithDescriptor:d error:&error];
    d.fragmentFunction = [s_ao_library newFunctionWithName:@"shaft_mask"];
    d.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA16Float;
    s_shaft_mask_pipeline = [s_device newRenderPipelineStateWithDescriptor:d error:&error];
    d.fragmentFunction = [s_ao_library newFunctionWithName:@"shaft_blur"];
    s_shaft_blur_pipeline = [s_device newRenderPipelineStateWithDescriptor:d error:&error];
    d.fragmentFunction = [s_ao_library newFunctionWithName:@"shaft_apply"];
    d.colorAttachments[0].blendingEnabled = YES; // added to the scene's colour; alpha untouched
    d.colorAttachments[0].sourceRGBBlendFactor = d.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOne;
    d.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorZero;
    d.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOne;
    s_shaft_apply_pipeline = [s_device newRenderPipelineStateWithDescriptor:d error:&error];
    if (!s_ao_linearize_pipeline || !s_ao_compute_pipeline || !s_ao_blur_pipeline || !s_shaft_mask_pipeline ||
        !s_shaft_blur_pipeline || !s_shaft_apply_pipeline) {
        complain("ambient occlusion pipeline failed: %s", error.localizedDescription.UTF8String);
        return s_ao = s_contact_shadows = s_light_shafts = false;
    }
    return true;
}

id<MTLRenderPipelineState> ao_apply_pipeline(MTLPixelFormat format, int samples)
{
    id<MTLRenderPipelineState> __strong &state = s_ao_apply_pipelines[{(int)format, samples}];
    if (!state) {
        MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
        d.vertexFunction = [s_ao_library newFunctionWithName:@"ao_vertex"];
        d.fragmentFunction = [s_ao_library newFunctionWithName:@"ao_apply"];
        d.rasterSampleCount = (NSUInteger)samples;
        MTLRenderPipelineColorAttachmentDescriptor *c = d.colorAttachments[0];
        c.pixelFormat = format;
        c.blendingEnabled = YES; // colour = colour already there * occlusion; alpha untouched
        c.sourceRGBBlendFactor = MTLBlendFactorDestinationColor;
        c.destinationRGBBlendFactor = MTLBlendFactorZero;
        c.sourceAlphaBlendFactor = MTLBlendFactorZero;
        c.destinationAlphaBlendFactor = MTLBlendFactorOne;
        NSError *error = nil;
        state = [s_device newRenderPipelineStateWithDescriptor:d error:&error];
        if (!state) complain("ambient occlusion apply pipeline failed: %s", error.localizedDescription.UTF8String);
    }
    return state;
}

// Runs the two passes on the scene buffer described by s_last. Called between
// render passes, with none open.
void apply_ambient_occlusion()
{
    // Without this frame's camera there is nothing sound to compute with; a
    // camera left over from another scene is worse than no effect.
    if (s_camera.projection_frame != s_frame_serial || !s_last.depth || !ao_setup()) return;
    AOParams params;
    params.a = (float)s_camera.a;
    params.b = (float)s_camera.b;
    params.p00 = (float)s_camera.p00;
    params.p11 = (float)s_camera.p11;
    // Distances are in the scene's own units, and scenes differ in scale: a
    // battlefield is life-size with a far plane around 1250, the world map a
    // miniature with one around 300. The radius, fade distance and shadow
    // length are scaled by the far plane so that they stay in proportion.
    double far = fabs(1.0 - s_camera.a) > 1e-9 ? fabs(s_camera.b / (1.0 - s_camera.a)) : 1250.0;
    float scale = (float)std::min(1.0, std::max(0.15, far / 1250.0));
    NSUInteger w = s_last.depth.width, h = s_last.depth.height;
    NSUInteger aw = s_ao_full ? w : (w + 1) / 2, ah = s_ao_full ? h : (h + 1) / 2;
    params.depth_size[0] = (float)w, params.depth_size[1] = (float)h;
    params.ao_size[0] = (float)aw, params.ao_size[1] = (float)ah;
    params.radius = s_ao_radius * scale;
    params.strength = s_ao ? s_ao_strength : 0.0f;
    // vSunDir points the way the light travels, so the direction to the sun is
    // its negative. No sun (night, interiors) means no contact shadows or shafts.
    memset(params.sun, 0, sizeof params.sun);
    memset(params.shaft, 0, sizeof params.shaft);
    params.shadow_strength = 0;
    params.shadow_length = s_contact_length * scale;
    NSUInteger sw = std::max<NSUInteger>(1, w / 4), sh = std::max<NSUInteger>(1, h / 4);
    params.shaft_size[0] = (float)sw, params.shaft_size[1] = (float)sh;
    float sun_brightness = s_sun_color[0] + s_sun_color[1] + s_sun_color[2];
    bool has_sun = s_sun_valid && sun_brightness > 0.05f && s_camera.view_frame == s_frame_serial;
    bool shafts = false;
    if (has_sun) {
        for (int r = 0; r < 3; r++)
            params.sun[r] = (float)-(s_camera.view[r][0] * s_sun_dir[0] + s_camera.view[r][1] * s_sun_dir[1] + s_camera.view[r][2] * s_sun_dir[2]);
        if (s_contact_shadows) params.shadow_strength = s_contact_strength;
        if (s_dyn_shadows && s_dyn_near[2] > 0) params.sun[3] = s_dyn_near_range * 1.5f; // as far ahead as the fine window reaches
        // (Not around a previewed model on an empty stage: with nothing but sky, the whole picture would be shaft.)
        if (s_light_shafts && params.sun[2] > 0.05f && !(s_previewing && !s_preview_keeps_scene)) { // the sun is in front of the camera
            float nx = params.sun[0] * params.p00 / params.sun[2], ny = params.sun[1] * params.p11 / params.sun[2];
            float off_screen = std::max(fabsf(nx), fabsf(ny));
            float visible = 1.0f - std::min(1.0f, std::max(0.0f, (off_screen - 1.0f) / 0.8f)); // fades out as the sun leaves the view
            params.shaft[0] = nx * 0.5f + 0.5f;
            params.shaft[1] = ny * 0.5f + 0.5f;
            params.shaft[2] = visible;
            params.shaft[3] = s_shaft_strength;
            shafts = visible > 0.01f;
        }
    }
    bool occlude = s_ao || params.shadow_strength > 0;
    if (!occlude && !shafts) return;
    params.reversed = s_reverse_depth ? 1.0f : 0.0f;
    params.fade_start = 60.0f * scale; // beyond this the effect fades out, and is gone at twice it

    if (!s_ao_texture || s_ao_texture.width != aw || s_ao_texture.height != ah) {
        auto make = [&](MTLPixelFormat format) {
            MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:aw height:ah mipmapped:NO];
            td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
            td.storageMode = MTLStorageModePrivate;
            return [s_device newTextureWithDescriptor:td];
        };
        s_ao_texture = make(MTLPixelFormatR8Unorm);
        s_ao_blurred_texture = make(MTLPixelFormatR8Unorm);
        s_ao_z_texture = make(MTLPixelFormatR32Float);
    }
    if (!s_shaft_texture || s_shaft_texture.width != sw || s_shaft_texture.height != sh) {
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:sw height:sh mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModePrivate;
        s_shaft_mask_texture = [s_device newTextureWithDescriptor:td];
        s_shaft_texture = [s_device newTextureWithDescriptor:td];
    }
    id<MTLRenderPipelineState> apply = ao_apply_pipeline(s_last.color.pixelFormat, 1);
    if (!apply) return;

    auto half_pass = [&](id<MTLTexture> target, id<MTLRenderPipelineState> pipeline, id<MTLTexture> input0, id<MTLTexture> input1) {
        MTLRenderPassDescriptor *d = [MTLRenderPassDescriptor renderPassDescriptor];
        d.colorAttachments[0].texture = target;
        d.colorAttachments[0].loadAction = MTLLoadActionDontCare;
        d.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> pass = [command_buffer() renderCommandEncoderWithDescriptor:d];
        [pass setRenderPipelineState:pipeline];
        [pass setFragmentTexture:input0 atIndex:0];
        if (input1) [pass setFragmentTexture:input1 atIndex:1];
        [pass setFragmentBytes:&params length:sizeof params atIndex:0];
        [pass drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [pass endEncoding];
    };
    // Both `onto` passes blend onto the resolved image, which is what
    // post-processing reads. The multisampled copy is left as it was: writing
    // there would blend every sample and resolve the whole buffer again, and the
    // engine does not draw into the scene once post-processing has begun.
    auto onto_scene = [&](id<MTLRenderPipelineState> pipeline, id<MTLTexture> input) {
        MTLRenderPassDescriptor *d = [MTLRenderPassDescriptor renderPassDescriptor];
        d.colorAttachments[0].texture = s_last.color;
        d.colorAttachments[0].loadAction = MTLLoadActionLoad;
        d.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> pass = [command_buffer() renderCommandEncoderWithDescriptor:d];
        [pass setRenderPipelineState:pipeline];
        [pass setFragmentTexture:input atIndex:0];
        [pass setFragmentBytes:&params length:sizeof params atIndex:0];
        [pass drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [pass endEncoding];
    };
    half_pass(s_ao_z_texture, s_ao_linearize_pipeline, s_last.depth, nil);
    if (occlude) {
        half_pass(s_ao_texture, s_ao_compute_pipeline, s_ao_z_texture, nil);
        half_pass(s_ao_blurred_texture, s_ao_blur_pipeline, s_ao_texture, s_ao_z_texture);
        onto_scene(apply, s_ao_blurred_texture);
    }
    if (shafts) { // after occlusion, so that the light is not darkened by it
        half_pass(s_shaft_mask_texture, s_shaft_mask_pipeline, s_last.color, s_ao_z_texture);
        half_pass(s_shaft_texture, s_shaft_blur_pipeline, s_shaft_mask_texture, nil);
        onto_scene(s_shaft_apply_pipeline, s_shaft_texture);
    }
}

// ---- the scene camera ------------------------------------------------------------------

// Column-major 4x4 matrices, as OpenGL passes them: element (row r, column c) is m[c * 4 + r].
void mat_mul(const double *x, const double *y, double *out)
{
    for (int c = 0; c < 4; c++)
        for (int r = 0; r < 4; r++) {
            double v = 0;
            for (int k = 0; k < 4; k++) v += x[k * 4 + r] * y[c * 4 + k];
            out[c * 4 + r] = v;
        }
}

bool mat_inverse(const double *m, double *out)
{
    double inv[16];
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    double det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if (fabs(det) < 1e-30) return false;
    for (int i = 0; i < 16; i++) out[i] = inv[i] / det;
    return true;
}

// The current value of a mat4 uniform of the vertex stage, or false if the program has none by that name.
bool vertex_matrix(Program &p, const char *name, double *out)
{
    auto it = p.location_by_name.find(name);
    if (it == p.location_by_name.end()) return false;
    const XlUniform *u = p.locations[(size_t)it->second].uniform[0];
    if (!u || u->columns != 4 || u->components != 4) return false;
    float m[16];
    memcpy(m, p.uniform_data[0].data() + u->offset, sizeof m);
    for (int i = 0; i < 16; i++) out[i] = m[i];
    return true;
}

// Called for draws into the scene buffer until the camera is known for this
// frame. The engine gives each program combined matrices, never the projection
// on its own, and which combinations exist depends on the scene (the world map
// never sets a view-projection, for one). Two that are nearly always there:
//
//   projection = (world-view-projection) x inverse(world-view)
//   view       = (world-view) x inverse(world)
//
// Anything that does not come out as a perspective projection is ignored and
// the next draw is tried.
void observe_camera(Program &p)
{
    double wvp[16], wv[16], w[16], inv[16], m[16];
    if (s_motion_blur && s_camera.view_proj_frame != s_frame_serial && vertex_matrix(p, "matWorldViewProj", wvp) &&
        vertex_matrix(p, "matWorld", w) && mat_inverse(w, inv)) {
        memcpy(s_camera.last_view_proj, s_camera.view_proj, sizeof s_camera.view_proj);
        s_camera.last_view_proj_frame = s_camera.view_proj_frame;
        mat_mul(wvp, inv, s_camera.view_proj);
        s_camera.view_proj_frame = s_frame_serial;
    }
    if (!vertex_matrix(p, "matWorldView", wv) || !mat_inverse(wv, inv)) return;
    if (s_camera.projection_frame != s_frame_serial && vertex_matrix(p, "matWorldViewProj", wvp)) {
        mat_mul(wvp, inv, m);
        // Rows 2 and 3 of a perspective projection are (0, 0, A, B) and (0, 0, C, 0) with C = +-1.
        double A = m[10], B = m[14], C = m[11], D = m[15];
        bool perspective = fabs(fabs(C) - 1.0) < 0.01 && fabs(D) < 0.01 && fabs(m[3]) + fabs(m[7]) < 0.01 && fabs(B) > 1e-6 &&
                           fabs(m[0]) > 1e-3 && fabs(m[5]) > 1e-3;
        if (perspective) {
            s_camera.a = A / C;
            s_camera.b = B - s_camera.a * D;
            s_camera.p00 = m[0];
            s_camera.p11 = m[5];
            s_camera.projection_frame = s_frame_serial;
            // The third view axis is the direction in which w grows.
            s_camera.view[2][0] = C;
        }
    }
    if (s_camera.view_frame != s_frame_serial && s_camera.projection_frame == s_frame_serial && vertex_matrix(p, "matWorld", w) &&
        mat_inverse(w, inv)) {
        mat_mul(wv, inv, m);
        double forward_sign = s_camera.view[2][0];
        s_camera.forward_sign = forward_sign < 0 ? -1.0 : 1.0;
        for (int r = 0; r < 3; r++) {
            double x = m[r], y = m[4 + r], z = m[8 + r], len = sqrt(x * x + y * y + z * z);
            if (len < 1e-9) return;
            double sign = r == 2 ? forward_sign : 1.0;
            s_camera.view[r][0] = sign * x / len, s_camera.view[r][1] = sign * y / len, s_camera.view[r][2] = sign * z / len;
        }
        s_camera.view_frame = s_frame_serial;
    }
}

// ---- motion blur ------------------------------------------------------------------------
//
// Blur from the camera's own movement. A pixel's depth gives its place in the
// world; the previous frame's camera says where on screen that place was a
// frame ago; the scene is smeared along the line between the two. Things that
// move by themselves are not known to move, so a sword's swing is not blurred,
// and whatever travels with the camera (the player's character, a horse) would
// be blurred as if it stood still in the world. That is avoided by distance:
// nothing closer than a few metres is blurred or smeared into what is.

const char k_blur_source[] = R"(
#include <metal_stdlib>
using namespace metal;

struct Params {
    float4x4 reproject; // this frame's clip space to the previous frame's
    float4 inverse_w;   // dot with a clip-space point at w = 1: one over its distance from the camera
    float2 size;        // of the scene, in pixels
    float reversed;
    float exposure;     // in frames
    float near_start, near_end; // distances between which the blur fades in
    float max_length;   // in screen heights
    float pad;
};

#if WBM_MSAA
typedef depth2d_ms<float> DepthTexture;
#else
typedef depth2d<float> DepthTexture;
#endif

struct VertexOut {
    float4 position [[position]];
};

vertex VertexOut blur_vertex(uint id [[vertex_id]])
{
    VertexOut out;
    out.position = float4(float2((id << 1) & 2, id & 2) * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

// The pixel as a point in OpenGL clip space with w = 1. Textures are stored in
// OpenGL's row order, so v grows with clip-space y.
static float4 clip_point(DepthTexture depth, float2 pixel, Params p)
{
    uint2 c = uint2(clamp(pixel, float2(0.0), p.size - 1.0));
    float d = depth.read(c, 0);
    return float4(pixel / p.size * 2.0 - 1.0, p.reversed > 0.5 ? 1.0 - 2.0 * d : 2.0 * d - 1.0, 1.0);
}

static float blurred(float4 n, Params p)
{
    float inverse = dot(p.inverse_w, n);
    float distance = inverse > 1.0e-9 ? 1.0 / inverse : 1.0e9;
    return smoothstep(p.near_start, p.near_end, distance);
}

fragment float4 blur_fragment(VertexOut in [[stage_in]], texture2d<float> scene [[texture(0)]], DepthTexture depth [[texture(1)]],
                              constant Params &p [[buffer(0)]])
{
    constexpr sampler smooth(filter::linear, address::clamp_to_edge);
    float2 uv = in.position.xy / p.size;
    float4 centre = scene.sample(smooth, uv);
    float4 n = clip_point(depth, in.position.xy, p);
    float amount = blurred(n, p);
    float4 before = p.reproject * n;
    if (amount <= 0.0 || before.w <= 1.0e-6) return centre;
    float2 velocity = (n.xy - before.xy / before.w) * 0.5 * p.exposure * amount; // in uv, per exposure
    float aspect = p.size.x / p.size.y;
    float length_ = length(velocity * float2(aspect, 1.0));
    if (length_ * p.size.y < 0.75) return centre;
    velocity *= min(1.0, p.max_length / length_);

    const int samples = 10;
    float noise = fract(52.9829189 * fract(dot(in.position.xy, float2(0.06711056, 0.00583715))));
    float3 sum = centre.rgb;
    float weight = 1.0;
    for (int i = 0; i < samples; i++) {
        float t = (float(i) + noise) / float(samples) - 0.5;
        float2 q = uv + velocity * t;
        if (any(q < float2(0.0)) || any(q > float2(1.0))) continue;
        float w = blurred(clip_point(depth, q * p.size, p), p); // near things keep their colour to themselves
        sum += scene.sample(smooth, q).rgb * w;
        weight += w;
    }
    return float4(sum / weight, centre.a);
}
)";

struct BlurParams { // must match Params in k_blur_source
    float reproject[16];
    float inverse_w[4];
    float size[2];
    float reversed, exposure, near_start, near_end, max_length, pad;
};

id<MTLLibrary> s_blur_library;
std::map<int, id<MTLRenderPipelineState>> s_blur_pipelines; // by colour format
id<MTLTexture> s_blur_texture;
double s_blur_time;
uint64_t s_blur_frames;

// Called between render passes, once the engine has finished the scene, after
// the other effects on it.
void apply_motion_blur()
{
    double now = CACurrentMediaTime(), frame_time = now - s_blur_time;
    s_blur_time = now;
    // Needs this frame's camera and the previous frame's; after a gap (a menu,
    // a loading screen) the two have nothing to do with each other.
    if (s_camera.view_proj_frame != s_frame_serial || s_camera.last_view_proj_frame + 1 != s_frame_serial || !s_last.depth || !s_last.color) return;
    if (frame_time <= 0 || frame_time > 0.1) return;
    double inverse[16], reproject[16];
    if (!mat_inverse(s_camera.view_proj, inverse)) return;
    mat_mul(s_camera.last_view_proj, inverse, reproject);
    // Nothing to do while the camera is still. And a jump (a cut to another
    // view) is no movement: the far point straight ahead would cross the screen.
    double change = 0;
    for (int i = 0; i < 16; i++) change = std::max(change, fabs(reproject[i] - (i % 5 == 0 ? reproject[15] : 0.0)));
    if (change < 1e-7 * fabs(reproject[15])) return;
    double qw = reproject[11] + reproject[15];
    if (qw <= 1e-9) return;
    double jump = hypot((reproject[8] + reproject[12]) / qw, (reproject[9] + reproject[13]) / qw);
    if (jump > 0.5) return;

    if (!s_blur_library) {
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.preprocessorMacros = @{@"WBM_MSAA" : @(s_last.depth.sampleCount > 1 ? 1 : 0)};
        NSError *error = nil;
        s_blur_library = [s_device newLibraryWithSource:@(k_blur_source) options:options error:&error];
        if (!s_blur_library) {
            complain("motion blur shaders did not compile: %s", error.localizedDescription.UTF8String);
            s_motion_blur = false;
            return;
        }
    }
    id<MTLRenderPipelineState> __strong &pipeline = s_blur_pipelines[(int)s_last.color.pixelFormat];
    if (!pipeline) {
        MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
        d.vertexFunction = [s_blur_library newFunctionWithName:@"blur_vertex"];
        d.fragmentFunction = [s_blur_library newFunctionWithName:@"blur_fragment"];
        d.colorAttachments[0].pixelFormat = s_last.color.pixelFormat;
        NSError *error = nil;
        pipeline = [s_device newRenderPipelineStateWithDescriptor:d error:&error];
        if (!pipeline) {
            complain("motion blur pipeline failed: %s", error.localizedDescription.UTF8String);
            s_motion_blur = false;
            return;
        }
    }
    NSUInteger w = s_last.color.width, h = s_last.color.height;
    if (s_last.depth.width != w || s_last.depth.height != h) return;
    if (!s_blur_texture || s_blur_texture.width != w || s_blur_texture.height != h || s_blur_texture.pixelFormat != s_last.color.pixelFormat) {
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:s_last.color.pixelFormat width:w height:h mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModePrivate;
        s_blur_texture = [s_device newTextureWithDescriptor:td];
    }

    BlurParams params;
    for (int i = 0; i < 16; i++) params.reproject[i] = (float)reproject[i];
    for (int c = 0; c < 4; c++) params.inverse_w[c] = (float)inverse[c * 4 + 3];
    params.size[0] = (float)w, params.size[1] = (float)h;
    params.reversed = s_reverse_depth ? 1.0f : 0.0f;
    // The exposure is a length of time, not a share of the frame, so that the
    // blur looks the same at any frame rate.
    params.exposure = (float)(s_motion_blur_strength / 48.0 / std::max(frame_time, 1.0 / 240.0));
    // Scenes differ in scale (see apply_ambient_occlusion); distances follow the far plane.
    double far_inverse = inverse[11] + inverse[15], far = fabs(far_inverse) > 1e-9 ? fabs(1.0 / far_inverse) : 1250.0;
    float scale = (float)std::min(1.0, std::max(0.15, far / 1250.0));
    params.near_start = 5.0f * scale;
    params.near_end = 10.0f * scale;
    params.max_length = 0.04f;
    params.pad = 0;

    MTLRenderPassDescriptor *d = [MTLRenderPassDescriptor renderPassDescriptor];
    d.colorAttachments[0].texture = s_blur_texture;
    d.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    d.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> pass = [command_buffer() renderCommandEncoderWithDescriptor:d];
    [pass setRenderPipelineState:pipeline];
    [pass setFragmentTexture:s_last.color atIndex:0];
    [pass setFragmentTexture:s_last.depth atIndex:1];
    [pass setFragmentBytes:&params length:sizeof params atIndex:0];
    [pass drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [pass endEncoding];
    id<MTLBlitCommandEncoder> blit = [command_buffer() blitCommandEncoder];
    [blit copyFromTexture:s_blur_texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(w, h, 1)
                toTexture:s_last.color destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];
    s_blur_frames++;
}

// ---- anti-aliasing after the fact ---------------------------------------------------------
//
// FXAA (Lottes): where neighbouring pixels differ enough in brightness to be
// an edge, the pixel is blended with its neighbour across the edge by as much
// as the edge's run along its own direction says it is covered. It works on
// the finished scene, so it costs the same whatever was drawn and smooths
// cut-out edges (leaves, hair) as well as geometry. It runs before the game's
// interface is drawn, which stays sharp. The scene buffer holds gamma-encoded
// colour, which is what the brightness comparison wants.

const char k_fxaa_source[] = R"(
#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
};

vertex VertexOut fxaa_vertex(uint id [[vertex_id]])
{
    VertexOut out;
    out.position = float4(float2((id << 1) & 2, id & 2) * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

static float luma(float3 c) { return dot(clamp(c, 0.0, 1.0), float3(0.299, 0.587, 0.114)); }

fragment float4 fxaa_fragment(VertexOut in [[stage_in]], texture2d<float> scene [[texture(0)]])
{
    constexpr sampler smooth(filter::linear, address::clamp_to_edge);
    float2 inv = 1.0 / float2(scene.get_width(), scene.get_height());
    float2 uv = in.position.xy * inv;
    float4 centre = scene.sample(smooth, uv);
    float m = luma(centre.rgb);
    float d = luma(scene.sample(smooth, uv + float2(0, -1) * inv).rgb), u = luma(scene.sample(smooth, uv + float2(0, 1) * inv).rgb);
    float l = luma(scene.sample(smooth, uv + float2(-1, 0) * inv).rgb), r = luma(scene.sample(smooth, uv + float2(1, 0) * inv).rgb);
    float lowest = min(m, min(min(d, u), min(l, r))), highest = max(m, max(max(d, u), max(l, r)));
    float range = highest - lowest;
    if (range < max(0.0312, highest * 0.125)) return centre;

    float dl = luma(scene.sample(smooth, uv + float2(-1, -1) * inv).rgb), ur = luma(scene.sample(smooth, uv + float2(1, 1) * inv).rgb);
    float ul = luma(scene.sample(smooth, uv + float2(-1, 1) * inv).rgb), dr = luma(scene.sample(smooth, uv + float2(1, -1) * inv).rgb);
    float horizontal = abs(-2.0 * l + dl + ul) + abs(-2.0 * m + d + u) * 2.0 + abs(-2.0 * r + dr + ur);
    float vertical = abs(-2.0 * u + ul + ur) + abs(-2.0 * m + l + r) * 2.0 + abs(-2.0 * d + dl + dr);
    bool is_horizontal = horizontal >= vertical;

    float luma1 = is_horizontal ? d : l, luma2 = is_horizontal ? u : r;
    float gradient1 = luma1 - m, gradient2 = luma2 - m;
    bool steepest1 = abs(gradient1) >= abs(gradient2);
    float gradient = 0.25 * max(abs(gradient1), abs(gradient2));
    float step_length = is_horizontal ? inv.y : inv.x;
    float average;
    if (steepest1) {
        step_length = -step_length;
        average = 0.5 * (luma1 + m);
    } else {
        average = 0.5 * (luma2 + m);
    }
    float2 at = uv;
    if (is_horizontal) at.y += step_length * 0.5;
    else at.x += step_length * 0.5;

    // Along the edge both ways, until it ends.
    float2 along = is_horizontal ? float2(inv.x, 0.0) : float2(0.0, inv.y);
    float2 uv1 = at - along, uv2 = at + along;
    float end1 = luma(scene.sample(smooth, uv1).rgb) - average, end2 = luma(scene.sample(smooth, uv2).rgb) - average;
    bool reached1 = abs(end1) >= gradient, reached2 = abs(end2) >= gradient;
    const float steps[10] = {1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 4.0, 8.0};
    for (int i = 0; i < 10 && !(reached1 && reached2); i++) {
        if (!reached1) {
            uv1 -= along * steps[i];
            end1 = luma(scene.sample(smooth, uv1).rgb) - average;
            reached1 = abs(end1) >= gradient;
        }
        if (!reached2) {
            uv2 += along * steps[i];
            end2 = luma(scene.sample(smooth, uv2).rgb) - average;
            reached2 = abs(end2) >= gradient;
        }
    }
    float distance1 = is_horizontal ? uv.x - uv1.x : uv.y - uv1.y, distance2 = is_horizontal ? uv2.x - uv.x : uv2.y - uv.y;
    bool nearer1 = distance1 < distance2;
    float offset = -min(distance1, distance2) / (distance1 + distance2) + 0.5;
    bool centre_darker = m < average;
    bool right_way = ((nearer1 ? end1 : end2) < 0.0) != centre_darker;
    offset = right_way ? offset : 0.0;

    // Single bright or dark pixels.
    float around = (2.0 * (d + u + l + r) + dl + ul + dr + ur) / 12.0;
    float s1 = clamp(abs(around - m) / range, 0.0, 1.0), s2 = (-2.0 * s1 + 3.0) * s1 * s1;
    offset = max(offset, s2 * s2 * 0.75);

    float2 final_uv = uv;
    if (is_horizontal) final_uv.y += offset * step_length;
    else final_uv.x += offset * step_length;
    return float4(scene.sample(smooth, final_uv).rgb, centre.a);
}
)";

id<MTLLibrary> s_fxaa_library;
std::map<int, id<MTLRenderPipelineState>> s_fxaa_pipelines; // by colour format
id<MTLTexture> s_fxaa_texture;
uint64_t s_fxaa_frames;

// Called between render passes, once the engine has finished the scene.
void apply_fxaa()
{
    if (!s_last.color) return;
    if (!s_fxaa_library) {
        NSError *error = nil;
        s_fxaa_library = [s_device newLibraryWithSource:@(k_fxaa_source) options:nil error:&error];
        if (!s_fxaa_library) {
            complain("anti-aliasing shaders did not compile: %s", error.localizedDescription.UTF8String);
            s_fxaa = false;
            return;
        }
    }
    id<MTLRenderPipelineState> __strong &pipeline = s_fxaa_pipelines[(int)s_last.color.pixelFormat];
    if (!pipeline) {
        MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
        d.vertexFunction = [s_fxaa_library newFunctionWithName:@"fxaa_vertex"];
        d.fragmentFunction = [s_fxaa_library newFunctionWithName:@"fxaa_fragment"];
        d.colorAttachments[0].pixelFormat = s_last.color.pixelFormat;
        NSError *error = nil;
        pipeline = [s_device newRenderPipelineStateWithDescriptor:d error:&error];
        if (!pipeline) {
            complain("anti-aliasing pipeline failed: %s", error.localizedDescription.UTF8String);
            s_fxaa = false;
            return;
        }
    }
    NSUInteger w = s_last.color.width, h = s_last.color.height;
    if (!s_fxaa_texture || s_fxaa_texture.width != w || s_fxaa_texture.height != h || s_fxaa_texture.pixelFormat != s_last.color.pixelFormat) {
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:s_last.color.pixelFormat width:w height:h mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModePrivate;
        s_fxaa_texture = [s_device newTextureWithDescriptor:td];
    }
    MTLRenderPassDescriptor *d = [MTLRenderPassDescriptor renderPassDescriptor];
    d.colorAttachments[0].texture = s_fxaa_texture;
    d.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    d.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> pass = [command_buffer() renderCommandEncoderWithDescriptor:d];
    [pass setRenderPipelineState:pipeline];
    [pass setFragmentTexture:s_last.color atIndex:0];
    [pass drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [pass endEncoding];
    id<MTLBlitCommandEncoder> blit = [command_buffer() blitCommandEncoder];
    [blit copyFromTexture:s_fxaa_texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(w, h, 1)
                toTexture:s_last.color destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];
    s_fxaa_frames++;
}

// Upscaling (WBM_UPSCALE): the scene was drawn smaller than the screen. MetalFX
// enlarges it to the screen's size into a texture of its own, which stands in
// for the scene's when the engine's last pass, the one onto the screen, reads
// it (see the binding of textures in prepare_draw). Everything before that
// pass, the engine's own effects and the shim's, works on the smaller picture,
// which is where the time is saved.
id<MTLFXSpatialScaler> s_scaler API_AVAILABLE(macos(13.0));
id<MTLTexture> s_upscaled;
uint64_t s_upscale_frames;

id<MTLTexture> apply_upscale()
{
    id<MTLTexture> scene = s_last.color;
    if (!scene) return nil;
    NSUInteger w = (NSUInteger)(s_screen_w * s_scale), h = (NSUInteger)(s_screen_h * s_scale);
    if (scene.width >= w || scene.height >= h) return nil;
    if (!s_scaler || s_scaler.inputWidth != scene.width || s_scaler.inputHeight != scene.height || s_scaler.outputWidth != w || s_scaler.outputHeight != h ||
        s_scaler.colorTextureFormat != scene.pixelFormat) {
        MTLFXSpatialScalerDescriptor *d = [MTLFXSpatialScalerDescriptor new];
        d.inputWidth = scene.width, d.inputHeight = scene.height, d.outputWidth = w, d.outputHeight = h;
        d.colorTextureFormat = d.outputTextureFormat = scene.pixelFormat;
        // The scene holds colours already encoded for the eye; in its 16-bit form they may also run past white.
        d.colorProcessingMode = scene.pixelFormat == MTLPixelFormatRGBA16Float ? MTLFXSpatialScalerColorProcessingModeHDR : MTLFXSpatialScalerColorProcessingModePerceptual;
        s_scaler = [d newSpatialScalerWithDevice:s_device];
        if (!s_scaler) {
            complain("WBM_UPSCALE: MetalFX would not make a scaler from %lux%lu to %lux%lu", (unsigned long)scene.width, (unsigned long)scene.height, (unsigned long)w, (unsigned long)h);
            s_upscale = false;
            return nil;
        }
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:scene.pixelFormat width:w height:h mipmapped:NO];
        td.usage = s_scaler.outputTextureUsage | MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
        td.storageMode = MTLStorageModePrivate;
        s_upscaled = [s_device newTextureWithDescriptor:td];
    }
    s_scaler.colorTexture = scene;
    s_scaler.outputTexture = s_upscaled;
    [s_scaler encodeToCommandBuffer:command_buffer()];
    for (auto &entry : s_textures)
        if (entry.second.mtl == scene) entry.second.upscaled = s_upscaled, entry.second.upscaled_frame = s_frame_serial;
    s_upscale_frames++;
    return s_upscaled;
}

// Sharpening (WBM_SHARPEN, 0 to 1), after AMD's contrast-adaptive sharpening:
// each pixel is pushed away from the average of its four neighbours, less
// where they already differ much (so edges do not ring and noise is not
// raised), and never past the darkest or brightest of them.
const char k_sharpen_source[] = R"(
#include <metal_stdlib>
using namespace metal;
struct VertexOut {
    float4 position [[position]];
};
vertex VertexOut sharpen_vertex(uint id [[vertex_id]])
{
    VertexOut out;
    out.position = float4(float2((id << 1) & 2, id & 2) * 2.0 - 1.0, 0.0, 1.0);
    return out;
}
fragment float4 sharpen_fragment(VertexOut in [[stage_in]], texture2d<float> picture [[texture(0)]], constant float &amount [[buffer(0)]])
{
    int2 at = int2(in.position.xy), most = int2(picture.get_width(), picture.get_height()) - 1;
    float4 here = picture.read(uint2(at));
    float3 a = picture.read(uint2(clamp(at + int2(0, -1), int2(0), most))).rgb, b = picture.read(uint2(clamp(at + int2(-1, 0), int2(0), most))).rgb;
    float3 c = picture.read(uint2(clamp(at + int2(1, 0), int2(0), most))).rgb, d = picture.read(uint2(clamp(at + int2(0, 1), int2(0), most))).rgb;
    float3 low = min(min(min(a, b), min(c, d)), here.rgb), high = max(max(max(a, b), max(c, d)), here.rgb);
    // How far there is to go before clipping, against how far apart they are already.
    float3 room = sqrt(clamp(min(low, 2.0 - high) / max(high, 1e-4), 0.0, 1.0));
    float3 weight = room * (-0.2 * amount);
    float3 sharp = (here.rgb + (a + b + c + d) * weight) / (1.0 + 4.0 * weight);
    return float4(clamp(sharp, low, high), here.a);
}
)";
id<MTLLibrary> s_sharpen_library;
std::map<int, id<MTLRenderPipelineState>> s_sharpen_pipelines; // by colour format
id<MTLTexture> s_sharpen_texture;
uint64_t s_sharpen_frames;

void apply_sharpen(id<MTLTexture> picture)
{
    if (!picture) return;
    if (!s_sharpen_library) {
        NSError *error = nil;
        s_sharpen_library = [s_device newLibraryWithSource:@(k_sharpen_source) options:nil error:&error];
        if (!s_sharpen_library) {
            complain("sharpening shaders did not compile: %s", error.localizedDescription.UTF8String);
            s_sharpen = 0;
            return;
        }
    }
    id<MTLRenderPipelineState> __strong &pipeline = s_sharpen_pipelines[(int)picture.pixelFormat];
    if (!pipeline) {
        MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
        d.vertexFunction = [s_sharpen_library newFunctionWithName:@"sharpen_vertex"];
        d.fragmentFunction = [s_sharpen_library newFunctionWithName:@"sharpen_fragment"];
        d.colorAttachments[0].pixelFormat = picture.pixelFormat;
        NSError *error = nil;
        pipeline = [s_device newRenderPipelineStateWithDescriptor:d error:&error];
        if (!pipeline) {
            complain("sharpening pipeline failed: %s", error.localizedDescription.UTF8String);
            s_sharpen = 0;
            return;
        }
    }
    NSUInteger w = picture.width, h = picture.height;
    if (!s_sharpen_texture || s_sharpen_texture.width != w || s_sharpen_texture.height != h || s_sharpen_texture.pixelFormat != picture.pixelFormat) {
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:picture.pixelFormat width:w height:h mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModePrivate;
        s_sharpen_texture = [s_device newTextureWithDescriptor:td];
    }
    MTLRenderPassDescriptor *d = [MTLRenderPassDescriptor renderPassDescriptor];
    d.colorAttachments[0].texture = s_sharpen_texture;
    d.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    d.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> pass = [command_buffer() renderCommandEncoderWithDescriptor:d];
    [pass setRenderPipelineState:pipeline];
    [pass setFragmentTexture:picture atIndex:0];
    [pass setFragmentBytes:&s_sharpen length:4 atIndex:0];
    [pass drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [pass endEncoding];
    id<MTLBlitCommandEncoder> blit = [command_buffer() blitCommandEncoder];
    [blit copyFromTexture:s_sharpen_texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(w, h, 1)
                toTexture:picture destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];
    s_sharpen_frames++;
}

// Foliage that the engine hands over in batches, many plants in one buffer and
// already placed in the world, carries nothing that says how high above its
// own foot a vertex is, which is what the wind needs. It is worked out here,
// once per buffer: a batch repeats one mesh, so the texture coordinates repeat
// with the mesh's vertex count, which gives the plants; a vertex's height above
// the lowest point of its plant, in decimetres, is then written into the alpha
// of its colour, which foliage leaves at full. The vertex shaders read it from
// there and take the alpha as full (see k_wind_function). For leaves that hang
// from a crown (palms) the lowest point is a leaf's tip, so there the distance
// from the plant's middle is used instead.
bool weigh_foliage(Buffer &b, bool radial)
{
    if (b.wind_weights == (__bridge void *)b.mtl) return true;
    const Attrib &pos = s_attribs[0], &col = s_attribs[2], &tex = s_attribs[4];
    if (!b.mtl || !pos.enabled || !col.enabled || !tex.enabled || pos.stride <= 0 || col.buffer != pos.buffer || tex.buffer != pos.buffer ||
        pos.type != GL_FLOAT || col.type != GL_UNSIGNED_BYTE || tex.type != GL_FLOAT)
        return false;
    size_t stride = (size_t)pos.stride, n = b.mtl.length / stride;
    if (n < 3 || std::max({pos.offset + 12, col.offset + 4, tex.offset + 8}) > stride) return false;
    uint8_t *base = (uint8_t *)b.mtl.contents;
    auto uv = [&](size_t i) { return base + i * stride + tex.offset; };
    auto z = [&](size_t i) {
        float v;
        memcpy(&v, base + i * stride + pos.offset + 8, 4);
        return v;
    };
    // The shortest period of the texture coordinates, tried only where the first vertex's recur.
    size_t period = n;
    for (size_t c = 3; c <= n / 2; c++) {
        if (n % c || memcmp(uv(0), uv(c), 8)) continue;
        bool repeats = true;
        for (size_t i = c; i < n && repeats; i++) repeats = !memcmp(uv(i), uv(i - c), 8);
        if (repeats) {
            period = c;
            break;
        }
    }
    for (size_t first = 0; first < n; first += period) {
        float lowest = 1e30f;
        double middle[2] = {0, 0};
        for (size_t i = first; i < first + period; i++) {
            float xy[2];
            memcpy(xy, base + i * stride + pos.offset, 8);
            lowest = std::min(lowest, z(i));
            middle[0] += xy[0] / (double)period, middle[1] += xy[1] / (double)period;
        }
        for (size_t i = first; i < first + period; i++) {
            float metres = z(i) - lowest;
            if (radial) { // leaves that hang from a crown are held at the middle and free at their tips, wherever those are
                float xy[2];
                memcpy(xy, base + i * stride + pos.offset, 8);
                metres = 1.6f * (float)hypot(xy[0] - middle[0], xy[1] - middle[1]);
            }
            base[i * stride + col.offset + 3] = (uint8_t)std::min(255.0f, std::max(0.0f, metres * 10.0f + 0.5f));
        }
    }
    b.wind_weights = (__bridge void *)b.mtl;
    return true;
}

// Where a pair of eyes is turned at a time: held still for a second or two,
// about half the time straight ahead, then moved to the next place in a
// twelfth of a second, as eyes do. In the face texture's coordinates.
void eye_gaze(double time, float phase, float *out)
{
    if (const char *fixed = getenv("WBM_GAZE")) { // "x y", for looking at the effect
        sscanf(fixed, "%f %f", &out[0], &out[1]);
        return;
    }
    auto hash = [](double n) {
        double v = sin(n) * 43758.5453;
        return (float)(v - floor(v));
    };
    auto target = [&](double step, float *xy) {
        bool ahead = hash(step * 1.7 + phase) < 0.5f;
        xy[0] = ahead ? 0.0f : (hash(step * 3.1 + phase + 4.0) - 0.5f) * 0.016f * s_gaze_strength;
        xy[1] = ahead ? 0.0f : (hash(step * 5.3 + phase + 9.0) - 0.5f) * 0.005f * s_gaze_strength;
    };
    double period = 1.1 + 1.6 * hash(phase + 2.0), at = time / period + hash(phase), step = floor(at);
    float now[2], before[2];
    target(step, now), target(step - 1, before);
    float t = (float)std::min(1.0, (at - step) * period / 0.08);
    t = t * t * (3 - 2 * t);
    out[0] = before[0] + (now[0] - before[0]) * t;
    out[1] = before[1] + (now[1] - before[1]) * t;
}

// Holds the frame rate to a limit by waiting at the end of a frame.
void limit_frame_rate()
{
    static const double start = CACurrentMediaTime();
    s_wind_time = fmod(CACurrentMediaTime() - start, 3600.0); // for the frame about to begin
    static std::chrono::steady_clock::time_point next;
    double limit = s_scene_frame + 1 >= s_frame_serial ? s_max_fps : (s_menu_fps > 0 ? s_menu_fps : s_max_fps); // s_frame_serial was just advanced
    auto now = std::chrono::steady_clock::now();
    if (limit <= 0) {
        next = now;
        return;
    }
    auto period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(1.0 / limit));
    next += period;
    if (next < now - period * 4) next = now; // fell behind: do not try to catch up
    if (next > now) std::this_thread::sleep_until(next);
}

#include "texture_packs.inc"

// Binds a buffer to a vertex (stage 0) or fragment (stage 1) slot, doing only
// as much as changed: nothing, the offset alone, or the whole binding.
void bind_buffer(int stage, id<MTLBuffer> buffer, NSUInteger offset, NSUInteger index)
{
    void *identity = (__bridge void *)buffer;
    bool same_buffer = index < 31 && s_set.buffer[stage][index] == identity;
    if (already(same_buffer && s_set.offset[stage][index] == offset)) return;
    if (same_buffer && s_state_cache) {
        if (stage == 0) [s_pass setVertexBufferOffset:offset atIndex:index];
        else [s_pass setFragmentBufferOffset:offset atIndex:index];
    } else if (stage == 0) {
        [s_pass setVertexBuffer:buffer offset:offset atIndex:index];
    } else {
        [s_pass setFragmentBuffer:buffer offset:offset atIndex:index];
    }
    if (index < 31) {
        s_set.buffer[stage][index] = identity;
        s_set.offset[stage][index] = offset;
    }
}

// ---- the per-frame shadow map -------------------------------------------------------------
//
// Before its main pass the engine draws every visible object once into a depth
// buffer (when "depth effects" are on). Those draws are recorded here, and then
// drawn again from the sun's direction into a shadow map of the shim's own, so
// that characters, horses and their gear cast shadows on everything, at any
// distance, where the engine gives them only a decal on the ground.
//
// The map lives in the engine's own shadow space, the one its matSunViewProj
// maps the world into, restricted to a window around the camera. A shader's
// existing shadow coordinate therefore indexes it directly (see k_xl_dyn_sampler).
//
// To move a recorded draw there, its world-view-projection has to be turned
// back into a world transform, which needs the camera's view-projection. The
// pre-pass shaders are never given that, so the map is drawn at the first draw
// of the main pass whose program has both a world-view-projection and a world
// matrix (the sky dome, normally the very first): view-projection is then
// WVP x inverse(W), exact for this frame. It is drawn in a command buffer of
// its own, committed at once, since a render pass is open on the main one.

struct CasterDraw {
    PipelineKey key;          // vertex layout; the other fields are zero
    id<MTLBuffer> vertex[MAX_VERTEX_SLOTS];
    NSUInteger vertex_offset[MAX_VERTEX_SLOTS];
    int slots;
    id<MTLBuffer> indices;
    NSUInteger index_offset, index_count;
    MTLIndexType index_type;
    float wvp[16];            // object to clip space; object to world space when `world` is set
    bool world;
    float cutoff;             // alpha below which a cut-out texel casts nothing
    int bones;                // index into s_caster_bones of 32 matrices, or -1
    id<MTLTexture> alpha;     // set for alpha-tested geometry
    id<MTLSamplerState> alpha_sampler;
    // A layer of long hair (see record_hair_casters): the mesh pushed out by
    // `push` metres, hanging as the drawn hair does at `share` of its length,
    // casting only where `alpha`, here the hair map, says long hair.
    bool hair = false;
    int strand = 0; // 1: a horse's mane and tail as strands (push: the mane's length, share: the tail's); 2: a person's (push: the longest strand)
    float drift[3] = {0, 0, 0};
    float push = 0, share = 0, down[3] = {0, 0, 0};
    std::shared_ptr<std::vector<float>> own_bones; // 32 matrices, kept with the record: it outlives the frame's own
};
std::vector<CasterDraw> s_casters;
std::vector<float> s_caster_bones;

// Grass never reaches the pre-pass: the engine draws it in the main pass only.
// Cut-out geometry that turns up there without having been in the pre-pass is
// recorded too, with its world matrix, which the main pass does supply, and
// goes into the next frame's map. Grass stands still, so a frame's delay does
// not show.
std::vector<CasterDraw> s_late_casters, s_late_ready; // from this frame, and from the last for use in this one
std::unordered_set<uint64_t> s_prepass_meshes;        // everything this frame's pre-pass drew

// Scenery is in the engine's own shadow map already, drawn there once per scene.
// Drawing it into this one as well would cost a draw and change the look of its
// shadow wherever the window happens to reach, so the meshes the engine drew
// into its map are remembered and passed over.
std::unordered_set<uint64_t> s_scenery;
uint64_t s_scenery_frame;

uint64_t mesh_id(NSUInteger count, NSUInteger index_offset)
{
    return ((uint64_t)s_element_buffer * 0x9E3779B97F4A7C15ull) ^ ((uint64_t)s_draw_binding.buffer[0] * 0xC2B2AE3D27D4EB4Full) ^
           ((uint64_t)index_offset * 0x165667B19E3779F9ull) ^ (uint64_t)count;
}

// Some of what the pre-pass draws is not part of the world at all: the aiming
// mark, for one. The main pass gives such things away by drawing them on top of
// everything, without a depth test, and a mesh seen drawn that way is kept out
// of the map from the next frame on. (Not writing depth is no such sign: grass
// is drawn that way, and grass should cast.)
std::unordered_set<uint64_t> s_overlays;

void record_overlay(NSUInteger count, NSUInteger index_offset)
{
    if (s_overlays.size() > 4096) s_overlays.clear();
    s_overlays.insert(mesh_id(count, index_offset));
}

void record_scenery(NSUInteger count, NSUInteger index_offset)
{
    if (s_scenery_frame + 1 < s_frame_serial) s_scenery.clear(); // the engine is drawing its map afresh
    s_scenery_frame = s_frame_serial;
    s_scenery.insert(mesh_id(count, index_offset));
}

const char k_caster_source[] = R"(
#include <metal_stdlib>
using namespace metal;

// m takes object space to the engine's shadow space: map coordinates in x and
// y, the depth its shaders compare against in z.
struct Caster {
    float4x4 m;
    float4 window[2]; // the wide window and the fine one. xy: centre, zw: 2 / (width, height), all in map coordinates
    float4 params;    // x: alpha below which a cut-out texel casts nothing; for hair, y: metres out, z: share of the hair's length
    float4 hang;      // for hair, xyz: which way is down, in the space the bones leave the vertex in
    float4 drift;     // for strands, xyz: how far their tips trail behind, as a share of their length, in the same space
};
struct SkinnedCaster {
    float4x4 m;
    float4 window[2];
    float4 params;
    float4 hang;
    float4 drift;
    float4x4 bones[32];
};
struct HairIn {
    float3 position [[attribute(0)]];
    float3 normal [[attribute(1)]];
    float2 uv [[attribute(4)]];
    float4 weights [[attribute(7)]];
    float4 indices [[attribute(8)]];
};
// Strands (see build_strands): every vertex sits at its strand's root, and its colour says which point of the strand it is.
struct StrandIn {
    float3 position [[attribute(0)]];
    float3 normal [[attribute(1)]];
    float4 colour [[attribute(2)]];
    float2 uv [[attribute(4)]];
    float4 weights [[attribute(7)]];
    float4 indices [[attribute(8)]];
};
struct PersonStrandIn {
    float3 position [[attribute(0)]];
    float3 normal [[attribute(1)]];
    float4 colour [[attribute(2)]];
    float2 uv [[attribute(4)]];
    float3 run [[attribute(6)]];
};
struct StaticIn {
    float3 position [[attribute(0)]];
    float2 uv [[attribute(4)]];
};
struct SkinnedIn {
    float3 position [[attribute(0)]];
    float2 uv [[attribute(4)]];
    float4 weights [[attribute(7)]];
    float4 indices [[attribute(8)]];
};
struct Out {
    float4 position [[position]];
    float depth;
    float2 uv;
    float cutoff [[flat]];
};

// Rows of the map are stored in OpenGL's order like every other texture the
// renderer draws, hence the flipped y. The depth buffer only has to order the
// casters; the comparison value is written as colour, unscaled.
static Out finish(float4 s, float2 uv, float4 window, float4 params)
{
    float3 q = s.xyz / s.w;
    Out out;
    out.position = float4((q.x - window.x) * window.z, -(q.y - window.y) * window.w, clamp(q.z * 0.25 + 0.5, 0.0, 1.0), 1.0);
    out.depth = q.z;
    out.uv = uv;
    out.cutoff = params.x;
    return out;
}

// Each draw is rasterised twice, once into either half of the map (vertex
// amplification): `side` says which, and only the position depends on it.
vertex Out caster_static(StaticIn in [[stage_in]], constant Caster &c [[buffer(0)]], ushort side [[amplification_id]])
{
    return finish(c.m * float4(in.position, 1.0), in.uv, c.window[side], c.params);
}

vertex Out caster_skinned(SkinnedIn in [[stage_in]], constant SkinnedCaster &c [[buffer(0)]], ushort side [[amplification_id]])
{
    float4 p = float4(in.position, 1.0);
    float4 skinned = c.bones[int(in.indices.x)] * p * in.weights.x + c.bones[int(in.indices.y)] * p * in.weights.y +
                     c.bones[int(in.indices.z)] * p * in.weights.z + c.bones[int(in.indices.w)] * p * in.weights.w;
    return finish(c.m * skinned, in.uv, c.window[side], c.params);
}

// A layer of long hair, moved as the renderer's own layers are (add_shell_push in shader_xlate.cpp).
vertex Out caster_hair(HairIn in [[stage_in]], constant SkinnedCaster &c [[buffer(0)]], ushort side [[amplification_id]])
{
    float4 p = float4(in.position, 1.0), q = float4(in.normal, 0.0);
    float4 skinned = c.bones[int(in.indices.x)] * p * in.weights.x + c.bones[int(in.indices.y)] * p * in.weights.y +
                     c.bones[int(in.indices.z)] * p * in.weights.z + c.bones[int(in.indices.w)] * p * in.weights.w;
    float3 n = normalize((c.bones[int(in.indices.x)] * q * in.weights.x + c.bones[int(in.indices.y)] * q * in.weights.y +
                          c.bones[int(in.indices.z)] * q * in.weights.z + c.bones[int(in.indices.w)] * q * in.weights.w).xyz);
    skinned.xyz += c.params.y * (n * (1.0 - 0.55 * c.params.z) + c.hang.xyz * (2.2 * c.params.z));
    return finish(c.m * skinned, in.uv, c.window[side], c.params);
}

// A strand is a ribbon turned to face the sun, wide enough for the map to hold.
static float3 strand_point(float3 out_, float3 down, float len, float k, float t, float side, float half_width, float3 drift, float4x4 m)
{
    float3 dir = out_ * (1.0 - k * t) + down * (k * t);
    float3 p = len * (out_ * (t - 0.5 * k * t * t) + down * (0.5 * k * t * t)) + drift * (len * t * t);
    p *= min(1.0, len * t / max(length(p), 1e-6)); // however it is blown about, a strand is no longer than it is
    float3 sunward = float3(m[0][2], m[1][2], m[2][2]);
    float3 across = cross(dir, sunward);
    across = dot(across, across) > 1e-12 ? normalize(across) : float3(1.0, 0.0, 0.0);
    return p + across * (side * half_width);
}
// Hair that streams behind does not hang as well: the faster, the less it droops.
static float streaming(float k, float3 drift) { return k * (1.0 - 0.9 * min(length(drift), 1.0)); }

// The mane and tail of a horse, laid out as add_shell_push in shader_xlate.cpp lays them out. params.y: the mane's length, z: the tail's.
vertex Out caster_strand(StrandIn in [[stage_in]], constant SkinnedCaster &c [[buffer(0)]], ushort side_ [[amplification_id]])
{
    float4 p = float4(in.position, 1.0);
    float4 skinned = c.bones[int(in.indices.x)] * p * in.weights.x + c.bones[int(in.indices.y)] * p * in.weights.y +
                     c.bones[int(in.indices.z)] * p * in.weights.z + c.bones[int(in.indices.w)] * p * in.weights.w;
    float4x4 b = c.bones[int(in.indices.x)] * in.weights.x + c.bones[int(in.indices.y)] * in.weights.y + c.bones[int(in.indices.z)] * in.weights.z + c.bones[int(in.indices.w)] * in.weights.w;
    float3x3 turn = float3x3(b[0].xyz, b[1].xyz, b[2].xyz);
    float v = floor(in.colour.x * 255.0 + 0.5), r = floor(in.colour.y * 255.0 + 0.5);
    float far = floor(v / 32.0) / 7.0;
    v -= floor(v / 32.0) * 32.0;
    float side = fmod(v, 2.0) * 2.0 - 1.0, t = floor(v / 2.0) / 8.0;
    float tail = step(128.0, r);
    r -= 128.0 * tail;
    float fall = step(64.0, r) * 2.0 - 1.0, chance = (r - 64.0 * step(64.0, r)) / 63.0;
    float3 n = normalize(turn * in.normal), down = c.hang.xyz;
    float3 out_ = tail > 0.5 ? normalize(normalize(turn * float3(0.0, -0.12, -1.0)) + n * 0.12)
                             : normalize(normalize(turn * (far > 0.5 ? float3(fall * 0.25, 1.0, 0.0) : float3(fall, (fract(chance * 5.7) - 0.5) * 0.9, 0.0))) * (0.3 + 0.35 * fract(chance * 3.1)) + n * 0.2 + down * 0.35);
    float len = (tail > 0.5 ? c.params.z * (1.0 - 0.6 * far) : c.params.y * (1.0 - 0.45 * far)) * (tail > 0.5 ? 0.7 + 0.3 * chance : 0.45 + 0.55 * chance);
    skinned.xyz += strand_point(out_, down, len, streaming(tail > 0.5 ? 0.85 : 1.0, c.drift.xyz), t, side, 0.006, c.drift.xyz, c.m);
    return finish(c.m * skinned, in.uv + float2(chance, far), c.window[side_], c.params);
}

// A person's, as add_hair_strands lays them out. params.y: the longest strand's length.
vertex Out caster_strand_person(PersonStrandIn in [[stage_in]], constant Caster &c [[buffer(0)]], ushort side_ [[amplification_id]])
{
    float v = floor(in.colour.x * 255.0 + 0.5), chance = in.colour.y;
    float side = fmod(v, 2.0) * 2.0 - 1.0, t = floor(v / 2.0) / 8.0;
    float3 n = normalize(in.normal), down = c.hang.xyz;
    float3 run = dot(in.run, in.run) > 0.0004 ? normalize(in.run) : down;
    float3 out_ = normalize(run + n * (fract(chance * 3.1) > 0.95 ? 0.45 : 0.02 + 0.14 * fract(chance * 3.1)));
    float len = c.params.y * in.colour.z * (0.4 + 0.6 * chance);
    float4 p = float4(in.position + strand_point(out_, down, len, streaming(0.4, c.drift.xyz), t, side, 0.003, c.drift.xyz, c.m), 1.0);
    return finish(c.m * p, in.uv + float2(chance, 0.0), c.window[side_], c.params);
}

// Hair lets about half the light past it: every other strand casts.
fragment float4 caster_strand_only(Out in [[stage_in]])
{
    if (fract(sin(dot(floor(in.uv * 4096.0), float2(12.9898, 78.233))) * 43758.5453) > 0.5) discard_fragment();
    return float4(in.depth);
}

fragment float4 caster_hair_only(Out in [[stage_in]], texture2d<float> hair [[texture(0)]], sampler s [[sampler(0)]])
{
    if (hair.sample(s, in.uv).b < 0.8) discard_fragment(); // long hair is 1 in a hair map's blue
    // Hair lets about half the light past it: every other strand casts, which
    // the receiving filter turns into half a shadow. Solid, a mane would put
    // itself and the neck under it in the dark.
    float strand = fract(sin(floor(in.uv.x * 700.0) * 12.9898 + floor(in.uv.y * 700.0) * 78.233) * 43758.5453);
    if (strand > 0.45) discard_fragment();
    return float4(in.depth);
}

fragment float4 caster_opaque(Out in [[stage_in]])
{
    return float4(in.depth);
}

fragment float4 caster_cutout(Out in [[stage_in]], texture2d<float> diffuse [[texture(0)]], sampler s [[sampler(0)]])
{
    if (diffuse.sample(s, in.uv).a < in.cutoff) discard_fragment();
    return float4(in.depth);
}
)";

id<MTLLibrary> s_caster_library;
id<MTLDepthStencilState> s_caster_depth_state;
NSUInteger s_caster_halves = 1; // 2 when the fine window is drawn too
std::unordered_map<PipelineKey, id<MTLRenderPipelineState>, PipelineKeyHash> s_caster_pipelines[7]; // [skinned * 2 + cutout], 4 for hair in layers, 5 and 6 for strands

// Called for each draw of the engine's depth pre-pass, after it has been issued.
void record_caster(NSUInteger count, MTLIndexType index_type, id<MTLBuffer> indices, NSUInteger index_offset, bool late)
{
    Program *p = find(s_programs, s_program);
    if (!p || !p->u_wvp || s_casters.size() >= 20000) return;
    uint64_t mesh = mesh_id(count, index_offset);
    if (late) {
        if (!p->u_world || p->u_bones || p->alpha_texture_location < 0 || p->is_blob_shadow || s_late_casters.size() >= 4000) return;
        if (s_prepass_meshes.count(mesh)) return;
    } else {
        s_prepass_meshes.insert(mesh);
    }
    // Cut-out scenery (bushes, grass, leaves) is drawn here all the same: the
    // engine's map is too coarse for twigs, whose shadows all but vanish in it.
    if (!late && p->alpha_texture_location < 0 && s_scenery.count(mesh)) {
        s_dyn_stats.scenery++;
        return;
    }
    if (!late && s_overlays.count(mesh)) {
        s_dyn_stats.overlays++;
        return;
    }
    CasterDraw c{};
    memcpy(c.key.format, s_last_key.format, sizeof c.key.format);
    memcpy(c.key.slot, s_last_key.slot, sizeof c.key.slot);
    memcpy(c.key.offset, s_last_key.offset, sizeof c.key.offset);
    memcpy(c.key.stride, s_last_key.stride, sizeof c.key.stride);
    if (!c.key.format[0]) return; // no position
    c.slots = s_draw_binding.slots;
    for (int i = 0; i < c.slots; i++) {
        Buffer *b = find(s_buffers, s_draw_binding.buffer[i]);
        if (!b || !b->mtl) return;
        c.vertex[i] = b->mtl;
        c.vertex_offset[i] = s_draw_binding.base[i];
    }
    c.indices = indices;
    c.index_offset = index_offset;
    c.index_count = count;
    c.index_type = index_type;
    memcpy(c.wvp, p->uniform_data[0].data() + (late ? p->u_world : p->u_wvp)->offset, sizeof c.wvp);
    c.world = late;
    c.cutoff = 0.03137255f; // the engine's own threshold in its pre-pass
    if (late) {
        c.cutoff = 0.5f;
        auto it = p->location_by_name.find("alpha_test_val");
        const XlUniform *u = it != p->location_by_name.end() ? p->locations[(size_t)it->second].uniform[1] : nullptr;
        if (u && u->components == 1 && !u->is_int) memcpy(&c.cutoff, p->uniform_data[1].data() + u->offset, 4);
        // Blended cut-outs test alpha against next to nothing and leave the soft
        // edge to the blend. A shadow has no such edge; half is where the shape ends.
        if (s_blend) c.cutoff = std::max(c.cutoff, 0.5f);
    }
    c.bones = -1;
    if (!late && p->u_bones && c.key.format[7] && c.key.format[8]) {
        c.bones = (int)(s_caster_bones.size() / (32 * 16));
        size_t at = s_caster_bones.size();
        s_caster_bones.resize(at + 32 * 16);
        size_t bytes = std::min<size_t>(32 * 64, p->uniform_data[0].size() - (size_t)p->u_bones->offset);
        memcpy(&s_caster_bones[at], p->uniform_data[0].data() + p->u_bones->offset, bytes);
    }
    if (p->alpha_texture_location >= 0) {
        UniformLocation &l = p->locations[(size_t)p->alpha_texture_location];
        Texture *t = (l.unit >= 0 && l.unit < MAX_UNITS) ? find(s_textures, s_bound_texture[l.unit]) : nullptr;
        if (t && t->mtl) {
            c.alpha = t->mtl;
            c.alpha_sampler = sampler_for(*t);
        }
    }
    (late ? s_late_casters : s_casters).push_back(c);
}

// Strands. Where a coat's hair map marks long hair, roots are scattered over
// the mesh's triangles, as many as their area asks for, and for each root a
// ribbon of k_strand_segments segments is made whose vertices are all copies of
// the mesh's vertex nearest the root, moved to the root. So the ribbon is
// drawn with the horse's own program, bones and lighting; what tells its
// vertices apart is their colour, from which the vertex shader lays the strand
// out (see add_shell_push in shader_xlate.cpp). The model itself is untouched.
const int k_strand_segments = 8; // the vertex shader divides by this

struct StrandMesh {
    GLuint buffer = 0; // in s_buffers, laid out like the mesh's own vertices
    id<MTLBuffer> indices;
    NSUInteger count = 0;
};
std::unordered_map<uint64_t, StrandMesh> s_strand_meshes;
GLuint s_strand_names = 0x7f000000; // buffer names of the shim's own, out of the game's way

// vertices, elements: the mesh as the game gave it. Attributes 0 (position),
// 2 (colour), 4 (texture coordinate), 7 and 8 (bone weights and numbers) must share the buffer.
// Without a map it is a person's hair (see add_hair_strands): a rigid mesh,
// all of it hair, with roots everywhere; the texture's clear parts are left to the shader.
const StrandMesh &build_strands(uint64_t key, const uint8_t *vertices, size_t vertex_bytes, const uint8_t *elements, size_t count, size_t width, id<MTLTexture> map)
{
    const bool person = map == nil;
    StrandMesh &mesh = s_strand_meshes[key];
    if (mesh.buffer) return mesh;
    mesh.buffer = s_strand_names++;
    const Attrib &pa = s_attribs[0], &ca = s_attribs[2], &ta = s_attribs[4], &wa = s_attribs[7], &ba = s_attribs[8];
    size_t stride = (size_t)pa.stride;
    bool fits = stride > 0 && pa.type == GL_FLOAT && ta.type == GL_FLOAT && ca.type == GL_UNSIGNED_BYTE && ca.size == 4 && ta.buffer == pa.buffer && ca.buffer == pa.buffer &&
                (person || (wa.buffer == pa.buffer && ba.buffer == pa.buffer && wa.enabled && ba.enabled)) && (size_t)ta.stride == stride && (size_t)ca.stride == stride &&
                (person || map.pixelFormat == MTLPixelFormatRGBA8Unorm);
    if (!fits) return mesh;
    size_t n = vertex_bytes / stride, mw = person ? 1 : map.width, mh = person ? 1 : map.height;
    std::vector<uint8_t> px(mw * mh * 4, 255); // a person's: long hair everywhere
    if (!person) [map getBytes:px.data() bytesPerRow:mw * 4 fromRegion:MTLRegionMake2D(0, 0, mw, mh) mipmapLevel:0];
    auto at = [&](float u, float v) {
        size_t x = std::min(mw - 1, (size_t)((u - floorf(u)) * (float)mw)), y = std::min(mh - 1, (size_t)((v - floorf(v)) * (float)mh));
        return &px[(y * mw + x) * 4];
    };
    auto component = [](const uint8_t *p, GLenum type, int k) {
        float f;
        if (type == GL_FLOAT) return memcpy(&f, p + k * 4, 4), f;
        return (float)p[k];
    };
    uint32_t seed = (uint32_t)key | 1;
    auto chance = [&] { // xorshift
        seed ^= seed << 13, seed ^= seed >> 17, seed ^= seed << 5;
        return (float)(seed & 0xffffff) / 16777216.0f;
    };
    std::vector<uint8_t> out;
    std::vector<uint32_t> indices;
    const size_t most = person ? 6000 : 12000; // strands to a mesh
    size_t strands = 0;
    float density = person ? s_people_hair_density : s_strand_density;
    // The bone a vertex hangs on most. The tail is what hangs on the last two bones of a horse's skeleton.
    auto bone_of = [&](const uint8_t *from) {
        int bone = 0;
        if (person) return bone;
        float heaviest = -1;
        for (int k = 0; k < std::min<int>(wa.size, 4); k++) {
            float weight = component(from + wa.offset, wa.type, k);
            if (weight > heaviest) heaviest = weight, bone = (int)component(from + ba.offset, ba.type, k);
        }
        return bone;
    };
    // A mane grows from the crest of the neck only, which lies on the model's
    // plane of symmetry (x = 0), and the hair there is closer set than a tail's.
    const float crest = 0.035f, closer = 6.0f;
    float tail_top = -1e9f, tail_bottom = 1e9f; // the model's z over the tail, found in the first pass
    for (int pass = 0; pass < 2; pass++) { // first to see how much skin there is to cover, so that the allowance is spread evenly
    double covered = 0;
    for (size_t i = 0; i + 2 < count && strands < most; i += 3) {
        uint32_t v[3];
        for (int k = 0; k < 3; k++) v[k] = width == 2 ? *(const uint16_t *)(elements + (i + (size_t)k) * 2) : *(const uint32_t *)(elements + (i + (size_t)k) * 4);
        if (v[0] >= n || v[1] >= n || v[2] >= n) continue;
        float p[3][3], t[3][2];
        for (int k = 0; k < 3; k++) memcpy(p[k], vertices + v[k] * stride + pa.offset, 12), memcpy(t[k], vertices + v[k] * stride + ta.offset, 8);
        // Any long hair on it? Its corners and middle are asked.
        bool any = at((t[0][0] + t[1][0] + t[2][0]) / 3, (t[0][1] + t[1][1] + t[2][1]) / 3)[2] > 204;
        for (int k = 0; k < 3; k++) any = any || at(t[k][0], t[k][1])[2] > 204;
        if (!any) continue;
        float e1[3] = {p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]}, e2[3] = {p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]};
        float cx = e1[1] * e2[2] - e1[2] * e2[1], cy = e1[2] * e2[0] - e1[0] * e2[2], cz = e1[0] * e2[1] - e1[1] * e2[0];
        // A person's hair runs the way its texture is painted: down the picture, which is towards v = 0.
        // (The band along the picture's top is a braid and a ribbon, lying sideways: nothing is rooted there.)
        float flow[3] = {0, 0, 0}, reach = 1; // reach: how long strands are here, as a share of the longest
        if (person) {
            float du1 = t[1][0] - t[0][0], dv1 = t[1][1] - t[0][1], du2 = t[2][0] - t[0][0], dv2 = t[2][1] - t[0][1], det = du1 * dv2 - du2 * dv1;
            if (fabsf(det) < 1e-9f) continue;
            if (std::max({t[0][1], t[1][1], t[2][1]}) > 0.93f) continue;
            for (int k = 0; k < 3; k++) flow[k] = -(e2[k] * du1 - e1[k] * du2) / det;
            float len = sqrtf(flow[0] * flow[0] + flow[1] * flow[1] + flow[2] * flow[2]);
            if (len < 1e-9f) continue;
            for (float &f : flow) f /= len;
            // len is how many metres the whole texture's length of hair is drawn over here: little for a
            // beard or a cropped head, whose painted hairs are short, and loose strands are short with them.
            reach = std::min(1.0f, std::max(0.12f, len * 0.12f / s_people_hair_length));
        }
        bool tail_here = bone_of(vertices + v[0] * stride) >= 26;
        if (!person && !tail_here && std::min({fabsf(p[0][0]), fabsf(p[1][0]), fabsf(p[2][0])}) > crest) continue;
        float area = 0.5f * sqrtf(cx * cx + cy * cy + cz * cz) * (tail_here || person ? 1.0f : closer);
        float wanted = area * density;
        covered += area;
        if (pass == 0) {
            if (tail_here)
                for (int k = 0; k < 3; k++) tail_top = std::max(tail_top, p[k][2]), tail_bottom = std::min(tail_bottom, p[k][2]);
            continue;
        }
        int roots = (int)wanted + (chance() < wanted - floorf(wanted) ? 1 : 0);
        for (int r = 0; r < roots && strands < most; r++) {
            float a = chance(), b = chance();
            if (a + b > 1) a = 1 - a, b = 1 - b;
            float w[3] = {1 - a - b, a, b};
            float root[3], uv[2];
            for (int k = 0; k < 3; k++) root[k] = w[0] * p[0][k] + w[1] * p[1][k] + w[2] * p[2][k];
            for (int k = 0; k < 2; k++) uv[k] = w[0] * t[0][k] + w[1] * t[1][k] + w[2] * t[2][k];
            const uint8_t *here = at(uv[0], uv[1]);
            float luck = chance(), luck2 = chance(); // drawn whether or not the root is kept, so that a map's edit moves no other strand
            if (here[2] <= 204) continue;
            int nearest = w[1] > w[0] ? (w[2] > w[1] ? 2 : 1) : (w[2] > w[0] ? 2 : 0);
            const uint8_t *from = vertices + v[nearest] * stride;
            int bone = bone_of(from);
            bool tail = bone >= 26;
            if (!person && !tail && fabsf(root[0]) > crest) continue;
            bool to_plus = fabsf(root[0]) > 0.004f ? root[0] > 0 : luck2 > 0.5f; // the side the mane falls to
            int far = tail && tail_top > tail_bottom ? std::min(7, std::max(0, (int)lroundf(7.0f * (tail_top - root[2]) / (tail_top - tail_bottom)))) : !tail && bone == 7 ? 7 : 0; // bone 7 is the head: the forelock
            uint32_t first = (uint32_t)(out.size() / stride);
            for (int segment = 0; segment <= k_strand_segments; segment++) {
                for (int side = 0; side < 2; side++) {
                    size_t o = out.size();
                    out.insert(out.end(), from, from + stride);
                    memcpy(&out[o + pa.offset], root, 12), memcpy(&out[o + ta.offset], uv, 8);
                    // Normal, tangent and binormal are blended between the corners as well, or the lighting would go in patches.
                    for (int attrib : {1, 5, 6}) {
                        const Attrib &na = s_attribs[attrib];
                        if (!na.enabled || na.buffer != pa.buffer || na.type != GL_FLOAT || na.size < 3) continue;
                        float blended[3] = {0, 0, 0}, corner[3];
                        for (int c = 0; c < 3; c++) {
                            memcpy(corner, vertices + v[c] * stride + na.offset, 12);
                            for (int k = 0; k < 3; k++) blended[k] += w[c] * corner[k];
                        }
                        float len = sqrtf(blended[0] * blended[0] + blended[1] * blended[1] + blended[2] * blended[2]);
                        if (len > 1e-6f) {
                            for (float &f : blended) f /= len;
                            memcpy(&out[o + na.offset], blended, 12);
                        }
                    }
                    uint8_t *colour = &out[o + ca.offset];
                    colour[0] = (uint8_t)(segment * 2 + side + far * 32), colour[1] = (uint8_t)((int)(luck * 63.0f) + (!tail && to_plus ? 64 : 0) + (tail ? 128 : 0)), colour[2] = here[0], colour[3] = here[1];
                    if (person) {
                        colour[0] = (uint8_t)(segment * 2 + side), colour[1] = (uint8_t)(luck * 255.0f), colour[2] = (uint8_t)(reach * 255.0f);
                        const Attrib &fa = s_attribs[6]; // the binormal, which this shader takes for the hair's run
                        if (fa.enabled && fa.buffer == pa.buffer && fa.type == GL_FLOAT && fa.size >= 3) memcpy(&out[o + fa.offset], flow, 12);
                    }
                }
                if (segment < k_strand_segments) {
                    uint32_t q = first + (uint32_t)segment * 2;
                    for (uint32_t k : {q, q + 1, q + 2, q + 1, q + 3, q + 2}) indices.push_back(k);
                }
            }
            strands++;
        }
    }
    if (pass == 0 && covered * density > (double)most * 0.9) density = (float)((double)most * 0.9 / covered);
    }
    if (indices.empty()) return mesh;
    s_buffers[mesh.buffer].mtl = [s_device newBufferWithBytes:out.data() length:out.size() options:MTLResourceStorageModeShared];
    mesh.indices = [s_device newBufferWithBytes:indices.data() length:indices.size() * 4 options:MTLResourceStorageModeShared];
    mesh.count = indices.size();
    if (getenv("WBM_DEBUG_STRANDS")) fprintf(stderr, "strands: %zu on a mesh of %zu triangles\n", strands, count / 3);
    return mesh;
}

// Long hair casts shadows too. Its layers exist only as draws of the main
// pass, so that is where they are taken from: a few of them per coat, to be
// drawn into the next frame's map like the grass (see s_late_casters).
void record_hair_casters(Program &p, Texture &coat, NSUInteger count, MTLIndexType index_type, id<MTLBuffer> indices, NSUInteger index_offset)
{
    if (!p.u_world || !p.u_bones || !coat.hair || s_late_casters.size() >= 4000) return;
    CasterDraw c{};
    memcpy(c.key.format, s_last_key.format, sizeof c.key.format);
    memcpy(c.key.slot, s_last_key.slot, sizeof c.key.slot);
    memcpy(c.key.offset, s_last_key.offset, sizeof c.key.offset);
    memcpy(c.key.stride, s_last_key.stride, sizeof c.key.stride);
    if (!c.key.format[0] || !c.key.format[1] || !c.key.format[7] || !c.key.format[8]) return;
    c.slots = s_draw_binding.slots;
    for (int i = 0; i < c.slots; i++) {
        Buffer *b = find(s_buffers, s_draw_binding.buffer[i]);
        if (!b || !b->mtl) return;
        c.vertex[i] = b->mtl;
        c.vertex_offset[i] = s_draw_binding.base[i];
    }
    c.indices = indices, c.index_offset = index_offset, c.index_count = count, c.index_type = index_type;
    memcpy(c.wvp, p.uniform_data[0].data() + p.u_world->offset, sizeof c.wvp);
    c.world = true;
    c.bones = -1;
    c.own_bones = std::make_shared<std::vector<float>>(32 * 16, 0.0f);
    memcpy(c.own_bones->data(), p.uniform_data[0].data() + p.u_bones->offset, std::min<size_t>(32 * 64, p.uniform_data[0].size() - (size_t)p.u_bones->offset));
    c.alpha = coat.hair;
    c.alpha_sampler = sampler_for(coat);
    c.hair = true;
    // Down, in the space the world matrix starts from: the world's, turned back by the matrix's rotation.
    float len = sqrtf(c.wvp[2] * c.wvp[2] + c.wvp[6] * c.wvp[6] + c.wvp[10] * c.wvp[10]);
    if (len > 1e-6f) c.down[0] = -c.wvp[2] / len, c.down[1] = -c.wvp[6] / len, c.down[2] = -c.wvp[10] / len;
    for (int layer = 1; layer <= 2; layer++) {
        c.share = (float)layer / 2.0f;
        c.push = s_hair_length * c.share;
        s_late_casters.push_back(c);
    }
}

// Strands cast shadows as strands: the ribbons just drawn are drawn again
// into the next frame's map (see s_late_casters), laid out by the map's own
// shaders. Called while the strands' buffer is still bound and prepared.
// kind: 1 a horse's, 2 a person's. drift: as the strands were given it, in the world's space.
void record_strand_caster(Program &p, const StrandMesh &strands, int kind, float first, float second, const float drift[3])
{
    if (!p.u_world || s_late_casters.size() >= 4000 || (kind == 1 && !p.u_bones)) return;
    CasterDraw c{};
    memcpy(c.key.format, s_last_key.format, sizeof c.key.format);
    memcpy(c.key.slot, s_last_key.slot, sizeof c.key.slot);
    memcpy(c.key.offset, s_last_key.offset, sizeof c.key.offset);
    memcpy(c.key.stride, s_last_key.stride, sizeof c.key.stride);
    const int horse[6] = {0, 1, 2, 4, 7, 8}, person[5] = {0, 1, 2, 4, 6};
    for (int k = 0; k < (kind == 1 ? 6 : 5); k++)
        if (!c.key.format[(kind == 1 ? horse : person)[k]]) return;
    c.slots = s_draw_binding.slots;
    for (int i = 0; i < c.slots; i++) {
        Buffer *b = find(s_buffers, s_draw_binding.buffer[i]);
        if (!b || !b->mtl) return;
        c.vertex[i] = b->mtl;
        c.vertex_offset[i] = s_draw_binding.base[i];
    }
    c.indices = strands.indices, c.index_offset = 0, c.index_count = strands.count, c.index_type = MTLIndexTypeUInt32;
    memcpy(c.wvp, p.uniform_data[0].data() + p.u_world->offset, sizeof c.wvp);
    c.world = true;
    c.bones = -1;
    if (kind == 1) {
        c.own_bones = std::make_shared<std::vector<float>>(32 * 16, 0.0f);
        memcpy(c.own_bones->data(), p.uniform_data[0].data() + p.u_bones->offset, std::min<size_t>(32 * 64, p.uniform_data[0].size() - (size_t)p.u_bones->offset));
    }
    c.strand = kind, c.push = first, c.share = second;
    // Down and the drift, in the space the world matrix starts from: the world's, turned back by the matrix's rotation.
    const float *w = c.wvp;
    float len = sqrtf(w[2] * w[2] + w[6] * w[6] + w[10] * w[10]);
    if (len > 1e-6f) c.down[0] = -w[2] / len, c.down[1] = -w[6] / len, c.down[2] = -w[10] / len;
    for (int k = 0; k < 3; k++) c.drift[k] = drift[0] * w[k * 4] + drift[1] * w[k * 4 + 1] + drift[2] * w[k * 4 + 2];
    s_late_casters.push_back(c);
}

// Hair has weight: when its owner moves off, the strands' tips are left
// behind a moment, and settle again when it stops. Each thing that has
// strands is followed from frame to frame by where its world matrix puts it
// (several may share a mesh: the nearest of last frame's is taken for the
// same one), and how far its strands trail is eased after how fast it goes.
// The result: which way and how far the tips trail, in the world's space, as a
// share of a strand's length.
struct Mover {
    float place[3], drift[3];
    double time;
    uint64_t frame;
};
std::unordered_map<uint64_t, std::vector<Mover>> s_movers;

void drift_of(uint64_t mesh, const float *world, float out[3])
{
    static const char *fixed = getenv("WBM_TEST_DRIFT"); // "x y z", to see it in a still
    if (fixed && sscanf(fixed, "%f %f %f", &out[0], &out[1], &out[2]) == 3) return;
    std::vector<Mover> &movers = s_movers[mesh];
    const float place[3] = {world[12], world[13], world[14]};
    Mover *same = nullptr;
    float nearest = 2.0f * 2.0f; // metres, squared: farther off than that it is another
    for (Mover &m : movers) {
        if (m.frame == s_frame_serial) continue; // taken by another this frame
        float d = 0;
        for (int k = 0; k < 3; k++) d += (place[k] - m.place[k]) * (place[k] - m.place[k]);
        if (d < nearest) nearest = d, same = &m;
    }
    if (!same) {
        for (size_t i = 0; i < movers.size();) // those not seen for a while make room
            if (movers[i].frame + 120 < s_frame_serial) movers[i] = movers.back(), movers.pop_back();
            else i++;
        movers.push_back(Mover{{place[0], place[1], place[2]}, {0, 0, 0}, s_wind_time, s_frame_serial});
        out[0] = out[1] = out[2] = 0;
        return;
    }
    double dt = s_wind_time - same->time;
    if (dt > 1e-4 && dt < 0.5) {
        float target[3], length = 0;
        float ahead = sqrtf(world[4] * world[4] + world[5] * world[5] + world[6] * world[6]); // the model's +y is the way it faces
        for (int k = 0; k < 3; k++) {
            float speed = (place[k] - same->place[k]) / (float)dt + (ahead > 1e-6f ? world[4 + k] / ahead * s_pretend_speed : 0.0f);
            target[k] = -speed * 0.09f, length += target[k] * target[k]; // at a gallop, 10 m/s, the tips trail by nine tenths of the length
        }
        length = sqrtf(length);
        float ease = 1.0f - expf(-(float)dt * 5.0f);
        for (int k = 0; k < 3; k++) same->drift[k] += ((length > 0.95f ? target[k] * 0.95f / length : target[k]) - same->drift[k]) * ease;
    } else if (dt >= 0.5) {
        same->drift[0] = same->drift[1] = same->drift[2] = 0;
    }
    if (dt > 1e-4 || dt < 0) memcpy(same->place, place, sizeof place), same->time = s_wind_time;
    same->frame = s_frame_serial;
    memcpy(out, same->drift, 12);
}
float s_drift_now[4]; // the shaders' wbm_drift for the draw being prepared

id<MTLRenderPipelineState> caster_pipeline(const PipelineKey &key, bool skinned, bool cutout, bool hair = false, int strand = 0)
{
    auto &cache = s_caster_pipelines[strand ? 4 + strand : hair ? 4 : skinned * 2 + cutout];
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = [s_caster_library newFunctionWithName:strand == 1 ? @"caster_strand" : strand == 2 ? @"caster_strand_person" : hair ? @"caster_hair" : skinned ? @"caster_skinned" : @"caster_static"];
    d.fragmentFunction = [s_caster_library newFunctionWithName:strand ? @"caster_strand_only" : hair ? @"caster_hair_only" : cutout ? @"caster_cutout" : @"caster_opaque"];
    MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
    const int plain[5] = {0, 4, 7, 8, 1}, horse[6] = {0, 4, 7, 8, 1, 2}, person[5] = {0, 4, 1, 2, 6};
    const int *wanted = strand == 1 ? horse : strand == 2 ? person : plain;
    bool uses_constant = false;
    for (int n = 0; n < (strand == 1 ? 6 : strand == 2 ? 5 : hair ? 5 : skinned ? 4 : 2); n++) {
        int i = wanted[n];
        if (key.format[i] && key.slot[i] != CONSTANT_SLOT) {
            vd.attributes[i].format = (MTLVertexFormat)key.format[i];
            vd.attributes[i].offset = key.offset[i];
            vd.attributes[i].bufferIndex = VERTEX_SLOT_BASE + key.slot[i];
        } else { // the mesh has no such attribute: it reads the constant buffer
            vd.attributes[i].format = MTLVertexFormatFloat4;
            vd.attributes[i].offset = 0;
            vd.attributes[i].bufferIndex = CONSTANT_SLOT;
            uses_constant = true;
        }
    }
    for (int s = 0; s < MAX_VERTEX_SLOTS; s++) {
        if (!key.stride[s]) continue;
        vd.layouts[VERTEX_SLOT_BASE + s].stride = key.stride[s];
        vd.layouts[VERTEX_SLOT_BASE + s].stepFunction = MTLVertexStepFunctionPerVertex;
    }
    if (uses_constant) {
        vd.layouts[CONSTANT_SLOT].stride = 16;
        vd.layouts[CONSTANT_SLOT].stepFunction = MTLVertexStepFunctionConstant;
        vd.layouts[CONSTANT_SLOT].stepRate = 0;
    }
    d.vertexDescriptor = vd;
    d.colorAttachments[0].pixelFormat = MTLPixelFormatR32Float;
    d.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    d.maxVertexAmplificationCount = s_caster_halves;
    NSError *error = nil;
    id<MTLRenderPipelineState> state = [s_device newRenderPipelineStateWithDescriptor:d error:&error];
    if (!state) complain("shadow caster pipeline failed: %s", error.localizedDescription.UTF8String);
    cache[key] = state;
    return state;
}

// p is the program of a main-pass draw that has both a world-view-projection
// and a world matrix set, from which the camera is recovered.
void render_dynamic_shadows(Program &p)
{
    s_dyn_frame = s_frame_serial; // one attempt per frame, whatever comes of it
    std::vector<CasterDraw> casters;
    casters.swap(s_casters);
    std::vector<float> bones;
    bones.swap(s_caster_bones);
    size_t from_prepass = casters.size();
    casters.insert(casters.end(), s_late_ready.begin(), s_late_ready.end());
    s_dyn_window[2] = s_dyn_window[3] = 0; // no map, until this succeeds
    s_dyn_near[2] = s_dyn_near[3] = 0;
    if (!s_sun_matrix_valid) return;

    double wvp[16], world[16], inv[16], view_proj[16], sun[16];
    if (!vertex_matrix(p, "matWorldViewProj", wvp) || !vertex_matrix(p, "matWorld", world) || !mat_inverse(world, inv)) return;
    mat_mul(wvp, inv, view_proj);
    for (int i = 0; i < 16; i++) sun[i] = s_sun_matrix[i];

    // Inverting the view-projection itself to get from a draw's clip space back
    // to the world loses the position along the view axis to rounding: a
    // centimetre or two at a hundred metres, different every frame the camera
    // moves, which made distant shadows shimmer. The parts of a projected
    // position that do not go through the depth mapping are enough: x and y
    // divided by the projection's scales, and w, are a rigid transform of the
    // world position (`frame`), and that inverts without loss.
    double p00 = sqrt(view_proj[0] * view_proj[0] + view_proj[4] * view_proj[4] + view_proj[8] * view_proj[8]);
    double p11 = sqrt(view_proj[1] * view_proj[1] + view_proj[5] * view_proj[5] + view_proj[9] * view_proj[9]);
    if (p00 < 1e-9 || p11 < 1e-9) return;
    auto rigid = [&](const double *clip, double *out) { // rows (x / p00, y / p11, w, 0 0 0 1) of a clip-space matrix
        for (int c = 0; c < 4; c++) {
            out[c * 4 + 0] = clip[c * 4 + 0] / p00;
            out[c * 4 + 1] = clip[c * 4 + 1] / p11;
            out[c * 4 + 2] = clip[c * 4 + 3];
            out[c * 4 + 3] = c == 3 ? 1.0 : 0.0;
        }
    };
    double frame[16], inv_frame[16];
    rigid(view_proj, frame);
    if (!mat_inverse(frame, inv_frame)) return;

    if (!s_caster_library) {
        NSError *error = nil;
        // The default language version follows the SDK the game was built with,
        // which is too old for vertex amplification.
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion2_2;
        s_caster_library = [s_device newLibraryWithSource:@(k_caster_source) options:options error:&error];
        if (!s_caster_library) {
            complain("shadow caster shaders did not compile: %s", error.localizedDescription.UTF8String);
            s_dyn_shadows = false;
            return;
        }
        MTLDepthStencilDescriptor *dd = [MTLDepthStencilDescriptor new];
        dd.depthCompareFunction = MTLCompareFunctionLess;
        dd.depthWriteEnabled = YES;
        s_caster_depth_state = [s_device newDepthStencilStateWithDescriptor:dd];
        // The fine window needs every draw rasterised twice in one go.
        s_caster_halves = s_dyn_near_range > 0 && [s_device supportsVertexAmplificationCount:2] ? 2 : 1;
        const NSUInteger size = k_xl_dyn_size;
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float width:size * 2 height:size mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModePrivate;
        s_dyn_map = [s_device newTextureWithDescriptor:td];
        td.pixelFormat = MTLPixelFormatDepth32Float;
        td.usage = MTLTextureUsageRenderTarget;
        s_dyn_depth = [s_device newTextureWithDescriptor:td];
        MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new]; // nearest: the shader compares depths itself
        sd.sAddressMode = sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
        s_dyn_sampler = [s_device newSamplerStateWithDescriptor:sd];
    }

    // Where the camera is, and which way it faces: the direction in which w grows.
    double eye[3] = {inv_frame[12], inv_frame[13], inv_frame[14]};
    for (int k = 0; k < 3; k++) s_eye[k] = (float)eye[k];
    s_eye_frame = s_frame_serial;
    double fx = view_proj[3], fy = view_proj[7], flat = sqrt(fx * fx + fy * fy); // Z is up
    double reach = s_dyn_range * 0.6;
    double focus[3] = {eye[0] + (flat > 1e-6 ? fx / flat * reach : 0), eye[1] + (flat > 1e-6 ? fy / flat * reach : 0), eye[2]};

    // The window: centred on that point in map coordinates, s_dyn_range metres to
    // each side, snapped to its own texel grid so that shadows do not crawl as
    // the camera moves.
    auto row_length = [&](int r) { return sqrt(sun[r] * sun[r] + sun[4 + r] * sun[4 + r] + sun[8 + r] * sun[8 + r]); };
    double half_u = s_dyn_range * row_length(0), half_v = s_dyn_range * row_length(1);
    if (half_u < 1e-9 || half_v < 1e-9) return;
    double cu = sun[0] * focus[0] + sun[4] * focus[1] + sun[8] * focus[2] + sun[12];
    double cv = sun[1] * focus[0] + sun[5] * focus[1] + sun[9] * focus[2] + sun[13];
    double texel_u = 2 * half_u / k_xl_dyn_size, texel_v = 2 * half_v / k_xl_dyn_size;
    cu = round(cu / texel_u) * texel_u;
    cv = round(cv / texel_v) * texel_v;
    // The fine window: the same, s_dyn_near_range metres to each side of a point
    // just ahead of the camera.
    double near_u = s_dyn_near_range * row_length(0), near_v = s_dyn_near_range * row_length(1), nu = 0, nv = 0;
    bool fine = s_caster_halves == 2 && near_u > 1e-9 && near_v > 1e-9;
    if (fine) {
        double ahead_by = s_dyn_near_range * 0.5;
        double point[3] = {eye[0] + (flat > 1e-6 ? fx / flat * ahead_by : 0), eye[1] + (flat > 1e-6 ? fy / flat * ahead_by : 0), eye[2]};
        nu = sun[0] * point[0] + sun[4] * point[1] + sun[8] * point[2] + sun[12];
        nv = sun[1] * point[0] + sun[5] * point[1] + sun[9] * point[2] + sun[13];
        double step_u = 2 * near_u / k_xl_dyn_size, step_v = 2 * near_v / k_xl_dyn_size;
        nu = round(nu / step_u) * step_u;
        nv = round(nv / step_v) * step_v;
    }
    float window[8] = {(float)cu, (float)cv, (float)(1 / half_u), (float)(1 / half_v), 0, 0, 0, 0};
    if (fine) window[4] = (float)nu, window[5] = (float)nv, window[6] = (float)(1 / near_u), window[7] = (float)(1 / near_v);

    double to_shadow[16]; // from `frame` to shadow space
    mat_mul(sun, inv_frame, to_shadow);

    // All the casters' uniforms go into one buffer.
    const size_t static_size = 256, skinned_size = 2304; // 144 and 2192 bytes, rounded up to the offset alignment
    size_t total = 0;
    for (const CasterDraw &c : casters) total += c.bones >= 0 || c.own_bones ? skinned_size : static_size;
    if (!total) return;
    id<MTLBuffer> uniforms = [s_device newBufferWithLength:total options:MTLResourceStorageModeShared];
    uint8_t *base = (uint8_t *)uniforms.contents;

    id<MTLCommandBuffer> cmd = [s_queue commandBuffer];
    MTLRenderPassDescriptor *d = [MTLRenderPassDescriptor renderPassDescriptor];
    d.colorAttachments[0].texture = s_dyn_map;
    d.colorAttachments[0].loadAction = MTLLoadActionClear;
    d.colorAttachments[0].storeAction = MTLStoreActionStore;
    d.colorAttachments[0].clearColor = MTLClearColorMake(1e9, 0, 0, 0); // nothing nearer the sun
    d.depthAttachment.texture = s_dyn_depth;
    d.depthAttachment.loadAction = MTLLoadActionClear;
    d.depthAttachment.storeAction = MTLStoreActionDontCare;
    d.depthAttachment.clearDepth = 1.0;
    id<MTLRenderCommandEncoder> pass = [cmd renderCommandEncoderWithDescriptor:d];
    [pass setDepthStencilState:s_caster_depth_state];
    [pass setCullMode:MTLCullModeNone];
    [pass setVertexBuffer:s_constant_attrib offset:0 atIndex:CONSTANT_SLOT];
    const double N = k_xl_dyn_size;
    const MTLViewport viewports[2] = {{0, 0, N, N, 0, 1}, {N, 0, N, N, 0, 1}};
    const MTLScissorRect scissors[2] = {{0, 0, k_xl_dyn_size, k_xl_dyn_size}, {k_xl_dyn_size, 0, k_xl_dyn_size, k_xl_dyn_size}};
    NSUInteger halves = fine ? 2 : 1;
    [pass setViewports:viewports count:halves];
    [pass setScissorRects:scissors count:halves];
    if (s_caster_halves == 2) {
        MTLVertexAmplificationViewMapping mappings[2] = {{0, 0}, {1, 0}};
        [pass setVertexAmplificationCount:halves viewMappings:mappings];
    }
    size_t at = 0;
    void *last_pipeline = nullptr;
    for (const CasterDraw &c : casters) {
        bool skinned = c.bones >= 0 || c.own_bones;
        size_t size = skinned ? skinned_size : static_size;
        id<MTLRenderPipelineState> pipeline = caster_pipeline(c.key, skinned, c.alpha != nil, c.hair, c.strand);
        if (!pipeline) {
            at += size;
            continue;
        }
        double m[16], object[16], placed[16];
        for (int i = 0; i < 16; i++) object[i] = c.wvp[i];
        if (!c.world) rigid(object, placed);
        mat_mul(c.world ? sun : to_shadow, c.world ? object : placed, m);
        float m32[16];
        for (int i = 0; i < 16; i++) m32[i] = (float)m[i];
        memcpy(base + at, m32, 64);
        memcpy(base + at + 64, window, 32);
        const float params[4] = {c.cutoff, c.push, c.share, 0}, hang[4] = {c.down[0], c.down[1], c.down[2], 0};
        memcpy(base + at + 96, params, 16);
        memcpy(base + at + 112, hang, 16);
        const float drift[4] = {c.drift[0], c.drift[1], c.drift[2], 0};
        memcpy(base + at + 128, drift, 16);
        if (skinned) memcpy(base + at + 144, c.own_bones ? c.own_bones->data() : &bones[(size_t)c.bones * 32 * 16], 32 * 64);
        if (last_pipeline != (__bridge void *)pipeline) {
            [pass setRenderPipelineState:pipeline];
            last_pipeline = (__bridge void *)pipeline;
        }
        [pass setVertexBuffer:uniforms offset:at atIndex:0];
        for (int i = 0; i < c.slots; i++) [pass setVertexBuffer:c.vertex[i] offset:c.vertex_offset[i] atIndex:(NSUInteger)(VERTEX_SLOT_BASE + i)];
        if (c.alpha) {
            [pass setFragmentTexture:c.alpha atIndex:0];
            [pass setFragmentSamplerState:c.alpha_sampler atIndex:0];
        }
        [pass drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:c.index_count indexType:c.index_type indexBuffer:c.indices indexBufferOffset:c.index_offset];
        at += size;
    }
    [pass endEncoding];
    [cmd commit];

    s_dyn_window[0] = (float)cu;
    s_dyn_window[1] = (float)cv;
    s_dyn_window[2] = (float)(1 / (2 * half_u));
    s_dyn_window[3] = (float)(1 / (2 * half_v));
    if (fine) {
        s_dyn_near[0] = (float)nu;
        s_dyn_near[1] = (float)nv;
        s_dyn_near[2] = (float)(1 / (2 * near_u));
        s_dyn_near[3] = (float)(1 / (2 * near_v));
    }
    s_dyn_bias = (float)(s_dyn_bias_m * row_length(2));
    s_dyn_stats.frames++;
    s_dyn_stats.casters += from_prepass;
    s_dyn_stats.late += casters.size() - from_prepass;
}

// Binds everything a draw needs. Returns false when the draw has to be skipped.
bool prepare_draw()
{
    Program *p = find(s_programs, s_program);
    if (!p || !p->linked) return false;
    if (!p->functions_ready && !finish_compile(*p)) return false;
    if (!ensure_pass()) return false;

    PipelineKey key;
    memset(&key, 0, sizeof key);
    VertexBinding binding;
    if (!build_vertex_layout(*p, key, binding)) return false;
    key.color_format = (uint16_t)(s_pass_color ? s_pass_color.pixelFormat : MTLPixelFormatInvalid);
    key.depth_format = (uint16_t)(s_pass_depth ? s_pass_depth.pixelFormat : MTLPixelFormatInvalid);
    key.samples = (uint8_t)s_pass_samples;
    if (s_blend) {
        key.blend = 1;
        key.blend_src = (uint16_t)s_blend_src;
        key.blend_dst = (uint16_t)s_blend_dst;
        key.blend_eq = (uint16_t)s_blend_eq;
    }
    id<MTLRenderPipelineState> pipeline;
    if (s_state_cache && s_last_key_program == s_program && s_last_key == key) {
        pipeline = s_last_pipeline;
    } else {
        pipeline = pipeline_for(*p, key);
        s_last_key_program = s_program;
        s_last_key = key;
        s_last_pipeline = pipeline;
    }
    if (!pipeline) return false;
    if (!already(s_set.pipeline == (__bridge void *)pipeline)) {
        [s_pass setRenderPipelineState:pipeline];
        s_set.pipeline = (__bridge void *)pipeline;
    }
    id<MTLDepthStencilState> depth = depth_state(s_pass_depth != nil);
    if (!already(s_set.depth_state == (__bridge void *)depth)) {
        [s_pass setDepthStencilState:depth];
        s_set.depth_state = (__bridge void *)depth;
    }

    // Textures are rendered upside down so that their rows are stored in OpenGL's
    // order (see wbm_yflip in shader_xlate.h). That mirrors the winding, and the
    // viewport then counts from the top like Metal's own.
    bool flip = !s_pass_screen;
    double target_h = s_pass_color ? (double)s_pass_color.height : (double)s_pass_depth.height;
    double k = s_pass_scale, vw = s_viewport.w * k, vh = s_viewport.h * k;
    double viewport[4] = {s_viewport.x * k, flip ? s_viewport.y * k : target_h - s_viewport.y * k - vh, vw, vh};
    if (!already(!memcmp(viewport, s_set.viewport, sizeof viewport))) {
        [s_pass setViewport:(MTLViewport){viewport[0], viewport[1], viewport[2], viewport[3], 0.0, 1.0}];
        memcpy(s_set.viewport, viewport, sizeof viewport);
    }
    int winding = flip ? MTLWindingClockwise : MTLWindingCounterClockwise;
    if (!already(s_set.winding == winding)) [s_pass setFrontFacingWinding:(MTLWinding)(s_set.winding = winding)];
    int cull = !s_cull ? MTLCullModeNone : s_cull_face == GL_FRONT ? MTLCullModeFront : MTLCullModeBack;
    if (!already(s_set.cull == cull)) [s_pass setCullMode:(MTLCullMode)(s_set.cull = cull)];
    int fill = s_polygon_mode == GL_LINE ? MTLTriangleFillModeLines : MTLTriangleFillModeFill;
    if (!already(s_set.fill == fill)) [s_pass setTriangleFillMode:(MTLTriangleFillMode)(s_set.fill = fill)];

    float samples = (float)s_pass_samples, yflip = flip ? -1.0f : 1.0f;
    if (p->samples_offset >= 0 && memcmp(p->uniform_data[1].data() + p->samples_offset, &samples, 4)) {
        memcpy(p->uniform_data[1].data() + p->samples_offset, &samples, 4);
        p->uniform_dirty[1] = true;
    }
    if (p->dyn_window_offset >= 0 && memcmp(p->uniform_data[1].data() + p->dyn_window_offset, s_dyn_window, 16)) {
        memcpy(p->uniform_data[1].data() + p->dyn_window_offset, s_dyn_window, 16);
        p->uniform_dirty[1] = true;
    }
    for (int stage = 0; stage < 2; stage++) {
        if (p->amb_offset[stage] >= 0 && memcmp(p->uniform_data[stage].data() + p->amb_offset[stage], s_amb, sizeof s_amb)) {
            memcpy(p->uniform_data[stage].data() + p->amb_offset[stage], s_amb, sizeof s_amb);
            p->uniform_dirty[stage] = true;
        }
    }
    Texture *diffuse = nullptr; // the diffuse texture says what the surface is
    if ((p->material_offset >= 0 || p->pbr_offset >= 0 || p->skin_offset[0] >= 0 || p->skin_offset[1] >= 0 || p->wind_offset >= 0 || p->leaf_sun_offset >= 0 || p->gaze_offset >= 0 || p->fur_offset >= 0 || s_debug_skin) && p->alpha_texture_location >= 0) {
        UniformLocation &l = p->locations[(size_t)p->alpha_texture_location];
        diffuse = (l.unit >= 0 && l.unit < MAX_UNITS) ? find(s_textures, s_bound_texture[l.unit]) : nullptr;
    }
    if (s_debug_skin && diffuse && diffuse->material == MAT_FLORA && (p->wind_offset < 0 || (p->skin_offset[0] < 0 && p->skin_offset[1] < 0 && p->leaf_sun_offset < 0)))
        complain("foliage drawn by program %u: wind %s, light through leaves %s", s_program, p->wind_offset >= 0 ? "yes" : "NO",
                 p->skin_offset[0] >= 0 || p->skin_offset[1] >= 0 || p->leaf_sun_offset >= 0 ? "yes" : "NO");
    bool unlit_leaf = diffuse && diffuse->material == MAT_FLORA && p->skin_offset[0] < 0 && p->skin_offset[1] < 0 && p->leaf_sun_offset < 0;
    if (s_debug_skin && diffuse && (unlit_leaf || (diffuse->material == MAT_SKIN && p->skin_offset[0] < 0 && p->skin_offset[1] < 0))) {
        // Skin drawn by a shader the scattering rewrite did not recognise: show how it lights.
        for (auto &sh : p->shaders) {
            std::string lines;
            size_t at = 0;
            while (at < sh->source.size() && lines.size() < 700) {
                size_t end = sh->source.find(';', at);
                if (end == std::string::npos) break;
                std::string st = sh->source.substr(at, end - at);
                if (st.find("Sun") != std::string::npos && st.find("uniform") == std::string::npos && st.find("varying") == std::string::npos) {
                    for (char &c : st)
                        if (c == '\n' || c == '\r') c = ' ';
                    lines += st + ";";
                }
                at = end + 1;
            }
            complain("%s without scattering, program %u %s: %s", unlit_leaf ? "foliage" : "skin", s_program, sh->type == GL_VERTEX_SHADER ? "vs" : "fs", lines.c_str());
        }
    }
    for (int stage = 0; stage < 2; stage++) {
        if (p->shell_offset[stage] < 0 || !memcmp(p->uniform_data[stage].data() + p->shell_offset[stage], s_shell_now, 16)) continue;
        memcpy(p->uniform_data[stage].data() + p->shell_offset[stage], s_shell_now, 16);
        p->uniform_dirty[stage] = true;
    }
    if (p->drift_offset >= 0 && memcmp(p->uniform_data[0].data() + p->drift_offset, s_drift_now, 16)) {
        memcpy(p->uniform_data[0].data() + p->drift_offset, s_drift_now, 16);
        p->uniform_dirty[0] = true;
    }
    if (p->fur_offset >= 0) {
        const float fur[4] = {diffuse && (diffuse->pbr || diffuse->hair) ? s_fur_strength : 0.0f, s_dyn_near[2] > 0 && s_dyn_bias > 0 ? s_dyn_bias_m / s_dyn_bias : 0.0f,
                              s_skin_backlight, diffuse && diffuse->hair ? s_texture_per_metre : -s_texture_per_metre};
        if (p->hair_colour_offset >= 0) {
            float colour[4] = {0, 0, 0, 0};
            if (diffuse && diffuse->hair_colour[3] > 0) {
                // In whatever form the shader gets the texture's own colour: decoded by the GPU, or as painted.
                for (int k = 0; k < 3; k++) colour[k] = s_shader_options.srgb_textures ? powf(diffuse->hair_colour[k], 2.2f) : diffuse->hair_colour[k];
                colour[3] = 1;
            }
            if (memcmp(p->uniform_data[1].data() + p->hair_colour_offset, colour, 16)) {
                memcpy(p->uniform_data[1].data() + p->hair_colour_offset, colour, 16);
                p->uniform_dirty[1] = true;
            }
        }
        if (p->horse_eye_offset >= 0) {
            const float eye[4] = {diffuse ? diffuse->horse_eye[0] : 0, diffuse ? diffuse->horse_eye[1] : 0, diffuse && diffuse->horse_eye[2] > 0 ? diffuse->horse_eye[2] : 1.0f,
                                  diffuse && diffuse->horse_eye[2] > 0 && s_shader_options.eyes ? 1.0f : 0.0f};
            if (memcmp(p->uniform_data[1].data() + p->horse_eye_offset, eye, 16)) {
                memcpy(p->uniform_data[1].data() + p->horse_eye_offset, eye, 16);
                p->uniform_dirty[1] = true;
            }
        }
        if (memcmp(p->uniform_data[1].data() + p->fur_offset, fur, 16)) {
            memcpy(p->uniform_data[1].data() + p->fur_offset, fur, 16);
            p->uniform_dirty[1] = true;
        }
    }
    if (p->gaze_offset >= 0) {
        float gaze[4] = {0, 0, 0, 0};
        float phase = 0; // a face's own, from where its head is, so that a crowd does not glance about as one
        if (p->u_world) {
            const float *w = (const float *)(p->uniform_data[0].data() + p->u_world->offset);
            phase = floorf(w[12] * 3.0f) * 12.9898f + floorf(w[13] * 3.0f) * 78.233f;
        }
        eye_gaze(s_wind_time, phase, gaze);
        gaze[2] = diffuse && diffuse->woman ? 1.0f : 0.0f;
        if (memcmp(p->uniform_data[1].data() + p->gaze_offset, gaze, 16)) {
            memcpy(p->uniform_data[1].data() + p->gaze_offset, gaze, 16);
            p->uniform_dirty[1] = true;
        }
    }
    if (p->leaf_sun_offset >= 0) {
        float sun[4] = {0, 0, 0, 0};
        // The direction to the sun in view space, from the camera worked out for this frame or the last.
        if (diffuse && diffuse->material == MAT_FLORA && s_sun_valid && s_camera.view_frame + 2 >= s_frame_serial) {
            double v[3], len = 0;
            for (int r = 0; r < 3; r++) {
                v[r] = -(s_camera.view[r][0] * s_sun_dir[0] + s_camera.view[r][1] * s_sun_dir[1] + s_camera.view[r][2] * s_sun_dir[2]);
                len += v[r] * v[r];
            }
            v[2] *= s_camera.forward_sign; // the third axis is kept pointing forwards; the shaders' own may point back
            if (len > 1e-12)
                for (int r = 0; r < 3; r++) sun[r] = (float)(v[r] / sqrt(len));
            sun[3] = s_foliage_backlight;
        }
        if (memcmp(p->uniform_data[0].data() + p->leaf_sun_offset, sun, 16)) {
            memcpy(p->uniform_data[0].data() + p->leaf_sun_offset, sun, 16);
            p->uniform_dirty[0] = true;
        }
    }
    if (p->wind_offset >= 0) {
        float wind[4] = {0, 0, 0, 0};
        if (diffuse && diffuse->material == MAT_FLORA && s_wind_strength > 0) {
            // A mesh in its own space has its foot at height 0. A batch placed in the
            // world (its vertices far from the origin) needs heights worked out for it.
            Buffer *vb = s_attribs[0].enabled ? find(s_buffers, s_attribs[0].buffer) : nullptr;
            bool batch = false;
            if (vb && vb->mtl && s_attribs[0].stride > 0 && s_attribs[0].type == GL_FLOAT && vb->mtl.length >= (NSUInteger)s_attribs[0].offset + 12) {
                float first[3];
                memcpy(first, (const uint8_t *)vb->mtl.contents + s_attribs[0].offset, 12);
                batch = fabsf(first[0]) + fabsf(first[1]) > 40.0f;
            }
            float phase = 0;
            if (p->u_world) {
                const float *w = (const float *)(p->uniform_data[0].data() + p->u_world->offset);
                phase = w[12] * 0.37f + w[13] * 0.71f;
            }
            // A palm's leaves are weighed whether batched or not: height above the
            // foot says nothing about them, their crown being at the top of a trunk.
            if (diffuse->radial_wind && vb && vb->mtl) batch = true;
            bool ready = !batch || weigh_foliage(*vb, diffuse->radial_wind);
            wind[0] = (float)s_wind_time, wind[1] = ready ? s_wind_strength : 0.0f, wind[2] = batch ? 1.0f : 0.0f, wind[3] = fmodf(phase, 6.2831853f);
            // Once a buffer's alpha holds heights, every shader has to be told, sway or not.
            if (vb && vb->wind_weights == (__bridge void *)vb->mtl) wind[2] = 1.0f;
        }
        if (memcmp(p->uniform_data[0].data() + p->wind_offset, wind, 16)) {
            memcpy(p->uniform_data[0].data() + p->wind_offset, wind, 16);
            p->uniform_dirty[0] = true;
        }
    }
    for (int stage = 0; stage < 2; stage++) {
        if (p->skin_offset[stage] < 0) continue;
        const float skin[4] = {s_shader_options.skin_scattering && diffuse && diffuse->material == MAT_SKIN ? 1.0f : 0.0f,
                               s_dyn_near[2] > 0 && s_dyn_bias > 0 ? s_dyn_bias_m / s_dyn_bias : 0.0f, s_skin_backlight,
                               diffuse && diffuse->material == MAT_FLORA ? s_foliage_backlight : 0.0f};
        if (memcmp(p->uniform_data[stage].data() + p->skin_offset[stage], skin, 16)) {
            memcpy(p->uniform_data[stage].data() + p->skin_offset[stage], skin, 16);
            p->uniform_dirty[stage] = true;
        }
    }
    if (p->pbr_offset >= 0) {
        const float has[4] = {diffuse && diffuse->pbr ? 1.0f : 0.0f, 0, 0, 0};
        if (memcmp(p->uniform_data[1].data() + p->pbr_offset, has, 16)) {
            memcpy(p->uniform_data[1].data() + p->pbr_offset, has, 16);
            p->uniform_dirty[1] = true;
        }
    }
    if (p->material_offset >= 0) {
        int cls = diffuse ? diffuse->material : MAT_UNKNOWN;
        if (memcmp(p->uniform_data[1].data() + p->material_offset, k_material_params[cls], 16)) {
            memcpy(p->uniform_data[1].data() + p->material_offset, k_material_params[cls], 16);
            p->uniform_dirty[1] = true;
        }
    }
    if (p->dyn_near_offset >= 0 && memcmp(p->uniform_data[1].data() + p->dyn_near_offset, s_dyn_near, 16)) {
        memcpy(p->uniform_data[1].data() + p->dyn_near_offset, s_dyn_near, 16);
        p->uniform_dirty[1] = true;
    }
    if (p->dyn_bias_offset >= 0 && memcmp(p->uniform_data[1].data() + p->dyn_bias_offset, &s_dyn_bias, 4)) {
        memcpy(p->uniform_data[1].data() + p->dyn_bias_offset, &s_dyn_bias, 4);
        p->uniform_dirty[1] = true;
    }
    float limit = s_pass_screen ? 1.0f : 65000.0f;
    if (p->limit_offset >= 0 && memcmp(p->uniform_data[1].data() + p->limit_offset, &limit, 4)) {
        memcpy(p->uniform_data[1].data() + p->limit_offset, &limit, 4);
        p->uniform_dirty[1] = true;
    }
    if (p->grade_offset >= 0 && memcmp(p->uniform_data[1].data() + p->grade_offset, s_grade, 16)) {
        memcpy(p->uniform_data[1].data() + p->grade_offset, s_grade, 16);
        p->uniform_dirty[1] = true;
    }
    float headroom = s_headroom.load(std::memory_order_relaxed);
    if (p->headroom_offset >= 0 && memcmp(p->uniform_data[1].data() + p->headroom_offset, &headroom, 4)) {
        memcpy(p->uniform_data[1].data() + p->headroom_offset, &headroom, 4);
        p->uniform_dirty[1] = true;
    }
    if (p->yflip_offset >= 0 && memcmp(p->uniform_data[0].data() + p->yflip_offset, &yflip, 4)) {
        memcpy(p->uniform_data[0].data() + p->yflip_offset, &yflip, 4);
        p->uniform_dirty[0] = true;
    }
    const XlStage *stages[2] = {&p->xl.vs, &p->xl.fs};
    for (int s = 0; s < 2; s++) {
        if (stages[s]->uniform_buffer_index < 0 || p->uniform_data[s].empty()) continue;
        if (!s_state_cache || p->uniform_dirty[s] || p->pushed_frame[s] != s_frame_serial) {
            p->pushed_buffer[s] = push_uniforms(p->uniform_data[s], &p->pushed_offset[s]);
            p->pushed_frame[s] = s_frame_serial;
            p->uniform_dirty[s] = false;
            s_call_stats.uniform_copies++;
        } else {
            s_call_stats.uniform_reuses++;
        }
        bind_buffer(s, p->pushed_buffer[s], p->pushed_offset[s], (NSUInteger)stages[s]->uniform_buffer_index);
    }
    for (UniformLocation &l : p->locations) {
        for (int s = 0; s < 2; s++) {
            int index = l.texture_index[s];
            if (index < 0) continue;
            Texture *t = (l.unit >= 0 && l.unit < MAX_UNITS) ? find(s_textures, s_bound_texture[l.unit]) : nullptr;
            id<MTLTexture> texture = t && t->mtl ? t->mtl : s_white;
            id<MTLSamplerState> sampler = t && t->mtl ? sampler_for(*t) : s_samplers.begin()->second;
            if (l.is_sky) { // not one of the game's textures
                texture = s_sky_atlas ? s_sky_atlas : s_white;
                if (s_sky_sampler) sampler = s_sky_sampler;
            } else if (l.is_dyn_map) {
                texture = s_dyn_map ? s_dyn_map : s_white;
                if (s_dyn_sampler) sampler = s_dyn_sampler;
            } else if (l.is_hair_map) {
                texture = diffuse && diffuse->hair ? diffuse->hair : s_white;
                sampler = diffuse && diffuse->mtl ? sampler_for(*diffuse) : s_samplers.begin()->second;
            } else if (l.is_pbr_map || l.is_fur_map) {
                texture = diffuse && diffuse->pbr ? diffuse->pbr : s_white;
                sampler = diffuse && diffuse->mtl ? sampler_for(*diffuse) : s_samplers.begin()->second;
            } else if (l.srgb && t && t->mtl) {
                // The shader no longer decodes this sampler's colour itself.
                if (id<MTLTexture> view = srgb_view_for(*t)) texture = view;
                else complain("a texture without an sRGB format is bound to %s; it will look too bright", l.name.c_str());
            }
            // The pass onto the screen reads the scene enlarged, when it has been (see apply_upscale).
            if (s_pass_screen && t && texture == t->mtl && t->upscaled && t->upscaled_frame == s_frame_serial) texture = t->upscaled;
            bool tracked = index < 16;
            if (!already(tracked && s_set.texture[s][index] == (__bridge void *)texture)) {
                if (s == 0) [s_pass setVertexTexture:texture atIndex:(NSUInteger)index];
                else [s_pass setFragmentTexture:texture atIndex:(NSUInteger)index];
                if (tracked) s_set.texture[s][index] = (__bridge void *)texture;
            }
            if (!already(tracked && s_set.sampler[s][index] == (__bridge void *)sampler)) {
                if (s == 0) [s_pass setVertexSamplerState:sampler atIndex:(NSUInteger)index];
                else [s_pass setFragmentSamplerState:sampler atIndex:(NSUInteger)index];
                if (tracked) s_set.sampler[s][index] = (__bridge void *)sampler;
            }
        }
    }
    for (int slot = 0; slot < binding.slots; slot++) {
        Buffer *b = find(s_buffers, binding.buffer[slot]);
        if (!b || !b->mtl) return false;
        bind_buffer(0, b->mtl, binding.base[slot], (NSUInteger)(VERTEX_SLOT_BASE + slot));
    }
    bind_buffer(0, s_constant_attrib, 0, CONSTANT_SLOT);
    s_last.draws++;
    s_draw_binding = binding;
    // (Played back with the camera moved, the sky only turns while the scene also
    // shifts, so the sky's matrices do not give the scene's camera; a draw that
    // tests depth does. In the game the two agree and the sky comes first.)
    if (s_dyn_shadows && s_pass_scene && s_dyn_frame != s_frame_serial && !s_casters.empty() && p->u_wvp && p->u_world && (!s_replaying || s_depth_test))
        render_dynamic_shadows(*p);
    if (s_pass_scene && (((s_ao || s_contact_shadows || s_light_shafts || s_foliage_backlight > 0) && s_camera.view_frame != s_frame_serial) ||
                         (s_motion_blur && s_camera.view_proj_frame != s_frame_serial)))
        observe_camera(*p);
    return true;
}

// ---- the sky atlas ------------------------------------------------------------------------

// Redraws the draw that was just issued, the engine's sky dome, six more times:
// once per face of the atlas, each with a 90-degree view along one axis. The
// dome's own shader, textures and vertex data are reused; only its
// world-view-projection matrix is replaced. The dome is centred on the camera,
// so its world matrix with the translation removed puts it around the origin.
//
// This runs in a command buffer of its own, committed at once: a render pass is
// open on the main one, and the atlas has to be finished before the draws that
// follow in that pass sample it.
void capture_sky(MTLPrimitiveType primitive, NSUInteger count, MTLIndexType index_type, id<MTLBuffer> indices, NSUInteger index_offset)
{
    Program *p = find(s_programs, s_program);
    double world[16];
    if (!p || !vertex_matrix(*p, "matWorld", world)) return;
    auto it = p->location_by_name.find("matWorldViewProj");
    const XlUniform *wvp = it == p->location_by_name.end() ? nullptr : p->locations[(size_t)it->second].uniform[0];
    if (!wvp) return;
    s_sky_frame = s_frame_serial;

    const int N = k_xl_sky_face_size;
    if (!s_sky_atlas) {
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:6 * N height:N mipmapped:YES];
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModePrivate;
        s_sky_atlas = [s_device newTextureWithDescriptor:td];
        MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
        sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterLinear;
        sd.mipFilter = MTLSamplerMipFilterLinear;
        s_sky_sampler = [s_device newSamplerStateWithDescriptor:sd];
    }
    PipelineKey key = s_last_key; // the dome's vertex layout, onto the atlas instead of the scene
    key.color_format = (uint16_t)MTLPixelFormatRGBA16Float;
    key.depth_format = (uint16_t)MTLPixelFormatInvalid;
    key.samples = 1;
    key.blend = 0;
    key.blend_src = key.blend_dst = key.blend_eq = 0;
    id<MTLRenderPipelineState> pipeline = pipeline_for(*p, key);
    if (!pipeline) return;

    world[12] = world[13] = world[14] = 0;
    const double near = 0.1, far = 10000;
    double projection[16] = {};
    projection[0] = projection[5] = 1; // 90 degrees, square
    projection[10] = -(far + near) / (far - near);
    projection[11] = -1;
    projection[14] = -2 * far * near / (far - near);
    static const double faces[6][2][3] = { // forward, up; see k_xl_sky_sampler
        {{1, 0, 0}, {0, 0, 1}}, {{-1, 0, 0}, {0, 0, 1}}, {{0, 1, 0}, {0, 0, 1}},
        {{0, -1, 0}, {0, 0, 1}}, {{0, 0, 1}, {0, 1, 0}}, {{0, 0, -1}, {0, 1, 0}}};

    id<MTLCommandBuffer> cmd = [s_queue commandBuffer];
    for (int f = 0; f < 6; f++) {
        const double *fw = faces[f][0], *up = faces[f][1];
        double right[3] = {fw[1] * up[2] - fw[2] * up[1], fw[2] * up[0] - fw[0] * up[2], fw[0] * up[1] - fw[1] * up[0]};
        double view[16] = {}, vw[16], m[16]; // rows: right, up, -forward
        for (int c = 0; c < 3; c++) view[c * 4 + 0] = right[c], view[c * 4 + 1] = up[c], view[c * 4 + 2] = -fw[c];
        view[15] = 1;
        mat_mul(view, world, vw);
        mat_mul(projection, vw, m);

        std::vector<uint8_t> vs = p->uniform_data[0], fs = p->uniform_data[1];
        float m32[16], yflip = -1.0f, limit = 65000.0f; // drawn like any other texture target: rows in OpenGL's order
        for (int i = 0; i < 16; i++) m32[i] = (float)m[i];
        memcpy(vs.data() + wvp->offset, m32, sizeof m32);
        if (p->yflip_offset >= 0) memcpy(vs.data() + p->yflip_offset, &yflip, 4);
        if (p->limit_offset >= 0) memcpy(fs.data() + p->limit_offset, &limit, 4);

        MTLRenderPassDescriptor *d = [MTLRenderPassDescriptor renderPassDescriptor];
        d.colorAttachments[0].texture = s_sky_atlas;
        d.colorAttachments[0].loadAction = f == 0 ? MTLLoadActionClear : MTLLoadActionLoad;
        d.colorAttachments[0].storeAction = MTLStoreActionStore;
        d.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
        id<MTLRenderCommandEncoder> pass = [cmd renderCommandEncoderWithDescriptor:d];
        [pass setViewport:(MTLViewport){(double)(f * N), 0.0, (double)N, (double)N, 0.0, 1.0}];
        [pass setScissorRect:(MTLScissorRect){(NSUInteger)(f * N), 0, (NSUInteger)N, (NSUInteger)N}];
        [pass setRenderPipelineState:pipeline];
        [pass setDepthStencilState:depth_state(false)];
        [pass setCullMode:MTLCullModeNone];
        if (p->xl.vs.uniform_buffer_index >= 0 && !vs.empty())
            [pass setVertexBuffer:[s_device newBufferWithBytes:vs.data() length:vs.size() options:MTLResourceStorageModeShared]
                           offset:0
                          atIndex:(NSUInteger)p->xl.vs.uniform_buffer_index];
        if (p->xl.fs.uniform_buffer_index >= 0 && !fs.empty())
            [pass setFragmentBuffer:[s_device newBufferWithBytes:fs.data() length:fs.size() options:MTLResourceStorageModeShared]
                             offset:0
                            atIndex:(NSUInteger)p->xl.fs.uniform_buffer_index];
        for (UniformLocation &l : p->locations) {
            for (int stage = 0; stage < 2; stage++) {
                int index = l.texture_index[stage];
                if (index < 0) continue;
                Texture *t = (l.unit >= 0 && l.unit < MAX_UNITS) ? find(s_textures, s_bound_texture[l.unit]) : nullptr;
                id<MTLTexture> texture = t && t->mtl ? t->mtl : s_white;
                id<MTLSamplerState> sampler = t && t->mtl ? sampler_for(*t) : s_samplers.begin()->second;
                if (l.srgb && t && t->mtl) {
                    if (id<MTLTexture> srgb = srgb_view_for(*t)) texture = srgb;
                }
                if (stage == 0) {
                    [pass setVertexTexture:texture atIndex:(NSUInteger)index];
                    [pass setVertexSamplerState:sampler atIndex:(NSUInteger)index];
                } else {
                    [pass setFragmentTexture:texture atIndex:(NSUInteger)index];
                    [pass setFragmentSamplerState:sampler atIndex:(NSUInteger)index];
                }
            }
        }
        bool complete = true;
        for (int slot = 0; slot < s_draw_binding.slots; slot++) {
            Buffer *b = find(s_buffers, s_draw_binding.buffer[slot]);
            if (!b || !b->mtl) {
                complete = false;
                break;
            }
            [pass setVertexBuffer:b->mtl offset:s_draw_binding.base[slot] atIndex:(NSUInteger)(VERTEX_SLOT_BASE + slot)];
        }
        [pass setVertexBuffer:s_constant_attrib offset:0 atIndex:CONSTANT_SLOT];
        if (complete)
            [pass drawIndexedPrimitives:primitive indexCount:count indexType:index_type indexBuffer:indices indexBufferOffset:index_offset];
        [pass endEncoding];
    }
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit generateMipmapsForTexture:s_sky_atlas];
    if (s_sky_ambient) { // the smallest level with a texel per face: each is that face's mean colour
        NSUInteger level = 0;
        while ((N >> level) > 1) level++;
        id<MTLBuffer> means = [s_device newBufferWithLength:6 * 8 options:MTLResourceStorageModeShared];
        [blit copyFromTexture:s_sky_atlas sourceSlice:0 sourceLevel:level sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(6, 1, 1)
                     toBuffer:means destinationOffset:0 destinationBytesPerRow:6 * 8 destinationBytesPerImage:6 * 8];
        [cmd addCompletedHandler:^(id<MTLCommandBuffer>) {
            const uint16_t *half = (const uint16_t *)means.contents;
            auto to_float = [](uint16_t h) {
                int e = (h >> 10) & 31, m = h & 1023;
                float v = e == 0 ? ldexpf((float)m, -24) : e == 31 ? 65504.0f : ldexpf((float)(m + 1024), e - 25);
                return (h & 0x8000) ? -v : v;
            };
            std::lock_guard<std::mutex> lock(s_amb_mu);
            for (int face = 0; face < 6; face++)
                for (int c = 0; c < 3; c++) s_amb_faces[face][c] = to_float(half[face * 4 + c]);
            s_amb_faces_valid = true;
        }];
    }
    [blit endEncoding];
    [cmd commit];
}

// Works out the shaders' wbm_amb for this frame. Called at present.
void update_sky_ambient()
{
    float amb[24] = {};
    float faces[6][3];
    bool valid;
    {
        std::lock_guard<std::mutex> lock(s_amb_mu);
        memcpy(faces, s_amb_faces, sizeof faces);
        valid = s_amb_faces_valid;
    }
    // Only while a sky is being drawn: indoors, and in menus, there is none.
    if (valid && s_sky_frame + 2 >= s_frame_serial) {
        float linear[5][3], mean[3] = {};
        for (int face = 0; face < 5; face++) // the sixth looks down, at the dome's mirrored underside
            for (int c = 0; c < 3; c++) mean[c] += (linear[face][c] = powf(std::max(faces[face][c], 0.0f), 2.2f)) / 5;
        float brightness = 0.2126f * mean[0] + 0.7152f * mean[1] + 0.0722f * mean[2];
        if (brightness > 1e-5f) {
            for (int face = 0; face < 5; face++)
                for (int c = 0; c < 3; c++) amb[face * 4 + c] = std::min(2.5f, std::max(0.4f, linear[face][c] / brightness));
            amb[23] = s_sky_ambient_strength;
            // vSunDir is the way the light travels; what reaches the ground is its
            // downward part. A fifth of that comes back up, a little warm.
            float down = s_sun_valid ? std::max(0.0f, -s_sun_dir[2]) : 0.0f;
            const float ground[3] = {0.22f, 0.19f, 0.14f};
            for (int c = 0; c < 3; c++) amb[20 + c] = std::max(0.0f, s_sun_color[c]) * down * ground[c] * s_bounce_strength;
            s_amb_stats_frames++;
        }
    }
    memcpy(s_amb, amb, sizeof s_amb);
}

bool primitive_type(GLenum mode, MTLPrimitiveType *type)
{
    switch (mode) {
    case GL_POINTS: *type = MTLPrimitiveTypePoint; return true;
    case GL_LINES: *type = MTLPrimitiveTypeLine; return true;
    case GL_LINE_STRIP: *type = MTLPrimitiveTypeLineStrip; return true;
    case GL_TRIANGLES: *type = MTLPrimitiveTypeTriangle; return true;
    case GL_TRIANGLE_STRIP: *type = MTLPrimitiveTypeTriangleStrip; return true;
    default: complain("unsupported draw mode %s", E_prim(mode)); return false;
    }
}

// Metal has no quads: each one is drawn as two triangles through this index buffer.
id<MTLBuffer> quad_indices(int quads)
{
    if (quads > s_quad_capacity) {
        s_quad_capacity = std::max(quads, 256);
        std::vector<uint16_t> idx((size_t)s_quad_capacity * 6);
        for (int q = 0; q < s_quad_capacity; q++) {
            const uint16_t v = (uint16_t)(q * 4), corners[6] = {0, 1, 2, 0, 2, 3};
            for (int i = 0; i < 6; i++) idx[(size_t)q * 6 + i] = (uint16_t)(v + corners[i]);
        }
        s_quad_indices = [s_device newBufferWithBytes:idx.data() length:idx.size() * 2 options:MTLResourceStorageModeShared];
    }
    return s_quad_indices;
}

float half_to_float(uint16_t h)
{
    int exponent = (h >> 10) & 31, mantissa = h & 1023;
    float v = exponent == 0 ? ldexpf((float)mantissa, -24) : exponent == 31 ? 65504.0f : ldexpf((float)(mantissa | 1024), exponent - 25);
    return (h & 0x8000) ? -v : v;
}

void save_capture(id<MTLTexture> copy, uint32_t frame, const char *name = "frame")
{
    int w = (int)copy.width, h = (int)copy.height;
    std::vector<unsigned char> pixels((size_t)w * h * 4);
    if (copy.pixelFormat == MTLPixelFormatRGBA16Float) {
        // An extended-range frame: a PNG can only hold the part up to display white.
        std::vector<uint16_t> halves((size_t)w * h * 4);
        [copy getBytes:halves.data() bytesPerRow:(NSUInteger)w * 8 fromRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0];
        for (size_t i = 0; i < halves.size(); i++)
            pixels[i] = (unsigned char)lrintf(std::min(1.0f, std::max(0.0f, half_to_float(halves[i]))) * 255.0f);
        wb_save_frame_png(pixels.data(), w, h, frame, false, name);
        return;
    }
    [copy getBytes:pixels.data() bytesPerRow:(NSUInteger)w * 4 fromRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0];
    wb_save_frame_png(pixels.data(), w, h, frame, true, name);
}

#include "frame_capture.inc"
#include "model_preview.inc"

} // namespace

// ---- window and frame ------------------------------------------------------------------------

void mtl_attach_window(void *nswindow, int width, int height)
{
    s_debug_clear = getenv("WBM_DEBUG_CLEAR") != nullptr;
    signal(SIGUSR2, [](int) { s_capture_requested = 1; }); // scripts/capture.sh
    s_device = MTLCreateSystemDefaultDevice();
    s_queue = [s_device newCommandQueue];
    s_viewport = {0, 0, width, height};
    s_screen_w = width;
    s_screen_h = height;
    const char *scale_env = getenv("WBM_SCALE");
    const char *shadows_env = getenv("WBM_SHADOWS"); // on unless set to 0
    // WBM_SHADOWS: 0 the engine's filter, 1 (default) a wider one, 2 softness that grows with distance.
    s_shader_options.soft_shadows = !(shadows_env && atoi(shadows_env) == 0);
    s_shader_options.pcss_shadows = shadows_env && atoi(shadows_env) >= 2;
    const char *msaa_env = getenv("WBM_MSAA");
    const char *reverse_env = getenv("WBM_REVERSE_Z"), *srgb_env = getenv("WBM_SRGB"), *cache_env = getenv("WBM_STATE_CACHE");
    s_reverse_depth = s_shader_options.reverse_depth = !(reverse_env && atoi(reverse_env) == 0); // on unless set to 0
    s_shader_options.srgb_textures = srgb_env && atoi(srgb_env) != 0;                            // off unless set
    s_state_cache = !(cache_env && atoi(cache_env) == 0);
    const char *hdr_env = getenv("WBM_HDR"), *tonemap_env = getenv("WBM_TONEMAP");
    s_hdr = s_shader_options.hdr_output = hdr_env && atoi(hdr_env) != 0;                  // off unless set
    s_shader_options.filmic_tonemap = tonemap_env && atoi(tonemap_env) != 0;              // off unless set
    if (const char *v = getenv("WBM_HDR_PEAK")) s_hdr_peak = std::max(1.0f, (float)atof(v));
    if (const char *v = getenv("WBM_CONTRAST")) s_grade[0] = std::min(1.0f, std::max(-1.0f, (float)atof(v)));
    if (const char *v = getenv("WBM_SATURATION")) s_grade[1] = std::min(1.0f, std::max(-1.0f, (float)atof(v)));
    if (const char *v = getenv("WBM_WARMTH")) s_grade[2] = std::min(1.0f, std::max(-1.0f, (float)atof(v)));
    s_shader_options.grade = s_grade[0] != 0 || s_grade[1] != 0 || s_grade[2] != 0;
    if (const char *v = getenv("WBM_LOD_BIAS")) s_lod_bias = (float)atof(v);
    const char *ao_env = getenv("WBM_AO");
    s_ao = ao_env && atoi(ao_env) != 0; // off unless set
    if (const char *v = getenv("WBM_AO_RADIUS")) s_ao_radius = (float)atof(v);
    if (const char *v = getenv("WBM_AO_STRENGTH")) s_ao_strength = (float)atof(v);
    if (const char *v = getenv("WBM_AO_FULL")) s_ao_full = atoi(v) != 0;
    const char *contact_env = getenv("WBM_CONTACT_SHADOWS");
    s_contact_shadows = contact_env && atoi(contact_env) != 0; // off unless set
    if (const char *v = getenv("WBM_CONTACT_STRENGTH")) s_contact_strength = (float)atof(v);
    if (const char *v = getenv("WBM_CONTACT_LENGTH")) s_contact_length = (float)atof(v);
    const char *pbr_env = getenv("WBM_PBR");
    s_shader_options.pbr_specular = pbr_env && atoi(pbr_env) >= 1;    // off unless set
    s_shader_options.pbr_environment = pbr_env && atoi(pbr_env) >= 2; // 2: metalness and environment reflection too
    s_shader_options.pbr_sky = pbr_env && atoi(pbr_env) >= 3;         // 3: the environment is the scene's own sky
    const char *dyn_env = getenv("WBM_DYNAMIC_SHADOWS");
    s_dyn_shadows = s_shader_options.dynamic_shadows = dyn_env && atoi(dyn_env) != 0; // off unless set
    if (const char *v = getenv("WBM_DYN_RANGE")) s_dyn_range = std::max(5.0f, (float)atof(v));
    if (const char *v = getenv("WBM_DYN_BIAS")) s_dyn_bias_m = (float)atof(v);
    if (const char *v = getenv("WBM_DYN_NEAR")) s_dyn_near_range = std::max(0.0f, (float)atof(v));
    const char *amb_env = getenv("WBM_SKY_AMBIENT");
    s_sky_ambient = s_shader_options.sky_ambient = amb_env && atoi(amb_env) != 0; // off unless set
    if (const char *v = getenv("WBM_SKY_AMBIENT_STRENGTH")) s_sky_ambient_strength = std::min(1.0f, std::max(0.0f, (float)atof(v)));
    if (const char *v = getenv("WBM_BOUNCE")) s_bounce_strength = std::max(0.0f, (float)atof(v));
    const char *mat_env = getenv("WBM_MATERIALS");
    s_materials = s_shader_options.materials = mat_env && atoi(mat_env) != 0 && s_shader_options.pbr_environment; // off unless set; needs WBM_PBR=2 or 3
    s_shader_options.debug_materials = getenv("WBM_DEBUG_MATERIALS") != nullptr && s_shader_options.pbr_environment;
    const char *sss_env = getenv("WBM_SSS");
    s_shader_options.skin_scattering = sss_env && atoi(sss_env) != 0; // off unless set
    s_shader_options.eyes = getenv("WBM_EYES") && atoi(getenv("WBM_EYES")) != 0; // off unless set
    if (const char *v = getenv("WBM_FUR")) s_fur_strength = std::max(0.0f, (float)atof(v));
    if (const char *v = getenv("WBM_FUR_SHELLS")) s_fur_shells = std::min(24, std::max(0, atoi(v)));
    if (const char *v = getenv("WBM_HAIR_LENGTH")) s_hair_length = std::max(0.0f, (float)atof(v));
    if (const char *v = getenv("WBM_HAIR_STRANDS")) s_hair_strands = atoi(v) != 0;
    if (const char *v = getenv("WBM_PEOPLE_HAIR")) s_people_hair = atoi(v) != 0;
    if (const char *v = getenv("WBM_PEOPLE_HAIR_RANGE")) s_people_hair_range = std::max(0.0f, (float)atof(v));
    if (const char *v = getenv("WBM_PEOPLE_HAIR_LENGTH")) s_people_hair_length = std::max(0.001f, (float)atof(v));
    if (const char *v = getenv("WBM_PEOPLE_HAIR_DENSITY")) s_people_hair_density = std::max(100.0f, (float)atof(v));
    if (const char *v = getenv("WBM_PEOPLE_HAIR_MOST")) s_people_hair_most = std::max(0, atoi(v));
    if (const char *v = getenv("WBM_TAIL_LENGTH")) s_tail_length = std::max(0.0f, (float)atof(v));
    if (const char *v = getenv("WBM_MANE_LENGTH")) s_mane_length = std::max(0.0f, (float)atof(v));
    if (const char *v = getenv("WBM_HAIR_STRANDS_DENSITY")) s_strand_density = std::max(100.0f, (float)atof(v));
    if (const char *v = getenv("WBM_FUR_LENGTH")) s_fur_length = std::max(0.0f, (float)atof(v));
    if (const char *v = getenv("WBM_FUR_RANGE")) s_fur_range = std::max(0.0f, (float)atof(v));
    if (const char *v = getenv("WBM_FUR_DENSITY")) s_fur_density = std::max(1.0f, (float)atof(v));
    s_shader_options.fur = s_fur_strength > 0 || s_shader_options.eyes; // the horse's eye rides on the same rewrite
    s_shader_options.hair_strands = s_people_hair;
    if (const char *v = getenv("WBM_EYES_MOVE")) s_gaze_strength = std::max(0.0f, (float)atof(v));
    if (const char *v = getenv("WBM_SSS_BACKLIGHT")) s_skin_backlight = std::max(0.0f, (float)atof(v));
    if (const char *v = getenv("WBM_FOLIAGE")) s_foliage_backlight = std::max(0.0f, (float)atof(v));
    if (const char *v = getenv("WBM_WIND")) s_wind_strength = std::max(0.0f, (float)atof(v));
    s_shader_options.foliage_backlight = s_foliage_backlight > 0;
    s_shader_options.wind = s_wind_strength > 0;
    index_texture_packs();
    if (s_materials || !s_pack_files.empty() || s_pack_self_test || s_shader_options.skin_scattering || s_shader_options.wind || s_shader_options.foliage_backlight || s_shader_options.eyes)
        scan_game_textures();
    const char *fxaa_env = getenv("WBM_FXAA");
    s_fxaa = fxaa_env && atoi(fxaa_env) != 0; // off unless set
    if (const char *v = getenv("WBM_MAX_FPS")) s_max_fps = std::max(0.0, atof(v));
    if (const char *v = getenv("WBM_MENU_FPS")) s_menu_fps = std::max(0.0, atof(v));
    const char *blur_env = getenv("WBM_MOTION_BLUR");
    s_motion_blur = blur_env && atoi(blur_env) != 0; // off unless set
    if (const char *v = getenv("WBM_MOTION_BLUR_STRENGTH")) s_motion_blur_strength = std::max(0.0f, (float)atof(v));
    const char *shafts_env = getenv("WBM_LIGHT_SHAFTS");
    s_light_shafts = shafts_env && atoi(shafts_env) != 0; // off unless set
    if (const char *v = getenv("WBM_SHAFT_STRENGTH")) s_shaft_strength = (float)atof(v);

    const uint8_t white[4] = {255, 255, 255, 255};
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                  width:1
                                                                                 height:1
                                                                              mipmapped:NO];
    s_white = [s_device newTextureWithDescriptor:td];
    [s_white replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:white bytesPerRow:4];
    Texture defaults;
    sampler_for(defaults); // guarantees s_samplers is never empty
    s_msaa = msaa_env ? atoi(msaa_env) : 4;
    if (s_msaa < 1 || ![s_device supportsTextureSampleCount:(NSUInteger)s_msaa]) {
        wb_note("  !! metal: WBM_MSAA=%s is not a sample count this GPU supports; multisampling is off", msaa_env);
        s_msaa = 1;
    }
    s_shader_options.smooth_alpha_test = s_msaa > 1;
    const float constant[4] = {0, 0, 0, 1}; // OpenGL's value for a disabled attribute array
    s_constant_attrib = [s_device newBufferWithBytes:constant length:sizeof constant options:MTLResourceStorageModeShared];
    if (!s_device.supportsBCTextureCompression) wb_note("  !! metal: this GPU does not support DXT textures");

    void (^attach)(void) = ^{
        NSWindow *window = (__bridge NSWindow *)nswindow;
        s_scale = scale_env ? atoi(scale_env) : (int)lround(window.backingScaleFactor);
        s_scale = std::max(1, std::min(4, s_scale));
        s_render_scale = s_scale;
        if (const char *v = getenv("WBM_UPSCALE")) {
            double share = atof(v);
            if (share >= 0.3 && share < 0.999) {
                bool can = false;
                if (@available(macOS 13.0, *)) can = [MTLFXSpatialScalerDescriptor supportsDevice:s_device];
                if (can) s_upscale = true, s_render_scale = s_scale * share;
                else complain("WBM_UPSCALE: MetalFX is not available on this machine");
            }
        }
        if (const char *v = getenv("WBM_SHARPEN")) s_sharpen = std::min(1.0f, std::max(0.0f, (float)atof(v)));
        CAMetalLayer *layer = [CAMetalLayer layer];
        layer.device = s_device;
        layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
        if (s_hdr) {
            layer.pixelFormat = MTLPixelFormatRGBA16Float;
            layer.wantsExtendedDynamicRangeContent = YES;
            CGColorSpaceRef space = CGColorSpaceCreateWithName(kCGColorSpaceExtendedSRGB);
            layer.colorspace = space;
            CGColorSpaceRelease(space);
        }
        s_window = window;
        layer.framebufferOnly = NO; // frame captures read the drawable back
        layer.opaque = YES;
        // The drawable is the game's size times s_scale; the layer stretches it to the view.
        layer.drawableSize = CGSizeMake(width * s_scale, height * s_scale);
        layer.contentsGravity = kCAGravityResize;
        layer.magnificationFilter = kCAFilterLinear;
        layer.minificationFilter = kCAFilterLinear;
        NSView *view = window.contentView;
        view.layer = layer; // set before wantsLayer: the view hosts our layer instead of making its own
        view.wantsLayer = YES;
        s_layer = layer;
    };
    if ([NSThread isMainThread]) attach();
    else dispatch_sync(dispatch_get_main_queue(), attach);
    wb_note("metal: game renders at %dx%d, drawable is %dx%d (scale %d); soft shadows %s; %dx multisampling; "
            "reversed depth %s; sRGB textures %s; state cache %s; microfacet specular %s; ambient occlusion %s; "
            "contact shadows %s; light shafts %s; per-frame shadow map %s; motion blur %s; sky ambient %s; material classes %s; skin scattering %s; wind %.1f; light through leaves %.1f; filmic tone map %s; extended-range output %s",
            width, height, width * s_scale, height * s_scale, s_scale, s_shader_options.pcss_shadows ? "PCSS" : s_shader_options.soft_shadows ? "on" : "off", s_msaa,
            s_reverse_depth ? "on" : "off", s_shader_options.srgb_textures ? "on" : "off", s_state_cache ? "on" : "off",
            s_shader_options.pbr_sky ? "on, with metalness and sky reflection" : s_shader_options.pbr_environment ? "on, with metalness and environment" : s_shader_options.pbr_specular ? "on" : "off",
            s_ao ? "on" : "off", s_contact_shadows ? "on" : "off", s_light_shafts ? "on" : "off", s_dyn_shadows ? "on" : "off", s_motion_blur ? "on" : "off", s_sky_ambient ? "on" : "off", s_materials ? "on" : "off", s_shader_options.skin_scattering ? "on" : "off", (double)s_wind_strength, (double)s_foliage_backlight,
            s_shader_options.filmic_tonemap ? "on" : "off",
            s_hdr ? "on" : "off");
}

const char *mtl_device_name()
{
    return s_device ? s_device.name.UTF8String : "(no device)";
}

void mtl_dump_stats(FILE *f)
{
    fprintf(f, "\n== Metal shaders ==\n%d programs linked, %d from the disk cache, %.0f ms translating\n", s_shader_stats.links,
            s_shader_stats.cache_hits, s_shader_stats.translate_ms);
    fprintf(f, "%d distinct shaders compiled by Metal in the background, %.0f ms spent waiting for them at draws\n",
            s_shader_stats.stage_compiles, s_shader_stats.wait_ms);
    fprintf(f, "\n== Metal state changes ==\n%llu state-setting calls made, %llu skipped as unchanged\n",
            (unsigned long long)s_call_stats.made, (unsigned long long)s_call_stats.skipped);
    fprintf(f, "%llu uniform blocks copied, %llu reused\n", (unsigned long long)s_call_stats.uniform_copies,
            (unsigned long long)s_call_stats.uniform_reuses);
    if (s_blur_frames) fprintf(f, "\nmotion blur applied in %llu frames\n", (unsigned long long)s_blur_frames);
    if (s_shell_draws) fprintf(f, "\n%llu layers of fur drawn\n", (unsigned long long)s_shell_draws);
    if (s_upscale_frames) fprintf(f, "\nupscaled by MetalFX in %llu frames, from %.0f%% of the screen's size each way\n", (unsigned long long)s_upscale_frames, 100.0 * s_render_scale / s_scale);
    if (s_sharpen_frames) fprintf(f, "\nsharpened in %llu frames\n", (unsigned long long)s_sharpen_frames);
    if (s_fxaa_frames) fprintf(f, "\nanti-aliasing (FXAA) applied in %llu frames\n", (unsigned long long)s_fxaa_frames);
    if (s_max_fps > 0 || s_menu_fps > 0) fprintf(f, "\nframe rate limited to %.0f, %.0f without a 3D scene (0: no limit)\n", s_max_fps, s_menu_fps);
    if (s_sky_ambient) {
        fprintf(f, "\n== sky ambient ==\napplied in %llu frames; tint now towards +X %.2f %.2f %.2f, -X %.2f %.2f %.2f, up %.2f %.2f %.2f; bounce %.3f %.3f %.3f\n",
                (unsigned long long)s_amb_stats_frames, s_amb[0], s_amb[1], s_amb[2], s_amb[4], s_amb[5], s_amb[6], s_amb[16], s_amb[17], s_amb[18],
                s_amb[20], s_amb[21], s_amb[22]);
    }
    if (!s_pack_files.empty() || s_pack_stats.replaced.load()) {
        fprintf(f, "\n== texture packs ==\n%zu files on offer; %d textures replaced, %d given a metalness map, %d could not be loaded\n", s_pack_files.size(),
                s_pack_stats.replaced.load(), s_pack_stats.pbr_maps.load(), s_pack_stats.failed.load());
        std::lock_guard<std::mutex> lock(s_pack_stats.mu);
        for (const std::string &name : s_pack_stats.names) fprintf(f, "  %s\n", name.c_str());
    }
    if (s_materials) {
        fprintf(f, "\n== material classes ==\n%d texture files seen, classed by name:", s_material_stats.files);
        for (int c = 1; c < MAT_COUNT; c++) fprintf(f, " %s %d,", k_material_names[c], s_material_stats.classed[c]);
        fprintf(f, "\n%d compressed textures loaded by the game: %d matched a file, %d of those have a class\n", s_material_stats.uploads.load(),
                s_material_stats.matched.load(), s_material_stats.recognised.load());
    }
    if (s_dyn_stats.frames)
        fprintf(f, "\n== per-frame shadow map ==\ndrawn in %llu frames, %.0f casters per frame on average, %.0f more left out as scenery, %.0f added from the main pass\n"
                   "%llu draws left out as overlays, %zu overlay meshes known at exit\n",
                (unsigned long long)s_dyn_stats.frames, (double)s_dyn_stats.casters / (double)s_dyn_stats.frames,
                (double)s_dyn_stats.scenery / (double)s_dyn_stats.frames, (double)s_dyn_stats.late / (double)s_dyn_stats.frames,
                (unsigned long long)s_dyn_stats.overlays, s_overlays.size());
}

void mtl_set_vsync(bool on)
{
    CAMetalLayer *layer = s_layer;
    dispatch_async(dispatch_get_main_queue(), ^{
      layer.displaySyncEnabled = on;
    });
}

void mtl_present(bool capture, uint32_t frame)
{
    @autoreleasepool {
        end_pass();
        if (!s_cmd) return; // the game issued nothing this frame
        id<MTLTexture> copy = nil;
        if (capture && s_drawable) {
            id<MTLTexture> src = s_drawable.texture;
            MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:src.pixelFormat
                                                                                         width:src.width
                                                                                        height:src.height
                                                                                     mipmapped:NO];
            d.storageMode = MTLStorageModeShared;
            copy = [s_device newTextureWithDescriptor:d];
            id<MTLBlitCommandEncoder> blit = [s_cmd blitCommandEncoder];
            [blit copyFromTexture:src
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(src.width, src.height, 1)
                        toTexture:copy
                 destinationSlice:0
                 destinationLevel:0
                destinationOrigin:MTLOriginMake(0, 0, 0)];
            [blit endEncoding];
        }
        if (s_frame_chunks.count) { // hand the uniform chunks back once the GPU is done with them
            NSArray<id<MTLBuffer>> *chunks = [s_frame_chunks copy];
            [s_cmd addCompletedHandler:^(id<MTLCommandBuffer>) {
              std::lock_guard<std::mutex> lock(s_chunk_mu);
              for (id<MTLBuffer> chunk in chunks) s_free_chunks.push_back(chunk);
            }];
            [s_frame_chunks removeAllObjects];
        }
        id<MTLTexture> sky_copy = nil;
        if (capture && s_sky_atlas) { // saved next to the frame, for checking what reflections sample
            MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:s_sky_atlas.pixelFormat
                                                                                         width:s_sky_atlas.width
                                                                                        height:s_sky_atlas.height
                                                                                     mipmapped:NO];
            d.storageMode = MTLStorageModeShared;
            sky_copy = [s_device newTextureWithDescriptor:d];
            id<MTLBlitCommandEncoder> blit = [s_cmd blitCommandEncoder];
            [blit copyFromTexture:s_sky_atlas
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(s_sky_atlas.width, s_sky_atlas.height, 1)
                        toTexture:sky_copy
                 destinationSlice:0
                 destinationLevel:0
                destinationOrigin:MTLOriginMake(0, 0, 0)];
            [blit endEncoding];
        }
        if (s_drawable) [s_cmd presentDrawable:s_drawable];
        [s_cmd commit];
        if (copy) {
            [s_cmd waitUntilCompleted];
            save_capture(copy, frame);
            if (sky_copy) save_capture(sky_copy, frame, "sky");
        }
        s_cmd = nil;
        s_drawable = nil;
        s_frame_serial++;
        if (!s_casters.empty()) { // recorded, but no main pass followed
            s_casters.clear();
            s_caster_bones.clear();
        }
        if (s_sky_ambient) update_sky_ambient();
        if (s_dyn_shadows) {
            s_late_ready.swap(s_late_casters);
            s_late_casters.clear();
            s_prepass_meshes.clear();
        }
        if (s_dyn_shadows && s_dyn_frame + 2 < s_frame_serial) // the map has gone stale
            s_dyn_window[2] = s_dyn_window[3] = s_dyn_near[2] = s_dyn_near[3] = 0;
        if (s_hdr && s_frame_serial % 30 == 0) {
            dispatch_async(dispatch_get_main_queue(), ^{
              CGFloat available = s_window.screen.maximumExtendedDynamicRangeColorComponentValue;
              s_headroom.store(std::min(s_hdr_peak, std::max(1.0f, (float)available)));
            });
        }
        capture_at_frame_end();
        s_preview_in_scene = false;
    }
    limit_frame_rate();
}

// ---- frame capture and playback (see frame_capture.inc) -------------------------------------------

void mtl_request_capture() { s_capture_requested = 1; }

bool mtl_replay_open(const char *path, int *width, int *height)
{
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    s_replay.file.resize((size_t)std::max(size, 0L));
    bool read = fread(s_replay.file.data(), 1, s_replay.file.size(), f) == s_replay.file.size();
    fclose(f);
    size_t at = 0;
    uint32_t kind;
    Rd r{};
    if (!read || !next_record(s_replay.file, &at, &kind, &r) || kind != CAP_HEADER || r.i() != CAP_VERSION) return false;
    s_replaying = true;
    s_replay.width = *width = (int)r.i();
    s_replay.height = *height = (int)r.i();
    s_replay.state_at = at;
    return r.ok;
}

bool mtl_replay_restore()
{
    if (!restore_state()) return false;
    find_camera();
    return true;
}

void mtl_replay_orbit(double yaw, double pitch, double pan_x, double pan_y, double forward, double pivot)
{
    const double o[6] = {yaw, pitch, pan_x, pan_y, forward, pivot};
    memcpy(s_replay.orbit, o, sizeof o);
}

void mtl_replay_show_interface(bool show) { s_replay.hide_ui = !show; }

bool mtl_preview_model(const char *name, int lod, bool keep_scene, double *distance)
{
    s_preview.keep_scene = s_preview_keeps_scene = keep_scene;
    s_previewing = false;
    if (!name || !*name) {
        s_preview.active = false;
        return true;
    }
    bool ok = preview_load(name, lod);
    s_previewing = ok;
    if (ok && distance) *distance = preview_distance();
    return ok;
}

bool mtl_preview_action(const char *action, double held)
{
    s_preview.held = held;
    return preview_set_action(action ? action : "");
}

const char *mtl_preview_next_action(const char *action)
{
    size_t n = sizeof k_horse_actions / sizeof k_horse_actions[0];
    for (size_t i = 0; i < n; i++)
        if (action && !strcmp(action, k_horse_actions[i].name)) return i + 1 < n ? k_horse_actions[i + 1].name : "";
    return k_horse_actions[0].name;
}

void mtl_preview_material(const char *name) { s_preview.material = name ? name : ""; }

// The mesh a name stands for: itself if it is a mesh; the first mesh of a
// resource file of that name otherwise; null if neither.
const char *mtl_preview_resolve(const char *name)
{
    preview_read_library();
    static std::string result;
    std::string want = lower_case(name ? name : "");
    if (want.size() > 4 && !want.compare(want.size() - 4, 4, ".brf")) want.resize(want.size() - 4);
    auto &meshes = s_preview.library.meshes;
    if (meshes.count(want)) return (result = want).c_str();
    for (size_t f = 0; f < s_preview.library.files.size(); f++) {
        const std::string &path = s_preview.library.files[f];
        std::string base = lower_case(path.substr(path.rfind('/') + 1));
        if (base != want + ".brf") continue;
        for (auto &m : meshes)
            if (m.second.file == (int)f && m.first.find(".lod") == std::string::npos) return (result = m.first).c_str();
    }
    return nullptr;
}

int mtl_preview_list(const char *pattern, FILE *out)
{
    preview_read_library();
    std::string want = lower_case(pattern ? pattern : "");
    int found = 0;
    for (auto &m : s_preview.library.meshes) {
        if (m.first.find(".lod") != std::string::npos || m.first.find(want) == std::string::npos) continue;
        const std::string &file = s_preview.library.files[(size_t)m.second.file];
        fprintf(out, "%-44s %s\n", m.first.c_str(), file.substr(file.rfind('/') + 1).c_str());
        found++;
    }
    for (auto &m : s_preview.library.materials) {
        if (m.first.find(want) == std::string::npos) continue;
        fprintf(out, "%-44s material: shader %s, textures %s %s %s %s\n", m.first.c_str(), m.second.shader.c_str(), m.second.diffuse.c_str(), m.second.diffuse2.c_str(),
                m.second.normal.c_str(), m.second.specular.c_str());
    }
    return found;
}

// The mesh `by` places after (or before) the named one in its resource file's list; lods and further parts passed over.
const char *mtl_preview_neighbour(const char *name, int by)
{
    preview_read_library();
    static std::string result;
    auto &meshes = s_preview.library.meshes;
    auto it = meshes.find(lower_case(name ? name : ""));
    if (it == meshes.end()) return nullptr;
    int file = it->second.file;
    std::vector<std::string> names;
    for (auto &m : meshes) {
        size_t dot = m.first.rfind('.');
        bool part = dot != std::string::npos && dot + 1 < m.first.size() && isdigit((unsigned char)m.first[dot + 1]);
        if (m.second.file == file && m.first.find(".lod") == std::string::npos && !part) names.push_back(m.first);
    }
    auto at = std::find(names.begin(), names.end(), it->first);
    if (at == names.end() || names.empty()) return nullptr;
    long index = ((at - names.begin()) + by) % (long)names.size();
    if (index < 0) index += (long)names.size();
    result = names[(size_t)index];
    return result.c_str();
}

void mtl_replay_frame(bool capture, uint32_t frame)
{
    replay_calls();
    mtl_present(capture, frame);
}

// ---- clears, viewport, fixed-function state ------------------------------------------------------

// A clear starts a new render pass whose load actions do the clearing; the
// draws that follow on the same target go into that pass.
void mtl_glClear(GLbitfield mask)
{
    REC(FN_glClear).i(mask);
    @autoreleasepool {
        Target t;
        if (resolve_target(t)) start_pass(t, mask);
    }
}

void mtl_glClearColor(GLclampf r, GLclampf g, GLclampf b, GLclampf a) { REC(FN_glClearColor).f(r).f(g).f(b).f(a); s_clear_color = MTLClearColorMake(r, g, b, a); }
void mtl_glViewport(GLint x, GLint y, GLsizei width, GLsizei height) { REC(FN_glViewport).i(x).i(y).i(width).i(height); s_viewport = {x, y, width, height}; }
void mtl_glDepthMask(GLboolean flag) { REC(FN_glDepthMask).i(flag); s_depth_mask = flag; }
void mtl_glDepthFunc(GLenum func) { REC(FN_glDepthFunc).i(func); s_depth_func = func; }
void mtl_glCullFace(GLenum mode) { REC(FN_glCullFace).i(mode); s_cull_face = mode; }
void mtl_glBlendEquation(GLenum mode) { REC(FN_glBlendEquation).i(mode); s_blend_eq = mode; }
void mtl_glPolygonMode(GLenum, GLenum mode) { REC(FN_glPolygonMode).i(mode); s_polygon_mode = mode; }

void mtl_glBlendFunc(GLenum sfactor, GLenum dfactor)
{
    REC(FN_glBlendFunc).i(sfactor).i(dfactor);
    s_blend_src = sfactor;
    s_blend_dst = dfactor;
}

static void set_capability(GLenum cap, bool on)
{
    switch (cap) {
    case GL_DEPTH_TEST: s_depth_test = on; break;
    case GL_CULL_FACE: s_cull = on; break;
    case GL_BLEND: s_blend = on; break;
    default: complain("glEnable/glDisable(%s) is ignored", E(cap));
    }
}

void mtl_glEnable(GLenum cap) { REC(FN_glEnable).i(cap); set_capability(cap, true); }
void mtl_glDisable(GLenum cap) { REC(FN_glDisable).i(cap); set_capability(cap, false); }

// ---- queries ------------------------------------------------------------------------------------------

void mtl_glGetIntegerv(GLenum pname, GLint *params)
{
    switch (pname) {
    case GL_MAX_TEXTURE_SIZE: *params = 16384; break;
    default:
        *params = 0;
        complain("unanswered glGetIntegerv %s", E(pname));
    }
}

void mtl_glGetFloatv(GLenum pname, GLfloat *params)
{
    switch (pname) {
    case GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT: *params = 16; break;
    default:
        *params = 0;
        complain("unanswered glGetFloatv %s", E(pname));
    }
}

// ---- buffers -------------------------------------------------------------------------------------------

void mtl_glGenBuffers(GLsizei n, GLuint *buffers) { gen_ids(s_next_buffer, n, buffers); REC(FN_glGenBuffers).i(n).i(n > 0 ? buffers[0] : 0); }

void mtl_glDeleteBuffers(GLsizei n, const GLuint *buffers)
{
    for (GLsizei i = 0; i < n; i++) s_buffers.erase(buffers[i]);
}

void mtl_glBindBuffer(GLenum target, GLuint buffer)
{
    REC(FN_glBindBuffer).i(target).i(buffer);
    if (target == GL_ARRAY_BUFFER) s_array_buffer = buffer;
    else if (target == GL_ELEMENT_ARRAY_BUFFER) s_element_buffer = buffer;
}

// Every glBufferData gets a fresh MTLBuffer, which is also what orphaning asks
// for: draws already encoded keep the old contents alive.
void mtl_glBufferData(GLenum target, GLsizeiptr size, const void *data, GLenum)
{
    REC(FN_glBufferData).i(target).i(size).blob(data, (size_t)std::max<GLsizeiptr>(size, 0));
    GLuint id = target == GL_ARRAY_BUFFER ? s_array_buffer : s_element_buffer;
    if (!id || size <= 0) return;
    @autoreleasepool {
        Buffer &b = s_buffers[id];
        b.mtl = data ? [s_device newBufferWithBytes:data length:(NSUInteger)size options:MTLResourceStorageModeShared]
                     : [s_device newBufferWithLength:(NSUInteger)size options:MTLResourceStorageModeShared];
    }
}

// ---- vertex arrays ---------------------------------------------------------------------------------------

void mtl_glEnableVertexAttribArray(GLuint index)
{
    REC(FN_glEnableVertexAttribArray).i(index);
    if (index < MAX_ATTRIBS) s_attribs[index].enabled = true;
}

void mtl_glDisableVertexAttribArray(GLuint index)
{
    REC(FN_glDisableVertexAttribArray).i(index);
    if (index < MAX_ATTRIBS) s_attribs[index].enabled = false;
}

void mtl_glVertexAttribPointer(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer)
{
    REC(FN_glVertexAttribPointer).i(index).i(size).i(type).i(normalized).i(stride).i((int64_t)(uintptr_t)pointer);
    if (index >= MAX_ATTRIBS) return;
    Attrib &a = s_attribs[index];
    a.buffer = s_array_buffer;
    a.size = size;
    a.type = type;
    a.normalized = normalized;
    a.stride = stride;
    a.offset = (uintptr_t)pointer;
}

// ---- textures ------------------------------------------------------------------------------------------------

void mtl_glGenTextures(GLsizei n, GLuint *textures) { gen_ids(s_next_texture, n, textures); REC(FN_glGenTextures).i(n).i(n > 0 ? textures[0] : 0); }

void mtl_glDeleteTextures(GLsizei n, const GLuint *textures)
{
    for (GLsizei i = 0; i < n; i++) s_textures.erase(textures[i]);
}

void mtl_glActiveTexture(GLenum texture) { REC(FN_glActiveTexture).i(texture); s_unit = texture - GL_TEXTURE0; }

void mtl_glBindTexture(GLenum target, GLuint texture)
{
    REC(FN_glBindTexture).i(target).i(texture);
    if (target != GL_TEXTURE_2D) {
        complain("texture target %s is not supported", E(target));
        return;
    }
    if (s_unit < MAX_UNITS) s_bound_texture[s_unit] = texture;
    if (texture) s_textures[texture]; // the object exists from its first bind
}

void mtl_glTexImage2D(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint, GLenum format,
                      GLenum type, const GLvoid *pixels)
{
    Texture *t = bound_texture();
    if (target != GL_TEXTURE_2D || !t || width <= 0 || height <= 0) return;
    FormatInfo f = pick_format(internalformat, format, type);
    if (f.format == MTLPixelFormatInvalid) {
        complain("unsupported texture format: internalformat %s, format %s, type %s", E_ifmt(internalformat), E(format), E(type));
        return;
    }
    REC(FN_glTexImage2D).i(target).i(level).i(internalformat).i(width).i(height).i(format).i(type).blob(
        pixels, (((size_t)width * (size_t)f.bytes_per_pixel + 3) & ~(size_t)3) * (size_t)height);
    if (level == 0) t->gl_internal = internalformat, t->gl_format = format, t->gl_type = type, t->gl_w = width, t->gl_h = height;
    @autoreleasepool {
        // Without data this is a render target: screen-sized ones get the screen's scale.
        double k = (!pixels && level == 0) ? scale_for(width, height) : 1;
        if (!ensure_storage(*t, f, level, (int)lround(width * k), (int)lround(height * k))) return;
        t->scale = k;
        if (level == 0) t->screen_sized = !pixels && is_screen_sized(width, height);
        if (!pixels) return;
        bool type_matches = type == GL_UNSIGNED_BYTE || type == GL_UNSIGNED_SHORT ||
                            (type == GL_FLOAT && (f.format == MTLPixelFormatR32Float || f.format == MTLPixelFormatRGBA32Float));
        if (!type_matches) {
            complain("texture data of type %s for internalformat %s is not converted", E(type), E_ifmt(internalformat));
            return;
        }
        size_t src_row = ((size_t)width * f.bytes_per_pixel + 3) & ~(size_t)3; // GL_UNPACK_ALIGNMENT is always 4
        MTLRegion region = MTLRegionMake2D(0, 0, (NSUInteger)width, (NSUInteger)height);
        if (f.expand_rgb) {
            std::vector<uint8_t> rgba((size_t)width * height * 4);
            for (int y = 0; y < height; y++) {
                const uint8_t *src = (const uint8_t *)pixels + src_row * y;
                uint8_t *dst = &rgba[(size_t)y * width * 4];
                for (int x = 0; x < width; x++, src += 3, dst += 4) dst[0] = src[0], dst[1] = src[1], dst[2] = src[2], dst[3] = 255;
            }
            [t->mtl replaceRegion:region mipmapLevel:(NSUInteger)level withBytes:rgba.data() bytesPerRow:(NSUInteger)width * 4];
        } else {
            [t->mtl replaceRegion:region mipmapLevel:(NSUInteger)level withBytes:pixels bytesPerRow:src_row];
        }
        if (level > t->top_level) {
            t->top_level = level;
            t->sampler = nil;
        }
    }
}

void mtl_glCompressedTexImage2D(GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLint,
                                GLsizei imageSize, const void *data)
{
    Texture *t = bound_texture();
    if (target != GL_TEXTURE_2D || !t || width <= 0 || height <= 0) return;
    FormatInfo f = pick_compressed_format(internalformat);
    if (f.format == MTLPixelFormatInvalid) {
        complain("unsupported compressed texture format %s", E(internalformat));
        return;
    }
    REC(FN_glCompressedTexImage2D).i(target).i(level).i(internalformat).i(width).i(height).blob(data, (size_t)std::max(imageSize, 0));
    @autoreleasepool {
        if (t->replaced && level > 0) return; // the pack's image came with its own smaller levels
        if (level == 0) t->replaced = false;
        if (!s_material_by_content.empty() && level == 0 && data) {
            int alpha = internalformat == 0x83F2 ? 3 : internalformat == 0x83F3 ? 5 : 0; // GL_COMPRESSED_RGBA_S3TC_DXT3_EXT, ..._DXT5_EXT
            auto it = s_material_by_content.find(content_key((uint32_t)width, (uint32_t)height, data, (size_t)imageSize, alpha));
            // The game loads some textures small first and in full later; whatever
            // was recognised stays until something else is.
            if (level == 0) t->pbr = nil;
            if (it != s_material_by_content.end()) {
                t->material = it->second.cls;
                t->radial_wind = s_texture_names[it->second.name].find("palm") != std::string::npos;
                t->woman = is_womans_face(s_texture_names[it->second.name]);
                horse_eye_for(s_texture_names[it->second.name], t->horse_eye);
                attach_pbr_map(*t, it->second);
                if (replace_from_pack(*t, it->second)) return;
            }
        }
        if (!ensure_storage(*t, f, level, width, height) || !data) return;
        if (s_materials && level == 0) {
            int alpha = internalformat == 0x83F2 ? 3 : internalformat == 0x83F3 ? 5 : 0;
            auto it = s_material_by_content.find(content_key((uint32_t)width, (uint32_t)height, data, (size_t)imageSize, alpha));
            s_material_stats.uploads++;
            if (it != s_material_by_content.end()) s_material_stats.matched++;
            else if (s_material_stats.strangers++ < 60 && imageSize >= 32) {
                const uint8_t *b = (const uint8_t *)data, *e = b + imageSize - 16;
                char first[40], last[40];
                for (int i = 0; i < 16; i++) snprintf(first + i * 2, 3, "%02x", b[i]), snprintf(last + i * 2, 3, "%02x", e[i]);
                wb_note("material classes: no file matches a %dx%d %s texture of %d bytes, first %s last %s", width, height, E(internalformat), imageSize, first, last);
                if (imageSize >= 16384 && s_material_stats.dumped < 12) { // a few of them whole, to compare with the files by hand
                    if (const char *dir = getenv("WBM_OUT")) {
                        char path[1200];
                        snprintf(path, sizeof path, "%s/unmatched_%02d_%dx%d_%s.bin", dir, s_material_stats.dumped++, width, height, E(internalformat));
                        if (FILE *out = fopen(path, "wb")) {
                            fwrite(data, 1, (size_t)imageSize, out);
                            fclose(out);
                        }
                    }
                }
            }
            if (t->material) s_material_stats.recognised++;
        }
        size_t blocks_w = ((size_t)width + 3) / 4, blocks_h = ((size_t)height + 3) / 4;
        if ((size_t)imageSize < blocks_w * blocks_h * f.block_bytes) {
            complain("compressed texture level is shorter than its size requires");
            return;
        }
        [t->mtl replaceRegion:MTLRegionMake2D(0, 0, (NSUInteger)width, (NSUInteger)height)
                  mipmapLevel:(NSUInteger)level
                        slice:0
                    withBytes:data
                  bytesPerRow:blocks_w * f.block_bytes
                bytesPerImage:0];
        if (level > t->top_level) {
            t->top_level = level;
            t->sampler = nil;
        }
    }
}

static void set_texture_parameter(GLenum pname, GLint value, GLfloat fvalue)
{
    Texture *t = bound_texture();
    if (!t) return;
    switch (pname) {
    case GL_TEXTURE_MIN_FILTER: t->min_filter = (GLenum)value; break;
    case GL_TEXTURE_MAG_FILTER: t->mag_filter = (GLenum)value; break;
    case GL_TEXTURE_WRAP_S: t->wrap_s = (GLenum)value; break;
    case GL_TEXTURE_WRAP_T: t->wrap_t = (GLenum)value; break;
    case GL_TEXTURE_MAX_LEVEL: t->max_level = value; break;
    case GL_TEXTURE_MAX_ANISOTROPY_EXT: t->anisotropy = fvalue; break;
    default: complain("texture parameter %s is ignored", E(pname)); return;
    }
    t->sampler = nil;
}

void mtl_glTexParameteri(GLenum, GLenum pname, GLint param) { REC(FN_glTexParameteri).i(pname).i(param); set_texture_parameter(pname, param, (GLfloat)param); }
void mtl_glTexParameterf(GLenum, GLenum pname, GLfloat param) { REC(FN_glTexParameterf).i(pname).f(param); set_texture_parameter(pname, (GLint)param, param); }

void mtl_glGenerateMipmap(GLenum)
{
    REC(FN_glGenerateMipmap);
    Texture *t = bound_texture();
    if (!t || !t->mtl || t->mtl.mipmapLevelCount < 2) return;
    @autoreleasepool {
        end_pass(); // the texture may have just been rendered to
        id<MTLBlitCommandEncoder> blit = [command_buffer() blitCommandEncoder];
        [blit generateMipmapsForTexture:t->mtl];
        [blit endEncoding];
        t->top_level = (int)t->mtl.mipmapLevelCount - 1;
        t->sampler = nil;
    }
}

// The only glTexEnv call the engine makes sets a LOD bias of 0; glPixelStorei
// only ever sets the unpack alignment to 4, which the uploads assume.
void mtl_glTexEnvf(GLenum, GLenum pname, GLfloat param)
{
    REC(FN_glTexEnvf).i(pname).f(param);
    if (pname != GL_TEXTURE_LOD_BIAS || param != 0) complain("glTexEnvf(%s, %g) is ignored", E(pname), param);
}

void mtl_glPixelStorei(GLenum pname, GLint param)
{
    REC(FN_glPixelStorei).i(pname).i(param);
    if (pname != GL_UNPACK_ALIGNMENT || param != 4) complain("glPixelStorei(%s, %d) is ignored", E(pname), param);
}

// ---- framebuffers ---------------------------------------------------------------------------------------------

void mtl_glGenFramebuffers(GLsizei n, GLuint *framebuffers) { gen_ids(s_next_framebuffer, n, framebuffers); REC(FN_glGenFramebuffers).i(n).i(n > 0 ? framebuffers[0] : 0); }
void mtl_glGenRenderbuffers(GLsizei n, GLuint *renderbuffers) { gen_ids(s_next_renderbuffer, n, renderbuffers); REC(FN_glGenRenderbuffers).i(n).i(n > 0 ? renderbuffers[0] : 0); }

void mtl_glDeleteFramebuffers(GLsizei n, const GLuint *framebuffers)
{
    for (GLsizei i = 0; i < n; i++) s_framebuffers.erase(framebuffers[i]);
}

void mtl_glDeleteRenderbuffers(GLsizei n, const GLuint *renderbuffers)
{
    for (GLsizei i = 0; i < n; i++) s_renderbuffers.erase(renderbuffers[i]);
}

void mtl_glBindFramebuffer(GLenum target, GLuint framebuffer)
{
    REC(FN_glBindFramebuffer).i(target).i(framebuffer);
    if (target == GL_READ_FRAMEBUFFER) return;
    s_fbo = framebuffer;
    if (framebuffer) s_framebuffers[framebuffer];
}

void mtl_glBindRenderbuffer(GLenum, GLuint renderbuffer) { REC(FN_glBindRenderbuffer).i(renderbuffer); s_rbo = renderbuffer; }

void mtl_glRenderbufferStorage(GLenum, GLenum internalformat, GLsizei width, GLsizei height)
{
    REC(FN_glRenderbufferStorage).i(internalformat).i(width).i(height);
    if (!s_rbo || width <= 0 || height <= 0) return;
    FormatInfo f = pick_format((GLint)internalformat, 0, 0);
    if (f.format != MTLPixelFormatDepth32Float) {
        complain("renderbuffer format %s is not supported", E(internalformat));
        return;
    }
    @autoreleasepool {
        Renderbuffer &rb = s_renderbuffers[s_rbo];
        rb.gl_internal = internalformat, rb.gl_w = width, rb.gl_h = height;
        rb.scale = scale_for(width, height);
        rb.mtl = new_depth_texture((NSUInteger)lround(width * rb.scale), (NSUInteger)lround(height * rb.scale),
                                   is_screen_sized(width, height) ? s_msaa : 1);
    }
}

void mtl_glFramebufferTexture2D(GLenum, GLenum attachment, GLenum, GLuint texture, GLint level)
{
    REC(FN_glFramebufferTexture2D).i(attachment).i(texture).i(level);
    Framebuffer *fb = find(s_framebuffers, s_fbo);
    if (!fb) return;
    if (level != 0) complain("rendering to mip level %d is not supported", level);
    if (attachment == GL_COLOR_ATTACHMENT0) fb->color_texture = texture;
    else if (attachment == GL_DEPTH_ATTACHMENT) fb->depth_texture = texture, fb->depth_renderbuffer = 0;
    else complain("framebuffer attachment %s is not supported", E(attachment));
}

void mtl_glFramebufferRenderbuffer(GLenum, GLenum attachment, GLenum, GLuint renderbuffer)
{
    REC(FN_glFramebufferRenderbuffer).i(attachment).i(renderbuffer);
    Framebuffer *fb = find(s_framebuffers, s_fbo);
    if (!fb) return;
    if (attachment == GL_DEPTH_ATTACHMENT) fb->depth_renderbuffer = renderbuffer, fb->depth_texture = 0;
    else complain("renderbuffer attachment %s is not supported", E(attachment));
}

// ---- shaders and programs ---------------------------------------------------------------------------------------

GLuint mtl_glCreateShader(GLenum type)
{
    GLuint id = ++s_next_object;
    s_shaders[id] = std::make_shared<Shader>();
    s_shaders[id]->type = type;
    REC(FN_glCreateShader).i(type).i(id);
    return id;
}

GLuint mtl_glCreateProgram(void)
{
    GLuint id = ++s_next_object;
    s_programs[id];
    REC(FN_glCreateProgram).i(id);
    return id;
}

void mtl_glShaderSource(GLuint shader, GLsizei count, const GLchar *const *string, const GLint *length)
{
    auto it = s_shaders.find(shader);
    if (it == s_shaders.end()) return;
    std::string &src = it->second->source;
    src.clear();
    for (GLsizei i = 0; i < count; i++) {
        if (!string[i]) continue;
        if (length && length[i] >= 0) src.append(string[i], (size_t)length[i]);
        else src.append(string[i]);
    }
    REC(FN_glShaderSource).i(shader).s(src);
}

// Translation needs both stages, so the work happens in glLinkProgram.
void mtl_glCompileShader(GLuint) {}

void mtl_glAttachShader(GLuint program, GLuint shader)
{
    REC(FN_glAttachShader).i(program).i(shader);
    Program *p = find(s_programs, program);
    auto it = s_shaders.find(shader);
    if (p && it != s_shaders.end()) p->shaders.push_back(it->second);
}

void mtl_glBindAttribLocation(GLuint program, GLuint index, const GLchar *name)
{
    REC(FN_glBindAttribLocation).i(program).i(index).s(name ? name : "");
    if (Program *p = find(s_programs, program)) p->attribs[name] = (int)index;
}

void mtl_glLinkProgram(GLuint program)
{
    REC(FN_glLinkProgram).i(program);
    Program *p = find(s_programs, program);
    if (!p) return;
    @autoreleasepool {
        link_program(program, *p);
    }
}

void mtl_glDeleteShader(GLuint shader) { s_shaders.erase(shader); } // programs keep their own reference
void mtl_glDeleteProgram(GLuint program)
{
    s_programs.erase(program);
    s_last_key_program = 0;
}
void mtl_glUseProgram(GLuint program) { REC(FN_glUseProgram).i(program); s_program = program; }

// Compile and link status are always reported as success: a program that
// failed to translate is logged and its draws are skipped, which keeps the
// engine running where a reported failure would not.
void mtl_glGetShaderiv(GLuint, GLenum pname, GLint *params) { *params = pname == GL_COMPILE_STATUS ? GL_TRUE : 0; }
void mtl_glGetProgramiv(GLuint, GLenum pname, GLint *params) { *params = pname == GL_LINK_STATUS ? GL_TRUE : 0; }

void mtl_glGetShaderInfoLog(GLuint, GLsizei bufSize, GLsizei *length, GLchar *infoLog)
{
    if (length) *length = 0;
    if (bufSize > 0 && infoLog) infoLog[0] = 0;
}

void mtl_glGetProgramInfoLog(GLuint, GLsizei bufSize, GLsizei *length, GLchar *infoLog)
{
    if (length) *length = 0;
    if (bufSize > 0 && infoLog) infoLog[0] = 0;
}

GLint mtl_glGetUniformLocation(GLuint program, const GLchar *name)
{
    Program *p = find(s_programs, program);
    if (!p || !p->linked) return -1;
    auto it = p->location_by_name.find(name);
    if (it == p->location_by_name.end()) {
        size_t n = strlen(name);
        if (n > 3 && !strcmp(name + n - 3, "[0]")) it = p->location_by_name.find(std::string(name, n - 3));
    }
    return it == p->location_by_name.end() ? -1 : it->second;
}

void mtl_glUniform1f(GLint location, GLfloat v0) { REC(FN_glUniform1f).s(uniform_name(location)).f(v0); set_uniform(location, &v0, 1, 1, false); }
void mtl_glUniform1i(GLint location, GLint v0) { REC(FN_glUniform1i).s(uniform_name(location)).i(v0); set_uniform(location, &v0, 1, 1, true); }
void mtl_glUniform1iv(GLint location, GLsizei count, const GLint *value) { REC(FN_glUniform1iv).s(uniform_name(location)).i(count).blob(value, (size_t)std::max(count, 0) * 4); set_uniform(location, value, count, 1, true); }
void mtl_glUniform4fv(GLint location, GLsizei count, const GLfloat *value) { REC(FN_glUniform4fv).s(uniform_name(location)).i(count).blob(value, (size_t)std::max(count, 0) * 16); set_uniform(location, value, count, 4, false); }

void mtl_glUniformMatrix4fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
    REC(FN_glUniformMatrix4fv).s(uniform_name(location)).i(count).blob(value, (size_t)std::max(count, 0) * 64);
    if (transpose) complain("glUniformMatrix4fv with transpose is not supported");
    set_uniform(location, value, count, 16, false);
}

// ---- draws -----------------------------------------------------------------------------------------------------------

void mtl_glDrawArrays(GLenum mode, GLint first, GLsizei count)
{
    REC(FN_glDrawArrays).i(mode).i(first).i(count);
    if (count <= 0) return;
    @autoreleasepool {
        if (mode == GL_QUADS) {
            int quads = count / 4;
            if (quads > 65535 / 4) {
                complain("glDrawArrays(GL_QUADS) with %d vertices is not supported", count);
                return;
            }
            if (!quads || !prepare_draw()) return;
            [s_pass drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                               indexCount:(NSUInteger)quads * 6
                                indexType:MTLIndexTypeUInt16
                              indexBuffer:quad_indices(quads)
                        indexBufferOffset:0
                            instanceCount:1
                               baseVertex:first
                             baseInstance:0];
            return;
        }
        MTLPrimitiveType type;
        if (!primitive_type(mode, &type) || !prepare_draw()) return;
        [s_pass drawPrimitives:type vertexStart:(NSUInteger)first vertexCount:(NSUInteger)count];
    }
}

void mtl_glDrawElements(GLenum mode, GLsizei count, GLenum type, const GLvoid *indices)
{
    REC(FN_glDrawElements).i(mode).i(count).i(type).i((int64_t)(uintptr_t)indices);
    if (count <= 0) return;
    @autoreleasepool {
        MTLPrimitiveType primitive;
        if (!primitive_type(mode, &primitive)) return;
        if (type != GL_UNSIGNED_SHORT && type != GL_UNSIGNED_INT) {
            complain("index type %s is not supported", E(type));
            return;
        }
        Buffer *b = find(s_buffers, s_element_buffer);
        if (!b || !b->mtl) {
            complain("glDrawElements without an element buffer is not supported");
            return;
        }
        // A character's ground decal would double up with its real shadow.
        if (s_dyn_shadows && s_dyn_window[2] > 0) {
            Program *p = find(s_programs, s_program);
            if (p && p->is_blob_shadow && s_pass_scene) return;
        }
        NSUInteger offset = (NSUInteger)(uintptr_t)indices, index_size = type == GL_UNSIGNED_SHORT ? 2 : 4;
        if (offset + (NSUInteger)count * index_size > b->mtl.length) {
            complain("glDrawElements reads past the end of its element buffer");
            return;
        }
        // A coat near the camera is drawn again in layers, as fur (see k_fur_function).
        // How much of the texture a metre of this mesh covers, from its triangles; once per vertex buffer.
        if (s_fur_strength > 0 && s_pass_scene) {
            Program *fp = find(s_programs, s_program);
            Buffer *vb = fp && fp->fur_offset >= 0 && s_attribs[0].enabled ? find(s_buffers, s_attribs[0].buffer) : nullptr;
            const Attrib &pa = s_attribs[0], &ta = s_attribs[4];
            if (vb && vb->mtl && vb->scale_of != (__bridge void *)vb->mtl) {
                vb->scale_of = (__bridge void *)vb->mtl;
                vb->texture_per_metre = 0.3f;
                if (ta.enabled && ta.buffer == pa.buffer && pa.type == GL_FLOAT && ta.type == GL_FLOAT && pa.stride > 0) {
                    const uint8_t *vertices = (const uint8_t *)vb->mtl.contents, *elements = (const uint8_t *)b->mtl.contents + (uintptr_t)indices;
                    size_t stride = (size_t)pa.stride, n = vb->mtl.length / stride, width = type == GL_UNSIGNED_SHORT ? 2 : 4;
                    double in_texture = 0, in_space = 0;
                    for (GLsizei i = 0; i + 2 < std::min<GLsizei>(count, 3000); i += 3) {
                        uint32_t v[3];
                        for (int k = 0; k < 3; k++) v[k] = width == 2 ? *(const uint16_t *)(elements + (size_t)(i + k) * 2) : *(const uint32_t *)(elements + (size_t)(i + k) * 4);
                        if (v[0] >= n || v[1] >= n || v[2] >= n) continue;
                        for (int k = 0; k < 3; k++) {
                            float p0[3], p1[3], t0[2], t1[2];
                            memcpy(p0, vertices + v[k] * stride + pa.offset, 12), memcpy(p1, vertices + v[(k + 1) % 3] * stride + pa.offset, 12);
                            memcpy(t0, vertices + v[k] * stride + ta.offset, 8), memcpy(t1, vertices + v[(k + 1) % 3] * stride + ta.offset, 8);
                            in_space += sqrt((p1[0] - p0[0]) * (p1[0] - p0[0]) + (p1[1] - p0[1]) * (p1[1] - p0[1]) + (p1[2] - p0[2]) * (p1[2] - p0[2]));
                            in_texture += hypot(t1[0] - t0[0], t1[1] - t0[1]);
                        }
                    }
                    if (in_space > 1e-6 && in_texture > 1e-6) vb->texture_per_metre = (float)(in_texture / in_space);
                }
            }
            if (vb) s_texture_per_metre = vb->texture_per_metre;
        }
        int shells = 0;
        bool long_hair = false; // the coat has a hair map, which says where mane and tail are
        if (s_fur_shells > 0 && s_fur_strength > 0 && s_pass_scene && !s_blend) {
            Program *fp = find(s_programs, s_program);
            if (fp && fp->shell_offset[0] >= 0 && fp->shell_offset[1] >= 0 && fp->u_wvp && fp->alpha_texture_location >= 0) {
                UniformLocation &l = fp->locations[(size_t)fp->alpha_texture_location];
                Texture *coat = (l.unit >= 0 && l.unit < MAX_UNITS) ? find(s_textures, s_bound_texture[l.unit]) : nullptr;
                float distance; // of the object's origin ahead of the camera
                memcpy(&distance, fp->uniform_data[0].data() + fp->u_wvp->offset + 60, 4);
                if (coat && (coat->pbr || coat->hair) && distance > 0 && distance < s_fur_range) {
                    shells = std::max(2, (int)lroundf((float)s_fur_shells * std::min(1.0f, 1.6f * (1.0f - distance / s_fur_range))));
                    long_hair = coat->hair != nil && s_hair_length > 0;
                }
            }
        }
        float clock = 1.0f + (float)fmod(s_wind_time, 600.0); // above 0.5, which is all the coat's layers ask of it
        const float surface[4] = {0, 0, s_fur_density, shells ? clock : 0.0f};
        memcpy(s_shell_now, surface, sizeof surface);
        if (!prepare_draw()) return;
        MTLIndexType index_type = type == GL_UNSIGNED_SHORT ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
        [s_pass drawIndexedPrimitives:primitive indexCount:(NSUInteger)count indexType:index_type indexBuffer:b->mtl indexBufferOffset:offset];
        if (long_hair && s_dyn_shadows && !s_hair_strands && primitive == MTLPrimitiveTypeTriangle) { // strands cast for themselves
            Program *fp = find(s_programs, s_program);
            UniformLocation &l = fp->locations[(size_t)fp->alpha_texture_location];
            if (Texture *coat = (l.unit >= 0 && l.unit < MAX_UNITS) ? find(s_textures, s_bound_texture[l.unit]) : nullptr)
                record_hair_casters(*fp, *coat, (NSUInteger)count, index_type, b->mtl, offset);
        }
        // The coat's layers are tested against the depth of the body but leave
        // none of their own: ambient occlusion and contact shadows, which work
        // from depth, would take the gaps between hairs for pits and speckle the horse.
        // Each layer is blended over what lies beneath it, by how much of the
        // pixel its hairs cover (wbm_shell_cover): the layers go from the body outwards.
        bool depth_mask = s_depth_mask, blend = s_blend;
        GLenum blend_src = s_blend_src, blend_dst = s_blend_dst, blend_eq = s_blend_eq;
        if (shells) s_depth_mask = false, s_blend = true, s_blend_src = GL_SRC_ALPHA, s_blend_dst = GL_ONE_MINUS_SRC_ALPHA, s_blend_eq = GL_FUNC_ADD;
        for (int layer = 1; layer <= shells; layer++) {
            float share = (float)layer / (float)shells;
            const float now[4] = {s_fur_length * share, share, s_fur_density, clock};
            memcpy(s_shell_now, now, sizeof now);
            if (!prepare_draw()) break;
            [s_pass drawIndexedPrimitives:primitive indexCount:(NSUInteger)count indexType:index_type indexBuffer:b->mtl indexBufferOffset:offset];
            s_shell_draws++;
        }
        s_depth_mask = depth_mask; // mane and tail do leave depth: their layers hang far from the body and must hide one another
        // People's hair as strands over the hair mesh, near the camera (see add_hair_strands).
        if (s_people_hair && s_pass_scene && primitive == MTLPrimitiveTypeTriangle) {
            Program *hp = find(s_programs, s_program);
            Buffer *vb = s_attribs[0].enabled ? find(s_buffers, s_attribs[0].buffer) : nullptr;
            const uint32_t needed = 1u << 0 | 1u << 1 | 1u << 2 | 1u << 4 | 1u << 6;
            // The hair shader is the one that takes wbm_shell in both stages without being a coat's or having bones.
            bool hair = hp && hp->shell_offset[0] >= 0 && hp->shell_offset[1] >= 0 && hp->fur_offset < 0 && !hp->u_bones && hp->u_wvp && (hp->input_mask & needed) == needed;
            float distance = 0;
            if (hair) memcpy(&distance, hp->uniform_data[0].data() + hp->u_wvp->offset + 60, 4);
            if (s_people_hair_frame != s_frame_serial) s_people_hair_frame = s_frame_serial, s_people_hair_drawn = 0;
            if (hair && vb && vb->mtl && distance > 0 && distance < s_people_hair_range && s_people_hair_drawn < s_people_hair_most) {
                const StrandMesh &strands = build_strands(mesh_id((NSUInteger)count, offset) ^ 0x5bd1e995a4c7f3d1ull, (const uint8_t *)vb->mtl.contents, vb->mtl.length,
                                                          (const uint8_t *)b->mtl.contents + offset, (size_t)count, type == GL_UNSIGNED_SHORT ? 2 : 4, nil);
                if (strands.count) {
                    Attrib attribs[MAX_ATTRIBS];
                    memcpy(attribs, s_attribs, sizeof attribs);
                    bool cull = s_cull, was_blend = s_blend, depth_mask = s_depth_mask;
                    GLenum src = s_blend_src, dst = s_blend_dst, eq = s_blend_eq;
                    for (Attrib &a : s_attribs)
                        if (a.enabled && a.buffer == attribs[0].buffer) a.buffer = strands.buffer;
                    s_cull = false;
                    if (hp->u_world) drift_of(mesh_id((NSUInteger)count, offset), (const float *)(hp->uniform_data[0].data() + hp->u_world->offset), s_drift_now);
                    for (int soft = 0; soft < 2; soft++) { // what is solid first, leaving depth; then the soft edges, blended, leaving none
                        s_blend = soft != 0, s_depth_mask = soft == 0;
                        if (soft) s_blend_src = GL_SRC_ALPHA, s_blend_dst = GL_ONE_MINUS_SRC_ALPHA, s_blend_eq = GL_FUNC_ADD;
                        const float now[4] = {-s_people_hair_length, s_wind_strength, soft ? 1.0f : -1.0f, clock}; // y: how hard the wind blows
                        memcpy(s_shell_now, now, sizeof now);
                        if (prepare_draw())
                            [s_pass drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:strands.count indexType:MTLIndexTypeUInt32 indexBuffer:strands.indices indexBufferOffset:0];
                        s_shell_draws++;
                    }
                    if (s_dyn_shadows) record_strand_caster(*hp, strands, 2, s_people_hair_length, 0, s_drift_now);
                    memset(s_drift_now, 0, sizeof s_drift_now);
                    memcpy(s_attribs, attribs, sizeof attribs);
                    s_cull = cull, s_blend = was_blend, s_depth_mask = depth_mask, s_blend_src = src, s_blend_dst = dst, s_blend_eq = eq;
                    s_people_hair_drawn++;
                }
            }
        }
        // Mane and tail as strands (see build_strands): one more draw of the same program, over ribbons.
        if (long_hair && s_hair_strands && primitive == MTLPrimitiveTypeTriangle) {
            Program *fp = find(s_programs, s_program);
            UniformLocation &l = fp->locations[(size_t)fp->alpha_texture_location];
            Texture *coat = (l.unit >= 0 && l.unit < MAX_UNITS) ? find(s_textures, s_bound_texture[l.unit]) : nullptr;
            Buffer *vb = s_attribs[0].enabled ? find(s_buffers, s_attribs[0].buffer) : nullptr;
            const uint32_t needed = 1u << 0 | 1u << 1 | 1u << 2 | 1u << 4 | 1u << 5 | 1u << 7 | 1u << 8;
            if (coat && coat->hair && vb && vb->mtl && (fp->input_mask & needed) == needed) {
                const StrandMesh &strands = build_strands(mesh_id((NSUInteger)count, offset) ^ (uint64_t)(uintptr_t)(__bridge void *)coat->hair, (const uint8_t *)vb->mtl.contents,
                                                          vb->mtl.length, (const uint8_t *)b->mtl.contents + offset, (size_t)count, type == GL_UNSIGNED_SHORT ? 2 : 4, coat->hair);
                if (strands.count) {
                    Attrib attribs[MAX_ATTRIBS];
                    memcpy(attribs, s_attribs, sizeof attribs);
                    bool cull = s_cull, was_blend = s_blend;
                    GLenum src = s_blend_src, dst = s_blend_dst, eq = s_blend_eq;
                    for (Attrib &a : s_attribs)
                        if (a.enabled && a.buffer == attribs[0].buffer) a.buffer = strands.buffer;
                    // What is solid first, leaving depth; then the soft edges, blended, leaving none (the sign of z tells the shader which).
                    bool depth_mask = s_depth_mask;
                    s_cull = false;
                    if (fp->u_world) drift_of(mesh_id((NSUInteger)count, offset), (const float *)(fp->uniform_data[0].data() + fp->u_world->offset), s_drift_now);
                    for (int soft = 0; soft < 2; soft++) {
                        if (soft) s_blend = true, s_blend_src = GL_SRC_ALPHA, s_blend_dst = GL_ONE_MINUS_SRC_ALPHA, s_blend_eq = GL_FUNC_ADD, s_depth_mask = false;
                        const float now[4] = {-std::max(s_mane_length, 0.001f), s_tail_length, soft ? s_fur_density : -s_fur_density, clock};
                        memcpy(s_shell_now, now, sizeof now);
                        if (prepare_draw())
                            [s_pass drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:strands.count indexType:MTLIndexTypeUInt32 indexBuffer:strands.indices indexBufferOffset:0];
                        s_shell_draws++;
                    }
                    if (s_dyn_shadows) record_strand_caster(*fp, strands, 1, s_mane_length, s_tail_length, s_drift_now);
                    memset(s_drift_now, 0, sizeof s_drift_now);
                    memcpy(s_attribs, attribs, sizeof attribs);
                    s_depth_mask = depth_mask;
                    s_cull = cull, s_blend = was_blend, s_blend_src = src, s_blend_dst = dst, s_blend_eq = eq;
                    long_hair = false; // no layers as well
                }
            }
        }
        // Mane and tail: layers of their own, reaching much farther out (see wbm_shell_bare).
        for (int layer = 1; long_hair && layer <= shells * 2; layer++) {
            float share = (float)layer / (float)(shells * 2);
            const float now[4] = {s_hair_length * share, share, -s_fur_density, clock};
            memcpy(s_shell_now, now, sizeof now);
            if (!prepare_draw()) break;
            [s_pass drawIndexedPrimitives:primitive indexCount:(NSUInteger)count indexType:index_type indexBuffer:b->mtl indexBufferOffset:offset];
            s_shell_draws++;
        }
        s_blend = blend, s_blend_src = blend_src, s_blend_dst = blend_dst, s_blend_eq = blend_eq;
        if (shells) memset(s_shell_now, 0, sizeof s_shell_now);
        if (s_dyn_shadows && s_pass_prepass && primitive == MTLPrimitiveTypeTriangle) record_caster((NSUInteger)count, index_type, b->mtl, offset, false);
        if (s_dyn_shadows && s_pass_scene && primitive == MTLPrimitiveTypeTriangle && s_depth_test && s_depth_mask)
            record_caster((NSUInteger)count, index_type, b->mtl, offset, true);
        if (s_dyn_shadows && s_pass_shadow_map) record_scenery((NSUInteger)count, offset);
        if (s_dyn_shadows && s_pass_scene && !s_depth_test) record_overlay((NSUInteger)count, offset);
        // The engine draws the sky first, with the depth test off. That draw is the one to capture.
        if ((s_shader_options.pbr_sky || s_sky_ambient) && s_pass_scene && !s_depth_test && s_sky_frame != s_frame_serial)
            capture_sky(primitive, (NSUInteger)count, index_type, b->mtl, offset);
    }
}
