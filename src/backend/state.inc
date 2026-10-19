// The library's objects (textures, buffers, programs, pipelines) and the device and frame state.
// Included by metal_backend.mm, inside namespace wbm.

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
    id<MTLTexture> sheen;  // a pack's sheen map (<name>_sheen, glTF's KHR_materials_sheen): colour, and roughness in alpha
    id<MTLTexture> pbr;    // a pack's metalness and roughness map for this texture, see k_xl_pbr_sampler
    id<MTLTexture> depth;  // a pack's map of the surface's heights (<name>_depth), for parallax
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
    uint8_t motion; // 0: the pass has no movement target; 1: it has, and this program leaves it alone; 2: and writes it
    uint8_t pad[1];
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
    bool is_sheen_map = false;                        // the wbm_sheen_map sampler, which gets the bound texture's sheen map
    bool is_depth_map = false;                        // the wbm_depth_map sampler, which gets the bound texture's map of heights
    bool is_pbr_map = false;                          // the wbm_pbr_map sampler, which gets the bound texture's metalness map
    bool is_scene_copy = false;                       // the wbm_scene sampler, which gets the scene as it was before water was drawn
    bool is_light_indices = false;                    // iLightIndices: the order in which a shader takes the point lights
    bool is_torch_map = false;                        // the wbm_torch_map sampler, which gets the point lights' cubes of depth
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
    int jitter_offset = -1;  // vertex stage, see k_xl_jitter_uniform
    int hash_frame_offset = -1; // fragment stage: wbm_hash_frame (XlOptions::alpha_hash)
    int reproject_offset = -1, wind_before_offset = -1; // vertex stage: wbm_reproject and wbm_wind_before, of the shaders that write their own movement (XlOptions::motion_vectors)
    int samples_offset = -1; // fragment stage, see k_xl_samples_uniform
    int headroom_offset = -1; // fragment stage, see k_xl_headroom_uniform
    int grade_offset = -1;    // fragment stage, wbm_grade
    int parallax_offset = -1; // fragment stage, wbm_parallax
    int water_offset = -1;    // fragment stage, wbm_water[3]: see copy_scene_for_water
    int limit_offset = -1;    // fragment stage, see k_xl_limit_uniform
    int dyn_window_offset = -1, dyn_near_offset = -1, dyn_bias_offset = -1;
    int torch_offset = -1, torch_light_offset = -1; // fragment stage, see k_xl_torch_sampler
    int light_pos_location = -1, light_colour_location = -1, light_count_location = -1; // vLightPosDir, vLightDiffuse, iLightPointCount
    int amb_offset[2] = {-1, -1}; // per stage, see k_xl_ambient_uniform
    int material_offset = -1;     // fragment stage, see k_xl_material_uniform
    int pbr_offset = -1;          // fragment stage, see k_xl_pbr_sampler
    int skin_offset[2] = {-1, -1}; // per stage, see k_xl_skin_uniform
    int wind_offset = -1;          // vertex stage, see k_xl_wind_uniform
    int hair_colour_offset = -1; // fragment stage, the same
    int horse_eye_offset = -1; // fragment stage, the same
    int sheen_offset = -1;     // fragment stage: wbm_sheen, see k_fur_function
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
// WBM_UPSCALE_MODE=temporal: MetalFX's temporal scaler in place of the spatial
// one. It builds each picture from several frames, so it is told what the
// spatial one is not: how far along each frame the picture was shifted
// (s_jitter, in pixels of the scene, a different fraction each frame), how far
// off every pixel is, and where on screen it was a frame ago. See
// apply_temporal_upscale. It smooths edges itself, so FXAA is left out.
bool s_temporal;
float s_jitter[2];
int s_jitter_sign = 1; // WBM_DEBUG_JITTER_SIGN
// The second colour target of the scene under temporal upscaling: what sways
// in the wind writes its own movement there (see k_motion_vertex_uniforms in
// shader_xlate.cpp). x, y: the step across the screen since the last frame, in
// clip space; z: the distance, below 0 where nothing wrote.
id<MTLTexture> s_scene_motion;
uint64_t s_scene_motion_frame = ~0ull; // the frame it was last cleared in
bool s_pass_motion;                    // the pass under way has it attached
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
