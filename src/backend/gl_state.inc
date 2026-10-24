// The OpenGL state the game sets, and the index of the game's textures by content.
// Included by metal_backend.mm, inside namespace wbm.

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

LastPass s_last; // the pass most recently rendered into (see shared.h)

// Extended-range output (WBM_HDR=1). The drawable is 16-bit float in extended
// sRGB, so the game's gamma-encoded colours mean what they did, and values
// above 1 are brighter than display white. How far above the screen can go
// right now (it depends on its brightness setting) is polled on the main
// thread and handed to the final shader.
bool s_hdr;
float s_hdr_peak = 4.0f; // WBM_HDR_PEAK: the most headroom to use
// Water reflects the scene and bends the view of its bed (WBM_WATER_REFLECTIONS, on with WBM_WATER): before the
// first water of a frame is drawn, the scene so far is copied for the water's shader to look into.
// WBM_WATER_STEPS: how far and how finely the reflection is searched for; WBM_WATER_EVERY: the copy is made afresh every so many frames.
bool s_water_reflections = true;
int s_water_steps = 24, s_water_every = 1;
id<MTLTexture> s_water_scene;
uint64_t s_water_frame, s_water_copies;
float s_parallax;
const float s_parallax_sign[2] = {-1.0f, -1.0f}; // which way the texture runs against the surface's own axes, found by trial // WBM_PARALLAX: depth of stone and brick, metres (0.05 is plenty); 0 for none
float s_grade[4];        // WBM_CONTRAST, WBM_SATURATION, WBM_WARMTH: each about -1 to 1, 0 for no change
std::atomic<float> s_headroom{1.0f};
NSWindow *s_window;
// WBM_TRILINEAR: the game asks for most of its textures to switch from one mipmap level to the next at a
// line (GL_LINEAR_MIPMAP_NEAREST); with this they are blended across the two levels instead, and the line goes.
bool s_trilinear;
float s_lod_bias; // WBM_LOD_BIAS: negative values pick sharper mip levels (macOS 26 and later)

// Ambient occlusion (WBM_AO=1), applied to the engine's HDR scene buffer when
// the engine moves on from it to post-processing.
bool s_ao;
float s_ao_radius = 0.6f, s_ao_strength = 0.5f; // metres; WBM_AO_RADIUS, WBM_AO_STRENGTH
bool s_ao_full; // WBM_AO_FULL: compute at the depth buffer's full resolution instead of half
Camera s_camera; // the scene camera (see shared.h)
bool s_pass_scene, s_pass_prepass, s_pass_shadow_map;
uint64_t s_prepass_frame = ~0ull; // the frame in which the engine last drew its depth pre-pass: what follows it in the scene's buffer is a scene, however few its draws (a small room)

// The per-frame shadow map (WBM_DYNAMIC_SHADOWS=1): see record_caster and render_dynamic_shadows.
bool s_dyn_shadows;
float s_dyn_range = 60.0f;     // WBM_DYN_RANGE: half the width of the window, in metres
float s_dyn_bias_m = 0.05f;    // WBM_DYN_BIAS: metres
// WBM_DYN_FOLIAGE: metres from the camera beyond which cut-out things without bones (leaves, bushes, grass
// tufts, fences) cast no shadow in the per-frame map. In a forest they are most of what it draws, two windows
// over; past this the engine's own, coarser map shades them as it always did. 0 (the default): all of them, at any distance.
float s_dyn_foliage_range = 0.0f;
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
    uint64_t frames, casters, scenery, overlays, late, far_foliage;
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
// Indoors there is no sky to reflect or to take the ambient light's colour from: the room itself stands in for it
// (WBM_ROOM_PROBE, see update_room_probe). An interior is where the engine gives the sun no colour.
bool s_room_probe, s_interior;
int s_room_every = 1; // WBM_ROOM_PROBE_EVERY: frames between updates
id<MTLTexture> s_room_atlas;
uint64_t s_room_frame, s_room_updates;
void update_room_probe();
id<MTLSamplerState> s_sky_sampler;
uint64_t s_sky_frame;
float s_sun_dir[4], s_sun_color[4];
// WBM_TORCH_SHADOWS, a strength: the engine's point lights (torches, candles, fires) cast no shadows. Each
// frame the scene is drawn from every such light into a cube of depths, and the shaders those lights reach look
// their own surface up in it (see render_torch_shadows, and add_torch_fragment in shader_xlate.cpp).
// WBM_TORCH_MAP: the width of a cube's face, texels.
float s_torch_shadows;
float s_torch_self = 0.5f; // WBM_TORCH_SELF: scenery with a corner within this many metres of a light casts no shadow from it (its own bracket); 0: all of it casts
int s_torch_size = 1024;
struct TorchPlace {
    float at[3];
};
std::vector<TorchPlace> s_torch_seen, s_torch_wanted; // the lights of this frame's draws so far, and those of the last frame's
float s_torch_cube[4][3];                             // where the light of each cube stood when it was drawn
int s_torch_cubes;
float s_torch_params[4]; // the shaders' wbm_torch
uint64_t s_torch_frame = ~0ull, s_torch_frames, s_torch_casters;
id<MTLTexture> s_torch_map;
id<MTLSamplerState> s_torch_sampler;
bool casting() { return s_dyn_shadows || s_torch_shadows > 0; } // whether what the pre-pass draws is recorded, to cast shadows
bool s_sun_valid;
bool s_bounce_half;   // WBM_BOUNCE_HALF: bounce light worked out at half resolution, not a quarter (finer, and more than twice the cost)
float s_bounce_light; // WBM_BOUNCE_LIGHT: light thrown by lit surfaces on those near them, a strength (1 is plenty); 0 for none

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
double s_wind_time_before; // and the frame before's
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
float s_fur_anisotropy = 0.8f; // WBM_FUR_ANISOTROPY: glTF's anisotropyStrength, times the hair map's blue (see k_fur_function)
float s_fur_sheen = 1.0f;      // WBM_FUR_SHEEN: the sheen map's strength
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
