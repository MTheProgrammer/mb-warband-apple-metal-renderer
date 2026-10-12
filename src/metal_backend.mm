// Metal backend: implements the game's 62 OpenGL functions on Metal.
//
// All state here is touched only by the game's render thread, except for the
// window setup, which runs on the main thread before rendering starts.

#include "backend.h"
#include "shader_xlate.h"

#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <atomic>
#include <cmath>
#include <cstdarg>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
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
};

struct Texture {
    id<MTLTexture> mtl;
    MTLPixelFormat format = MTLPixelFormatInvalid;
    int top_level = -1; // highest mip level that has content
    int scale = 1;      // storage is this many times the size the game asked for (see s_scale)
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
    int scale = 1;
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
    bool is_view_proj = false;                        // the uniform named matViewProj, which ambient occlusion reads
    bool is_sun_dir = false, is_sun_color = false;    // vSunDir and vSunColor, which contact shadows read
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
    int limit_offset = -1;    // fragment stage, see k_xl_limit_uniform
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
int s_pass_scale = 1, s_pass_samples = 1;

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
int s_screen_w, s_screen_h; // the size the game believes the screen has

bool is_screen_sized(int w, int h)
{
    return w == s_screen_w && h == s_screen_h;
}

int scale_for(int w, int h)
{
    return is_screen_sized(w, h) ? s_scale : 1;
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
std::atomic<float> s_headroom{1.0f};
NSWindow *s_window;
float s_lod_bias; // WBM_LOD_BIAS: negative values pick sharper mip levels (macOS 26 and later)

// Ambient occlusion (WBM_AO=1), applied to the engine's HDR scene buffer when
// the engine moves on from it to post-processing.
bool s_ao;
float s_ao_radius = 0.6f, s_ao_strength = 0.5f; // metres; WBM_AO_RADIUS, WBM_AO_STRENGTH
bool s_ao_full; // WBM_AO_FULL: compute at the depth buffer's full resolution instead of half
float s_view_proj[16];                          // the camera's matViewProj, as last set while drawing the scene
bool s_view_proj_valid;
// Screen-space contact shadows (WBM_CONTACT_SHADOWS=1), computed in the same
// pass as ambient occlusion. They need the sun: its direction and colour as
// last given to a shader while the scene was being drawn.
bool s_contact_shadows;
float s_contact_strength = 0.5f, s_contact_length = 0.4f; // WBM_CONTACT_STRENGTH; WBM_CONTACT_LENGTH in metres
float s_sun_dir[4], s_sun_color[4];
bool s_sun_valid;
void apply_ambient_occlusion();

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
    int scale = 1, samples = 1;
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
    if ((s_ao || s_contact_shadows) && s_last.scene && s_last.color != t.color && s_last.draws > 100 &&
        s_last.occluded_in_frame != s_frame_serial) {
        s_last.occluded_in_frame = s_frame_serial;
        apply_ambient_occlusion();
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
    p.yflip_offset = p.samples_offset = p.headroom_offset = p.limit_offset = -1;
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
                p.locations.back().is_view_proj = name == "matViewProj";
                p.locations.back().is_sun_dir = name == "vSunDir";
                p.locations.back().is_sun_color = name == "vSunColor";
            }
            return p.locations[(size_t)it->second];
        };
        for (const XlUniform &u : stages[s]->uniforms) {
            if (u.name == k_xl_yflip_uniform) p.yflip_offset = u.offset;
            else if (u.name == k_xl_samples_uniform) p.samples_offset = u.offset;
            else if (u.name == k_xl_headroom_uniform) p.headroom_offset = u.offset;
            else if (u.name == k_xl_limit_uniform) p.limit_offset = u.offset;
            else location(u.name).uniform[s] = &u;
        }
        for (const XlSampler &smp : stages[s]->samplers) location(smp.name).texture_index[s] = smp.texture_index;
        for (const std::string &name : stages[s]->srgb_samplers) location(name).srgb = true;
        p.uniform_dirty[s] = true;
        p.pushed_frame[s] = 0;
    }
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
    if ((s_ao || s_contact_shadows) && (l.is_view_proj || l.is_sun_dir || l.is_sun_color) && count >= 1) {
        // Other cameras (shadows, water reflection) and the UI set these too; only
        // the values in use while the scene buffer is bound describe the scene.
        Target t;
        if (s_fbo != 0 && resolve_target(t) && t.scene) {
            if (l.is_view_proj && components == 16) {
                memcpy(s_view_proj, data, sizeof s_view_proj);
                s_view_proj_valid = true;
            } else if (l.is_sun_dir && components == 4) {
                memcpy(s_sun_dir, data, sizeof s_sun_dir);
                s_sun_valid = true;
            } else if (l.is_sun_color && components == 4) {
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
    float4 sun;        // xyz: direction to the sun in view space
    float shadow_strength, shadow_length;
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

    float radius_px = clamp(p.radius * p.p00 / P.z * 0.5 * p.ao_size.x, 2.0, 96.0 * p.ao_size.x / p.depth_size.x);
    float noise = fract(52.9829189 * fract(dot(in.position.xy, float2(0.06711056, 0.00583715))));
    float bias = 0.02 + 0.004 * P.z;
    float sum = 0.0;
    const int directions = 4, steps = 4;
    for (int i = 0; i < directions; i++) {
        float angle = (float(i) + noise) * (6.2831853 / float(directions));
        float2 dir = float2(cos(angle), sin(angle));
        for (int s = 0; s < steps; s++) {
            float t = (float(s) + fract(noise * 7.0 + 0.37 * float(i)) + 0.2) / float(steps);
            float3 v = view_position(z, c + int2(dir * radius_px * t), p) - P;
            float dd = dot(v, v);
            sum += max(0.0, dot(v, n) - bias) / (dd + 0.0001) * saturate(1.0 - dd / (p.radius * p.radius));
        }
    }
    float occlusion = sum * p.radius * 2.0 / float(directions * steps);
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
            ao *= 1.0 - p.shadow_strength * hit * smoothstep(0.05, 0.3, facing);
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
)";

struct AOParams { // must match Params in k_ao_source
    float a, b, p00, p11;
    float depth_size[2], ao_size[2];
    float radius, strength, reversed, fade_start;
    float sun[4];
    float shadow_strength, shadow_length, pad[2];
};

id<MTLLibrary> s_ao_library;
id<MTLRenderPipelineState> s_ao_linearize_pipeline, s_ao_compute_pipeline, s_ao_blur_pipeline;
id<MTLTexture> s_ao_z_texture, s_ao_blurred_texture;
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
        return s_ao = s_contact_shadows = false;
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
    if (!s_ao_linearize_pipeline || !s_ao_compute_pipeline || !s_ao_blur_pipeline) {
        complain("ambient occlusion pipeline failed: %s", error.localizedDescription.UTF8String);
        return s_ao = s_contact_shadows = false;
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
    if (!s_view_proj_valid || !s_last.depth || !ao_setup()) return;

    // The projection's depth terms and scales, read out of view-projection. With
    // rows r0..r3 of that matrix: r3.xyz is the camera's forward direction, clip
    // w is distance along it, and clip z = a * w + b.
    const float *m = s_view_proj; // column-major: row r is m[r], m[4 + r], m[8 + r], m[12 + r]
    auto row = [&](int r, float out[4]) {
        for (int c = 0; c < 4; c++) out[c] = m[c * 4 + r];
    };
    float r0[4], r1[4], r2[4], r3[4];
    row(0, r0), row(1, r1), row(2, r2), row(3, r3);
    auto dot3 = [](const float *x, const float *y) { return x[0] * y[0] + x[1] * y[1] + x[2] * y[2]; };
    float forward2 = dot3(r3, r3);
    if (forward2 < 1e-12f) return;
    AOParams params;
    params.a = dot3(r2, r3) / forward2;
    params.b = r2[3] - params.a * r3[3];
    params.p00 = sqrtf(dot3(r0, r0) / forward2);
    params.p11 = sqrtf(dot3(r1, r1) / forward2);
    if (!(fabsf(params.b) > 1e-6f) || !(params.p00 > 1e-4f)) {
        complain("ambient occlusion: the camera matrix is not a perspective projection it understands");
        return;
    }
    NSUInteger w = s_last.depth.width, h = s_last.depth.height;
    NSUInteger aw = s_ao_full ? w : (w + 1) / 2, ah = s_ao_full ? h : (h + 1) / 2;
    params.depth_size[0] = (float)w, params.depth_size[1] = (float)h;
    params.ao_size[0] = (float)aw, params.ao_size[1] = (float)ah;
    params.radius = s_ao_radius;
    params.strength = s_ao ? s_ao_strength : 0.0f;
    // The sun in view space: rows 0, 1 and 3 of view-projection are the camera's
    // right, up and forward axes, each scaled. vSunDir points the way the light
    // travels, so the direction to the sun is its negative. No sun (night,
    // interiors) means no contact shadows.
    memset(params.sun, 0, sizeof params.sun);
    params.shadow_strength = 0;
    params.shadow_length = s_contact_length;
    params.pad[0] = params.pad[1] = 0;
    float sun_brightness = s_sun_color[0] + s_sun_color[1] + s_sun_color[2];
    if (s_contact_shadows && s_sun_valid && sun_brightness > 0.05f) {
        float to_sun[3] = {-s_sun_dir[0], -s_sun_dir[1], -s_sun_dir[2]};
        params.sun[0] = dot3(r0, to_sun) / sqrtf(dot3(r0, r0));
        params.sun[1] = dot3(r1, to_sun) / sqrtf(dot3(r1, r1));
        params.sun[2] = dot3(r3, to_sun) / sqrtf(forward2);
        params.shadow_strength = s_contact_strength;
    }
    params.reversed = s_reverse_depth ? 1.0f : 0.0f;
    params.fade_start = 60.0f; // metres: beyond this the effect fades out, and is gone at twice it

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
    half_pass(s_ao_z_texture, s_ao_linearize_pipeline, s_last.depth, nil);
    half_pass(s_ao_texture, s_ao_compute_pipeline, s_ao_z_texture, nil);
    half_pass(s_ao_blurred_texture, s_ao_blur_pipeline, s_ao_texture, s_ao_z_texture);

    // Blended onto the resolved image, which is what post-processing reads. The
    // multisampled copy is left as it was: writing there would blend every
    // sample and resolve the whole buffer a second time, and the engine does not
    // draw into the scene again once post-processing has begun.
    MTLRenderPassDescriptor *onto = [MTLRenderPassDescriptor renderPassDescriptor];
    onto.colorAttachments[0].texture = s_last.color;
    onto.colorAttachments[0].loadAction = MTLLoadActionLoad;
    onto.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> pass = [command_buffer() renderCommandEncoderWithDescriptor:onto];
    [pass setRenderPipelineState:apply];
    [pass setFragmentTexture:s_ao_blurred_texture atIndex:0];
    [pass setFragmentBytes:&params length:sizeof params atIndex:0];
    [pass drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [pass endEncoding];
}

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
    float limit = s_pass_screen ? 1.0f : 65000.0f;
    if (p->limit_offset >= 0 && memcmp(p->uniform_data[1].data() + p->limit_offset, &limit, 4)) {
        memcpy(p->uniform_data[1].data() + p->limit_offset, &limit, 4);
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
            if (l.srgb && t && t->mtl) {
                // The shader no longer decodes this sampler's colour itself.
                if (id<MTLTexture> view = srgb_view_for(*t)) texture = view;
                else complain("a texture without an sRGB format is bound to %s; it will look too bright", l.name.c_str());
            }
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
    return true;
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

void save_capture(id<MTLTexture> copy, uint32_t frame)
{
    int w = (int)copy.width, h = (int)copy.height;
    std::vector<unsigned char> pixels((size_t)w * h * 4);
    if (copy.pixelFormat == MTLPixelFormatRGBA16Float) {
        // An extended-range frame: a PNG can only hold the part up to display white.
        std::vector<uint16_t> halves((size_t)w * h * 4);
        [copy getBytes:halves.data() bytesPerRow:(NSUInteger)w * 8 fromRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0];
        for (size_t i = 0; i < halves.size(); i++)
            pixels[i] = (unsigned char)lrintf(std::min(1.0f, std::max(0.0f, half_to_float(halves[i]))) * 255.0f);
        wb_save_frame_png(pixels.data(), w, h, frame, false);
        return;
    }
    [copy getBytes:pixels.data() bytesPerRow:(NSUInteger)w * 4 fromRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0];
    wb_save_frame_png(pixels.data(), w, h, frame, true);
}

} // namespace

// ---- window and frame ------------------------------------------------------------------------

void mtl_attach_window(void *nswindow, int width, int height)
{
    s_debug_clear = getenv("WBM_DEBUG_CLEAR") != nullptr;
    s_device = MTLCreateSystemDefaultDevice();
    s_queue = [s_device newCommandQueue];
    s_viewport = {0, 0, width, height};
    s_screen_w = width;
    s_screen_h = height;
    const char *scale_env = getenv("WBM_SCALE");
    const char *shadows_env = getenv("WBM_SHADOWS"); // on unless set to 0
    s_shader_options.soft_shadows = !(shadows_env && atoi(shadows_env) == 0);
    const char *msaa_env = getenv("WBM_MSAA");
    const char *reverse_env = getenv("WBM_REVERSE_Z"), *srgb_env = getenv("WBM_SRGB"), *cache_env = getenv("WBM_STATE_CACHE");
    s_reverse_depth = s_shader_options.reverse_depth = !(reverse_env && atoi(reverse_env) == 0); // on unless set to 0
    s_shader_options.srgb_textures = srgb_env && atoi(srgb_env) != 0;                            // off unless set
    s_state_cache = !(cache_env && atoi(cache_env) == 0);
    const char *hdr_env = getenv("WBM_HDR"), *tonemap_env = getenv("WBM_TONEMAP");
    s_hdr = s_shader_options.hdr_output = hdr_env && atoi(hdr_env) != 0;                  // off unless set
    s_shader_options.filmic_tonemap = tonemap_env && atoi(tonemap_env) != 0;              // off unless set
    if (const char *v = getenv("WBM_HDR_PEAK")) s_hdr_peak = std::max(1.0f, (float)atof(v));
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
            "contact shadows %s; filmic tone map %s; extended-range output %s",
            width, height, width * s_scale, height * s_scale, s_scale, s_shader_options.soft_shadows ? "on" : "off", s_msaa,
            s_reverse_depth ? "on" : "off", s_shader_options.srgb_textures ? "on" : "off", s_state_cache ? "on" : "off",
            s_shader_options.pbr_environment ? "on, with metalness and environment" : s_shader_options.pbr_specular ? "on" : "off",
            s_ao ? "on" : "off", s_contact_shadows ? "on" : "off", s_shader_options.filmic_tonemap ? "on" : "off",
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
        if (s_drawable) [s_cmd presentDrawable:s_drawable];
        [s_cmd commit];
        if (copy) {
            [s_cmd waitUntilCompleted];
            save_capture(copy, frame);
        }
        s_cmd = nil;
        s_drawable = nil;
        s_frame_serial++;
        if (s_hdr && s_frame_serial % 30 == 0) {
            dispatch_async(dispatch_get_main_queue(), ^{
              CGFloat available = s_window.screen.maximumExtendedDynamicRangeColorComponentValue;
              s_headroom.store(std::min(s_hdr_peak, std::max(1.0f, (float)available)));
            });
        }
    }
}

// ---- clears, viewport, fixed-function state ------------------------------------------------------

// A clear starts a new render pass whose load actions do the clearing; the
// draws that follow on the same target go into that pass.
void mtl_glClear(GLbitfield mask)
{
    @autoreleasepool {
        Target t;
        if (resolve_target(t)) start_pass(t, mask);
    }
}

void mtl_glClearColor(GLclampf r, GLclampf g, GLclampf b, GLclampf a) { s_clear_color = MTLClearColorMake(r, g, b, a); }
void mtl_glViewport(GLint x, GLint y, GLsizei width, GLsizei height) { s_viewport = {x, y, width, height}; }
void mtl_glDepthMask(GLboolean flag) { s_depth_mask = flag; }
void mtl_glDepthFunc(GLenum func) { s_depth_func = func; }
void mtl_glCullFace(GLenum mode) { s_cull_face = mode; }
void mtl_glBlendEquation(GLenum mode) { s_blend_eq = mode; }
void mtl_glPolygonMode(GLenum, GLenum mode) { s_polygon_mode = mode; }

void mtl_glBlendFunc(GLenum sfactor, GLenum dfactor)
{
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

void mtl_glEnable(GLenum cap) { set_capability(cap, true); }
void mtl_glDisable(GLenum cap) { set_capability(cap, false); }

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

void mtl_glGenBuffers(GLsizei n, GLuint *buffers) { gen_ids(s_next_buffer, n, buffers); }

void mtl_glDeleteBuffers(GLsizei n, const GLuint *buffers)
{
    for (GLsizei i = 0; i < n; i++) s_buffers.erase(buffers[i]);
}

void mtl_glBindBuffer(GLenum target, GLuint buffer)
{
    if (target == GL_ARRAY_BUFFER) s_array_buffer = buffer;
    else if (target == GL_ELEMENT_ARRAY_BUFFER) s_element_buffer = buffer;
}

// Every glBufferData gets a fresh MTLBuffer, which is also what orphaning asks
// for: draws already encoded keep the old contents alive.
void mtl_glBufferData(GLenum target, GLsizeiptr size, const void *data, GLenum)
{
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
    if (index < MAX_ATTRIBS) s_attribs[index].enabled = true;
}

void mtl_glDisableVertexAttribArray(GLuint index)
{
    if (index < MAX_ATTRIBS) s_attribs[index].enabled = false;
}

void mtl_glVertexAttribPointer(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer)
{
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

void mtl_glGenTextures(GLsizei n, GLuint *textures) { gen_ids(s_next_texture, n, textures); }

void mtl_glDeleteTextures(GLsizei n, const GLuint *textures)
{
    for (GLsizei i = 0; i < n; i++) s_textures.erase(textures[i]);
}

void mtl_glActiveTexture(GLenum texture) { s_unit = texture - GL_TEXTURE0; }

void mtl_glBindTexture(GLenum target, GLuint texture)
{
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
    @autoreleasepool {
        // Without data this is a render target: screen-sized ones get the screen's scale.
        int k = (!pixels && level == 0) ? scale_for(width, height) : 1;
        if (!ensure_storage(*t, f, level, width * k, height * k)) return;
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
    @autoreleasepool {
        if (!ensure_storage(*t, f, level, width, height) || !data) return;
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

void mtl_glTexParameteri(GLenum, GLenum pname, GLint param) { set_texture_parameter(pname, param, (GLfloat)param); }
void mtl_glTexParameterf(GLenum, GLenum pname, GLfloat param) { set_texture_parameter(pname, (GLint)param, param); }

void mtl_glGenerateMipmap(GLenum)
{
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
    if (pname != GL_TEXTURE_LOD_BIAS || param != 0) complain("glTexEnvf(%s, %g) is ignored", E(pname), param);
}

void mtl_glPixelStorei(GLenum pname, GLint param)
{
    if (pname != GL_UNPACK_ALIGNMENT || param != 4) complain("glPixelStorei(%s, %d) is ignored", E(pname), param);
}

// ---- framebuffers ---------------------------------------------------------------------------------------------

void mtl_glGenFramebuffers(GLsizei n, GLuint *framebuffers) { gen_ids(s_next_framebuffer, n, framebuffers); }
void mtl_glGenRenderbuffers(GLsizei n, GLuint *renderbuffers) { gen_ids(s_next_renderbuffer, n, renderbuffers); }

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
    if (target == GL_READ_FRAMEBUFFER) return;
    s_fbo = framebuffer;
    if (framebuffer) s_framebuffers[framebuffer];
}

void mtl_glBindRenderbuffer(GLenum, GLuint renderbuffer) { s_rbo = renderbuffer; }

void mtl_glRenderbufferStorage(GLenum, GLenum internalformat, GLsizei width, GLsizei height)
{
    if (!s_rbo || width <= 0 || height <= 0) return;
    FormatInfo f = pick_format((GLint)internalformat, 0, 0);
    if (f.format != MTLPixelFormatDepth32Float) {
        complain("renderbuffer format %s is not supported", E(internalformat));
        return;
    }
    @autoreleasepool {
        Renderbuffer &rb = s_renderbuffers[s_rbo];
        rb.scale = scale_for(width, height);
        rb.mtl = new_depth_texture((NSUInteger)(width * rb.scale), (NSUInteger)(height * rb.scale),
                                   is_screen_sized(width, height) ? s_msaa : 1);
    }
}

void mtl_glFramebufferTexture2D(GLenum, GLenum attachment, GLenum, GLuint texture, GLint level)
{
    Framebuffer *fb = find(s_framebuffers, s_fbo);
    if (!fb) return;
    if (level != 0) complain("rendering to mip level %d is not supported", level);
    if (attachment == GL_COLOR_ATTACHMENT0) fb->color_texture = texture;
    else if (attachment == GL_DEPTH_ATTACHMENT) fb->depth_texture = texture, fb->depth_renderbuffer = 0;
    else complain("framebuffer attachment %s is not supported", E(attachment));
}

void mtl_glFramebufferRenderbuffer(GLenum, GLenum attachment, GLenum, GLuint renderbuffer)
{
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
    return id;
}

GLuint mtl_glCreateProgram(void)
{
    GLuint id = ++s_next_object;
    s_programs[id];
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
}

// Translation needs both stages, so the work happens in glLinkProgram.
void mtl_glCompileShader(GLuint) {}

void mtl_glAttachShader(GLuint program, GLuint shader)
{
    Program *p = find(s_programs, program);
    auto it = s_shaders.find(shader);
    if (p && it != s_shaders.end()) p->shaders.push_back(it->second);
}

void mtl_glBindAttribLocation(GLuint program, GLuint index, const GLchar *name)
{
    if (Program *p = find(s_programs, program)) p->attribs[name] = (int)index;
}

void mtl_glLinkProgram(GLuint program)
{
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
void mtl_glUseProgram(GLuint program) { s_program = program; }

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

void mtl_glUniform1f(GLint location, GLfloat v0) { set_uniform(location, &v0, 1, 1, false); }
void mtl_glUniform1i(GLint location, GLint v0) { set_uniform(location, &v0, 1, 1, true); }
void mtl_glUniform1iv(GLint location, GLsizei count, const GLint *value) { set_uniform(location, value, count, 1, true); }
void mtl_glUniform4fv(GLint location, GLsizei count, const GLfloat *value) { set_uniform(location, value, count, 4, false); }

void mtl_glUniformMatrix4fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
    if (transpose) complain("glUniformMatrix4fv with transpose is not supported");
    set_uniform(location, value, count, 16, false);
}

// ---- draws -----------------------------------------------------------------------------------------------------------

void mtl_glDrawArrays(GLenum mode, GLint first, GLsizei count)
{
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
        NSUInteger offset = (NSUInteger)(uintptr_t)indices, index_size = type == GL_UNSIGNED_SHORT ? 2 : 4;
        if (offset + (NSUInteger)count * index_size > b->mtl.length) {
            complain("glDrawElements reads past the end of its element buffer");
            return;
        }
        if (!prepare_draw()) return;
        [s_pass drawIndexedPrimitives:primitive
                           indexCount:(NSUInteger)count
                            indexType:type == GL_UNSIGNED_SHORT ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32
                          indexBuffer:b->mtl
                    indexBufferOffset:offset];
    }
}
