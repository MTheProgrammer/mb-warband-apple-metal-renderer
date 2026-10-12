// Metal backend: implements the game's 62 OpenGL functions on Metal.
//
// All state here is touched only by the game's render thread, except for the
// window setup, which runs on the main thread before rendering starts.

#include "backend.h"
#include "shader_xlate.h"

#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

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
    std::vector<UniformLocation> locations;
    std::unordered_map<std::string, GLint> location_by_name;
    int yflip_offset = -1;   // vertex stage, see k_xl_yflip_uniform
    int samples_offset = -1; // fragment stage, see k_xl_samples_uniform
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
    d.usage = MTLTextureUsageRenderTarget;
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
        if (!s_screen_depth || s_screen_depth.width != t.color.width || s_screen_depth.height != t.color.height) {
            s_screen_depth = new_depth_texture(t.color.width, t.color.height, s_msaa);
            s_screen_ms = s_msaa > 1 ? new_render_texture(t.color.pixelFormat, t.color.width, t.color.height, s_msaa) : nil;
        }
        t.depth = s_screen_depth;
        t.ms_color = s_screen_ms;
        t.samples = s_msaa;
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
        d.depthAttachment.clearDepth = 1.0;
    }
    s_pass = [command_buffer() renderCommandEncoderWithDescriptor:d];
    s_pass_color = t.color;
    s_pass_depth = t.depth;
    s_pass_screen = t.screen;
    s_pass_scale = t.scale;
    s_pass_samples = t.samples;
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
    d.swizzle = f.swizzle;
    t.mtl = [s_device newTextureWithDescriptor:d];
    t.format = f.format;
    t.top_level = -1;
    t.sampler = nil;
    t.ms = nil;
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
    p.yflip_offset = p.samples_offset = -1;
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
            }
            return p.locations[(size_t)it->second];
        };
        for (const XlUniform &u : stages[s]->uniforms) {
            if (u.name == k_xl_yflip_uniform) p.yflip_offset = u.offset;
            else if (u.name == k_xl_samples_uniform) p.samples_offset = u.offset;
            else location(u.name).uniform[s] = &u;
        }
        for (const XlSampler &smp : stages[s]->samplers) location(smp.name).texture_index[s] = smp.texture_index;
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
    for (int s = 0; s < 2; s++) {
        const XlUniform *u = l.uniform[s];
        if (!u) continue;
        int words = std::min(components, u->components * u->columns);
        int n = std::min(count, std::max(1, u->array_size));
        int stride = u->array_size ? u->array_stride : words * 4;
        uint8_t *dst = p->uniform_data[s].data() + u->offset;
        const uint32_t *src = (const uint32_t *)data;
        for (int i = 0; i < n; i++, dst += stride, src += components) {
            if (u->is_int == data_is_int) {
                memcpy(dst, src, (size_t)words * 4);
            } else if (data_is_int) {
                float v = (float)*(const GLint *)src;
                memcpy(dst, &v, 4);
            } else {
                GLint v = (GLint) * (const GLfloat *)src;
                memcpy(dst, &v, 4);
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
        d.depthCompareFunction = compare[func];
        d.depthWriteEnabled = s_depth_mask;
        state = [s_device newDepthStencilStateWithDescriptor:d];
    }
    return state;
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
    id<MTLRenderPipelineState> pipeline = pipeline_for(*p, key);
    if (!pipeline) return false;
    [s_pass setRenderPipelineState:pipeline];
    [s_pass setDepthStencilState:depth_state(s_pass_depth != nil)];

    // Textures are rendered upside down so that their rows are stored in OpenGL's
    // order (see wbm_yflip in shader_xlate.h). That mirrors the winding, and the
    // viewport then counts from the top like Metal's own.
    bool flip = !s_pass_screen;
    double target_h = s_pass_color ? (double)s_pass_color.height : (double)s_pass_depth.height;
    double k = s_pass_scale, vx = s_viewport.x * k, vy = s_viewport.y * k, vw = s_viewport.w * k, vh = s_viewport.h * k;
    [s_pass setViewport:(MTLViewport){vx, flip ? vy : target_h - vy - vh, vw, vh, 0.0, 1.0}];
    [s_pass setFrontFacingWinding:flip ? MTLWindingClockwise : MTLWindingCounterClockwise];
    [s_pass setCullMode:!s_cull ? MTLCullModeNone : s_cull_face == GL_FRONT ? MTLCullModeFront : MTLCullModeBack];
    [s_pass setTriangleFillMode:s_polygon_mode == GL_LINE ? MTLTriangleFillModeLines : MTLTriangleFillModeFill];

    if (p->samples_offset >= 0) {
        float samples = (float)s_pass_samples;
        memcpy(p->uniform_data[1].data() + p->samples_offset, &samples, 4);
    }
    if (p->yflip_offset >= 0) {
        float yflip = flip ? -1.0f : 1.0f;
        memcpy(p->uniform_data[0].data() + p->yflip_offset, &yflip, 4);
    }
    const XlStage *stages[2] = {&p->xl.vs, &p->xl.fs};
    for (int s = 0; s < 2; s++) {
        if (stages[s]->uniform_buffer_index < 0 || p->uniform_data[s].empty()) continue;
        NSUInteger offset;
        id<MTLBuffer> buffer = push_uniforms(p->uniform_data[s], &offset);
        if (s == 0) [s_pass setVertexBuffer:buffer offset:offset atIndex:(NSUInteger)stages[s]->uniform_buffer_index];
        else [s_pass setFragmentBuffer:buffer offset:offset atIndex:(NSUInteger)stages[s]->uniform_buffer_index];
    }
    for (UniformLocation &l : p->locations) {
        for (int s = 0; s < 2; s++) {
            if (l.texture_index[s] < 0) continue;
            Texture *t = (l.unit >= 0 && l.unit < MAX_UNITS) ? find(s_textures, s_bound_texture[l.unit]) : nullptr;
            id<MTLTexture> texture = t && t->mtl ? t->mtl : s_white;
            id<MTLSamplerState> sampler = t && t->mtl ? sampler_for(*t) : s_samplers.begin()->second;
            if (s == 0) {
                [s_pass setVertexTexture:texture atIndex:(NSUInteger)l.texture_index[s]];
                [s_pass setVertexSamplerState:sampler atIndex:(NSUInteger)l.texture_index[s]];
            } else {
                [s_pass setFragmentTexture:texture atIndex:(NSUInteger)l.texture_index[s]];
                [s_pass setFragmentSamplerState:sampler atIndex:(NSUInteger)l.texture_index[s]];
            }
        }
    }
    for (int slot = 0; slot < binding.slots; slot++) {
        Buffer *b = find(s_buffers, binding.buffer[slot]);
        if (!b || !b->mtl) return false;
        [s_pass setVertexBuffer:b->mtl offset:binding.base[slot] atIndex:(NSUInteger)(VERTEX_SLOT_BASE + slot)];
    }
    [s_pass setVertexBuffer:s_constant_attrib offset:0 atIndex:CONSTANT_SLOT];
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

void save_capture(id<MTLTexture> copy, uint32_t frame)
{
    int w = (int)copy.width, h = (int)copy.height;
    std::vector<unsigned char> pixels((size_t)w * h * 4);
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
    wb_note("metal: game renders at %dx%d, drawable is %dx%d (scale %d); soft shadows %s; %dx multisampling", width, height,
            width * s_scale, height * s_scale, s_scale, s_shader_options.soft_shadows ? "on" : "off", s_msaa);
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
void mtl_glDeleteProgram(GLuint program) { s_programs.erase(program); }
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
