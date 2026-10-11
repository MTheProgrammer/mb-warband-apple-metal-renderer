// Metal backend. Milestone 2: window, frame pacing, glClear/glClearColor/
// glViewport on the screen. Everything else is accepted and ignored, with
// answers that keep the engine running (ids, compile and link status, limits).

#include "backend.h"

#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <vector>

namespace {

id<MTLDevice> s_device;
id<MTLCommandQueue> s_queue;
CAMetalLayer *s_layer;
id<MTLTexture> s_depth;

// The frame in progress. All of it is touched only by the game's render thread.
id<CAMetalDrawable> s_drawable;
id<MTLCommandBuffer> s_cmd;
id<MTLRenderCommandEncoder> s_pass;

MTLClearColor s_clear_color = MTLClearColorMake(0, 0, 0, 0);
bool s_depth_mask = true;
struct {
    GLint x, y;
    GLsizei w, h;
} s_viewport;
GLuint s_fbo;         // 0 is the screen; FBO rendering arrives in milestone 6
GLuint s_next_object; // shaders and programs share one id space, as in OpenGL
GLuint s_next_texture, s_next_buffer, s_next_framebuffer, s_next_renderbuffer;
bool s_debug_clear;   // WBM_DEBUG_CLEAR: replace the game's clear colour with a visible one

void gen_ids(GLuint &counter, GLsizei n, GLuint *out)
{
    for (GLsizei i = 0; i < n; i++) out[i] = ++counter;
}

bool begin_frame()
{
    if (s_cmd) return s_drawable != nil;
    if (!s_layer) return false;
    s_drawable = [s_layer nextDrawable];
    if (!s_drawable) return false;
    s_cmd = [s_queue commandBuffer];
    return true;
}

void end_pass()
{
    if (!s_pass) return;
    [s_pass endEncoding];
    s_pass = nil;
}

id<MTLTexture> depth_for(id<MTLTexture> color)
{
    if (!s_depth || s_depth.width != color.width || s_depth.height != color.height) {
        MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                                                     width:color.width
                                                                                    height:color.height
                                                                                 mipmapped:NO];
        d.usage = MTLTextureUsageRenderTarget;
        d.storageMode = MTLStorageModePrivate;
        s_depth = [s_device newTextureWithDescriptor:d];
    }
    return s_depth;
}

// OpenGL viewports are measured from the bottom-left corner, Metal's from the top-left.
void apply_viewport()
{
    if (!s_pass || s_viewport.w <= 0 || s_viewport.h <= 0) return;
    double target_h = (double)s_drawable.texture.height;
    [s_pass setViewport:(MTLViewport){(double)s_viewport.x, target_h - s_viewport.y - s_viewport.h, (double)s_viewport.w,
                                      (double)s_viewport.h, 0.0, 1.0}];
}

void save_capture(id<MTLTexture> copy, uint32_t frame)
{
    int w = (int)copy.width, h = (int)copy.height;
    std::vector<unsigned char> pixels((size_t)w * h * 4);
    [copy getBytes:pixels.data() bytesPerRow:(NSUInteger)w * 4 fromRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0];
    wb_save_frame_png(pixels.data(), w, h, frame, true);
}

} // namespace

// ---- window and frame ------------------------------------------------------

void mtl_attach_window(void *nswindow, int width, int height)
{
    s_debug_clear = getenv("WBM_DEBUG_CLEAR") != nullptr;
    s_device = MTLCreateSystemDefaultDevice();
    s_queue = [s_device newCommandQueue];
    s_viewport = {0, 0, width, height};

    void (^attach)(void) = ^{
        NSWindow *window = (__bridge NSWindow *)nswindow;
        CAMetalLayer *layer = [CAMetalLayer layer];
        layer.device = s_device;
        layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
        layer.framebufferOnly = NO; // frame captures read the drawable back
        layer.opaque = YES;
        // The game renders at the size it asked SDL for; the layer stretches that to the view.
        layer.drawableSize = CGSizeMake(width, height);
        layer.contentsGravity = kCAGravityResize;
        NSView *view = window.contentView;
        view.layer = layer; // set before wantsLayer: the view hosts our layer instead of making its own
        view.wantsLayer = YES;
        s_layer = layer;
    };
    if ([NSThread isMainThread]) attach();
    else dispatch_sync(dispatch_get_main_queue(), attach);
}

const char *mtl_device_name()
{
    return s_device ? s_device.name.UTF8String : "(no device)";
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
        if (!s_cmd) return; // the game drew nothing to the screen this frame
        id<MTLTexture> copy = nil;
        if (capture) {
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
        [s_cmd presentDrawable:s_drawable];
        [s_cmd commit];
        if (copy) {
            [s_cmd waitUntilCompleted];
            save_capture(copy, frame);
        }
        s_cmd = nil;
        s_drawable = nil;
    }
}

// ---- implemented -----------------------------------------------------------

// Starts a new render pass on the screen whose load actions do the clearing.
// Later draws of the frame go into the same pass.
void mtl_glClear(GLbitfield mask)
{
    if (s_fbo != 0) return;
    @autoreleasepool {
        if (!begin_frame()) return;
        end_pass();
        bool color = mask & GL_COLOR_BUFFER_BIT, depth = (mask & GL_DEPTH_BUFFER_BIT) && s_depth_mask;
        MTLRenderPassDescriptor *d = [MTLRenderPassDescriptor renderPassDescriptor];
        d.colorAttachments[0].texture = s_drawable.texture;
        d.colorAttachments[0].loadAction = color ? MTLLoadActionClear : MTLLoadActionLoad;
        d.colorAttachments[0].storeAction = MTLStoreActionStore;
        d.colorAttachments[0].clearColor = s_debug_clear ? MTLClearColorMake(0.10, 0.30, 0.60, 1.0) : s_clear_color;
        d.depthAttachment.texture = depth_for(s_drawable.texture);
        d.depthAttachment.loadAction = depth ? MTLLoadActionClear : MTLLoadActionLoad;
        d.depthAttachment.storeAction = MTLStoreActionStore;
        d.depthAttachment.clearDepth = 1.0;
        s_pass = [s_cmd renderCommandEncoderWithDescriptor:d];
        apply_viewport();
    }
}

void mtl_glClearColor(GLclampf r, GLclampf g, GLclampf b, GLclampf a)
{
    s_clear_color = MTLClearColorMake(r, g, b, a);
}

void mtl_glViewport(GLint x, GLint y, GLsizei width, GLsizei height)
{
    s_viewport = {x, y, width, height};
    if (s_fbo == 0) apply_viewport();
}

void mtl_glDepthMask(GLboolean flag)
{
    s_depth_mask = flag;
}

void mtl_glBindFramebuffer(GLenum target, GLuint framebuffer)
{
    if (target != GL_READ_FRAMEBUFFER) s_fbo = framebuffer;
}

// ---- answers the engine depends on -------------------------------------------

void mtl_glGetIntegerv(GLenum pname, GLint *params)
{
    switch (pname) {
    case GL_MAX_TEXTURE_SIZE: *params = 16384; break;
    default:
        *params = 0;
        wb_note("  !! metal: unanswered glGetIntegerv %s", E(pname));
    }
}

void mtl_glGetFloatv(GLenum pname, GLfloat *params)
{
    switch (pname) {
    case GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT: *params = 16; break;
    default:
        *params = 0;
        wb_note("  !! metal: unanswered glGetFloatv %s", E(pname));
    }
}

GLuint mtl_glCreateShader(GLenum) { return ++s_next_object; }
GLuint mtl_glCreateProgram(void) { return ++s_next_object; }

void mtl_glGetShaderiv(GLuint, GLenum pname, GLint *params)
{
    *params = pname == GL_COMPILE_STATUS ? GL_TRUE : 0;
}

void mtl_glGetProgramiv(GLuint, GLenum pname, GLint *params)
{
    *params = pname == GL_LINK_STATUS ? GL_TRUE : 0;
}

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

GLint mtl_glGetUniformLocation(GLuint, const GLchar *) { return -1; }

void mtl_glGenTextures(GLsizei n, GLuint *textures) { gen_ids(s_next_texture, n, textures); }
void mtl_glGenBuffers(GLsizei n, GLuint *buffers) { gen_ids(s_next_buffer, n, buffers); }
void mtl_glGenFramebuffers(GLsizei n, GLuint *framebuffers) { gen_ids(s_next_framebuffer, n, framebuffers); }
void mtl_glGenRenderbuffers(GLsizei n, GLuint *renderbuffers) { gen_ids(s_next_renderbuffer, n, renderbuffers); }

// ---- not implemented yet: accepted and ignored ---------------------------------

void mtl_glBindTexture(GLenum, GLuint) {}
void mtl_glBlendFunc(GLenum, GLenum) {}
void mtl_glCullFace(GLenum) {}
void mtl_glDeleteTextures(GLsizei, const GLuint *) {}
void mtl_glDepthFunc(GLenum) {}
void mtl_glDisable(GLenum) {}
void mtl_glDrawArrays(GLenum, GLint, GLsizei) {}
void mtl_glDrawElements(GLenum, GLsizei, GLenum, const GLvoid *) {}
void mtl_glEnable(GLenum) {}
void mtl_glPixelStorei(GLenum, GLint) {}
void mtl_glPolygonMode(GLenum, GLenum) {}
void mtl_glTexEnvf(GLenum, GLenum, GLfloat) {}
void mtl_glTexImage2D(GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const GLvoid *) {}
void mtl_glTexParameterf(GLenum, GLenum, GLfloat) {}
void mtl_glTexParameteri(GLenum, GLenum, GLint) {}

void mtl_glActiveTexture(GLenum) {}
void mtl_glAttachShader(GLuint, GLuint) {}
void mtl_glBindAttribLocation(GLuint, GLuint, const GLchar *) {}
void mtl_glBindBuffer(GLenum, GLuint) {}
void mtl_glBindRenderbuffer(GLenum, GLuint) {}
void mtl_glBlendEquation(GLenum) {}
void mtl_glBufferData(GLenum, GLsizeiptr, const void *, GLenum) {}
void mtl_glCompileShader(GLuint) {}
void mtl_glCompressedTexImage2D(GLenum, GLint, GLenum, GLsizei, GLsizei, GLint, GLsizei, const void *) {}
void mtl_glDeleteBuffers(GLsizei, const GLuint *) {}
void mtl_glDeleteFramebuffers(GLsizei, const GLuint *) {}
void mtl_glDeleteProgram(GLuint) {}
void mtl_glDeleteRenderbuffers(GLsizei, const GLuint *) {}
void mtl_glDeleteShader(GLuint) {}
void mtl_glDisableVertexAttribArray(GLuint) {}
void mtl_glEnableVertexAttribArray(GLuint) {}
void mtl_glFramebufferRenderbuffer(GLenum, GLenum, GLenum, GLuint) {}
void mtl_glFramebufferTexture2D(GLenum, GLenum, GLenum, GLuint, GLint) {}
void mtl_glGenerateMipmap(GLenum) {}
void mtl_glLinkProgram(GLuint) {}
void mtl_glRenderbufferStorage(GLenum, GLenum, GLsizei, GLsizei) {}
void mtl_glShaderSource(GLuint, GLsizei, const GLchar *const *, const GLint *) {}
void mtl_glUniform1f(GLint, GLfloat) {}
void mtl_glUniform1i(GLint, GLint) {}
void mtl_glUniform1iv(GLint, GLsizei, const GLint *) {}
void mtl_glUniform4fv(GLint, GLsizei, const GLfloat *) {}
void mtl_glUniformMatrix4fv(GLint, GLsizei, GLboolean, const GLfloat *) {}
void mtl_glUseProgram(GLuint) {}
void mtl_glVertexAttribPointer(GLuint, GLint, GLenum, GLboolean, GLsizei, const void *) {}
