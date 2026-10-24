// MetalFX frame interpolation (see frame_interpolation.h).

#include "frame_interpolation.h"
#include "shared.h"

#import <MetalFX/MetalFX.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>

namespace wbm __attribute__((visibility("hidden"))) {

int s_framegen;
uint64_t s_framegen_frames, s_framegen_resets, s_framegen_alone;
std::atomic<int> s_display_hz{60};

void framegen_set_display_rate(int hz)
{
    if (hz > 0) s_display_hz.store(hz);
}

#if defined(__MAC_26_0) && __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0

// API_AVAILABLE gives a symbol default visibility, which would export it from the library.
__attribute__((visibility("hidden"))) id<MTLFXFrameInterpolator> s_interpolator API_AVAILABLE(macos(26.0));

// The screen as the engine draws it; the scene alone, as it was before the interface went over it, for
// this frame and the last by turns (what the interpolator moves); the interface alone; the frame made.
id<MTLTexture> s_screen, s_scenes[2], s_ui, s_made;
int s_current;
// The frames in which: the screen was drawn into; the interface texture was cleared; the scene was brought
// onto the screen; it was copied before the interface; the scene copy held a picture to interpolate from.
uint64_t s_target_frame = ~0ull, s_ui_frame = ~0ull, s_brought_frame = ~0ull, s_copied_frame = ~0ull, s_shown_frame = ~0ull;
bool s_made_now;      // framegen_present made a frame this time
bool s_remade = true; // the textures or the interpolator are new: no history

// What apply_temporal_upscale handed over this frame.
id<MTLTexture> s_scene_depth, s_scene_motion_vectors;
uint64_t s_scene_noted_frame = ~0ull;
bool s_scene_reset;

// Pacing. The time between the game's frames, smoothed (what the interpolator is told); and what a frame
// costs the game, smoothed: the time the render thread spends away from presenting or the GPU's time for
// it, whichever is longer. The cost leaves out the waits for a drawable, which the pacing itself causes;
// measured with them, it would hold the game at whatever rate it first fell to.
double s_interval = 1.0 / 60.0, s_cost = 1.0 / 60.0;
std::chrono::steady_clock::time_point s_last_present, s_presented;
uint64_t s_presented_frame = ~0ull, s_encoded_frame = ~0ull;
double s_gpu_before = -1;
bool s_making; // the game is slower than the display, so frames are made (see framegen_present)

bool framegen_supported()
{
    if (@available(macOS 26.0, *)) return [MTLFXFrameInterpolatorDescriptor supportsDevice:s_device];
    return false;
}

bool framegen_in_use()
{
    return s_framegen && s_target_frame == s_frame_serial;
}

id<MTLTexture> new_target(MTLPixelFormat format, NSUInteger w, NSUInteger h, MTLTextureUsage usage)
{
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h mipmapped:NO];
    td.usage = usage | MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModePrivate;
    return [s_device newTextureWithDescriptor:td];
}

id<MTLTexture> framegen_screen_target(CAMetalLayer *layer)
{
    if (!s_framegen || !layer) return nil;
    NSUInteger w = (NSUInteger)layer.drawableSize.width, h = (NSUInteger)layer.drawableSize.height;
    MTLPixelFormat format = layer.pixelFormat;
    if (!w || !h) return nil;
    if (!s_screen || s_screen.width != w || s_screen.height != h || s_screen.pixelFormat != format) {
        // The interpolator's own usage is only known once it is made (see ready_interpolator).
        s_screen = new_target(format, w, h, 0);
        for (id<MTLTexture> __strong &scene : s_scenes) scene = new_target(format, w, h, 0);
        s_ui = new_target(format, w, h, 0);
        s_made = new_target(format, w, h, MTLTextureUsageShaderWrite);
        s_ui_frame = s_copied_frame = ~0ull;
        s_remade = true;
        if (!s_screen || !s_scenes[0] || !s_scenes[1] || !s_ui || !s_made) {
            complain("WBM_FRAMEGEN: could not make the screen's textures at %lux%lu; frame interpolation is off", (unsigned long)w, (unsigned long)h);
            s_framegen = 0;
            return nil;
        }
    }
    s_target_frame = s_frame_serial;
    return s_screen;
}

bool framegen_attach_ui(MTLRenderPassDescriptor *d)
{
    if (!framegen_in_use() || !s_ui) return false;
    d.colorAttachments[3].texture = s_ui;
    d.colorAttachments[3].loadAction = s_ui_frame == s_frame_serial ? MTLLoadActionLoad : MTLLoadActionClear;
    d.colorAttachments[3].storeAction = MTLStoreActionStore;
    d.colorAttachments[3].clearColor = MTLClearColorMake(0, 0, 0, 0);
    s_ui_frame = s_frame_serial;
    return true;
}

void copy(id<MTLCommandBuffer> cmd, id<MTLTexture> from, id<MTLTexture> to)
{
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit copyFromTexture:from sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(std::min(from.width, to.width), std::min(from.height, to.height), 1)
                toTexture:to destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];
}

bool framegen_scene_draw(bool reads_scene, bool *copy_due)
{
    *copy_due = false;
    if (!framegen_in_use()) return false;
    if (reads_scene && s_brought_frame != s_frame_serial) {
        s_brought_frame = s_frame_serial;
        return true;
    }
    *copy_due = s_brought_frame == s_frame_serial && s_copied_frame != s_frame_serial;
    return false;
}

void framegen_copy_scene(id<MTLCommandBuffer> cmd)
{
    copy(cmd, s_screen, s_scenes[s_current]);
    s_copied_frame = s_frame_serial;
}

// The scene's depth and movements are stored upside down, as the engine's render targets are (rows in
// OpenGL's order, see wbm_yflip), and the screen the right way up. The interpolator takes them all the
// same way up, so it is handed copies of the two turned over, made as the scaler hands them on (which
// also keeps the depth as it was, whatever the engine draws into its buffer afterwards). A movement
// turned over goes the other way up and down.
const char k_turn_source[] = R"(
#include <metal_stdlib>
using namespace metal;
struct VertexOut { float4 position [[position]]; };
vertex VertexOut turn_vertex(uint id [[vertex_id]])
{
    VertexOut out;
    out.position = float4(float2((id << 1) & 2, id & 2) * 2.0 - 1.0, 0.0, 1.0);
    return out;
}
struct DepthOut { float depth [[depth(any)]]; };
fragment DepthOut turn_depth(VertexOut in [[stage_in]], depth2d<float> depth [[texture(0)]])
{
    uint2 at = uint2(in.position.xy);
    DepthOut out;
    out.depth = depth.read(uint2(at.x, depth.get_height() - 1 - at.y));
    return out;
}
fragment float4 turn_motion(VertexOut in [[stage_in]], texture2d<float> motion [[texture(0)]])
{
    uint2 at = uint2(in.position.xy);
    float2 m = motion.read(uint2(at.x, motion.get_height() - 1 - at.y)).xy;
    return float4(m.x, -m.y, 0.0, 0.0);
}
)";
id<MTLRenderPipelineState> s_turn_depth, s_turn_motion;
MTLPixelFormat s_turn_depth_format, s_turn_motion_format;
id<MTLDepthStencilState> s_turn_write;
id<MTLTexture> s_upright_depth, s_upright_motion;

bool turn_pipelines(MTLPixelFormat depth_format, MTLPixelFormat motion_format)
{
    if (s_turn_depth && s_turn_motion && s_turn_depth_format == depth_format && s_turn_motion_format == motion_format) return true;
    NSError *error = nil;
    id<MTLLibrary> lib = [s_device newLibraryWithSource:@(k_turn_source) options:nil error:&error];
    if (!lib) {
        complain("WBM_FRAMEGEN: shaders did not compile: %s; frame interpolation is off", error.localizedDescription.UTF8String);
        return false;
    }
    MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = [lib newFunctionWithName:@"turn_vertex"];
    d.fragmentFunction = [lib newFunctionWithName:@"turn_depth"];
    d.depthAttachmentPixelFormat = depth_format;
    s_turn_depth = [s_device newRenderPipelineStateWithDescriptor:d error:&error];
    d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = [lib newFunctionWithName:@"turn_vertex"];
    d.fragmentFunction = [lib newFunctionWithName:@"turn_motion"];
    d.colorAttachments[0].pixelFormat = motion_format;
    if (s_turn_depth) s_turn_motion = [s_device newRenderPipelineStateWithDescriptor:d error:&error];
    MTLDepthStencilDescriptor *dd = [MTLDepthStencilDescriptor new];
    dd.depthCompareFunction = MTLCompareFunctionAlways;
    dd.depthWriteEnabled = YES;
    s_turn_write = [s_device newDepthStencilStateWithDescriptor:dd];
    if (!s_turn_depth || !s_turn_motion) {
        complain("WBM_FRAMEGEN: pipelines failed: %s; frame interpolation is off", error.localizedDescription.UTF8String);
        return false;
    }
    s_turn_depth_format = depth_format, s_turn_motion_format = motion_format;
    return true;
}

// Into `to`, the right size and format for `from`, made as needed.
void turn_over(id<MTLTexture> from, id<MTLTexture> __strong &to, bool depth)
{
    if (!to || to.width != from.width || to.height != from.height || to.pixelFormat != from.pixelFormat) {
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:from.pixelFormat width:from.width height:from.height mipmapped:NO];
        td.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
        td.storageMode = MTLStorageModePrivate;
        to = [s_device newTextureWithDescriptor:td];
    }
    MTLRenderPassDescriptor *d = [MTLRenderPassDescriptor renderPassDescriptor];
    if (depth) {
        d.depthAttachment.texture = to;
        d.depthAttachment.loadAction = MTLLoadActionDontCare;
        d.depthAttachment.storeAction = MTLStoreActionStore;
    } else {
        d.colorAttachments[0].texture = to;
        d.colorAttachments[0].loadAction = MTLLoadActionDontCare;
        d.colorAttachments[0].storeAction = MTLStoreActionStore;
    }
    id<MTLRenderCommandEncoder> pass = [command_buffer() renderCommandEncoderWithDescriptor:d];
    [pass setRenderPipelineState:depth ? s_turn_depth : s_turn_motion];
    if (depth) [pass setDepthStencilState:s_turn_write];
    [pass setFragmentTexture:from atIndex:0];
    [pass drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [pass endEncoding];
}

void framegen_note_scene(id<MTLTexture> depth, id<MTLTexture> motion, bool reset)
{
    if (!s_framegen || !depth || !motion || !turn_pipelines(depth.pixelFormat, motion.pixelFormat)) {
        if (s_framegen && depth && motion) s_framegen = 0;
        return;
    }
    turn_over(depth, s_upright_depth, true);
    turn_over(motion, s_upright_motion, false);
    s_scene_depth = s_upright_depth;
    s_scene_motion_vectors = s_upright_motion;
    s_scene_reset = reset;
    s_scene_noted_frame = s_frame_serial;
}

// Made, or made again, for this frame's sizes and formats. False when there is none to use this frame.
bool ready_interpolator() API_AVAILABLE(macos(26.0))
{
    id<MTLTexture> depth = s_scene_depth, motion = s_scene_motion_vectors, screen = s_screen;
    if (s_interpolator && s_interpolator.inputWidth == depth.width && s_interpolator.inputHeight == depth.height &&
        s_interpolator.outputWidth == screen.width && s_interpolator.outputHeight == screen.height && s_interpolator.colorTextureFormat == screen.pixelFormat &&
        s_interpolator.depthTextureFormat == depth.pixelFormat && s_interpolator.motionTextureFormat == motion.pixelFormat)
        return true;
    MTLFXFrameInterpolatorDescriptor *d = [MTLFXFrameInterpolatorDescriptor new];
    d.colorTextureFormat = d.outputTextureFormat = d.uiTextureFormat = screen.pixelFormat;
    d.depthTextureFormat = depth.pixelFormat;
    d.motionTextureFormat = motion.pixelFormat;
    d.inputWidth = depth.width, d.inputHeight = depth.height;
    d.outputWidth = screen.width, d.outputHeight = screen.height;
    // Not given the scaler (d.scaler): the depth and movements are handed over as they are, and in replays the
    // frames made came out the same either way.
    s_interpolator = [d newFrameInterpolatorWithDevice:s_device];
    s_remade = true;
    if (!s_interpolator) {
        complain("WBM_FRAMEGEN: MetalFX would not make an interpolator from %lux%lu to %lux%lu; frame interpolation is off", (unsigned long)depth.width,
                 (unsigned long)depth.height, (unsigned long)screen.width, (unsigned long)screen.height);
        s_framegen = 0;
        return false;
    }
    // Textures made before the interpolator was: remade, empty, if they lack a use it asks for.
    MTLTextureUsage color = s_interpolator.colorTextureUsage, ui = s_interpolator.uiTextureUsage, out = s_interpolator.outputTextureUsage;
    if ((s_scenes[0].usage & color) != color || (s_ui.usage & ui) != ui || (s_made.usage & out) != out) {
        for (id<MTLTexture> __strong &scene : s_scenes) scene = new_target(scene.pixelFormat, scene.width, scene.height, color);
        s_ui = new_target(s_ui.pixelFormat, s_ui.width, s_ui.height, ui);
        s_made = new_target(s_made.pixelFormat, s_made.width, s_made.height, out | MTLTextureUsageShaderWrite);
        s_ui_frame = s_copied_frame = ~0ull;
        return false;
    }
    return true;
}

bool framegen_present(id<MTLCommandBuffer> cmd, CAMetalLayer *layer, double gpu_seconds)
{
    s_made_now = false;
    if (!framegen_in_use()) {
        s_shown_frame = ~0ull;
        return false;
    }
    auto now = std::chrono::steady_clock::now();
    double since = std::chrono::duration<double>(now - s_last_present).count();
    s_last_present = now;
    if (since > 0 && since < 0.25) s_interval += (since - s_interval) * 0.1;
    double gpu = s_gpu_before >= 0 ? gpu_seconds - s_gpu_before : 0; // finished a frame or two late, which the smoothing evens out
    s_gpu_before = gpu_seconds;
    if (s_presented_frame + 1 == s_frame_serial) {
        double work = std::chrono::duration<double>(now - s_presented).count(), cost = std::max(work, gpu);
        if (cost > 0 && cost < 0.25) s_cost += (cost - s_cost) * 0.1;
    }
    // Frames are made only while the game cannot keep up with the display by itself: when it can, two
    // shown for each of its own would only halve its own rate. (With some room between the two thresholds,
    // so that it does not go back and forth.) WBM_FRAMEGEN=2 always makes them, for looking at.
    double refresh = 1.0 / std::max(1, s_display_hz.load());
    s_making = s_framegen == 2 || (s_making ? s_cost > refresh * 1.2 : s_cost > refresh * 1.5);

    bool made = false, scene = s_scene_noted_frame == s_frame_serial && s_scene_depth && s_scene_motion_vectors;
    if (@available(macOS 26.0, *)) {
        if (scene && !s_making) s_framegen_alone++;
        if (scene && s_making && ready_interpolator()) {
            if (s_copied_frame != s_frame_serial) framegen_copy_scene(cmd); // no interface came after the scene
            if (s_ui_frame != s_frame_serial) { // nothing was drawn onto the screen that could have cleared it
                MTLRenderPassDescriptor *d = [MTLRenderPassDescriptor renderPassDescriptor];
                d.colorAttachments[0].texture = s_ui;
                d.colorAttachments[0].loadAction = MTLLoadActionClear;
                d.colorAttachments[0].storeAction = MTLStoreActionStore;
                d.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
                [[cmd renderCommandEncoderWithDescriptor:d] endEncoding];
                s_ui_frame = s_frame_serial;
            }
            bool reset = s_remade || s_scene_reset || s_shown_frame + 1 != s_frame_serial || s_encoded_frame + 1 != s_frame_serial;
            s_interpolator.colorTexture = s_scenes[s_current];
            s_interpolator.prevColorTexture = s_scenes[1 - s_current];
            s_interpolator.depthTexture = s_scene_depth;
            s_interpolator.motionTexture = s_scene_motion_vectors;
            // The interface goes over the made frame as it is over this one. (Handing MetalFX the screen with it
            // drawn in, uiTextureComposited, came out black but for the interface.)
            s_interpolator.uiTexture = s_ui;
            s_interpolator.uiTextureComposited = NO;
            s_interpolator.outputTexture = s_made;
            // Steps in texture coordinates of the scene, as for the scaler (see apply_temporal_upscale). The colour
            // is the scaler's output, from which the shift of a fraction of a pixel has been taken out.
            s_interpolator.motionVectorScaleX = (float)s_scene_depth.width;
            s_interpolator.motionVectorScaleY = (float)s_scene_depth.height;
            s_interpolator.jitterOffsetX = s_interpolator.jitterOffsetY = 0;
            s_interpolator.depthReversed = s_reverse_depth;
            // The camera: clip z = a w + b, as OpenGL has it (-w at the near plane, w at the far one).
            const Camera &c = s_camera;
            if (fabs(c.p11) > 1e-6 && fabs(c.p00) > 1e-6) {
                s_interpolator.fieldOfView = (float)(2.0 * atan(1.0 / fabs(c.p11)) * 180.0 / M_PI);
                s_interpolator.aspectRatio = (float)fabs(c.p11 / c.p00);
            }
            double near = fabs(1.0 + c.a) > 1e-9 ? fabs(c.b / (1.0 + c.a)) : 0.1, far = fabs(1.0 - c.a) > 1e-9 ? fabs(c.b / (1.0 - c.a)) : 1250.0;
            s_interpolator.nearPlane = (float)std::max(near, 1e-3);
            s_interpolator.farPlane = (float)std::max(far, near + 1.0);
            s_interpolator.deltaTime = (float)s_interval;
            s_interpolator.shouldResetHistory = reset;
            [s_interpolator encodeToCommandBuffer:cmd]; // on a reset too, which starts its history; there is no frame before to show then
            s_encoded_frame = s_frame_serial;
            made = !reset;
            if (made) s_framegen_frames++;
            else s_framegen_resets++;
            s_remade = false;
        }
    }

    // Each drawable is taken only when it is needed, so that the two shown never hold up the game more than
    // one would. Shown evenly: each half a frame after the one before, a frame being what the game takes,
    // and at least two of the display's.
    double half = std::max(s_cost, refresh * 2.0) * 0.5;
    if (made) {
        id<CAMetalDrawable> between = [layer nextDrawable];
        if (between) {
            copy(cmd, s_made, between.texture);
            [cmd presentDrawable:between afterMinimumDuration:half];
        }
    }
    id<CAMetalDrawable> drawable = [layer nextDrawable];
    if (drawable) {
        copy(cmd, made && s_framegen == 2 ? s_made : s_screen, drawable.texture);
        if (made) [cmd presentDrawable:drawable afterMinimumDuration:half];
        else [cmd presentDrawable:drawable];
    }
    s_presented = std::chrono::steady_clock::now();
    s_presented_frame = s_frame_serial;
    s_made_now = made;
    s_shown_frame = scene && s_copied_frame == s_frame_serial ? s_frame_serial : ~0ull;
    s_current = 1 - s_current;
    return true;
}

id<MTLTexture> framegen_made()
{
    return s_made_now ? s_made : nil;
}

#else // built with an SDK before macOS 26's

bool framegen_supported() { return false; }
bool framegen_in_use() { return false; }
id<MTLTexture> framegen_screen_target(CAMetalLayer *) { return nil; }
bool framegen_attach_ui(MTLRenderPassDescriptor *) { return false; }
bool framegen_scene_draw(bool, bool *copy_due)
{
    *copy_due = false;
    return false;
}
void framegen_copy_scene(id<MTLCommandBuffer>) {}
void framegen_note_scene(id<MTLTexture>, id<MTLTexture>, bool) {}
bool framegen_present(id<MTLCommandBuffer>, CAMetalLayer *, double) { return false; }
id<MTLTexture> framegen_made() { return nil; }

#endif

} // namespace wbm
