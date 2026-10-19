// FXAA, MetalFX upscaling (spatial and temporal) and sharpening (see upscale.h).

#include "upscale.h"
#include "shared.h"

#import <MetalFX/MetalFX.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>

namespace wbm __attribute__((visibility("hidden"))) {

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
WBM_HIDDEN id<MTLFXSpatialScaler> s_scaler API_AVAILABLE(macos(13.0));
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
    note_upscaled(scene, s_upscaled);
    s_upscale_frames++;
    return s_upscaled;
}

// Temporal upscaling (WBM_UPSCALE_MODE=temporal). The scene was drawn smaller
// and shifted by a fraction of a pixel that differs from frame to frame (see
// s_jitter), so over a few frames every part of each screen pixel gets looked
// at. MetalFX puts those frames together, which takes knowing where each pixel
// was a frame ago: a texture of movements is made for it here, from the
// camera's own movement and from the things that moved by themselves (the
// motion blur's, see render_object_velocities). What has no such record (grass
// in the wind, particles, water) is taken to stand still in the world.
//
// The history is dropped whenever this frame does not follow from the last:
// no scene in the frame before, a cut to another view, another size.
WBM_HIDDEN id<MTLFXTemporalScaler> s_temporal_scaler API_AVAILABLE(macos(13.0));
id<MTLTexture> s_motion_texture;
id<MTLRenderPipelineState> s_motion_pipeline;
uint64_t s_temporal_frames, s_temporal_resets, s_temporal_last_frame = ~0ull;

id<MTLTexture> apply_temporal_upscale() API_AVAILABLE(macos(13.0))
{
    id<MTLTexture> scene = s_last.color, depth = s_last.depth;
    if (!scene || !depth || depth.sampleCount > 1 || depth.width != scene.width || depth.height != scene.height) return nil;
    NSUInteger w = (NSUInteger)(s_screen_w * s_scale), h = (NSUInteger)(s_screen_h * s_scale);
    if (scene.width > w || scene.height > h) return nil;
    if (!blur_library()) return nil;
    bool reset = s_temporal_last_frame + 1 != s_frame_serial;
    if (!s_temporal_scaler || s_temporal_scaler.inputWidth != scene.width || s_temporal_scaler.inputHeight != scene.height || s_temporal_scaler.outputWidth != w ||
        s_temporal_scaler.outputHeight != h || s_temporal_scaler.colorTextureFormat != scene.pixelFormat || s_temporal_scaler.depthTextureFormat != depth.pixelFormat) {
        MTLFXTemporalScalerDescriptor *d = [MTLFXTemporalScalerDescriptor new];
        d.inputWidth = scene.width, d.inputHeight = scene.height, d.outputWidth = w, d.outputHeight = h;
        d.colorTextureFormat = d.outputTextureFormat = scene.pixelFormat;
        d.depthTextureFormat = depth.pixelFormat;
        d.motionTextureFormat = MTLPixelFormatRG16Float;
        d.autoExposureEnabled = YES; // the scene's colours are as shown, and nothing here knows the exposure the engine will choose
        s_temporal_scaler = [d newTemporalScalerWithDevice:s_device];
        if (!s_temporal_scaler) {
            complain("WBM_UPSCALE_MODE=temporal: MetalFX would not make a scaler from %lux%lu to %lux%lu; the spatial one is used", (unsigned long)scene.width, (unsigned long)scene.height,
                     (unsigned long)w, (unsigned long)h);
            s_temporal = false;
            return nil;
        }
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:scene.pixelFormat width:w height:h mipmapped:NO];
        td.usage = s_temporal_scaler.outputTextureUsage | MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
        td.storageMode = MTLStorageModePrivate;
        s_upscaled = [s_device newTextureWithDescriptor:td];
        td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG16Float width:scene.width height:scene.height mipmapped:NO];
        td.usage = s_temporal_scaler.motionTextureUsage | MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
        td.storageMode = MTLStorageModePrivate;
        s_motion_texture = [s_device newTextureWithDescriptor:td];
        reset = true;
    }
    if (!s_motion_pipeline) {
        MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
        d.vertexFunction = [s_blur_library newFunctionWithName:@"blur_vertex"];
        d.fragmentFunction = [s_blur_library newFunctionWithName:@"motion_fragment"];
        d.colorAttachments[0].pixelFormat = MTLPixelFormatRG16Float;
        NSError *error = nil;
        s_motion_pipeline = [s_device newRenderPipelineStateWithDescriptor:d error:&error];
        if (!s_motion_pipeline) {
            complain("movement texture pipeline failed: %s; the spatial scaler is used", error.localizedDescription.UTF8String);
            s_temporal = false;
            return nil;
        }
    }

    // This frame's clip space to the last one's, as for motion blur; where the two cameras
    // have nothing to do with each other, nothing has moved and the history goes.
    BlurParams params = {};
    double inverse[16], reproject[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    bool known = s_camera.view_proj_frame == s_frame_serial && s_camera.last_view_proj_frame + 1 == s_frame_serial && mat_inverse(s_camera.view_proj, inverse);
    if (known) {
        mat_mul(s_camera.last_view_proj, inverse, reproject);
        double qw = reproject[11] + reproject[15];
        if (qw <= 1e-9 || hypot((reproject[8] + reproject[12]) / qw, (reproject[9] + reproject[13]) / qw) > 0.5) known = false; // a cut
    }
    if (!known) {
        reset = true;
        for (int i = 0; i < 16; i++) reproject[i] = i % 5 == 0 ? 1.0 : 0.0, inverse[i] = reproject[i];
    }
    for (int i = 0; i < 16; i++) params.reproject[i] = (float)reproject[i];
    for (int c = 0; c < 4; c++) params.inverse_w[c] = (float)inverse[c * 4 + 3];
    params.size[0] = (float)scene.width, params.size[1] = (float)scene.height;
    params.reversed = s_reverse_depth ? 1.0f : 0.0f;
    id<MTLTexture> objects = known ? render_object_velocities(reproject, scene.width, scene.height) : nil;
    params.has_objects = objects ? 1.0f : 0.0f;
    bool swayed = known && s_scene_motion && s_scene_motion_frame == s_frame_serial && s_scene_motion.width == scene.width && s_scene_motion.height == scene.height;
    params.has_swayed = swayed ? 1.0f : 0.0f;

    MTLRenderPassDescriptor *d = [MTLRenderPassDescriptor renderPassDescriptor];
    d.colorAttachments[0].texture = s_motion_texture;
    d.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    d.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> pass = [command_buffer() renderCommandEncoderWithDescriptor:d];
    [pass setRenderPipelineState:s_motion_pipeline];
    [pass setFragmentTexture:depth atIndex:1];
    [pass setFragmentTexture:objects ? objects : s_white atIndex:2];
    [pass setFragmentTexture:swayed ? s_scene_motion : s_white atIndex:3];
    [pass setFragmentBytes:&params length:sizeof params atIndex:0];
    [pass drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [pass endEncoding];

    s_temporal_scaler.colorTexture = scene;
    s_temporal_scaler.depthTexture = depth;
    s_temporal_scaler.motionTexture = s_motion_texture;
    s_temporal_scaler.outputTexture = s_upscaled;
    s_temporal_scaler.motionVectorScaleX = (float)scene.width; // the texture holds steps in texture coordinates
    s_temporal_scaler.motionVectorScaleY = (float)scene.height;
    s_temporal_scaler.jitterOffsetX = (float)s_jitter_sign * s_jitter[0];
    s_temporal_scaler.jitterOffsetY = (float)s_jitter_sign * s_jitter[1];
    s_temporal_scaler.depthReversed = s_reverse_depth;
    s_temporal_scaler.reset = reset;
    [s_temporal_scaler encodeToCommandBuffer:command_buffer()];
    static const bool show = getenv("WBM_DEBUG_MOTION") != nullptr;
    if (show) {
        static id<MTLRenderPipelineState> pipeline;
        if (!pipeline) {
            MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
            pd.vertexFunction = [s_blur_library newFunctionWithName:@"blur_vertex"];
            pd.fragmentFunction = [s_blur_library newFunctionWithName:@"motion_show"];
            pd.colorAttachments[0].pixelFormat = s_upscaled.pixelFormat;
            pipeline = [s_device newRenderPipelineStateWithDescriptor:pd error:nil];
        }
        if (pipeline) {
            params.size[0] = (float)w, params.size[1] = (float)h;
            MTLRenderPassDescriptor *sd = [MTLRenderPassDescriptor renderPassDescriptor];
            sd.colorAttachments[0].texture = s_upscaled;
            sd.colorAttachments[0].loadAction = MTLLoadActionDontCare;
            sd.colorAttachments[0].storeAction = MTLStoreActionStore;
            id<MTLRenderCommandEncoder> shown = [command_buffer() renderCommandEncoderWithDescriptor:sd];
            [shown setRenderPipelineState:pipeline];
            [shown setFragmentTexture:s_motion_texture atIndex:0];
            [shown setFragmentBytes:&params length:sizeof params atIndex:0];
            [shown drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
            [shown endEncoding];
        }
    }
    note_upscaled(scene, s_upscaled);
    s_temporal_last_frame = s_frame_serial;
    s_temporal_frames++;
    if (reset) s_temporal_resets++;
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

} // namespace wbm
