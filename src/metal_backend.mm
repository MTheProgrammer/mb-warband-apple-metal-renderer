// Metal backend: implements the game's 62 OpenGL functions on Metal.
//
// All state here is touched only by the game's render thread, except for the
// window setup, which runs on the main thread before rendering starts.

#include "backend.h"
#include "brf.h"
#include "material_names.h"
#include "shader_xlate.h"
#include "backend/shared.h"
#include "backend/upscale.h"

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
#include <dlfcn.h>
#include <execinfo.h>
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

// The backend's own names; hidden from the game, like everything the library
// does not export (-fvisibility=hidden). The parts in src/backend/*.mm share it.
namespace wbm __attribute__((visibility("hidden"))) {

#include "backend/state.inc"

#include "backend/gl_state.inc"

#include "backend/passes.inc"

#include "backend/textures.inc"

#include "backend/pipelines.inc"

#include "backend/ambient_occlusion.inc"

#include "backend/camera.inc"

#include "backend/motion_blur.inc"

void note_upscaled(id<MTLTexture> scene, id<MTLTexture> upscaled)
{
    for (auto &entry : s_textures)
        if (entry.second.mtl == scene) entry.second.upscaled = upscaled, entry.second.upscaled_frame = s_frame_serial;
}

#include "backend/scene_extras.inc"

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

#include "backend/shadow_casters.inc"

#include "backend/object_velocity.inc"

#include "backend/strands.inc"

#include "backend/shadow_maps.inc"

#include "backend/draw.inc"

#include "backend/sky_room.inc"

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

} // namespace wbm

using namespace wbm;

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
    if (const char *v = getenv("WBM_BOUNCE_HALF")) s_bounce_half = atoi(v) != 0;
    if (const char *v = getenv("WBM_BOUNCE_LIGHT")) s_bounce_light = std::min(4.0f, std::max(0.0f, (float)atof(v)));
    if (const char *v = getenv("WBM_CONTRAST")) s_grade[0] = std::min(1.0f, std::max(-1.0f, (float)atof(v)));
    if (const char *v = getenv("WBM_SATURATION")) s_grade[1] = std::min(1.0f, std::max(-1.0f, (float)atof(v)));
    if (const char *v = getenv("WBM_WARMTH")) s_grade[2] = std::min(1.0f, std::max(-1.0f, (float)atof(v)));
    s_shader_options.grade = s_grade[0] != 0 || s_grade[1] != 0 || s_grade[2] != 0;
    if (const char *v = getenv("WBM_LOD_BIAS")) s_lod_bias = (float)atof(v);
    if (const char *v = getenv("WBM_TRILINEAR")) s_trilinear = atoi(v) != 0;
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
    if (const char *v = getenv("WBM_DYN_FOLIAGE")) s_dyn_foliage_range = std::max(0.0f, (float)atof(v));
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
    if (const char *v = getenv("WBM_FUR_ANISOTROPY")) s_fur_anisotropy = std::min(1.0f, std::max(0.0f, (float)atof(v)));
    if (const char *v = getenv("WBM_FUR_SHEEN")) s_fur_sheen = std::max(0.0f, (float)atof(v));
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
    if (const char *v = getenv("WBM_ROOM_PROBE")) s_room_probe = atoi(v) != 0;
    if (const char *v = getenv("WBM_TORCH_SHADOWS")) s_torch_shadows = std::min(1.0f, std::max(0.0f, (float)atof(v)));
    s_callsites = getenv("WBM_CALLSITES") != nullptr && atoi(getenv("WBM_CALLSITES")) != 0;
    if (const char *v = getenv("WBM_TORCH_SELF")) s_torch_self = std::min(10.0f, std::max(0.0f, (float)atof(v)));
    if (const char *v = getenv("WBM_TORCH_MAP")) s_torch_size = std::min(4096, std::max(128, atoi(v)));
    s_shader_options.torch_shadows = s_torch_shadows > 0;
    if (const char *v = getenv("WBM_TORCH_LIGHTING")) s_shader_options.torch_lighting = s_torch_shadows > 0 && atoi(v) != 0; // off unless set; rides on the shadows' rewrite
    if (const char *v = getenv("WBM_ROOM_PROBE_EVERY")) s_room_every = std::max(1, atoi(v));
    if (const char *v = getenv("WBM_WATER_REFLECTIONS")) s_water_reflections = atoi(v) != 0;
    if (const char *v = getenv("WBM_WATER_STEPS")) s_water_steps = std::min(64, std::max(4, atoi(v)));
    if (const char *v = getenv("WBM_WATER_EVERY")) s_water_every = std::max(1, atoi(v));
    if (const char *v = getenv("WBM_PARALLAX")) s_parallax = std::min(0.3f, std::max(0.0f, (float)atof(v)));
    s_shader_options.parallax = s_parallax > 0;
    if (const char *v = getenv("WBM_WATER")) s_shader_options.water = atoi(v) != 0;
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
            const char *mode = getenv("WBM_UPSCALE_MODE");
            if (mode && !strcmp(mode, "temporal") && share >= 0.3 && share <= 1.0) { // at 1 it only smooths edges
                bool can = false;
                if (@available(macOS 13.0, *)) can = [MTLFXTemporalScalerDescriptor supportsDevice:s_device];
                if (can && s_msaa > 1) complain("WBM_UPSCALE_MODE=temporal does not go with multisampling (WBM_MSAA); the spatial scaler is used"), can = false;
                if (can) {
                    s_upscale = s_temporal = true, s_render_scale = s_scale * std::min(share, 1.0);
                    s_shader_options.motion_vectors = !getenv("WBM_UPSCALE_SWAY") || atoi(getenv("WBM_UPSCALE_SWAY")) != 0; // 0: what sways in the wind does not say how it moved (cheaper; leaves smear)
                    // The reactive mask, for what has no movement the scaler could follow (see upscale.h); it needs the movement target.
                    const char *reactive = getenv("WBM_REACTIVE");
                    if (reactive) s_reactive_cap = std::min(1.0f, std::max(0.0f, (float)atof(reactive)));
                    if (@available(macOS 14.4, *)) s_reactive = s_shader_options.motion_vectors && s_reactive_cap > 0;
                    s_shader_options.reactive_mask = s_reactive;
                    if (const char *v = getenv("WBM_REACTIVE_WATER")) s_shader_options.reactive_water = std::min(1.0f, std::max(0.0f, (float)atof(v)));
                    // Textures are picked as for the full-size picture, or they come out soft.
                    if (!getenv("WBM_LOD_BIAS") && share < 0.999) s_lod_bias = (float)log2(share);
                } else if (s_msaa <= 1) complain("WBM_UPSCALE_MODE=temporal: MetalFX's temporal scaler is not available on this machine");
            }
            if (const char *sign = getenv("WBM_DEBUG_JITTER_SIGN")) s_jitter_sign = atoi(sign) < 0 ? -1 : 1;
        }
        if (const char *v = getenv("WBM_SHARPEN")) s_sharpen = std::min(1.0f, std::max(0.0f, (float)atof(v)));
        // Dithered cut-out edges only where the temporal scaler averages the dither away: anywhere else it
        // stands still and only trades stair steps for dots. Off unless set; with multisampling the samples do it.
        const char *hash = getenv("WBM_ALPHA_HASH");
        s_shader_options.alpha_hash = s_temporal && s_msaa <= 1 && hash && atoi(hash) != 0;
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
    if (s_gpu_buffers && s_frame_serial)
        fprintf(f, "\nGPU time: %.2f ms a frame on average, over %llu frames\n", s_gpu_seconds.load() * 1000.0 / (double)s_frame_serial, (unsigned long long)s_frame_serial);
    fprintf(f, "\n== Metal shaders ==\n%d programs linked, %d from the disk cache, %.0f ms translating\n", s_shader_stats.links,
            s_shader_stats.cache_hits, s_shader_stats.translate_ms);
    fprintf(f, "%d distinct shaders compiled by Metal in the background, %.0f ms spent waiting for them at draws\n",
            s_shader_stats.stage_compiles, s_shader_stats.wait_ms);
    fprintf(f, "\n== Metal state changes ==\n%llu state-setting calls made, %llu skipped as unchanged\n",
            (unsigned long long)s_call_stats.made, (unsigned long long)s_call_stats.skipped);
    fprintf(f, "%llu uniform blocks copied, %llu reused\n", (unsigned long long)s_call_stats.uniform_copies,
            (unsigned long long)s_call_stats.uniform_reuses);
    if (s_blur_frames) fprintf(f, "\nmotion blur applied in %llu frames\n", (unsigned long long)s_blur_frames);
    if (s_velocity_objects) fprintf(f, "moving objects blurred by their own movement: %llu in all\n", (unsigned long long)s_velocity_objects);
    if (s_shell_draws) fprintf(f, "\n%llu layers of fur drawn\n", (unsigned long long)s_shell_draws);
    if (s_water_copies) fprintf(f, "\nscene copied for water to reflect in %llu frames\n", (unsigned long long)s_water_copies);
    if (s_torch_frames)
        fprintf(f, "\nshadows from point lights drawn in %llu frames, %.0f casters per frame on average; %d lights now, in cubes %d texels wide\n", (unsigned long long)s_torch_frames,
                (double)s_torch_casters / (double)s_torch_frames, s_torch_cubes, s_torch_size);
    if (s_room_updates) fprintf(f, "\nroom probe updated in %llu frames\n", (unsigned long long)s_room_updates);
    if (s_bounce_frames) fprintf(f, "\nbounce light applied in %llu frames\n", (unsigned long long)s_bounce_frames);
    if (!s_callsite_counts.empty()) {
        fprintf(f, "\n== call sites (WBM_CALLSITES): times taken, first frame, what, then addresses in the game's executable, innermost first ==\n");
        std::vector<std::pair<std::string, CallSite>> sites(s_callsite_counts.begin(), s_callsite_counts.end());
        std::sort(sites.begin(), sites.end(), [](const auto &a, const auto &b) { return a.second.count > b.second.count; });
        for (const auto &site : sites) fprintf(f, "%10llu %8llu  %s\n", (unsigned long long)site.second.count, (unsigned long long)site.second.first_frame, site.first.c_str());
    }
    if (s_temporal_frames)
        fprintf(f, "\nupscaled by MetalFX's temporal scaler in %llu frames, from %.0f%% of the screen's size each way; history dropped %llu times\n", (unsigned long long)s_temporal_frames,
                100.0 * s_render_scale / s_scale, (unsigned long long)s_temporal_resets);
    if (s_reactive_frames) fprintf(f, "the reactive mask drawn in %llu of them (WBM_REACTIVE)\n", (unsigned long long)s_reactive_frames);
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
    if (s_dyn_stats.far_foliage)
        fprintf(f, "%.0f cut-out casters per frame left to the engine's map, being more than %.0f m off (WBM_DYN_FOLIAGE)\n", (double)s_dyn_stats.far_foliage / (double)s_dyn_stats.frames,
                s_dyn_foliage_range);
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
    if (s_callsites) note_callsite("frame shown");
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
        if (s_temporal) { // the next frame's shift: a Halton sequence in bases 2 and 3, sixteen steps long
            auto halton = [](unsigned i, unsigned base) {
                float f = 1, r = 0;
                for (; i; i /= base) f /= (float)base, r += f * (float)(i % base);
                return r;
            };
            unsigned step = (unsigned)(s_frame_serial % 16) + 1;
            s_jitter[0] = halton(step, 2) - 0.5f, s_jitter[1] = halton(step, 3) - 0.5f;
        }
        if (!s_casters.empty()) { // recorded, but no main pass followed
            s_casters.clear();
            s_caster_bones.clear();
        }
        if (!casting()) s_caster_bones.clear(), s_prepass_meshes.clear();
        s_prepass_solid.clear();
        s_past_objects.clear();
        for (const VelocityDraw &v : s_velocity_draws) {
            PastObject o{v.c.mesh, {}, v.bones, false, {v.origin[0], v.origin[1], v.origin[2], v.origin[3]}};
            memcpy(o.wvp, v.c.wvp, sizeof o.wvp);
            s_past_objects.push_back(o);
        }
        s_past_bones.swap(s_velocity_bones);
        s_velocity_bones.clear();
        s_velocity_draws.clear();
        s_past_frame = s_frame_serial;
        if (s_sky_ambient) update_sky_ambient();
        if (casting()) {
            s_late_ready.swap(s_late_casters);
            s_late_casters.clear();
            s_prepass_meshes.clear();
        }
        s_torch_wanted.swap(s_torch_seen);
        s_torch_seen.clear();
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

#include "backend/replay_api.inc"

#include "backend/gl_calls.inc"

#include "backend/gl_textures.inc"

#include "backend/gl_programs.inc"

#include "backend/gl_draws.inc"
