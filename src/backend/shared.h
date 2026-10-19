// What the backend's separately compiled parts (src/backend/*.mm) share with
// metal_backend.mm: the state and helpers they read, declared here and defined
// there. Everything is in namespace wbm, which the library does not export.
#pragma once

#import <Metal/Metal.h>

#include <cstdint>

namespace wbm __attribute__((visibility("hidden"))) {

// The pass most recently rendered into, remembered after it has ended so that
// work can be appended to its target (see apply_ambient_occlusion).
struct LastPass {
    id<MTLTexture> color, ms_color, depth;
    bool scene = false;
    int samples = 1, draws = 0;
    uint64_t occluded_in_frame = 0;
};

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

// The device and the frame.
extern id<MTLDevice> s_device;
extern uint64_t s_frame_serial;
extern int s_scale;                  // the screen's pixels to a point
extern int s_screen_w, s_screen_h;   // the size the game believes the screen has
extern LastPass s_last;
extern Camera s_camera;
extern bool s_reverse_depth;
extern id<MTLTexture> s_white;
id<MTLCommandBuffer> command_buffer();
void complain(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void mat_mul(const double *x, const double *y, double *out);
bool mat_inverse(const double *m, double *out);

// Upscaling's settings (see mtl_attach_window) and what it reads of the scene's movement.
extern bool s_upscale, s_temporal, s_fxaa;
extern float s_sharpen;
extern float s_jitter[2];
extern int s_jitter_sign;
extern id<MTLTexture> s_scene_motion;
extern uint64_t s_scene_motion_frame;
extern id<MTLLibrary> s_blur_library;
// The movement shaders' parameters, as Params in k_blur_source (motion_blur.inc) has them.
struct BlurParams {
    float reproject[16];
    float inverse_w[4];
    float size[2];
    float reversed, exposure, near_start, near_end, max_length, has_objects, has_swayed, spare[3];
};
bool blur_library(); // makes s_blur_library
id<MTLTexture> render_object_velocities(const double *reproject, NSUInteger w, NSUInteger h);
// Remembers that the scene in `scene` was enlarged into `upscaled` this frame, for the pass onto the screen.
void note_upscaled(id<MTLTexture> scene, id<MTLTexture> upscaled);

} // namespace wbm
