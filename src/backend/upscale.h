// Anti-aliasing after the fact and upscaling (src/backend/upscale.mm): FXAA,
// MetalFX's spatial and temporal scalers, and sharpening. Each works on the
// last pass's target (s_last) and is called as the frame ends (see end_pass).
#pragma once

#import <Metal/Metal.h>

#include <cstdint>

// API_AVAILABLE gives a symbol default visibility, which would export it from the library.
#define WBM_HIDDEN __attribute__((visibility("hidden")))

namespace wbm __attribute__((visibility("hidden"))) {

void apply_fxaa();                                                  // WBM_FXAA=1, into s_last.color
WBM_HIDDEN id<MTLTexture> apply_upscale() API_AVAILABLE(macos(13.0)); // WBM_UPSCALE: the enlarged scene, or nil
WBM_HIDDEN id<MTLTexture> apply_temporal_upscale() API_AVAILABLE(macos(13.0)); // WBM_UPSCALE_MODE=temporal; nil when it can't
void apply_sharpen(id<MTLTexture> picture);                         // WBM_SHARPEN, in place

// The temporal scaler's reactive mask (WBM_REACTIVE, on with temporal upscaling on macOS 14.4 and
// later): how much of each pixel to take from this frame alone, 0 to 1. The scene's passes draw it
// as colour attachment 2: the shaders write their alpha there (XlOptions::reactive_mask), and the
// draws without movement of their own the scaler could follow (smoke, sparks, fire, water) let it
// through, the largest share winning. Without it these leave trails behind them.
extern bool s_reactive;
extern float s_reactive_cap; // WBM_REACTIVE: the most a pixel is marked (default 0.9, so a little history stays)
// Attaches the mask, cleared at the frame's first pass, to a scene pass drawn into `color`.
bool attach_reactive_mask(MTLRenderPassDescriptor *d, id<MTLTexture> color);

// For the statistics (mtl_dump_stats).
extern uint64_t s_fxaa_frames, s_upscale_frames, s_temporal_frames, s_temporal_resets, s_reactive_frames, s_sharpen_frames;

} // namespace wbm
