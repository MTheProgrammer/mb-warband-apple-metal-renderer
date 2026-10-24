// Frame interpolation (src/backend/frame_interpolation.mm, WBM_FRAMEGEN): MetalFX
// makes a frame between each two the game draws, and both are shown. It works
// on top of the temporal scaler, whose depth and movements it reuses, and needs
// macOS 26 (and its SDK to be built with; without it, these do nothing).
//
// While the game is drawing scenes, the screen is drawn into a texture of the
// shim's own instead of the drawable. What the engine draws onto it after the
// scene (its interface) goes into a second texture as well (colour attachment
// 3), and the screen is copied just before that, so that MetalFX moves the
// scene alone and puts the interface back over it unmoved. At the frame's end
// the frame made between the last one and this is shown first, and this one,
// as the engine drew it, half a frame later. Frames without a scene are shown
// as before.
#pragma once

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <cstdint>

namespace wbm __attribute__((visibility("hidden"))) {

extern int s_framegen; // WBM_FRAMEGEN: 0 off, 1 on, 2 the made frames in place of the real ones too (for looking at them)
extern uint64_t s_framegen_frames, s_framegen_resets;

bool framegen_supported(); // by this machine and this build
bool framegen_in_use();    // this frame's screen is framegen_screen_target's
// The screen for this frame, at the layer's size and format; nil when there is none to be had.
id<MTLTexture> framegen_screen_target(CAMetalLayer *layer);
// Attaches the interface texture, cleared at the frame's first screen pass, to a pass onto the screen.
bool framegen_attach_ui(MTLRenderPassDescriptor *d);
// Called for each draw onto that screen, saying whether it reads the scene: true when it is the one that
// brings the scene there, the first such in the frame (later ones have it bound, left over, and are the
// interface). *copy_due when the screen is to be copied now, before the interface goes over the scene
// (the caller ends the pass, calls framegen_copy_scene, and opens it again).
bool framegen_scene_draw(bool reads_scene, bool *copy_due);
void framegen_copy_scene(id<MTLCommandBuffer> cmd);
// From apply_temporal_upscale: the scene's depth and movements as the scaler had them, which the
// interpolator reads too, and whether the scaler started its history afresh.
void framegen_note_scene(id<MTLTexture> depth, id<MTLTexture> motion, bool reset);
// At present, in place of presenting the drawable: shows the made frame, then this one. Returns
// false when this frame was drawn into the drawable as before, for the caller to present it.
bool framegen_present(id<MTLCommandBuffer> cmd, CAMetalLayer *layer);
// The frame framegen_present just made, or nil when it made none (for captures under WBM_FRAMEGEN=2).
id<MTLTexture> framegen_made();

} // namespace wbm
