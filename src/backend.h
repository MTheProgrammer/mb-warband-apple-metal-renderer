#pragma once

// The Metal backend: one mtl_gl* function per OpenGL function the game uses,
// with the same signature, plus the window and frame calls the SDL hooks need.

#include "trace.h"

#include <type_traits>

#define GL_DIRECT(n) extern decltype(gl##n) mtl_gl##n;
#define GL_GLEW(n) extern std::remove_pointer_t<decltype(__glew##n)> mtl_gl##n;
#include "gl_api.def"

// Attaches a CAMetalLayer to the window's content view. nswindow is the
// NSWindow; width and height are the size the game asked for and renders at.
void mtl_attach_window(void *nswindow, int width, int height);
const char *mtl_device_name();
void mtl_set_vsync(bool on);
void mtl_dump_stats(FILE *f); // shader translation and compile timings, for the summary
// Ends the frame and presents it; saves it to frames/ first when capture is set.
void mtl_present(bool capture, uint32_t frame);

// Frame captures (see frame_capture.inc). mtl_request_capture may be called
// from a signal handler: the frame after the current one is written, with all
// the state it depends on, to capture_<frame>.wbcap in the output directory.
void mtl_request_capture();
// Playback, for tools/replay.mm: open a capture to learn the size the game
// rendered at, attach a window of that size, restore, then run the frame as
// often as wanted. Each run ends with mtl_present.
bool mtl_replay_open(const char *path, int *width, int *height);
bool mtl_replay_restore();
void mtl_replay_frame(bool capture, uint32_t frame);
// Moves the camera for the runs that follow: the scene is turned about a point
// `pivot` metres ahead of the captured camera by yaw (about the world's up
// axis) and pitch, in radians, then shifted sideways and up by pan_x and pan_y
// and brought nearer by `forward`, all in metres. Zeros restore the capture's own view.
void mtl_replay_orbit(double yaw, double pitch, double pan_x, double pan_y, double forward, double pivot);
void mtl_replay_show_interface(bool show);
