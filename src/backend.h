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

// The model previewer (model_preview.inc), for the frame player. After
// mtl_replay_restore, and between frames: shows the named mesh of the game's
// resource files in the captured frame in place of its scene (or with it, for
// keep_scene), `lod` 0 being the full mesh. An empty name goes back to the
// frame as captured. *distance receives how far ahead of the camera the model
// is, for the camera to turn about.
bool mtl_preview_model(const char *name, int lod, bool keep_scene, double *distance);
// An action for a skinned model to play: for a horse stand, walk, trot, canter,
// gallop, rear, jump or turn; or "animation:from:to:seconds" for any animation
// of the game's files; empty for the rest pose. held: a fixed moment of it, 0
// to 1, or below 0 to let it run. False if there is no such action.
bool mtl_preview_gallery(const char *path, bool keep_scene); // several models at once, placed as a file says (see preview_load_gallery)
bool mtl_preview_action(const char *action, double held);
const char *mtl_preview_next_action(const char *action); // the horse's actions in turn, ending with "" for the rest pose
void mtl_preview_material(const char *name); // before mtl_preview_model: a material to use in place of the mesh's own; empty for the mesh's own
const char *mtl_preview_resolve(const char *name); // a mesh name, or a resource file's name for its first mesh; null if there is none
int mtl_preview_list(const char *pattern, FILE *out); // prints mesh names containing the pattern
const char *mtl_preview_neighbour(const char *name, int by);
