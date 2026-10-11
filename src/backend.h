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
