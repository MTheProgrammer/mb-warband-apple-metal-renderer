#pragma once

#define GLEW_NO_GLU
#include <glew.h>

#include <atomic>
#include <cstdint>
#include <cstdio>

enum Fn {
#define GL_DIRECT(n) FN_gl##n,
#define GL_GLEW(n) FN_gl##n,
#include "gl_api.def"
    FN_glewInit,
    FN_SDL_CreateWindow,
    FN_SDL_GL_CreateContext,
    FN_SDL_GL_DeleteContext,
    FN_SDL_GL_MakeCurrent,
    FN_SDL_GL_SetAttribute,
    FN_SDL_GL_SetSwapInterval,
    FN_SDL_GL_SwapWindow,
    FN_COUNT
};
enum { FN_GL_COUNT = FN_glewInit };

extern const char *const g_fn_names[FN_COUNT];

// ---- trace.cpp ----------------------------------------------------------

extern std::atomic<bool> g_verbose;       // current frame is logged call by call
extern std::atomic<uint32_t> g_frame;     // frame 0 is everything before the first swap
extern std::atomic<uint64_t> g_frame_draws;
extern bool g_metal;                      // WBM_MODE=metal: calls go to the Metal backend, not OpenGL
extern bool g_fast;                       // Metal without tracing: calls skip the wrappers
extern bool g_native;                     // WBM_MODE=native: time frames only, trace nothing
extern bool g_glerr;                      // WBM_GLERR: glGetError after setup calls
extern const char *g_out_dir;

int wb_thread();                          // small index of the calling thread
void wb_count(int fn);
bool wb_count_foreign(int fn);              // call from another library; true the first time
void wb_log(int fn, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
void wb_note(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void wb_flush();

// Histogram of distinct argument tuples. wb_hist counts the tuple and returns
// true the first time it is seen; the caller then supplies its description.
bool wb_hist(int fn, uint64_t k0, uint64_t k1, uint64_t k2, uint64_t k3);
void wb_hist_text(int fn, uint64_t k0, uint64_t k1, uint64_t k2, uint64_t k3, const char *fmt, ...)
    __attribute__((format(printf, 6, 7)));

void wb_set_context_thread();
void wb_set_context_info(const char *text);
void wb_frame_end();                      // called once per SDL_GL_SwapWindow
void wb_write_summary();

#define HIST(fn, k0, k1, k2, k3, ...)                                              \
    do {                                                                           \
        if (wb_hist(fn, (uint64_t)(k0), (uint64_t)(k1), (uint64_t)(k2), (uint64_t)(k3))) \
            wb_hist_text(fn, (uint64_t)(k0), (uint64_t)(k1), (uint64_t)(k2), (uint64_t)(k3), __VA_ARGS__); \
    } while (0)

#define VLOG(fn, ...)                                                              \
    do {                                                                           \
        if (g_verbose.load(std::memory_order_relaxed)) wb_log(fn, __VA_ARGS__);    \
    } while (0)

// ---- gl_enums.cpp -------------------------------------------------------

const char *E(GLenum e);          // generic enum; unknown values print as hex
const char *E_blend(GLenum e);    // blend factor (0 and 1 are GL_ZERO and GL_ONE)
const char *E_prim(GLenum e);     // primitive mode
const char *E_ifmt(GLint e);      // internal format (1..4 are legacy component counts)
const char *E_clear(GLbitfield m);
uint64_t wb_hash(const void *data, size_t len);

// ---- trace_gl.cpp -------------------------------------------------------

void wb_gl_install_metal();                         // routes all 62 functions to the Metal backend
void wb_gl_context_created();                       // logs GL strings and limits
void wb_gl_dump_state(FILE *f);                     // FBO layouts for the summary
void wb_gl_take_screen_extent(int *w, int *h);
bool wb_gl_read_backbuffer(int w, int h, unsigned char *rgba);

// ---- capture.cpp --------------------------------------------------------

void wb_capture_frame(int w, int h, uint32_t frame); // reads the OpenGL back buffer
// Writes frames/frame_<n>.png from top-down 32-bit pixels, RGBA or BGRA.
// `name` is the file's prefix: "frame" for the screen, anything else for a debugging image.
void wb_save_frame_png(const unsigned char *pixels, int w, int h, uint32_t frame, bool bgra, const char *name = "frame");
