// Tracing wrappers for the 7 SDL window and GL-context calls the game makes.

#include "trace.h"

#include "interpose.h"

#include <SDL.h>

#include <cstdlib>

namespace {

const char *attr_name(int attr)
{
    static const char *const names[] = {
        "RED_SIZE",           "GREEN_SIZE",           "BLUE_SIZE",              "ALPHA_SIZE",
        "BUFFER_SIZE",        "DOUBLEBUFFER",         "DEPTH_SIZE",             "STENCIL_SIZE",
        "ACCUM_RED_SIZE",     "ACCUM_GREEN_SIZE",     "ACCUM_BLUE_SIZE",        "ACCUM_ALPHA_SIZE",
        "STEREO",             "MULTISAMPLEBUFFERS",   "MULTISAMPLESAMPLES",     "ACCELERATED_VISUAL",
        "RETAINED_BACKING",   "CONTEXT_MAJOR_VERSION", "CONTEXT_MINOR_VERSION", "CONTEXT_EGL",
        "CONTEXT_FLAGS",      "CONTEXT_PROFILE_MASK", "SHARE_WITH_CURRENT_CONTEXT", "FRAMEBUFFER_SRGB_CAPABLE",
        "CONTEXT_RELEASE_BEHAVIOR", "CONTEXT_RESET_NOTIFICATION", "CONTEXT_NO_ERROR"};
    return attr >= 0 && attr < (int)(sizeof names / sizeof *names) ? names[attr] : "?";
}

SDL_Window *wb_SDL_CreateWindow(const char *title, int x, int y, int w, int h, Uint32 flags)
{
    wb_count(FN_SDL_CreateWindow);
    SDL_Window *win = SDL_CreateWindow(title, x, y, w, h, flags);
    int dw = 0, dh = 0;
    if (win) SDL_GL_GetDrawableSize(win, &dw, &dh);
    wb_log(FN_SDL_CreateWindow, "\"%s\", x=%d, y=%d, %dx%d, flags=0x%X%s%s%s%s%s -> %p (drawable %dx%d)", title ? title : "",
           x, y, w, h, flags, (flags & SDL_WINDOW_OPENGL) ? " OPENGL" : "", (flags & SDL_WINDOW_FULLSCREEN) ? " FULLSCREEN" : "",
           (flags & SDL_WINDOW_FULLSCREEN_DESKTOP) == SDL_WINDOW_FULLSCREEN_DESKTOP ? "_DESKTOP" : "",
           (flags & SDL_WINDOW_ALLOW_HIGHDPI) ? " HIGHDPI" : "", (flags & SDL_WINDOW_BORDERLESS) ? " BORDERLESS" : "",
           (void *)win, dw, dh);
    wb_flush();
    return win;
}

SDL_GLContext wb_SDL_GL_CreateContext(SDL_Window *window)
{
    wb_count(FN_SDL_GL_CreateContext);
    SDL_GLContext ctx = SDL_GL_CreateContext(window);
    wb_log(FN_SDL_GL_CreateContext, "window %p -> %p", (void *)window, ctx);
    if (ctx) {
        wb_set_context_thread();
        wb_gl_context_created();
        // WBM_VSYNC overrides the swap interval. The game sets its own before the
        // context exists, which fails, so this is the only call that takes effect.
        if (const char *v = getenv("WBM_VSYNC")) {
            int r = SDL_GL_SetSwapInterval(atoi(v));
            wb_note("WBM_VSYNC: SDL_GL_SetSwapInterval(%d) -> %d", atoi(v), r);
        }
    }
    wb_flush();
    return ctx;
}

void wb_SDL_GL_DeleteContext(SDL_GLContext context)
{
    wb_count(FN_SDL_GL_DeleteContext);
    wb_log(FN_SDL_GL_DeleteContext, "%p", context);
    wb_write_summary();
    wb_flush();
    SDL_GL_DeleteContext(context);
}

int wb_SDL_GL_MakeCurrent(SDL_Window *window, SDL_GLContext context)
{
    wb_count(FN_SDL_GL_MakeCurrent);
    int r = SDL_GL_MakeCurrent(window, context);
    HIST(FN_SDL_GL_MakeCurrent, wb_thread(), context != nullptr, 0, 0, "on thread t%d, context %s", wb_thread(),
         context ? "set" : "NULL");
    VLOG(FN_SDL_GL_MakeCurrent, "window %p, context %p -> %d", (void *)window, context, r);
    return r;
}

int wb_SDL_GL_SetAttribute(SDL_GLattr attr, int value)
{
    wb_count(FN_SDL_GL_SetAttribute);
    int r = SDL_GL_SetAttribute(attr, value);
    wb_log(FN_SDL_GL_SetAttribute, "SDL_GL_%s = %d -> %d", attr_name((int)attr), value, r);
    return r;
}

int wb_SDL_GL_SetSwapInterval(int interval)
{
    wb_count(FN_SDL_GL_SetSwapInterval);
    if (const char *v = getenv("WBM_VSYNC")) interval = atoi(v);
    int r = SDL_GL_SetSwapInterval(interval);
    wb_log(FN_SDL_GL_SetSwapInterval, "%d -> %d", interval, r);
    return r;
}

// One call per presented frame: the frame counter, verbose frames, captures and
// the periodic summary all hang off it.
void wb_SDL_GL_SwapWindow(SDL_Window *window)
{
    wb_count(FN_SDL_GL_SwapWindow);
    int w = 0, h = 0;
    wb_gl_take_screen_extent(&w, &h);
    if (g_verbose.load(std::memory_order_relaxed)) {
        int dw = 0, dh = 0;
        SDL_GL_GetDrawableSize(window, &dw, &dh);
        wb_log(FN_SDL_GL_SwapWindow, "window %p, drawable %dx%d, game viewports cover %dx%d%s", (void *)window, dw, dh, w,
               h, (w > dw || h > dh) ? " (LARGER THAN THE DRAWABLE: the frame is clipped)" : "");
        // The drawable is the real size of the back buffer; rows beyond it read as garbage.
        wb_capture_frame(dw > 0 ? dw : w, dh > 0 ? dh : h, g_frame.load());
    }
    SDL_GL_SwapWindow(window);
    wb_frame_end();
}

WB_INTERPOSE(wb_SDL_CreateWindow, SDL_CreateWindow);
WB_INTERPOSE(wb_SDL_GL_CreateContext, SDL_GL_CreateContext);
WB_INTERPOSE(wb_SDL_GL_DeleteContext, SDL_GL_DeleteContext);
WB_INTERPOSE(wb_SDL_GL_MakeCurrent, SDL_GL_MakeCurrent);
WB_INTERPOSE(wb_SDL_GL_SetAttribute, SDL_GL_SetAttribute);
WB_INTERPOSE(wb_SDL_GL_SetSwapInterval, SDL_GL_SetSwapInterval);
WB_INTERPOSE(wb_SDL_GL_SwapWindow, SDL_GL_SwapWindow);

} // namespace
