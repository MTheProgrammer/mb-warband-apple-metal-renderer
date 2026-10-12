// Plays back a frame captured from the game (see src/frame_capture.inc)
// through the Metal renderer, without the game:
//
//   build/replay capture.wbcap                 a window showing the frame, redrawn continuously
//   build/replay capture.wbcap --png out.png   draws it a few times, saves the last, and exits
//
// The WBM_* variables select the renderer's options exactly as they do in the
// game, so the same frame can be looked at with any combination of them.

#include "backend.h"

#import <AppKit/AppKit.h>

#include <cstdarg>
#include <string>
#include <sys/stat.h>
#include <thread>

// What the renderer expects from the tracing half of the library.
std::atomic<bool> g_verbose{false};
std::atomic<uint32_t> g_frame{0};
std::atomic<uint64_t> g_frame_draws{0};
bool g_metal = true, g_fast = true, g_native = false, g_glerr = false;
const char *g_out_dir = ".";
static bool s_quiet;

int wb_thread() { return 0; }
void wb_count(int) {}
void wb_log(int, const char *, ...) {}
void wb_note(const char *fmt, ...)
{
    if (s_quiet && !strstr(fmt, "!!") && !strstr(fmt, "replay")) return;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}
void wb_flush() {}
bool wb_gl_read_backbuffer(int, int, unsigned char *) { return false; }
void wb_gl_take_screen_extent(int *w, int *h) { *w = *h = 0; }

int main(int argc, char **argv)
{
    const char *path = nullptr, *png = nullptr;
    int frames = 8;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--png") && i + 1 < argc) png = argv[++i];
        else if (!strcmp(argv[i], "--frames") && i + 1 < argc) frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--quiet")) s_quiet = true;
        else path = argv[i];
    }
    int width = 0, height = 0;
    if (!path || !mtl_replay_open(path, &width, &height)) {
        fprintf(stderr, "usage: replay capture.wbcap [--png out.png] [--frames n] [--quiet]\n%s\n", path ? "that file is not a frame capture" : "");
        return 2;
    }
    static std::string out_dir = png ? std::string(png) + ".d" : std::string(".");
    if (png) mkdir(out_dir.c_str(), 0755), mkdir((out_dir + "/frames").c_str(), 0755);
    g_out_dir = out_dir.c_str();
    setenv("WBM_OUT", g_out_dir, 0);

    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:png ? NSApplicationActivationPolicyAccessory : NSApplicationActivationPolicyRegular];
        NSWindow *window = [[NSWindow alloc] initWithContentRect:NSMakeRect(80, 80, width, height)
                                                       styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable
                                                         backing:NSBackingStoreBuffered
                                                           defer:NO];
        window.title = [NSString stringWithFormat:@"%s (%dx%d)", path, width, height];
        mtl_attach_window((__bridge void *)window, width, height);
        if (!png) {
            [window makeKeyAndOrderFront:nil];
            [NSApp activateIgnoringOtherApps:YES];
        }
        static std::atomic<bool> closed{false};
        [[NSNotificationCenter defaultCenter] addObserverForName:NSWindowWillCloseNotification object:window queue:nil usingBlock:^(NSNotification *) {
          closed = true;
        }];
        std::thread([=] {
            if (!mtl_replay_restore()) {
                fprintf(stderr, "the capture could not be restored\n");
                exit(1);
            }
            // Some of what a frame shows comes from the frame before it (the sky's
            // colours, grass in the shadow map), so a still needs a few runs.
            for (uint32_t n = 1; png ? n <= (uint32_t)frames : !closed.load(); n++) {
                g_frame = n;
                mtl_replay_frame(png && n == (uint32_t)frames, n);
            }
            if (png) {
                char saved[1200];
                snprintf(saved, sizeof saved, "%s/frames/frame_%06u.png", g_out_dir, (unsigned)frames);
                usleep(300000); // the frame is written from a completion handler
                if (rename(saved, png)) fprintf(stderr, "no image was written (%s)\n", saved);
                else rmdir((out_dir + "/frames").c_str()), rmdir(out_dir.c_str());
            }
            exit(0);
        }).detach();
        [NSApp run];
    }
    return 0;
}
