// Plays back a frame captured from the game (see src/frame_capture.inc)
// through the Metal renderer, without the game:
//
//   build/replay capture.wbcap                 a window showing the frame, redrawn continuously
//   build/replay capture.wbcap --png out.png   draws it a few times, saves the last, and exits
//
// In the window the camera can be moved about the scene:
//   drag                 turn the scene about a point ahead of the camera
//   shift-drag, or drag with the right button   slide sideways and up
//   scroll               move nearer and farther
//   [ and ]              bring the point the scene turns about nearer and farther
//   u                    hide and show the game's interface
//   r                    back to the view as captured
//   n and p              with --model: the next and previous mesh of the same resource file
//   l                    with --model: the mesh's simpler versions in turn
// The same from the command line, for stills: --yaw and --pitch (degrees),
// --pan x y, --forward and --pivot (metres), --no-ui.
//
// Only what the game drew for its own view is there to look at: turn far
// enough and the scene ends where the game's camera stopped seeing it.
//
//   build/replay capture.wbcap --model sword_medieval_a
//                                              one of the game's models in the frame's light,
//                                              in place of what the game drew (--keep-scene: with it)
//   build/replay --list helmet                 the names of meshes containing a word
//
// The WBM_* variables select the renderer's options exactly as they do in the
// game, so the same frame can be looked at with any combination of them.

#include "backend.h"

#import <AppKit/AppKit.h>

#include <cmath>
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
static bool s_quiet, s_stats;

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

// The camera, set from the main thread's events and read by the thread that draws.
struct View {
    std::atomic<double> yaw{0}, pitch{0}, pan_x{0}, pan_y{0}, forward{0}, pivot{5};
    std::atomic<bool> interface{true};
    std::atomic<int> step{0}, lod_step{0}; // asked for with n, p and l
};
static View s_view;

int main(int argc, char **argv)
{
    const char *path = nullptr, *png = nullptr, *list = nullptr;
    static std::string model;
    static int lod = 0;
    static bool keep_scene = false;
    bool pivot_given = false;
    int frames = 8;
    const double degrees = M_PI / 180.0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--png") && i + 1 < argc) png = argv[++i];
        else if (!strcmp(argv[i], "--yaw") && i + 1 < argc) s_view.yaw = atof(argv[++i]) * degrees;
        else if (!strcmp(argv[i], "--pitch") && i + 1 < argc) s_view.pitch = atof(argv[++i]) * degrees;
        else if (!strcmp(argv[i], "--pan") && i + 2 < argc) s_view.pan_x = atof(argv[i + 1]), s_view.pan_y = atof(argv[i + 2]), i += 2;
        else if (!strcmp(argv[i], "--forward") && i + 1 < argc) s_view.forward = atof(argv[++i]);
        else if (!strcmp(argv[i], "--pivot") && i + 1 < argc) s_view.pivot = atof(argv[++i]), pivot_given = true;
        else if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--lod") && i + 1 < argc) lod = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--material") && i + 1 < argc) mtl_preview_material(argv[++i]);
        else if (!strcmp(argv[i], "--keep-scene")) keep_scene = true;
        else if (!strcmp(argv[i], "--list") && i + 1 < argc) list = argv[++i];
        else if (!strcmp(argv[i], "--no-ui")) s_view.interface = false;
        else if (!strcmp(argv[i], "--frames") && i + 1 < argc) frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--quiet")) s_quiet = true;
        else if (!strcmp(argv[i], "--stats")) s_stats = true;
        else path = argv[i];
    }
    if (list) {
        int found = mtl_preview_list(list, stdout);
        fprintf(stderr, "%d meshes\n", found);
        return 0;
    }
    if (!model.empty()) s_view.interface = false;
    int width = 0, height = 0;
    if (!path || !mtl_replay_open(path, &width, &height)) {
        fprintf(stderr, "usage: replay capture.wbcap [--png out.png] [--frames n] [--quiet] [--stats]\n"
                        "              [--yaw deg] [--pitch deg] [--pan x y] [--forward m] [--pivot m] [--no-ui]\n"
                        "              [--model mesh [--material name] [--lod n] [--keep-scene]]\n       replay --list word\n%s\n",
                path ? "that file is not a frame capture" : "");
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
        NSEventMask mask = NSEventMaskLeftMouseDragged | NSEventMaskRightMouseDragged | NSEventMaskScrollWheel | NSEventMaskKeyDown;
        [NSEvent addLocalMonitorForEventsMatchingMask:mask handler:^NSEvent *(NSEvent *e) {
          double reach = std::max(1.0, s_view.pivot.load()); // movements are in proportion to how far away the subject is
          if (e.type == NSEventTypeScrollWheel) {
              s_view.forward = s_view.forward + e.scrollingDeltaY * reach * (e.hasPreciseScrollingDeltas ? 0.004 : 0.04);
          } else if (e.type == NSEventTypeKeyDown) {
              NSString *key = e.charactersIgnoringModifiers.lowercaseString;
              if ([key isEqualToString:@"r"]) s_view.yaw = 0, s_view.pitch = 0, s_view.pan_x = 0, s_view.pan_y = 0, s_view.forward = 0;
              else if ([key isEqualToString:@"u"]) s_view.interface = !s_view.interface.load();
              else if ([key isEqualToString:@"["]) s_view.pivot = std::max(0.5, s_view.pivot.load() / 1.25);
              else if ([key isEqualToString:@"]"]) s_view.pivot = s_view.pivot.load() * 1.25;
              else if ([key isEqualToString:@"n"]) s_view.step = 1;
              else if ([key isEqualToString:@"p"]) s_view.step = -1;
              else if ([key isEqualToString:@"l"]) s_view.lod_step = 1;
              else if ([key isEqualToString:@"q"]) exit(0);
              else return e;
              return nil;
          } else if (e.type == NSEventTypeRightMouseDragged || (e.modifierFlags & NSEventModifierFlagShift)) {
              s_view.pan_x = s_view.pan_x + e.deltaX * reach * 0.002;
              s_view.pan_y = s_view.pan_y - e.deltaY * reach * 0.002;
          } else {
              s_view.yaw = s_view.yaw + e.deltaX * 0.006;
              s_view.pitch = std::min(1.4, std::max(-1.4, s_view.pitch + e.deltaY * 0.006));
          }
          return e;
        }];
        static std::atomic<bool> closed{false};
        [[NSNotificationCenter defaultCenter] addObserverForName:NSWindowWillCloseNotification object:window queue:nil usingBlock:^(NSNotification *) {
          closed = true;
        }];
        std::thread([=] {
            if (!mtl_replay_restore()) {
                fprintf(stderr, "the capture could not be restored\n");
                exit(1);
            }
            auto show = [&]() {
                double distance = 0;
                if (!mtl_preview_model(model.c_str(), lod, keep_scene, &distance)) {
                    if (lod > 0) { // no such simpler version: back to the full mesh
                        lod = 0;
                        if (mtl_preview_model(model.c_str(), 0, keep_scene, &distance)) goto shown;
                    }
                    fprintf(stderr, "no mesh named %s could be shown (replay --list <word> finds names)\n", model.c_str());
                    return false;
                }
            shown:
                if (!pivot_given) s_view.pivot = distance;
                fprintf(stderr, "showing %s%s\n", model.c_str(), lod ? (" (lod " + std::to_string(lod) + ")").c_str() : "");
                return true;
            };
            if (!model.empty() && !show() && png) exit(1);
            // Some of what a frame shows comes from the frame before it (the sky's
            // colours, grass in the shadow map), so a still needs a few runs.
            for (uint32_t n = 1; png ? n <= (uint32_t)frames : !closed.load(); n++) {
                g_frame = n;
                if (!model.empty() && (s_view.step.load() || s_view.lod_step.load())) {
                    int step = s_view.step.exchange(0);
                    if (s_view.lod_step.exchange(0)) lod++;
                    if (step) {
                        if (const char *next = mtl_preview_neighbour(model.c_str(), step)) model = next;
                        lod = 0;
                    }
                    show();
                }
                mtl_replay_orbit(s_view.yaw, s_view.pitch, s_view.pan_x, s_view.pan_y, s_view.forward, s_view.pivot);
                mtl_replay_show_interface(s_view.interface);
                mtl_replay_frame(png && n == (uint32_t)frames, n);
            }
            if (s_stats) mtl_dump_stats(stdout);
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
