# warband-metal

A Metal renderer for Mount & Blade: Warband on macOS. The Steam version of the game draws with OpenGL, which Apple deprecated years ago. This project loads a library into the game when it starts that takes over its drawing and does it with Metal instead. The game's files are not modified, and the game still runs as an Intel program under Rosetta 2.

In a 200 vs 200 custom battle on a MacBook Pro M5:

| | OpenGL (1512x982, limited to 120 fps) | Metal (3024x1964, unlimited) |
|---|---|---|
| Average | 110 fps | 199 fps |
| Median second | 117 fps | 210 fps |
| Slowest 10% of seconds | 86 fps or less | 131 fps or less |
| Slowest second | 55 fps | 109 fps |

Metal also draws at the display's native resolution, four times the pixels of the OpenGL path. The runs were played by hand, one each, so read the result as "roughly twice as fast"; details are in `docs/baseline.md`.

## Using it

You do not need to compile anything. A ready-made app is on the Releases page.

1. Download `warband-metal-<version>.zip` from the Releases page. Safari unzips it for you; otherwise double-click the zip. You get `Warband Metal.app`.
2. Allow the app to run. macOS blocks apps that were downloaded and are not signed by a registered Apple developer, and this one is not, so the first double-click only shows a warning. Do one of the following, once:

   **With Terminal** (the route that is known to work). Open Terminal, paste this and press Return:

       xattr -dr com.apple.quarantine ~/Downloads/"Warband Metal.app"

   If the app is not in Downloads, type `xattr -dr com.apple.quarantine ` with a space at the end, drag the app onto the Terminal window, and press Return. The command removes the "downloaded from the internet" mark from the app; it changes nothing else.

   **Without Terminal.** Double-click the app and close the warning. Open System Settings → Privacy & Security, scroll down to Security, and click "Open Anyway" next to the message about Warband Metal. Confirm with your password or Touch ID, then open the app again. This route has not been verified; if the game still does not start afterwards, use the Terminal command.

3. Double-click `Warband Metal.app`. It starts the game on Metal. If the game is not in the default Steam library, it asks for the "MountBlade Warband" folder once and remembers it.

You can keep the app anywhere, for example in Applications.

The module and the graphics options are still set in the game's own launcher, the window Steam's Play button opens: the game reads them when it starts, whichever way it is started.

If something looks wrong, the log of the latest run is in `~/Library/Logs/warband-metal/trace.log`.

**Do not use this in multiplayer.** It loads code into the game process, which is the kind of thing anti-cheat systems object to.

Tested on one machine: MacBook Pro M5, macOS 26.5, game version 1.174 from Steam, Native module (menus, world map, towns, battles). Other modules, Intel Macs and older macOS versions are untested.

## Status

The Metal renderer draws the whole game and passes Metal's API validation (`MTL_DEBUG_LAYER=1`) in a 200 vs 200 battle and on the world map. The rest of this document is for developers.

## Build

Needs the Xcode Command Line Tools, git and python3. The game does not have to be installed to build, only to run.

    make

The first build downloads its dependencies at pinned versions into `third_party/` and compiles the two shader libraries, glslang and SPIRV-Cross (`scripts/build_deps.sh`); cmake is not needed. It also fetches the headers of GLEW 2.1.0 and SDL 2.0.8, the versions the game bundles; `stubs/*.tbd` stand in for those two libraries at link time.

This produces `build/libwbmetal.dylib` for x86_64 (the game runs under Rosetta 2). `make app` builds `build/Warband Metal.app` and `make dist` zips it.

Releases are built by GitHub Actions (`.github/workflows/build.yml`): pushing a tag that starts with `v` publishes the zip on the Releases page. The steps are in `docs/releasing.md`.

## Run

    scripts/run.sh

This starts the game on Metal with the library just built. `WBM_MODE=trace scripts/run.sh` runs the game's own OpenGL renderer with every call traced, which is how the renderer was developed and is still the reference to compare against.

Each run writes to a new directory, `traces/<timestamp>/`, also reachable as `traces/latest`:

| File | Content |
|---|---|
| `trace.log` | Setup calls (shaders, textures, buffers, framebuffers, SDL) and any frames logged call by call |
| `summary.txt` | Call counts per function and thread, every distinct argument combination, framebuffer layouts, frame times. Rewritten every 300 frames and at exit |
| `shaders/` | The exact source of every shader passed to `glShaderSource` |
| `programs.txt` | Uniforms and attributes of every linked program, as reported by OpenGL |
| `fps.log` | One line per second: frame rate, average and worst frame time |
| `extensions.txt` | Extensions of the game's GL context |
| `frames/` | PNG of each frame that was logged call by call |

While the game runs, `scripts/snap.sh` logs the next frame call by call and saves it as a PNG. `scripts/snap.sh 10` waits ten seconds first, which leaves time to switch back to the game.

## Environment variables

| Variable | Meaning |
|---|---|
| `WBM_MODE` | `metal` (default): the Metal renderer. `trace`: OpenGL with tracing. `native`: the game's own OpenGL calls are left untouched and only frame times are recorded |
| `WBM_TRACE` | `1` traces the calls in Metal mode too, at a cost in frame rate. Off by default there |
| `WBM_VSYNC` | Overrides the swap interval. `0` asks for an uncapped frame rate |
| `WBM_SCALE` | Metal mode: resolution multiplier for the screen and screen-sized render targets. Defaults to the display's scale (2 on Retina); `1` renders at the game's own resolution, as OpenGL does |
| `WBM_MSAA` | Metal mode: samples per pixel for the screen and screen-sized render targets, `2` or `4`. Smooths geometry edges, and alpha-tested edges (foliage, hair, fur) through a rewrite of the shaders' alpha tests. Default `4`; `1` turns it off |
| `WBM_REVERSE_Z` | Metal mode: `0` uses the conventional depth mapping. By default depth is reversed (far is 0), which spreads the float depth buffer's precision evenly and avoids flicker between distant, nearly coplanar surfaces |
| `WBM_SRGB` | Metal mode: `1` removes the shaders' own `pow(colour, 2.2)` on texture samples and binds sRGB views of the textures instead, for an exact decode before filtering. Off by default: it shifts dark tones slightly |
| `WBM_STATE_CACHE` | Metal mode: `0` issues every state-setting Metal call on every draw, as a baseline for measuring the saving. By default calls whose value is unchanged within a render pass are skipped |
| `WBM_SHADOWS` | Metal mode: `0` keeps the game's own 4-sample shadow filter. By default shadows are filtered over a 4x4 block of shadow-map texels, for softer edges |
| `WBM_CACHE` | Directory of the translated-shader cache. Default `~/Library/Caches/warband-metal`; `0` disables it |
| `WBM_DEBUG_CLEAR` | Metal mode: clears the screen to blue instead of the game's colour, to show that frames arrive |
| `WBM_OUT` | Output directory. `run.sh` sets it to `traces/<timestamp>`; otherwise `~/Library/Logs/warband-metal` |
| `WBM_TRACE_FRAMES` | Frames to log call by call and capture, for example `1-3,600` |
| `WBM_SNAP_EVERY` | Captures a frame to `frames/` every this many seconds, for example `10`. In trace mode each captured frame is also logged call by call |
| `WBM_GLERR` | `1` checks `glGetError` after every setup call |
| `WBM_LOG_MAX_MB` | Size limit of `trace.log` outside call-by-call frames. Default 256 |
| `WBM_MODULE` | Module to load, for `run.sh`, for example `Native`. Works like the launcher's choice: it stays selected for later launches |
| `WBM_GAME` | Game directory, for `run.sh` and the launcher app, when the game is not in the default Steam library |

## How it works

`run.sh` starts the game binary with `DYLD_INSERT_LIBRARIES`. The game's OpenGL surface is 62 functions, listed in `src/gl_api.def`:

- 22 are imported from OpenGL.framework and replaced through a dyld interpose table, as are 7 SDL window and context functions.
- 40 are called through pointer variables exported by the game's bundled GLEW. The shim interposes `glewInit` and swaps those pointers after GLEW has filled them in.

In Metal mode the window is created without an OpenGL surface and gets a `CAMetalLayer`; the game receives a placeholder context, and the 62 functions are routed to `src/metal_backend.mm`.

Shaders are translated when a program is linked (`src/shader_xlate.cpp`), and the result is cached on disk (`src/shader_cache.cpp`), so later launches skip the translation. Metal then compiles the translated source in the background; a program waits for that only if it draws before its shaders are ready. The translation itself: the engine's GLSL 1.20 is rewritten to GLSL 4.50, compiled to SPIR-V by glslang and turned into Metal Shading Language by SPIRV-Cross. `make build/xlate_test && build/xlate_test traces/latest` runs that translation over every program a trace recorded and compiles the results with Metal.

The translation step can also change what a shader computes. The engine's shadow filter is recognised by its shape and replaced with a wider one (`soften_shadows` in `src/shader_xlate.cpp`). With multisampling on it also rewrites alpha tests into per-sample coverage (`smooth_alpha_tests`). Two more rewrites are optional: reversed depth (the clip-space mapping appended to every vertex shader) and hardware sRGB decoding (`decode_srgb_in_hardware`). `build/xlate_test traces/latest -s -a -z -g` checks that every program still compiles with all of them on.

Render-to-texture passes are drawn upside down, so that textures keep OpenGL's row order and the game's texture coordinates work unchanged.

AppKit also calls a few of the interposed functions when it presents the GL surface. Calls that do not come from the game binary are forwarded and counted separately.

Do not use the shim on multiplayer servers: it changes how the game process loads libraries.

## License

MIT, see `LICENSE`. The build downloads glslang and SPIRV-Cross, which carry their own licences. Mount & Blade: Warband is not included and is required; this project is not affiliated with TaleWorlds Entertainment.
