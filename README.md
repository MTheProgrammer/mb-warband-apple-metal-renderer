# warband-metal

An injected library for Mount & Blade: Warband (Steam, macOS) that will replace the game's OpenGL renderer with Metal. The game's files are not modified.

Status: the Metal renderer draws the game: menus, the world map, battles with shadows and post-processing, HUD and text. A 200 vs 200 battle and the world map of the Native module pass Metal's API validation (`MTL_DEBUG_LAYER=1`) without a violation. Frame rates against OpenGL are in `docs/baseline.md`. Not done yet: modules other than Native are unchecked.

## Build

Needs the Xcode Command Line Tools, git, python3 and the game installed through Steam.

    make

The first build downloads glslang and SPIRV-Cross at pinned versions into `third_party/` and compiles them (`scripts/build_deps.sh`); cmake is not needed.

This produces `build/libwbmetal.dylib` for x86_64 (the game runs under Rosetta 2). If the game is not in the default Steam library, pass `GAME=/path/to/MountBlade\ Warband`.

## Run

    scripts/run.sh

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
| `WBM_MODE` | `trace` (default): OpenGL with tracing. `native`: the game's own OpenGL calls are left untouched and only frame times are recorded. `metal`: the Metal renderer |
| `WBM_TRACE` | `1` traces the calls in Metal mode too, at a cost in frame rate. Off by default there |
| `WBM_VSYNC` | Overrides the swap interval. `0` asks for an uncapped frame rate |
| `WBM_SCALE` | Metal mode: resolution multiplier for the screen and screen-sized render targets. Defaults to the display's scale (2 on Retina); `1` renders at the game's own resolution, as OpenGL does |
| `WBM_CACHE` | Directory of the translated-shader cache. Default `~/Library/Caches/warband-metal`; `0` disables it |
| `WBM_DEBUG_CLEAR` | Metal mode: clears the screen to blue instead of the game's colour, to show that frames arrive |
| `WBM_OUT` | Output directory. `run.sh` sets it |
| `WBM_TRACE_FRAMES` | Frames to log call by call and capture, for example `1-3,600` |
| `WBM_GLERR` | `1` checks `glGetError` after every setup call |
| `WBM_LOG_MAX_MB` | Size limit of `trace.log` outside call-by-call frames. Default 256 |
| `WBM_MODULE` | Module to load, for `run.sh`, for example `Native`. Works like the launcher's choice: it stays selected for later launches |
| `WBM_GAME` | Game directory, for `run.sh` |

## How it works

`run.sh` starts the game binary with `DYLD_INSERT_LIBRARIES`. The game's OpenGL surface is 62 functions, listed in `src/gl_api.def`:

- 22 are imported from OpenGL.framework and replaced through a dyld interpose table, as are 7 SDL window and context functions.
- 40 are called through pointer variables exported by the game's bundled GLEW. The shim interposes `glewInit` and swaps those pointers after GLEW has filled them in.

In Metal mode the window is created without an OpenGL surface and gets a `CAMetalLayer`; the game receives a placeholder context, and the 62 functions are routed to `src/metal_backend.mm`.

Shaders are translated when a program is linked (`src/shader_xlate.cpp`), and the result is cached on disk (`src/shader_cache.cpp`), so later launches skip the translation. Metal then compiles the translated source in the background; a program waits for that only if it draws before its shaders are ready. The translation itself: the engine's GLSL 1.20 is rewritten to GLSL 4.50, compiled to SPIR-V by glslang and turned into Metal Shading Language by SPIRV-Cross. `make build/xlate_test && build/xlate_test traces/latest` runs that translation over every program a trace recorded and compiles the results with Metal.

Render-to-texture passes are drawn upside down, so that textures keep OpenGL's row order and the game's texture coordinates work unchanged.

AppKit also calls a few of the interposed functions when it presents the GL surface. Calls that do not come from the game binary are forwarded and counted separately.

Do not use the shim on multiplayer servers: it changes how the game process loads libraries.
