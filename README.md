# warband-metal

An injected library for Mount & Blade: Warband (Steam, macOS) that will replace the game's OpenGL renderer with Metal. The game's files are not modified.

Status: milestone 1 of 7. The library traces every graphics call the game makes and forwards it to real OpenGL. The Metal backend does not exist yet.

## Build

Needs the Xcode Command Line Tools and the game installed through Steam.

    make

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
| `WBM_MODE` | `trace` (default), or `native`: the game's own OpenGL calls are left untouched and only frame times are recorded. `metal` is reserved for the Metal backend |
| `WBM_VSYNC` | Overrides the swap interval. `0` asks for an uncapped frame rate |
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

AppKit also calls a few of the interposed functions when it presents the GL surface. Calls that do not come from the game binary are forwarded and counted separately.

Do not use the shim on multiplayer servers: it changes how the game process loads libraries.
