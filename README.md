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
| `WBM_DYNAMIC_SHADOWS` | Metal mode: `1` draws a second shadow map every frame from the objects of the engine's depth pre-pass, so that characters, horses and gear cast shadows on everything and at any distance, and suppresses the engine's ground decals for them. Needs the game's "depth effects" option. The map has two parts: a wide one, `WBM_DYN_RANGE` metres to each side of a point ahead of the camera (default 60), and a fine one for sharp shadows of thin things close by, `WBM_DYN_NEAR` metres to each side (default 12, `0` for none). `WBM_DYN_BIAS` (metres, default 0.05) moves surfaces towards the sun before the comparison. Grass, which the pre-pass leaves out, is taken from the main pass a frame late. Uses about 260 MB of video memory. Off by default |
| `WBM_REVERSE_Z` | Metal mode: `0` uses the conventional depth mapping. By default depth is reversed (far is 0), which spreads the float depth buffer's precision evenly and avoids flicker between distant, nearly coplanar surfaces |
| `WBM_CONTACT_SHADOWS` | Metal mode: `1` adds short screen-space shadows towards the sun, for contact detail the shadow map is too coarse for. Computed with ambient occlusion and subject to the same limits. `WBM_CONTACT_STRENGTH` (default 0.5) and `WBM_CONTACT_LENGTH` (metres, default 0.4) tune it. Off by default |
| `WBM_SKY_AMBIENT` | Metal mode: `1` gives the ambient light a direction. What the engine adds as flat ambient and sky light is tinted by the colour of the scene's sky on the side a surface faces, and surfaces facing sideways or down receive sunlight bounced off the ground. `WBM_SKY_AMBIENT_STRENGTH` (0 to 1, default 0.6) sets the tint, `WBM_BOUNCE` (default 0.5) the bounce. Outdoor scenes only. Off by default |
| `WBM_TEXTURES` | Metal mode: folders of replacement textures, separated by colons, the first one winning. A file named after one of the game's texture files (`helmets_new_a.png` or `.dds` for the game's `helmets_new_a.dds`; PNG, JPEG, TIFF, BMP and DDS in DXT1/3/5 or 32-bit are read) is loaded in place of the game's texture, at whatever size it has, when the game uploads it. Subfolders are searched too. Without the variable, `~/Library/Application Support/warband-metal/textures` is used if it exists. The game's files are not touched. Applies to frame captures played back as well, where `WBM_GAME` names the game's folder if Steam did not put it in the usual place |
| (texture packs, continued) | A pack may also hold `<name>_pbr.png` for the game's texture `<name>`: metalness in red and roughness in green, laid out like the texture. With `WBM_MATERIALS=1` such a map replaces the renderer's own estimates for that texture. `make build/texture_tool` builds a tool that writes these maps for the game's textures from their specular maps (`build/texture_tool pbr <folder>`), to be used as they are or corrected by hand, and that exports the game's textures as PNG for editing or enlarging (`build/texture_tool export <folder> [name...]`) |
| `WBM_WIND` | Metal mode: a strength, for example `1`: trees, bushes and grass sway, the more the higher up. Foliage is whatever is drawn with a texture whose file name says so (tree, bush, grass, plant, leaf and the like; bark and trunks stay still). Only the position on screen moves; shadows stay as they are. Off by default |
| `WBM_FOLIAGE` | Metal mode: how much sunlight comes through leaves, for example `0.6`: on the side turned from the sun where a shader lights by the surface's direction, and for the engine's own grass and leaf shaders, which do not, when looking towards the sun. Off by default |
| `WBM_FUR` | Metal mode: a strength, for example `1`: a horse's coat is lit softly, with a velvety rim where it is seen edge-on, and the engine's hard streaks of highlight toned down. Applies where a texture pack's `<name>_pbr.png` says in its blue channel that the surface is coat, which `build/texture_tool pbr` works out for horses (not armour, cloth or saddle); without such a map nothing changes. With `WBM_DYNAMIC_SHADOWS=1` a little sunlight also comes through thin parts of the coat (ears) seen against the sun. `WBM_EYES=1` gives the game's plain horses a highlight in the eye as well. `WBM_FUR_SHELLS` (for example `10`; 0, the default, for none) draws the coat of horses near the camera again in that many layers, each a little farther out and holding only the hairs that reach it, which gives short fur with depth; `WBM_FUR_LENGTH` (metres, default 0.012) and `WBM_FUR_RANGE` (metres from the camera, default 9) set how long and how far, and `WBM_FUR_DENSITY` (hairs per metre, default 1400) how fine: it costs nothing, but finer than the screen can show turns to noise. Each layer is one more draw of the horse. A pack file `<name>_hair.png` refines all of this for a texture: red and green give the way the hair is combed (128 for none; red rightwards, green down the picture), blue what grows (255 long hair such as mane and tail, about 128 coat, 0 nothing). Long hair then gets twice as many layers of its own (`WBM_HAIR_LENGTH`, metres off the body, default 0.07): thicker strands that hang along the combing and downwards and swing a little; it shines in bands across its strands, the fur leans the way it is combed and droops, and blue replaces the `_pbr` map's coat mask. `build/texture_tool hair <picture> <name> <folder>` makes such a map from a picture of the texture with all but the long hair blacked out. Off by default |
| `WBM_EYES` | Metal mode: `1` gives the eyes of faces a small sharp highlight from the sun and a faint reflection of the sky. The eyes are painted on the face texture at the same place for every face, which is where the effect is applied, in the engine's skin shader. The eyes also wander: held still for a second or two, then turned a little to one side or back to straight ahead, each face in its own time (`WBM_EYES_MOVE`, default 1, sets how far; `0` keeps them still). Shows in close-ups. Off by default |
| `WBM_SSS` | Metal mode: `1` gives skin subsurface scattering, done where it is lit: sunlight wraps a little way round the terminator, farthest in red; the red is lit from the smooth surface instead of the normal map's relief; and the edge of a shadow on skin is tinted red. It applies to whatever the engine draws with its skin shader (the faces of troops), and in the general shaders to draws whose texture's file name says face, body, hand or skin. With `WBM_DYNAMIC_SHADOWS=1`, sunlight also comes through thin parts (ears, the wings of a nose) seen against the sun, within the shadow map's fine window, which measures how thick they are; `WBM_SSS_BACKLIGHT` (default 0.6, `0` for none) sets how strongly. Off by default |
| `WBM_MATERIALS` | Metal mode, with `WBM_PBR=2` or `3`: `1` sets metalness, roughness and highlight strength by what a texture shows (metal, cloth, leather, skin, wood, stone, horse) instead of estimating them from the specular mask alone. Textures are recognised by content against the files in the game's texture folders and classed by words in the file names; names that fit no class, or more than one, keep the estimate. Within those limits a surface counts as metal where the material is shiny and its colour is grey, and a metal loses most of its diffuse colour to reflection. Where a shader has a specular map, metalness is also read from it. `WBM_DEBUG_MATERIALS=1` shows the estimated metalness instead of the picture, white for metal and black for none. Off by default |
| `WBM_FXAA` | Metal mode: `1` smooths edges after the scene is drawn (FXAA), geometry and cut-outs such as leaves and hair alike, before the interface goes on top. A cheap stand-in for `WBM_MSAA`, and usable with it. Needs the game's HDR option. Off by default |
| `WBM_MAX_FPS` | Metal mode: a frame-rate limit, for example `60`; less work per second means a cooler, quieter machine. `WBM_MENU_FPS` is a separate limit for frames in which no 3D scene is drawn (menus, loading). No limits by default, beyond the game's own `max_framerate` |
| `WBM_MOTION_BLUR` | Metal mode: `1` blurs the scene along the camera's own movement, from each pixel's depth and the previous frame's camera. Objects that move by themselves are not blurred, and nothing within a few metres of the camera is, so that the player's character stays sharp. `WBM_MOTION_BLUR_STRENGTH` (default 1) is the exposure time in units of 1/48 s. Needs the game's HDR option. Off by default |
| `WBM_LIGHT_SHAFTS` | Metal mode: `1` adds light shafts from the sun: bright sky near the sun is smeared along lines towards the sun's position on screen. Needs the game's HDR mode and a visible sun. `WBM_SHAFT_STRENGTH` (default 0.7) tunes it. Off by default |
| `WBM_TONEMAP` | Metal mode: `1` replaces the tone-mapping operator in the engine's final post-processing shader with one that leaves shadows and mid-tones as they are and rolls highlights off smoothly instead of clipping them. Needs the game's HDR mode. Off by default |
| `WBM_HDR` | Metal mode: `1` presents in extended range on displays that support it: highlights in the tone-mapped image are stretched above display white, up to the screen's current headroom or `WBM_HDR_PEAK` (default 4). Needs the game's HDR mode. Off by default |
| `WBM_LOD_BIAS` | Metal mode: mip level bias for all textures, for example `-0.5` for sharper texturing. macOS 26 or later. Rendering at Retina resolution already selects sharper mip levels, so this is rarely needed |
| `WBM_AO` | Metal mode: `1` adds screen-space ambient occlusion to the scene before the engine's post-processing. Needs the game's HDR mode. `WBM_AO_RADIUS` (metres, default 0.6) and `WBM_AO_STRENGTH` (default 0.5) tune it; `WBM_AO_FULL=1` computes it at full instead of half resolution, which is sharper on fine detail and costs about four times as much. Off by default |
| `WBM_PBR` | Metal mode: `1` replaces the engine's Blinn-Phong highlights with microfacet ones (GGX, Smith shadowing, Schlick Fresnel), keeping each material's highlight strength and deriving roughness from its shininess. `2` also estimates metalness from the specular mask, darkens the base colour of metals and adds a reflection of a sky-and-ground environment built from the scene's light colours. `3` reflects the scene's actual sky instead: the engine's sky dome is redrawn each frame into a small six-face atlas that the shaders sample, blurred by roughness. Off by default: it changes how armour and weapons look |
| `WBM_SRGB` | Metal mode: `1` removes the shaders' own `pow(colour, 2.2)` on texture samples and binds sRGB views of the textures instead, for an exact decode before filtering. Off by default: it shifts dark tones slightly |
| `WBM_STATE_CACHE` | Metal mode: `0` issues every state-setting Metal call on every draw, as a baseline for measuring the saving. By default calls whose value is unchanged within a render pass are skipped |
| `WBM_SHADOWS` | Metal mode: `0` keeps the game's own 4-sample shadow filter. `1`, the default, filters over a 4x4 block of shadow-map texels for softer edges. `2` uses percentage-closer soft shadows: sharp where an object meets the ground, wider with distance from it |
| `WBM_CACHE` | Directory of the translated-shader cache. Default `~/Library/Caches/warband-metal`; `0` disables it |
| `WBM_DEBUG_CLEAR` | Metal mode: clears the screen to blue instead of the game's colour, to show that frames arrive |
| `WBM_OUT` | Output directory. `run.sh` sets it to `traces/<timestamp>`; otherwise `~/Library/Logs/warband-metal` |
| `WBM_TRACE_FRAMES` | Frames to log call by call and capture, for example `1-3,600` |
| `WBM_SNAP_EVERY` | Captures a frame to `frames/` every this many seconds, for example `10`. In trace mode each captured frame is also logged call by call |

A single frame can also be captured whole and looked at later without the game. While the game runs in Metal mode, `scripts/capture.sh [delay-seconds]` writes the next frame and everything it depends on to `capture_<n>.wbcap` in the run's output directory (several hundred MB). `make build/replay` builds a player for such files: `build/replay file.wbcap` shows the frame in a window, and `build/replay file.wbcap --png out.png` saves it as an image. The `WBM_*` variables apply to the player as they do to the game, so one captured frame can be compared across settings. Capture with the features of interest switched on: the player can turn them off, but cannot supply what the game was never asked for.

The same player shows single models from the game's files: `scripts/preview.sh sword_medieval_a` draws that mesh, with the game's own shader and textures for its material, in the newest captured frame in place of what the game drew there, so it is lit as in that scene and gets every feature of the renderer (`--keep-scene` leaves the scene in; `WBM_STAGE` names another capture). `scripts/preview.sh --list helmet` finds mesh names. In the window, `n` and `p` step through the meshes of the same resource file and `l` through a mesh's simpler versions. Skinned pieces are shown in their rest pose; whole characters are not assembled.
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
