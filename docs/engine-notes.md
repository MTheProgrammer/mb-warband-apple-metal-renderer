# Notes on the game's executable

What has been learnt about the engine by reading the Mac executable, for hooks
the shim may one day place in it. Addresses are those the file has on disk (a
disassembler's, image base 0x100000000). Nothing of the game's is kept here or
anywhere in this repository: no code, no decompiled listings.

## The file

`Mount and Blade.app/Contents/MacOS/Mount and Blade`, Steam's build:

- SHA-256 `890af55036a0526f5f3559d8e39fe5ca35c7c05b7635487b14d16b11383b9bc7`
- Mach-O x86_64, position-independent, 22 MB, built against the macOS 10.13 SDK
- Links OpenGL, GLEW 2.1, SDL2, FMOD Ex, the Steam API, libcurl
- Stripped: about 250 named functions, nearly all Havok's (which also carries
  its type names). The engine's own classes have neither names nor type names.

Everything below holds for that hash only.

## Finding one's way: call sites

With no names, the way in is the shim's own knowledge of what each OpenGL call
is for. `WBM_CALLSITES=1` notes, at chosen moments, the addresses inside the
executable that the call came through, and the run's `summary.txt` lists each
distinct path with how often it was taken:

    WBM_CALLSITES=1 scripts/run-all.sh      # play a minute: a menu, the map, a battle, a town
    less traces/latest/summary.txt          # the section "call sites"

Moments noted: a pass begun (on the screen, the scene, the depth pre-pass, the
shadow map, another target); a draw, by the kind of pass, with or without bones,
blended or not; a matrix or four-vector handed to a shader, by its name; a
buffer filled; the frame shown.

Read it so: the innermost address is the engine's wrapper for the OpenGL call;
the next ones out are who asked. The function that hands over `matWorldArray`
draws what has bones; the one that begins the pass on the shadow map draws the
scenery's shadows; what draws onto the screen after the scene has been put
there is the interface.

## Findings

Ghidra 12.1 (Homebrew's `ghidra`, with `openjdk@21`) analysed the file in two
minutes into `~/Projects/warband-decomp` (not part of this repository) and found 11,246
functions.

By the names of the shaders' uniforms, which the file holds as plain strings:

| Address | Size | What refers to it | Taken to be |
|---|---|---|---|
| `100927083` | 3,449 bytes | every uniform name looked for: `matWorldViewProj`, `matWorldArray`, `matSunViewProj`, `vSunDir`, `vLightPosDir`, `iLightIndices`, `fShadowMapSize`, `shadowmap_texture`, `vCameraPos`, `time_var` and the rest, each once | the routine that asks a linked shader for the locations of all its uniforms and keeps them in a table |
| `100a53888` | 7,103 bytes | `matWorldViewProj`, `matViewProj`, `matWorldArray` only | a second, smaller set-up of the same kind; which shaders it serves is not known yet |

Neither has been read through. Which functions set those uniforms each frame,
and so draw, is what the first run with call sites will show.
