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

### Flora: how generated scenes are planted (read 2026-10-19)

- `1009e1750` loads `Data/flora_kinds.txt`, first from the module's folder
  (`DAT_103839490` holds `Modules/<module>/`), then from the game's own `Data`.
  Kinds go into a table of 0xb8-byte records at `DAT_102826718`, their number at
  `DAT_102826720`; the flags with the density in their upper 32 bits sit at +0x98.
  The table is allocated to the number of kinds: no fixed limit.
- `10097a608` decides, from the scene's region (bits 28-31 of a terrain word:
  2 steppe, 3 plain, 4 snow, 5 desert, 10 steppe forest, 11 plain forest, 12 snow
  forest, 13 desert forest), how many groups of each category to plant:

  | region | trees | other plants | grass |
  |---|---|---|---|
  | plain, steppe, snow (default) | 3-6 | 4-5 | 4 |
  | desert (5) | 0 | 2 | 0 |
  | steppe forest (10) | 2-5 | 2-3 | 2 |
  | plain forest (11) | 2-5 | 2-3 | 4 |
  | snow forest (12) | 2 | 2-3 | 0 |
  | desert forest (13) | 3-4 | 2 | 0 |

  Grass groups only when a setting allows (`DAT_101464ca8`, or `DAT_101465178` above 9).
- `10097e5ae` fills the groups of one category. `10097eb76` says whether a kind
  belongs to the category (0 trees: 0x400000; 1 others: neither tree nor grass
  0x80000; 2 grass; 3: 0x2000000) and may grow in the region (the region bits of the
  flags), and gives every such kind the same weight, 1: the density takes no part in
  the choice. Kinds marked 0x1000000 (guarantee) are chosen first, for the first group.
  Each group then gets one kind and an amount:
  `density / 1000 x (1 + r) x (0.4 + 0.05 r) x 15 x 3 / (groups + 2)` (r random in 0..1;
  the constants are at `100b46af0`, `100b4b590`, `100b440cc`, `100b4a034`, `100b47664`),
  and a radius of 50-100 and noise seeds. How the amount turns into plants on the ground
  (`10097a7ec`) has not been read.
- So adding kinds changes which plants are seen, not how many; raising density raises
  the amount. `scripts/forest_flora.py` and `scripts/flora_preview.py` follow this.
