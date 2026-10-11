# Benchmark scene and baseline

The scene used to compare renderers: **Native module, Custom Battle, map "Village", 200 vs 200**.

Run it the same way each time:

    WBM_MODULE=Native WBM_MODE=native scripts/run.sh

then read the battle seconds out of `traces/latest/fps.log`. The game limits itself to `max_framerate` from `~/Library/Application Support/MBWarband/rgl_config.txt` (120 here), so only the seconds below that limit carry information unless the limit is raised.

## Game's own OpenGL renderer

| Date | Trace | Battle length | Frame rate in battle |
|---|---|---|---|
| 2026-10-11 | `20261011-203728` | about 20 s (seconds 26–45) | 55–105 fps, mostly 80–105 |

Machine: Apple M5, macOS 26.5.2, game drawing at 1512x982, under Rosetta 2. Troop types and the other custom-battle settings were not recorded for this run, and the battle was short; repeat with a longer one before relying on the numbers.

## Comparison, 2026-10-11

Same battle, same routine, `max_framerate = 120`:

| Renderer | Trace | Length | Seconds below 110 fps | Frame rate while fighting |
|---|---|---|---|---|
| OpenGL (`WBM_MODE=native`) | `20261011-232509` | 79 s | 41 | 55-75 fps |
| Metal, `WBM_SCALE=1` | `20261011-232651` | 79 s | 2 (loading) | 120 fps, at the limit |
| Metal, Retina (scale 2) | `20261011-232844` | 124 s | 2 (loading) | 120 fps, at the limit |
| Metal, Retina, `WBM_VSYNC=0` | `20261011-233259` | 197 s | 5 | 120 fps, at the limit |

Metal never left the game's own limit, so its real ceiling is not measured yet.
