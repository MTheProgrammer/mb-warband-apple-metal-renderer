# Benchmark scene and baseline

The scene used to compare renderers: **Native module, Custom Battle, map "Village", 200 vs 200**.

Run it the same way each time:

    WBM_MODULE=Native WBM_MODE=native scripts/run.sh

then read the battle seconds out of `traces/latest/fps.log`. The display caps the frame rate at 120 fps, so only the seconds below the cap carry information.

## Game's own OpenGL renderer

| Date | Trace | Battle length | Frame rate in battle |
|---|---|---|---|
| 2026-10-11 | `20261011-203728` | about 20 s (seconds 26–45) | 55–105 fps, mostly 80–105 |

Machine: Apple M5, macOS 26.5.2, game drawing at 1512x982, under Rosetta 2. Troop types and the other custom-battle settings were not recorded for this run, and the battle was short; repeat with a longer one before relying on the numbers.
