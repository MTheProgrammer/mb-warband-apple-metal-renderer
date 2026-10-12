#!/bin/sh
# Shows one of the game's models through the renderer, lit as in a captured
# frame (see scripts/capture.sh), with every optional feature on.
#   scripts/preview.sh sword_medieval_a [options of build/replay]
#   scripts/preview.sh courser --anim gallop      (a horse in motion; key a for the next action)
#   scripts/preview.sh --list helmet
# The newest capture under traces/ is the stage, or the file named in WBM_STAGE.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
[ -x "$ROOT/build/replay" ] || { echo "player not built: run make build/replay" >&2; exit 1; }
if [ "$1" = "--list" ]; then exec "$ROOT/build/replay" --list "${2:-}"; fi
[ -n "$1" ] || { echo "usage: scripts/preview.sh <mesh name> [options], or --list <word>" >&2; exit 2; }
STAGE=${WBM_STAGE:-$(ls -t "$ROOT"/traces/*/capture_*.wbcap 2>/dev/null | head -1)}
[ -f "$STAGE" ] || { echo "no captured frame to use as a stage: make one with scripts/capture.sh" >&2; exit 1; }
MODEL=$1; shift
exec "$ROOT/scripts/replay.sh" "$STAGE" --model "$MODEL" "$@"
