#!/bin/sh
# Shows a captured frame (see scripts/capture.sh) with every optional feature
# on, as scripts/run-all.sh runs the game. Variables set by the caller win.
# Usage: scripts/replay.sh capture.wbcap [options of build/replay]
ROOT=$(cd "$(dirname "$0")/.." && pwd)
[ -x "$ROOT/build/replay" ] || { echo "player not built: run make build/replay" >&2; exit 1; }
# Not extended-range output, unless asked for: in a window the system changes
# how much range it grants from moment to moment, which shows as flicker.
export WBM_HDR=${WBM_HDR:-0}
eval "$(grep '^export WBM_' "$ROOT/scripts/run-all.sh")"
exec "$ROOT/build/replay" "$@"
