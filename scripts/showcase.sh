#!/bin/sh
# The material showcase: a ball for each kind of material the renderer treats
# differently and a row of real models, set in a castle hall lit by its torches,
# with every feature on as scripts/run-all.sh has them. The models and their
# places are in scripts/showcase.txt (or the file named in WBM_GALLERY); the
# hall is a captured frame (WBM_STAGE for another, an outdoor one for sunlight).
#   scripts/showcase.sh                    in a window: drag to turn, scroll to come nearer
#   scripts/showcase.sh --png out.png      a still
#   WBM_SSS=0 scripts/showcase.sh ...      any setting, to compare with and without it
ROOT=$(cd "$(dirname "$0")/.." && pwd)
STAGE=${WBM_STAGE:-"$ROOT/traces/20261013-interior/capture_006450.wbcap"}
GALLERY=${WBM_GALLERY:-"$ROOT/scripts/showcase.txt"}
[ -f "$STAGE" ] || { echo "no stage: $STAGE (set WBM_STAGE to a capture, see scripts/capture.sh)" >&2; exit 1; }
exec "$ROOT/scripts/replay.sh" "$STAGE" --gallery "$GALLERY" --keep-scene "$@"
