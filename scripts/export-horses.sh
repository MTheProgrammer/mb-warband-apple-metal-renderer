#!/bin/sh
# Writes the game's horse textures as PNG files, to paint hair maps over:
#   scripts/export-horses.sh [folder]        (default: ~/Desktop/warband-horses)
# For each texture <name>.png paint a map of the same size and save it as
# <name>_hair.png (see README: the coat of horses).
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-"$HOME/Desktop/warband-horses"}
[ -x "$ROOT/build/texture_tool" ] || make -C "$ROOT" build/texture_tool || exit 1
exec "$ROOT/build/texture_tool" export "$OUT" \
    horse_a horse_b hunting_horse steppe_horse saddle_horse courser sumpter_horse \
    arabian_horse_a arabian_horse_b charger war_horse warhorse_new warhorse_sarranid warhorse_steppe
