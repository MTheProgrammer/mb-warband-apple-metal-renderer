#!/bin/sh
# Writes the game's models as glTF files (.glb) for Blender, one per model, with
# the pictures they wear written once into <folder>/textures and shared.
#   scripts/export-models.sh <folder> [word]
# With a word, only the models whose names contain it (arabian, castle, horse);
# without, all of them: a few thousand files and a while. Simpler versions of a
# model (name.lod1 ...) are left out, and a model's further parts (name.1 ...)
# go into its file (some models are nothing but numbered parts: name.24 alone). Models already there are passed over, so a run can be taken
# up again. Options of build/gltf_export go in WBM_EXPORT ("--module Native").
ROOT=$(cd "$(dirname "$0")/.." && pwd)
[ -x "$ROOT/build/gltf_export" ] || { echo "exporter not built: run make build/gltf_export" >&2; exit 1; }
[ -n "$1" ] || { echo "usage: scripts/export-models.sh <folder> [word]" >&2; exit 2; }
mkdir -p "$1" && OUT=$(cd "$1" && pwd) || exit 1
"$ROOT/build/gltf_export" $WBM_EXPORT --list "${2:-}" 2>/dev/null | awk '$2 ~ /\.brf$/ {print $1}' | grep -vE '\.lod[0-9]+(\.[0-9]+)?$' | sed -E 's/\.[0-9]+$//' | sort -u > "$OUT/.names"
echo "$(wc -l < "$OUT/.names" | tr -d ' ') models to $OUT"
# Several at once: each reads the game's resource files anew, which is most of its time.
xargs -P "$(sysctl -n hw.ncpu 2>/dev/null || echo 4)" -I{} sh -c '
    [ -f "$1/$2.glb" ] || "$0" $WBM_EXPORT "$2" --textures textures -o "$1/$2.glb" > "$1/$2.log" 2>&1 || echo "failed: $2 (see $1/$2.log)" >&2
    grep -qvE "\.glb: [0-9]+ mesh" "$1/$2.log" 2>/dev/null || rm -f "$1/$2.log"' "$ROOT/build/gltf_export" "$OUT" {} < "$OUT/.names"
rm -f "$OUT/.names"
echo "$(ls "$OUT" | grep -c "\.glb$") models in $OUT; notes on those that had any are beside them, as .log"
