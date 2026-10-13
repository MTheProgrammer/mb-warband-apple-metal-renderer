#!/bin/sh
# Shows one of the game's models with pictures from a folder in place of the
# game's textures: the folder is made into a texture pack (build/texture_tool
# import, which says how the pictures are to be named) and the model shown as
# scripts/preview.sh shows it.
#   scripts/preview-textures.sh <folder of pictures> <mesh name> [options of build/replay]
#   scripts/preview-textures.sh <folder> arabian_house_a        (a building that wears stucco_6)
# The model has to be one that uses the textures: scripts/preview.sh --list <word>
# names meshes and, for materials, the textures they take.
# WBM_IMPORT holds options for the import, "--gain 2" for one.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
[ -x "$ROOT/build/texture_tool" ] || { echo "tool not built: run make build/texture_tool" >&2; exit 1; }
[ -d "$1" ] && [ -n "$2" ] || { echo "usage: scripts/preview-textures.sh <folder of pictures> <mesh name> [options]" >&2; exit 2; }
FOLDER=$(cd "$1" && pwd); shift
PACK="$ROOT/build/texture-preview/$(printf '%s' "$FOLDER" | cksum | cut -d' ' -f1)"
rm -rf "$PACK"
"$ROOT/build/texture_tool" import "$FOLDER" "$PACK" $WBM_IMPORT || exit 1
export WBM_TEXTURES="$PACK" WBM_MATERIALS=${WBM_MATERIALS:-1}
exec "$ROOT/scripts/preview.sh" "$@"
