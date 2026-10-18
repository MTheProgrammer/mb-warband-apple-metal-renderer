#!/bin/sh
# Shows a model that has been to Blender and back, as the game would draw it,
# without touching the game: the .glb files of a folder (or one file) are taken
# into a stand-in for the game's folder under build/, and the model is shown
# from there as scripts/preview.sh shows it.
#   scripts/preview-models.sh <folder or file.glb> <mesh name> [options of build/replay]
#   scripts/preview-models.sh <folder> <mesh name> --original      the game's own, to compare
#   scripts/preview-models.sh <folder> <mesh name> --png out.png   stills of both: out.png and out_original.png
# --lod n shows a simpler version; the changed texture mapping is carried over to those (gltf_import --lods uv;
# --lods keep leaves the game's own simpler versions as they are, --lods copy puts the full mesh in their place).
# WBM_MODULE names the module (the one last played, else Native); WBM_TEXTURES a texture pack, as ever.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
[ -x "$ROOT/build/gltf_import" ] || { echo "importer not built: run make build/gltf_import" >&2; exit 1; }
[ -e "$1" ] && [ -n "$2" ] || { echo "usage: scripts/preview-models.sh <folder or file.glb> <mesh name> [--original] [options]" >&2; exit 2; }
SOURCE=$1; MESH=$2; shift 2
GAME=${WBM_GAME:-"$HOME/Library/Application Support/Steam/steamapps/common/MountBlade Warband"}
# The module the previewer reads: the one asked for, or the one the game last played, as the previewer itself decides.
LAST="$HOME/Library/Application Support/MBWarband/last_module_warband"
MODULE=${WBM_MODULE:-$( [ -s "$LAST" ] && cat "$LAST" || echo Native )}
[ -d "$GAME/Modules/$MODULE" ] || MODULE=Native
ORIGINAL=; PNG=; REST=; LODS=uv
while [ $# -gt 0 ]; do
    case "$1" in
        --original) ORIGINAL=1 ;;
        --png) PNG=$2; shift ;;
        --lods) LODS=$2; shift ;;
        *) REST="$REST $1" ;;
    esac
    shift
done
if [ -n "$ORIGINAL" ]; then WBM_MODULE=$MODULE exec "$ROOT/scripts/preview.sh" "$MESH" $REST ${PNG:+--png "$PNG"}; fi

# The stand-in: the game's folder by links, but for the module, whose list of resource files gets changed.
STAND="$ROOT/build/model-preview"
rm -rf "$STAND"; mkdir -p "$STAND/Modules/$MODULE/Resource"
for e in "$GAME"/*; do [ "$(basename "$e")" = Modules ] || ln -s "$e" "$STAND/"; done
for e in "$GAME/Modules/$MODULE"/*; do
    case "$(basename "$e")" in Resource) ;; module.ini) cp "$e" "$STAND/Modules/$MODULE/" ;; *) ln -s "$e" "$STAND/Modules/$MODULE/" ;; esac
done
for e in "$GAME/Modules/$MODULE/Resource"/*; do [ -e "$e" ] && ln -s "$e" "$STAND/Modules/$MODULE/Resource/"; done
if [ -d "$SOURCE" ]; then set -- "$SOURCE"/*.glb; else set -- "$SOURCE"; fi
WBM_GAME="$STAND" "$ROOT/build/gltf_import" "$@" -o "$STAND/taken" --lods "$LODS" --install "$MODULE" | grep -v -- '->' || exit 1
if [ -n "$PNG" ]; then
    WBM_MODULE=$MODULE "$ROOT/scripts/preview.sh" "$MESH" $REST --png "${PNG%.png}_original.png" || exit 1
    WBM_GAME="$STAND" WBM_MODULE=$MODULE exec "$ROOT/scripts/preview.sh" "$MESH" $REST --png "$PNG"
fi
WBM_GAME="$STAND" WBM_MODULE=$MODULE exec "$ROOT/scripts/preview.sh" "$MESH" $REST
