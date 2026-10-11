#!/bin/sh
# Launches Warband with the shim injected. No game file is modified.
# Extra arguments are passed to the game. See README.md for the WBM_* variables.
set -e

ROOT=$(cd "$(dirname "$0")/.." && pwd)
GAME=${WBM_GAME:-"$HOME/Library/Application Support/Steam/steamapps/common/MountBlade Warband"}
BIN="$GAME/Mount and Blade.app/Contents/MacOS/Mount and Blade"
LIB="$ROOT/build/libwbmetal.dylib"

[ -x "$BIN" ] || { echo "game binary not found: $BIN (set WBM_GAME)" >&2; exit 1; }
[ -f "$LIB" ] || { echo "shim not built: run make" >&2; exit 1; }
pgrep -q steam_osx || echo "warning: Steam is not running; the game may refuse to start" >&2

export WBM_MODE=${WBM_MODE:-trace}
export WBM_OUT=${WBM_OUT:-"$ROOT/traces/$(date +%Y%m%d-%H%M%S)"}
mkdir -p "$WBM_OUT"
ln -sfn "$WBM_OUT" "$ROOT/traces/latest"
echo "output: $WBM_OUT"

# WBM_MODULE picks the module, as the game's launcher does: the game reads the
# name from this file. The choice stays in effect for later launches.
if [ -n "$WBM_MODULE" ]; then
    [ -d "$GAME/Modules/$WBM_MODULE" ] || { echo "no such module: $WBM_MODULE (see $GAME/Modules)" >&2; exit 1; }
    mkdir -p "$HOME/Library/Application Support/MBWarband"
    printf '%s' "$WBM_MODULE" > "$HOME/Library/Application Support/MBWarband/last_module_warband"
    echo "module: $WBM_MODULE"
fi

# steam_appid.txt is read from the working directory.
cd "$(dirname "$BIN")"
DYLD_INSERT_LIBRARIES="$LIB" exec "$BIN" "$@"
