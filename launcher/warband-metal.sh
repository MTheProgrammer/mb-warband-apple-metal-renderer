#!/bin/sh
# Warband Metal launcher: starts Mount & Blade: Warband with the Metal renderer
# library from this app bundle loaded into it. No game file is modified.

APP_SUPPORT="$HOME/Library/Application Support/warband-metal"
SAVED_PATH="$APP_SUPPORT/game-path"
LIB="$(cd "$(dirname "$0")/../Resources" && pwd)/libwbmetal.dylib"
DEFAULT_GAME="$HOME/Library/Application Support/Steam/steamapps/common/MountBlade Warband"

binary_in() { echo "$1/Mount and Blade.app/Contents/MacOS/Mount and Blade"; }

alert() {
    osascript -e 'on run argv' -e 'display dialog (item 1 of argv) with title "Warband Metal" buttons {"OK"} default button 1 with icon caution' \
        -e 'end run' "$1" >/dev/null 2>&1
}

# WBM_GAME, then the Steam default, then the folder chosen on an earlier run.
GAME=""
for candidate in "$WBM_GAME" "$DEFAULT_GAME" "$(cat "$SAVED_PATH" 2>/dev/null)"; do
    if [ -n "$candidate" ] && [ -x "$(binary_in "$candidate")" ]; then
        GAME="$candidate"
        break
    fi
done

if [ -z "$GAME" ]; then
    CHOSEN=$(osascript -e 'POSIX path of (choose folder with prompt "Warband was not found in the default Steam library. Select the \"MountBlade Warband\" folder (the one that contains \"Mount and Blade.app\").")' 2>/dev/null) || exit 0
    CHOSEN=${CHOSEN%/}
    if [ ! -x "$(binary_in "$CHOSEN")" ]; then
        alert "That folder does not contain \"Mount and Blade.app\". Nothing was started."
        exit 1
    fi
    mkdir -p "$APP_SUPPORT"
    printf '%s' "$CHOSEN" > "$SAVED_PATH"
    GAME="$CHOSEN"
fi

if [ ! -f "$LIB" ]; then
    alert "The renderer library is missing from this copy of Warband Metal. Download the app again."
    exit 1
fi

# steam_appid.txt is read from the working directory.
cd "$(dirname "$(binary_in "$GAME")")" || exit 1
DYLD_INSERT_LIBRARIES="$LIB" exec "$(binary_in "$GAME")" "$@"
