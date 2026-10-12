#!/bin/sh
# Asks the running game (Metal mode) to capture its next frame, with everything
# the frame depends on, into the run's trace directory as capture_<n>.wbcap.
# build/replay shows such a file without the game. Usage: scripts/capture.sh [delay-seconds]
[ -n "$1" ] && sleep "$1"
pkill -USR2 -f "MacOS/Mount and Blade$" || { echo "the game is not running" >&2; exit 1; }
