#!/bin/sh
# Asks the running game to log its next full frame call by call and save it as
# a PNG in the trace directory. Usage: scripts/snap.sh [delay-seconds]
[ -n "$1" ] && sleep "$1"
pkill -USR1 -f "MacOS/Mount and Blade$" || { echo "the game is not running" >&2; exit 1; }
