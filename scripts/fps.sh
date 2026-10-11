#!/bin/sh
# Summarises frame rates of one or more runs from their fps.log.
# Usage: scripts/fps.sh [trace-dir ...]        (default: traces/latest)
# Seconds at the display's cap say nothing about the renderer, so the summary
# also reports the seconds that fell below it.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
[ $# -eq 0 ] && set -- "$ROOT/traces/latest"
for dir in "$@"; do
    [ -f "$dir/fps.log" ] || { echo "$dir: no fps.log" >&2; continue; }
    mode=$(sed -n 's/^mode: //p' "$dir/summary.txt" 2>/dev/null)
    awk '{print $5}' "$dir/fps.log" | sort -n | awk -v name="$(basename "$(cd "$dir" && pwd -P)")" -v mode="$mode" '
        { v[NR] = $1; sum += $1; if ($1 < 110) { low++; lowsum += $1 } }
        END {
            if (!NR) exit
            printf "%s  [%s]\n", name, mode
            printf "  %d s recorded: average %.0f fps, median %.0f, slowest second %.0f, slowest 10%% of seconds <= %.0f\n",
                   NR, sum / NR, v[int((NR + 1) / 2)], v[1], v[int(NR * 0.1) + 1]
            if (low) printf "  %d s below 110 fps, averaging %.0f fps\n", low, lowsum / low
            else printf "  never below 110 fps\n"
        }'
done
