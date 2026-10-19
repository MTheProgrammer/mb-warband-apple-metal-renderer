#!/bin/sh
# Static analysis of the library and the tools (see `make analyze`, `make lint`, `make cppcheck`,
# `make metrics`). Needs `brew install llvm cppcheck` and `pipx install lizard`.
# Usage: scripts/check.sh compile-db|analyze|lint|cppcheck|metrics [file...]
# With files, only those are checked (analyze, lint, metrics); otherwise all of src/, tools/ and launcher/.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
LLVM=${LLVM:-$(brew --prefix llvm 2>/dev/null || echo /opt/homebrew/opt/llvm)}
SDK=$(xcrun --show-sdk-path)
what=$1
[ -n "$what" ] && shift

# The flags each file is built with, as the Makefile has them.
LIB="-arch x86_64 -std=c++17 -O0 -Wall -Wextra -Wno-deprecated-declarations -mmacosx-version-min=11.0 -fvisibility=hidden
     -Ithird_party/glew-2.1.0/include/GL -Ithird_party/SDL/include -Ithird_party/glslang -Ithird_party/SPIRV-Cross -Ibuild/deps/include"
flags_for()
{
    case $1 in
        src/*.mm) echo "$LIB -fobjc-arc" ;;
        src/*.cpp) echo "$LIB" ;;
        tools/replay.mm|tools/xlate_test.mm) echo "$LIB -fobjc-arc -Isrc" ;;
        tools/*.mm) echo "-std=c++17 -mmacosx-version-min=11.0 -fobjc-arc" ;;
        tools/*.cpp) echo "-std=c++17 -Wall -Wextra -mmacosx-version-min=11.0" ;;
        launcher/*.m) echo "-mmacosx-version-min=11.0 -Wall -Wextra -fobjc-arc" ;;
    esac
}
compiler_for() { case $1 in *.m) echo clang ;; *) echo clang++ ;; esac; }
all_files() { ls src/*.cpp src/*.mm tools/*.cpp tools/*.mm launcher/*.m; }
files=${*:-$(all_files)}

case $what in
compile-db)
    mkdir -p build
    {
        echo "["
        sep=""
        for f in $(all_files); do
            printf '%s  {"directory": "%s", "file": "%s", "command": "%s %s -isysroot %s -c %s"}' "$sep" "$ROOT" "$f" "$(compiler_for "$f")" "$(echo $(flags_for "$f"))" "$SDK" "$f"
            sep=",
"
        done
        echo
        echo "]"
    } > build/compile_commands.json
    echo "build/compile_commands.json"
    ;;
analyze)
    # The Clang Static Analyzer: follows paths through each function, for null dereferences, leaks,
    # use after free, values never set and Objective-C reference mistakes.
    mkdir -p build/analyze
    status=0
    for f in $files; do
        $(compiler_for "$f") --analyze -Xanalyzer -analyzer-output=text $(flags_for "$f") -isysroot "$SDK" \
            -o "build/analyze/$(basename "$f").plist" "$f" || status=1
    done
    exit $status
    ;;
lint)
    "$0" compile-db > /dev/null
    exec "$LLVM/bin/clang-tidy" -p build --quiet $files
    ;;
cppcheck)
    # Objective-C++ is beyond cppcheck, so it reads only the plain C++ files.
    exec cppcheck --quiet --std=c++17 --language=c++ --enable=warning,performance,portability --inline-suppr \
        --suppress=missingIncludeSystem --suppress='*:*third_party/*' -Isrc -Ithird_party/SPIRV-Cross -Ithird_party/glslang src/*.cpp tools/*.cpp
    ;;
metrics)
    # Functions longer than 150 lines or with more than 30 paths through them, the first to split.
    exec lizard --warnings_only --length 150 --CCN 30 ${*:-src tools launcher}
    ;;
*)
    sed -n '2,5p' "$0" >&2
    exit 1
    ;;
esac
