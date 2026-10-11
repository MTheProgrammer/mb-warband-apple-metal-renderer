#!/bin/sh
# Compiles glslang and SPIRV-Cross for x86_64 into build/deps/libdeps.a.
# Their sources are compiled directly, so cmake is not needed.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
"$ROOT/scripts/fetch_deps.sh"
TP="$ROOT/third_party"
OUT="$ROOT/build/deps"
mkdir -p "$OUT/obj" "$OUT/include/glslang"

python3 "$TP/glslang/build_info.py" "$TP/glslang" -i "$TP/glslang/build_info.h.tmpl" -o "$OUT/include/glslang/build_info.h"

G="$TP/glslang"
SOURCES=$(ls "$G"/glslang/MachineIndependent/*.cpp "$G"/glslang/MachineIndependent/preprocessor/*.cpp \
    "$G"/glslang/GenericCodeGen/*.cpp "$G"/glslang/OSDependent/Unix/ossource.cpp \
    "$G"/glslang/ResourceLimits/ResourceLimits.cpp \
    "$G"/SPIRV/GlslangToSpv.cpp "$G"/SPIRV/InReadableOrder.cpp "$G"/SPIRV/Logger.cpp "$G"/SPIRV/SpvBuilder.cpp \
    "$G"/SPIRV/SpvPostProcess.cpp "$G"/SPIRV/SpvTools.cpp "$G"/SPIRV/doc.cpp \
    "$TP"/SPIRV-Cross/spirv_cross.cpp "$TP"/SPIRV-Cross/spirv_cfg.cpp "$TP"/SPIRV-Cross/spirv_cross_parsed_ir.cpp \
    "$TP"/SPIRV-Cross/spirv_parser.cpp "$TP"/SPIRV-Cross/spirv_cross_util.cpp "$TP"/SPIRV-Cross/spirv_glsl.cpp \
    "$TP"/SPIRV-Cross/spirv_msl.cpp)

export OUT G
echo "$SOURCES" | tr ' ' '\n' | xargs -n 1 -P "$(sysctl -n hw.ncpu)" sh -c '
    o="$OUT/obj/$(basename "$1" .cpp).o"
    [ "$o" -nt "$1" ] || clang++ -arch x86_64 -std=c++17 -O2 -w -mmacosx-version-min=11.0 -fvisibility=hidden \
        -I"$G" -I"$OUT/include" -c "$1" -o "$o"' _
rm -f "$OUT/libdeps.a"
ar rcs "$OUT/libdeps.a" "$OUT"/obj/*.o
echo "built $OUT/libdeps.a"
