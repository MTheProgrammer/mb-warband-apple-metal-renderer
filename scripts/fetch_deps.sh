#!/bin/sh
# Fetches the shader-translation libraries at pinned versions into third_party/.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT/third_party"
cd "$ROOT/third_party"
fetch() { # name url tag
    [ -d "$1" ] && return 0
    git clone -q --depth 1 --branch "$3" -c advice.detachedHead=false "$2" "$1"
}
fetch glslang https://github.com/KhronosGroup/glslang.git 15.1.0
fetch SPIRV-Cross https://github.com/KhronosGroup/SPIRV-Cross.git vulkan-sdk-1.4.304.0
