#!/bin/sh
# Fetches the build's dependencies at pinned versions into third_party/.
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

# Headers of the two libraries the game bundles and the shim hooks: SDL 2.0.8
# and GLEW 2.1.0. Only headers are used; at run time the game's own copies are.
fetch SDL https://github.com/libsdl-org/SDL.git release-2.0.8
if [ ! -f glew-2.1.0/include/GL/glew.h ]; then
    # glew.h is generated, so it is in the release archive but not in GLEW's git.
    curl -fsSL https://github.com/nigels-com/glew/releases/download/glew-2.1.0/glew-2.1.0.tgz | tar xz glew-2.1.0/include glew-2.1.0/LICENSE.txt
fi
