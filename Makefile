# Builds build/libwbmetal.dylib for x86_64 (the game runs under Rosetta 2).
# The game does not have to be installed: headers for the GLEW and SDL2 it
# bundles are fetched into third_party/, and stubs/*.tbd stand in for the two
# libraries at link time.

# -fno-c++-static-destructors: the render thread keeps calling into the shim
# while the process exits, so its locks and tables must never be destroyed.
CXX      := clang++
CXXFLAGS := -arch x86_64 -std=c++17 -O2 -g -Wall -Wextra -Wno-deprecated-declarations \
            -mmacosx-version-min=11.0 -fvisibility=hidden -fno-c++-static-destructors
FRAMEWORKS := -framework OpenGL -framework CoreGraphics -framework ImageIO -framework CoreFoundation \
              -framework Metal -framework QuartzCore -framework AppKit -weak_framework MetalFX

HDRS := $(wildcard src/*.h) $(wildcard src/*.inc) src/gl_api.def
DEPS := build/deps/libdeps.a
INCLUDES := -Ithird_party/glew-2.1.0/include/GL -Ithird_party/SDL/include \
            -Ithird_party/glslang -Ithird_party/SPIRV-Cross -Ibuild/deps/include
OBJS := $(patsubst src/%.cpp,build/%.o,$(wildcard src/*.cpp)) \
        $(patsubst src/%.mm,build/%.o,$(wildcard src/*.mm))
LIB  := build/libwbmetal.dylib

all: $(LIB)

# The rpath is relative to the game's executable, so the built file works on
# any Mac and in any location. GLEW and SDL2 are linked weakly: when the game
# is started through its own launcher (as Steam does), the library is loaded
# into that launcher first, which has neither, and must load there harmlessly.
# Linked under a temporary name and renamed, so that rebuilding does not
# disturb a game that is running with the old library mapped.
$(LIB): $(OBJS) $(DEPS) stubs/GLEW.tbd stubs/SDL2.tbd
	$(CXX) -arch x86_64 -dynamiclib -mmacosx-version-min=11.0 -o $@.tmp $(OBJS) $(DEPS) \
	    -Wl,-weak_library,stubs/GLEW.tbd -Wl,-weak_library,stubs/SDL2.tbd $(FRAMEWORKS) \
	    -Wl,-rpath,@executable_path/../Resources -install_name @rpath/libwbmetal.dylib
	mv $@.tmp $@

build/%.o: src/%.cpp $(HDRS) Makefile | build $(DEPS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

build/%.o: src/%.mm $(HDRS) Makefile | build $(DEPS)
	$(CXX) $(CXXFLAGS) -fobjc-arc $(INCLUDES) -c $< -o $@

# The double-clickable launcher: a window for the options and a Play button, with the library inside the bundle.
APP := build/Warband Metal.app
VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)

build/launcher: launcher/launcher.m
	@mkdir -p build
	clang -arch arm64 -arch x86_64 -mmacosx-version-min=11.0 -O2 -Wall -Wextra -fobjc-arc launcher/launcher.m -framework AppKit -o $@

app: $(LIB) build/launcher launcher/Info.plist
	rm -rf "$(APP)"
	mkdir -p "$(APP)/Contents/MacOS" "$(APP)/Contents/Resources"
	sed 's/@VERSION@/$(VERSION)/' launcher/Info.plist > "$(APP)/Contents/Info.plist"
	cp build/launcher "$(APP)/Contents/MacOS/warband-metal"
	cp $(LIB) "$(APP)/Contents/Resources/"
	strip -S "$(APP)/Contents/Resources/libwbmetal.dylib"  # drops debug info, which names build paths

# The file to attach to a release.
dist: app
	rm -f "build/warband-metal-$(VERSION).zip"
	cd build && ditto -c -k --norsrc --noextattr --keepParent "Warband Metal.app" "warband-metal-$(VERSION).zip"
	@echo "build/warband-metal-$(VERSION).zip"

# Fetches third_party/ (glslang, SPIRV-Cross, GLEW and SDL headers) and
# compiles glslang and SPIRV-Cross, without cmake.
$(DEPS):
	scripts/build_deps.sh

# Translates every program of a recorded trace and compiles the result with Metal:
#   build/xlate_test traces/latest
build/xlate_test: tools/xlate_test.mm src/shader_xlate.cpp src/shader_xlate.h $(DEPS)
	$(CXX) -arch x86_64 -std=c++17 -O1 -mmacosx-version-min=11.0 -fobjc-arc $(INCLUDES) \
	    tools/xlate_test.mm src/shader_xlate.cpp $(DEPS) -framework Metal -framework Foundation -o $@

# Plays back a frame captured with scripts/capture.sh, without the game:
#   build/replay traces/latest/capture_000123.wbcap [--png out.png]
REPLAY_OBJS := build/metal_backend.o build/shader_xlate.o build/shader_cache.o build/gl_enums.o build/capture.o build/brf.o
build/replay: tools/replay.mm $(REPLAY_OBJS) $(DEPS)
	$(CXX) $(CXXFLAGS) -fobjc-arc $(INCLUDES) -Isrc tools/replay.mm $(REPLAY_OBJS) $(DEPS) $(FRAMEWORKS) -o $@

# Exports the game's textures and makes metalness maps for texture packs:
#   build/texture_tool pbr ~/Library/Application\ Support/warband-metal/textures/pbr
build/texture_tool: tools/texture_tool.mm tools/dds.h src/material_names.h src/brf.cpp src/brf.h
	$(CXX) -std=c++17 -O2 -mmacosx-version-min=11.0 -fobjc-arc tools/texture_tool.mm src/brf.cpp -framework Foundation -framework ImageIO -framework CoreGraphics -o $@

# Writes one of the game's models, with its rig and textures, as a glTF file for Blender:
#   build/gltf_export horse_a -o horse_a.glb
build/gltf_export: tools/gltf_export.cpp tools/dds.h src/brf.cpp src/brf.h
	@mkdir -p build
	$(CXX) -std=c++17 -O2 -Wall -Wextra -mmacosx-version-min=11.0 tools/gltf_export.cpp src/brf.cpp -lz -o $@

# The way back: models from Blender into copies of the game's resource files.
#   build/gltf_import horse_a.glb -o out
build/gltf_import: tools/gltf_import.cpp src/brf.cpp src/brf.h
	@mkdir -p build
	$(CXX) -std=c++17 -O2 -Wall -Wextra -mmacosx-version-min=11.0 tools/gltf_import.cpp src/brf.cpp -o $@

build:
	mkdir -p build

clean:
	rm -f build/*.o build/*.dylib build/xlate_test build/replay build/texture_tool build/gltf_export build/gltf_import build/launcher build/*.zip
	rm -rf "$(APP)"

distclean:
	rm -rf build

.PHONY: all app dist clean distclean
