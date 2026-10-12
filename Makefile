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
              -framework Metal -framework QuartzCore -framework AppKit

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

# The double-clickable launcher: a script, with the library inside the bundle.
APP := build/Warband Metal.app
VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)

app: $(LIB) launcher/warband-metal.sh launcher/Info.plist
	rm -rf "$(APP)"
	mkdir -p "$(APP)/Contents/MacOS" "$(APP)/Contents/Resources"
	sed 's/@VERSION@/$(VERSION)/' launcher/Info.plist > "$(APP)/Contents/Info.plist"
	cp launcher/warband-metal.sh "$(APP)/Contents/MacOS/warband-metal"
	chmod +x "$(APP)/Contents/MacOS/warband-metal"
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
REPLAY_OBJS := build/metal_backend.o build/shader_xlate.o build/shader_cache.o build/gl_enums.o build/capture.o
build/replay: tools/replay.mm $(REPLAY_OBJS) $(DEPS)
	$(CXX) $(CXXFLAGS) -fobjc-arc $(INCLUDES) -Isrc tools/replay.mm $(REPLAY_OBJS) $(DEPS) $(FRAMEWORKS) -o $@

build:
	mkdir -p build

clean:
	rm -f build/*.o build/*.dylib build/xlate_test build/replay build/*.zip
	rm -rf "$(APP)"

distclean:
	rm -rf build

.PHONY: all app dist clean distclean
