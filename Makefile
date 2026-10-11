# Builds build/libwbmetal.dylib for x86_64 (the game runs under Rosetta 2).
# Headers and link targets come from the game's own GLEW and SDL2 frameworks.
# Steam ships their top-level symlinks as plain files, so the Versions/ paths
# are used directly instead of -F/-framework.

GAME ?= $(HOME)/Library/Application Support/Steam/steamapps/common/MountBlade Warband
RES  := $(GAME)/Mount and Blade.app/Contents/Resources
GLEW := $(RES)/GLEW.framework/Versions/2.1.0
SDL  := $(RES)/SDL2.framework/Versions/A

# -fno-c++-static-destructors: the render thread keeps calling into the shim
# while the process exits, so its locks and tables must never be destroyed.
CXX      := clang++
CXXFLAGS := -arch x86_64 -std=c++17 -O2 -g -Wall -Wextra -Wno-deprecated-declarations \
            -mmacosx-version-min=11.0 -fvisibility=hidden -fno-c++-static-destructors
FRAMEWORKS := -framework OpenGL -framework CoreGraphics -framework ImageIO -framework CoreFoundation \
              -framework Metal -framework QuartzCore -framework AppKit

HDRS := $(wildcard src/*.h) src/gl_api.def
DEPS := build/deps/libdeps.a
INCLUDES := -I"$(GLEW)/Headers" -I"$(SDL)/Headers" -Ithird_party/glslang -Ithird_party/SPIRV-Cross -Ibuild/deps/include
OBJS := $(patsubst src/%.cpp,build/%.o,$(wildcard src/*.cpp)) \
        $(patsubst src/%.mm,build/%.o,$(wildcard src/*.mm))
LIB  := build/libwbmetal.dylib

all: $(LIB)

# Linked under a temporary name and renamed, so that rebuilding does not
# disturb a game that is running with the old library mapped.
$(LIB): $(OBJS) $(DEPS)
	$(CXX) -arch x86_64 -dynamiclib -mmacosx-version-min=11.0 -o $@.tmp $(OBJS) $(DEPS) \
	    "$(GLEW)/GLEW" "$(SDL)/SDL2" $(FRAMEWORKS) \
	    -Wl,-rpath,"$(RES)" -install_name @rpath/libwbmetal.dylib
	mv $@.tmp $@

build/%.o: src/%.cpp $(HDRS) Makefile | build $(DEPS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

build/%.o: src/%.mm $(HDRS) Makefile | build $(DEPS)
	$(CXX) $(CXXFLAGS) -fobjc-arc $(INCLUDES) -c $< -o $@

# glslang and SPIRV-Cross, fetched into third_party/ and compiled without cmake.
$(DEPS):
	scripts/build_deps.sh

# Translates every program of a recorded trace and compiles the result with Metal:
#   build/xlate_test traces/latest
build/xlate_test: tools/xlate_test.mm src/shader_xlate.cpp src/shader_xlate.h $(DEPS)
	$(CXX) -arch x86_64 -std=c++17 -O1 -mmacosx-version-min=11.0 -fobjc-arc $(INCLUDES) \
	    tools/xlate_test.mm src/shader_xlate.cpp $(DEPS) -framework Metal -framework Foundation -o $@

build:
	mkdir -p build

clean:
	rm -f build/*.o build/*.dylib build/xlate_test

distclean:
	rm -rf build

.PHONY: all clean distclean
