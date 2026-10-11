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
            -mmacosx-version-min=10.13 -fvisibility=hidden -fno-c++-static-destructors
FRAMEWORKS := -framework OpenGL -framework CoreGraphics -framework ImageIO -framework CoreFoundation

SRCS := $(wildcard src/*.cpp)
OBJS := $(SRCS:src/%.cpp=build/%.o)
LIB  := build/libwbmetal.dylib

all: $(LIB)

$(LIB): $(OBJS)
	$(CXX) -arch x86_64 -dynamiclib -mmacosx-version-min=10.13 -o $@ $(OBJS) \
	    "$(GLEW)/GLEW" "$(SDL)/SDL2" $(FRAMEWORKS) \
	    -Wl,-rpath,"$(RES)" -install_name @rpath/libwbmetal.dylib

build/%.o: src/%.cpp src/trace.h src/interpose.h src/gl_api.def | build
	$(CXX) $(CXXFLAGS) -I"$(GLEW)/Headers" -I"$(SDL)/Headers" -c $< -o $@

build:
	mkdir -p build

clean:
	rm -rf build

.PHONY: all clean
