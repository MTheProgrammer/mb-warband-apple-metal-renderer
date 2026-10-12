#pragma once

// Translates one of the game's GLSL programs (vertex + fragment source as
// passed to glShaderSource) into Metal Shading Language, with the layout
// information the backend needs to feed it.

#include <map>
#include <string>
#include <vector>

struct XlUniform {
    std::string name;
    int offset = 0;       // bytes into the stage's uniform buffer
    bool is_int = false;  // int, uint or bool; otherwise float
    int components = 1;   // per column: 1 for scalars, 4 for vec4 and mat4
    int columns = 1;      // 4 for mat4
    int array_size = 0;   // 0 when not an array
    int array_stride = 0; // bytes between array elements
};

struct XlSampler {
    std::string name;
    int texture_index = 0; // Metal texture and sampler slot
};

struct XlStage {
    std::string msl;
    std::string entry;              // name of the entry function in msl
    int uniform_buffer_index = -1;  // Metal buffer slot of the uniform block, -1 if the stage has none
    int uniform_buffer_size = 0;
    std::vector<XlUniform> uniforms;
    std::vector<XlSampler> samplers;
    std::vector<int> input_locations; // vertex stage: attribute locations the shader reads
    std::vector<std::string> srgb_samplers; // fragment stage: samplers that need an sRGB view bound
};

struct XlProgram {
    bool ok = false;
    std::string log;
    XlStage vs, fs;
};

// Optional changes to what the game's shaders compute. Each one alters the
// generated code, so the set of options is part of the cache key.
struct XlOptions {
    // Replace the engine's 2x2-texel shadow filter with a 4x4-texel one.
    bool soft_shadows = false;
    // Turn alpha tests (discard below a threshold) into partial coverage of a
    // multisampled pixel, so cut-out edges are smoothed like geometry edges.
    bool smooth_alpha_test = false;
    // Map depth so that far is 0 and near is 1. With a floating-point depth
    // buffer this spreads precision evenly instead of crowding it at the near
    // plane. The backend must then reverse its depth comparisons and clear to 0.
    bool reverse_depth = false;
    // Drop the shader's own pow(colour, 2.2) on texture samples, leaving the
    // decode to the GPU through an sRGB view of the texture. The samplers this
    // was done for are listed in XlStage::srgb_samplers.
    bool srgb_textures = false;
};

// The vertex stage gets an extra uniform, wbm_yflip, that multiplies clip-space
// Y: 1 when drawing to the screen, -1 when drawing to a texture.
extern const char *const k_xl_yflip_uniform;

// With smooth_alpha_test, fragment shaders that test alpha get a uniform,
// wbm_samples, that must hold the sample count of the target being drawn to.
extern const char *const k_xl_samples_uniform;

// attrib_locations: names given to glBindAttribLocation and their indices.
XlProgram xl_translate(const std::string &vs_glsl, const std::string &fs_glsl,
                       const std::map<std::string, int> &attrib_locations, const XlOptions &options);

// The same, through a cache on disk (~/Library/Caches/warband-metal, or the
// directory in WBM_CACHE; WBM_CACHE=0 turns it off). Failed translations are
// not cached.
XlProgram xl_translate_cached(const std::string &vs_glsl, const std::string &fs_glsl,
                              const std::map<std::string, int> &attrib_locations, const XlOptions &options,
                              bool *from_cache);
