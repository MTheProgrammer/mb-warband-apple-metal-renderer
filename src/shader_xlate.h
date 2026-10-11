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
};

struct XlProgram {
    bool ok = false;
    std::string log;
    XlStage vs, fs;
};

// The vertex stage gets an extra uniform, wbm_yflip, that multiplies clip-space
// Y: 1 when drawing to the screen, -1 when drawing to a texture.
extern const char *const k_xl_yflip_uniform;

// attrib_locations: names given to glBindAttribLocation and their indices.
XlProgram xl_translate(const std::string &vs_glsl, const std::string &fs_glsl,
                       const std::map<std::string, int> &attrib_locations);
