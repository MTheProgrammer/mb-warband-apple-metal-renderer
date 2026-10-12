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
    // Instead of that: shadows whose softness grows with the distance between
    // the surface and whatever shades it (percentage-closer soft shadows).
    bool pcss_shadows = false;
    // Also sample a second shadow map that the backend draws itself each frame
    // with everything visible in it, characters included. Adds the sampler and
    // uniforms described at k_xl_dyn_sampler.
    bool dynamic_shadows = false;
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
    // Replace the engine's Blinn-Phong highlight with a microfacet one (GGX
    // distribution, Smith shadowing, Schlick Fresnel), keeping each material's
    // highlight strength and deriving roughness from its shininess exponent.
    bool pbr_specular = false;
    // On top of pbr_specular: estimate metalness from the specular mask, darken
    // the base colour of metals, and add a reflection of a sky-and-ground
    // environment built from the scene's own light colours.
    bool pbr_environment = false;
    // On top of pbr_environment: reflect a captured image of the scene's sky
    // instead of a two-colour gradient. Adds the sampler wbm_sky, which must be
    // given the atlas described at k_xl_sky_sampler.
    bool pbr_sky = false;
    // In the engine's final post-processing shader: replace its tone-mapping
    // operator with a filmic curve.
    bool filmic_tonemap = false;
    // In the same shader: let the brightest parts of the image exceed display
    // white, for an extended-range (HDR) drawable. Adds the wbm_headroom uniform.
    bool hdr_output = false;
};

// The vertex stage gets an extra uniform, wbm_yflip, that multiplies clip-space
// Y: 1 when drawing to the screen, -1 when drawing to a texture.
extern const char *const k_xl_yflip_uniform;

// With smooth_alpha_test, fragment shaders that test alpha get a uniform,
// wbm_samples, that must hold the sample count of the target being drawn to.
extern const char *const k_xl_samples_uniform;

// With hdr_output, the final shader gets a uniform, wbm_headroom: how many
// times brighter than display white the screen can currently show (1 = none).
extern const char *const k_xl_headroom_uniform;

// With hdr_output, every other fragment shader gets a uniform, wbm_limit: the
// largest colour value it may write. The backend sets it to 1 when drawing to
// the screen and to a very large number otherwise.
extern const char *const k_xl_limit_uniform;

// With pbr_sky: a sampler for the sky atlas, six square faces side by side in
// the order +X, -X, +Y, -Y, +Z, -Z (Z is up). Face f looks along its axis with
// up = +Z for the four side faces and +Y for the top and bottom; a direction d
// lands at (dot(d, right), dot(d, up)) / dot(d, forward) * 0.5 + 0.5 within
// the face, with right = cross(forward, up) and v = 0 on the texture's first
// row. Colours are stored as the sky shader wrote them, gamma-encoded.
extern const char *const k_xl_sky_sampler;

// With dynamic_shadows, shaders that receive shadows get:
//   wbm_dyn_map   a square map, k_xl_dyn_size texels a side, holding for a window
//                 of the engine's shadow space the depth nearest the sun, in the
//                 units of the z the engine's shaders compare against;
//   wbm_dyn       xy: the window's centre in the engine's shadow-map coordinates,
//                 zw: 1 / its width and height in the same; z = 0 means no map;
//   wbm_dyn_bias  the depth by which a surface is moved towards the sun before
//                 the comparison.
extern const char *const k_xl_dyn_sampler;
extern const char *const k_xl_dyn_window_uniform;
extern const char *const k_xl_dyn_bias_uniform;
enum { k_xl_dyn_size = 4096 };
enum { k_xl_sky_face_size = 128 };

// attrib_locations: names given to glBindAttribLocation and their indices.
XlProgram xl_translate(const std::string &vs_glsl, const std::string &fs_glsl,
                       const std::map<std::string, int> &attrib_locations, const XlOptions &options);

// The same, through a cache on disk (~/Library/Caches/warband-metal, or the
// directory in WBM_CACHE; WBM_CACHE=0 turns it off). Failed translations are
// not cached.
XlProgram xl_translate_cached(const std::string &vs_glsl, const std::string &fs_glsl,
                              const std::map<std::string, int> &attrib_locations, const XlOptions &options,
                              bool *from_cache);
