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
    // In its place, with one sample a pixel: the alpha test against noise, for the
    // temporal scaler to average (adds the wbm_hash_frame uniform). See smooth_alpha_tests.
    bool alpha_hash = false;
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
    // Colour the ambient light by direction: tint what the engine adds as flat
    // ambient and sky light with the colour of the captured sky on that side,
    // and add sunlight bounced off the ground to surfaces that face downwards
    // or sideways. Adds the wbm_amb uniform (see k_xl_ambient_uniform).
    bool sky_ambient = false;
    // On top of pbr_environment: take metalness, roughness and highlight
    // strength per material class from the backend, which knows what each
    // texture is. Adds the wbm_material uniform (see k_xl_material_uniform).
    bool materials = false;
    // Skin that scatters light: in shaders that light per pixel, the sun's
    // diffuse term can wrap a little way round the terminator, farthest in red,
    // with the red taken from the smooth surface instead of the normal map's
    // pores, and the edge of a shadow tinted the same way. Whether it does is
    // up to the wbm_skin uniform, which the backend sets per draw.
    bool skin_scattering = false;
    // A horse's coat: in the engine's shaders for horses, where a map says the
    // surface is coat (the blue channel of the texture's _pbr map, see
    // k_xl_pbr_sampler), light wraps softly round the body, a velvety rim shows
    // where the coat is seen edge-on, and the engine's two hard streaks of
    // highlight are toned down. Adds the wbm_fur_map sampler and wbm_fur
    // uniform (x: strength, 0 without a map).
    bool fur = false;
    // People's hair as strands: the engine's hair shader learns to lay out
    // ribbons the backend draws over the hair mesh of characters near the
    // camera (see add_hair_strands). Adds the wbm_shell uniform to both stages.
    bool hair_strands = false;
    // Water: the engine's water shaders get the sun's glitter on the waves,
    // a paler, greener colour where the water is shallow, and foam along the
    // shore (the last two where the game's depth effects are on).
    bool water = false;
    // Parallax: in the engine's normal-mapped shaders the texture is looked up
    // where the line of sight meets a surface with depth, the texture's own
    // brightness standing for height (mortar and cracks are dark), so that
    // stones and bricks shift against each other as the eye moves. Adds the
    // wbm_parallax uniform (x: depth in texture coordinates, 0 for none).
    bool parallax = false;
    // Eyes that catch the light: faces have their eyes painted on, at the same
    // place in every face texture, since all heads share one layout. There the
    // engine's skin shader gets a small sharp highlight from the sun and a
    // faint reflection of the sky, as from a wet, bulging surface.
    bool eyes = false;
    // Leaves lit from behind: where skin_scattering rewrites the sun's diffuse
    // term, foliage also lets sunlight through to the side turned from the sun.
    // Either option brings the rewrite and the wbm_skin uniform in.
    bool foliage_backlight = false;
    // Wind: vertex shaders move what the backend marks as foliage, more the
    // higher up it is. Adds the wbm_wind uniform (see k_xl_wind_uniform).
    bool wind = false;
    // With wind: the shaders that sway also write, into a second colour target,
    // how far each of their pixels moved across the screen since the last frame
    // (xy, in clip space), and its distance (z). Adds the vertex uniforms
    // wbm_reproject and wbm_wind_before. For temporal upscaling.
    bool motion_vectors = false;
    // Shadows from the engine's point lights: shaders that are lit by them look
    // their surface up in a cube of depths per light, which the backend draws.
    // Adds the sampler and uniforms described at k_xl_torch_sampler.
    bool torch_shadows = false;
    // On top of torch_shadows: in the engine's standard shaders, the point
    // lights that are weighed at the vertices are weighed at each pixel
    // instead, with the normal map where there is one, and each gets a
    // highlight of the kind the sun's has.
    bool torch_lighting = false;
    // For tuning: shaders with an environment reflection show their estimated
    // metalness in place of their colour, white for metal and black for none.
    bool debug_materials = false;
    // In the engine's final post-processing shader: replace its tone-mapping
    // operator with a filmic curve.
    bool filmic_tonemap = false;
    // In the same shader: let the brightest parts of the image exceed display
    // white, for an extended-range (HDR) drawable. Adds the wbm_headroom uniform.
    bool hdr_output = false;
    // Colour grading in the final shader, after the tone map: contrast,
    // saturation and warmth, from the wbm_grade uniform (x, y, z; each 0 for
    // no change; w is spare).
    bool grade = false;
};

// The vertex stage gets an extra uniform, wbm_yflip, that multiplies clip-space
// Y: 1 when drawing to the screen, -1 when drawing to a texture.
extern const char *const k_xl_yflip_uniform;

// And wbm_jitter, added to clip-space x and y (times w) after the flip: zero
// but while the scene is drawn for temporal upscaling, when it shifts the whole
// picture by a different fraction of a pixel each frame (see apply_temporal_upscale
// in metal_backend.mm).
extern const char *const k_xl_jitter_uniform;

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

// With sky_ambient, shaders that add ambient light get wbm_amb, six vec4:
//   [0..4] xyz  the sky's colour towards +X, -X, +Y, -Y and +Z (Z is up),
//               linear, divided by the mean brightness of the five;
//   [5]    xyz  light bounced off the ground, to add to a surface facing
//               straight down; w: how far to apply the tint, 0 for not at all.
extern const char *const k_xl_ambient_uniform;

// With materials, shaders with an environment reflection get wbm_material:
//   x  factor on the estimated metalness (0: never a metal);
//   y  factor on the squared roughness derived from the material's shininess;
//   z  factor on the highlight and reflection strength;
//   w  0 when the material is unknown, and the rest is then ignored; otherwise
//      1 + the metalness of the texture's grey parts (so 1 for a non-metal).
extern const char *const k_xl_material_uniform;

// With materials as well: a sampler, wbm_pbr_map, for a map laid out like the
// surface's base texture with metalness in red and roughness in green, and a
// uniform, wbm_pbr, whose x is 1 when the surface has such a map and 0 when it
// has not (the sampler then holds anything). A map overrides the estimates.
extern const char *const k_xl_pbr_sampler;
extern const char *const k_xl_pbr_uniform;

// With skin_scattering: wbm_skin. x: 1 when the surface being drawn is skin,
// 0 otherwise; y: metres per unit of shadow-map depth while the per-frame
// shadow map's fine window is there to measure thickness with, else 0;
// z: strength of the light coming through thin parts; w: above 0 when the
// surface is foliage, and then how much sunlight comes through a leaf.
extern const char *const k_xl_skin_uniform;

// With wind: wbm_wind, in the vertex stage. x: time in seconds; y: strength, 0
// for whatever is not to move; z: 0 when positions are in the object's own
// space with its foot at height 0, 1 when the vertex colour's alpha holds the
// height above the plant's foot instead (batches placed in the world), and is
// to be read as full by the rest of the shader; w: a phase for this object, so
// that neighbours do not move as one.
extern const char *const k_xl_wind_uniform;

// With pbr_sky: a sampler for the sky atlas, six square faces side by side in
// the order +X, -X, +Y, -Y, +Z, -Z (Z is up). Face f looks along its axis with
// up = +Z for the four side faces and +Y for the top and bottom; a direction d
// lands at (dot(d, right), dot(d, up)) / dot(d, forward) * 0.5 + 0.5 within
// the face, with right = cross(forward, up) and v = 0 on the texture's first
// row. Colours are stored as the sky shader wrote them, gamma-encoded.
extern const char *const k_xl_sky_sampler;

// With dynamic_shadows, shaders that receive shadows get:
//   wbm_dyn_map   a map twice as wide as high, k_xl_dyn_size texels high, of two
//                 square halves. Each holds, for a window of the engine's shadow
//                 space, the depth nearest the sun, in the units of the z the
//                 engine's shaders compare against. The left half is a wide
//                 window, the right half a small one around the camera;
//   wbm_dyn       the wide window. xy: its centre in the engine's shadow-map
//                 coordinates, zw: 1 / its width and height in the same; z = 0
//                 means there is no map;
//   wbm_dyn_near  the same for the small window; z = 0 means there is none;
//   wbm_dyn_bias  the depth by which a surface is moved towards the sun before
//                 the comparison.
extern const char *const k_xl_dyn_sampler;
extern const char *const k_xl_dyn_window_uniform;
extern const char *const k_xl_dyn_near_uniform;
extern const char *const k_xl_dyn_bias_uniform;

// With torch_shadows, shaders lit by the engine's point lights get:
//   wbm_torch_map    an array of cube maps of depth, one cube per light, to be
//                    sampled with a comparison (less or equal passes). Seen from
//                    the light, a face holds y + z / (distance along its axis),
//                    y and z being those of wbm_torch;
//   wbm_torch_light  four vec4, one per light as the engine numbers them. xyz:
//                    the light's place in the world; w: 1 + the number of its
//                    cube, 0 when it has none;
//   wbm_torch        x: strength of the shadows, 0 while there are no maps;
//                    y, z: see above; w: 2 / a face's width in texels.
extern const char *const k_xl_torch_sampler;
extern const char *const k_xl_torch_uniform;
extern const char *const k_xl_torch_light_uniform;
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
