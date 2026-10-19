// GLSL 1.20 (as written by the engine's shader optimiser) -> GLSL 4.50 text ->
// SPIR-V (glslang) -> MSL (SPIRV-Cross).

#include "shader_xlate.h"

#include <SPIRV/GlslangToSpv.h>
#include <glslang/Public/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <spirv_msl.hpp>

#include <algorithm>
#include <regex>
#include <set>
#include <sstream>

const char *const k_xl_yflip_uniform = "wbm_yflip";
const char *const k_xl_jitter_uniform = "wbm_jitter";
const char *const k_xl_samples_uniform = "wbm_samples";
const char *const k_xl_headroom_uniform = "wbm_headroom";
const char *const k_xl_limit_uniform = "wbm_limit";
const char *const k_xl_sky_sampler = "wbm_sky";
const char *const k_xl_ambient_uniform = "wbm_amb";
const char *const k_xl_material_uniform = "wbm_material";
const char *const k_xl_pbr_sampler = "wbm_pbr_map";
const char *const k_xl_pbr_uniform = "wbm_pbr";
const char *const k_xl_skin_uniform = "wbm_skin";
const char *const k_xl_wind_uniform = "wbm_wind";
const char *const k_xl_dyn_sampler = "wbm_dyn_map";
const char *const k_xl_dyn_window_uniform = "wbm_dyn";
const char *const k_xl_dyn_near_uniform = "wbm_dyn_near";
const char *const k_xl_dyn_bias_uniform = "wbm_dyn_bias";
const char *const k_xl_torch_sampler = "wbm_torch_map";
const char *const k_xl_torch_uniform = "wbm_torch";
const char *const k_xl_torch_light_uniform = "wbm_torch_light";

namespace {

struct Varyings {
    std::map<std::string, int> location;
    int next = 0;
    // From add_torch_vertex to add_torch_fragment: the varying the vertex shader puts the point lights' sum in
    // (alone, or with the rest of the light when torch_shared), and those it puts their highlights in.
    std::string torch_total;
    bool torch_shared = false;
    std::vector<std::string> torch_gloss;
    // With torch_lighting: 1 when the vertex shader passes on the surface's own axes (wbm_torch_tbn), for a
    // fragment shader that works in them, 2 when it works in the world's, 0 when it does neither.
    int torch_space = 0;
};

// Locations a declaration of this type occupies.
int location_count(const std::string &type, const std::string &array)
{
    int n = type == "mat4" ? 4 : type == "mat3" ? 3 : type == "mat2" ? 2 : 1;
    if (!array.empty()) n *= std::max(1, atoi(array.c_str() + 1));
    return n;
}

std::string replace_word(const std::string &src, const char *word, const char *with)
{
    return std::regex_replace(src, std::regex(std::string("\\b") + word + "\\b"), with);
}

// The engine filters its shadow map with four samples, blended by where the
// pixel falls between them. This does the same over a 4x4 block of texels: the
// average of that 2x2 filter at nine positions one texel apart, which gives a
// soft edge about three texels wide instead of one.
//
// Samples away from the centre are compared against the receiving surface's
// own depth at that spot (estimated from screen-space derivatives), not the
// centre's. Without that, a surface sloping away from the sun would shadow
// itself. The estimate is clamped, since it grows without bound where the
// surface is nearly parallel to the light.
const char k_soft_shadow_function[] = R"(
float wbm_shadow(sampler2D map, vec3 coord, float size)
{
    vec2 p = coord.xy * size;
    vec2 base = floor(p);
    vec2 f = p - base;
    vec3 dx = dFdx(coord), dy = dFdy(coord);
    float det = dx.x * dy.y - dx.y * dy.x;
    vec2 slope = vec2(0.0); // change of the surface's depth per shadow-map texel
    if (abs(det) > 1e-14) slope = vec2(dy.y * dx.z - dx.y * dy.z, dx.x * dy.z - dy.x * dx.z) / (det * size);
    slope = clamp(slope, vec2(-0.002), vec2(0.002));
    vec4 wx = vec4(1.0 - f.x, 1.0, 1.0, f.x);
    vec4 wy = vec4(1.0 - f.y, 1.0, 1.0, f.y);
    float lit = 0.0;
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            vec2 texel = base + vec2(float(i) - 0.5, float(j) - 0.5); // a texel centre
            float surface = coord.z + dot(slope, texel - p);
            float nearest = textureLod(map, texel / size, 0.0).x;
            lit += wx[i] * wy[j] * (nearest < surface ? 0.0 : 1.0);
        }
    }
    return lit / 9.0;
}
)";

// Percentage-closer soft shadows. A real shadow is sharp where the object
// casting it touches the ground and blurs with distance, because the sun is a
// disc and not a point. Three steps per pixel:
//
//   1. Look around in the shadow map for anything nearer to the sun than this
//      surface, and average how near it is.
//   2. The further away that is, the wider the penumbra. The sun's projection
//      is orthographic, so the width is simply the gap times the tangent of the
//      sun's angular radius; the matrix gives the map's size in world units,
//      which turns that into texels. The radius used (about 2.3 degrees) is
//      several times the real sun's, as is usual, so that the effect reads.
//   3. Filter the shadow over a disc of that width.
//
// Pixels where step 1 finds nothing, or nothing but occluders, are fully lit
// or fully shadowed and skip step 3. The sample pattern is rotated per pixel,
// which trades banding for fine noise.
const char k_pcss_shadow_function[] = R"(
const vec2 wbm_disc[16] = vec2[16](
    vec2(-0.9420, -0.3991), vec2(0.9456, -0.7689), vec2(-0.0942, -0.9294), vec2(0.3450, 0.2939),
    vec2(-0.9159, 0.4577), vec2(-0.8154, -0.8791), vec2(-0.3828, 0.2768), vec2(0.9748, 0.7565),
    vec2(0.4432, -0.9751), vec2(0.5374, -0.4737), vec2(-0.2650, -0.4189), vec2(0.7920, 0.1909),
    vec2(-0.2419, 0.9971), vec2(-0.8141, 0.9144), vec2(0.1998, 0.7864), vec2(0.1438, -0.1410));

float wbm_shadow_pcss(sampler2D map, vec3 coord, float size, mat4 sun)
{
    vec3 dx = dFdx(coord), dy = dFdy(coord);
    float det = dx.x * dy.y - dx.y * dy.x;
    vec2 slope = vec2(0.0); // change of the surface's depth per shadow-map texel
    if (abs(det) > 1e-14) slope = vec2(dy.y * dx.z - dx.y * dy.z, dx.x * dy.z - dy.x * dx.z) / (det * size);
    slope = clamp(slope, vec2(-0.002), vec2(0.002));

    float angle = 6.2831853 * fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    mat2 rotate = mat2(cos(angle), sin(angle), -sin(angle), cos(angle));

    const float search = 8.0; // texels
    float blockers = 0.0, blocker_depth = 0.0;
    for (int i = 0; i < 16; i++) {
        vec2 o = rotate * wbm_disc[i] * search;
        float nearest = textureLod(map, coord.xy + o / size, 0.0).x;
        if (nearest < coord.z + dot(slope, o)) {
            blockers += 1.0;
            blocker_depth += nearest;
        }
    }
    if (blockers < 0.5) return 1.0;

    // Texels of penumbra per unit of depth. The matrix maps world space to map
    // coordinates (0..1 across the map) and depth, so the lengths of its rows
    // are map widths per metre and depth units per metre:
    //   metres of gap = gap / |row 2|;  texels = metres * tan(sun radius) * |row 0| * size.
    float texels_per_depth = 0.04 * size * length(vec3(sun[0][0], sun[1][0], sun[2][0])) /
                             max(length(vec3(sun[0][2], sun[1][2], sun[2][2])), 1e-6);
    float gap = max(coord.z - blocker_depth / blockers, 0.0);
    float radius = clamp(gap * texels_per_depth, 1.5, 12.0);
    if (blockers > 15.5 && radius < search * 0.5) return 0.0;

    float lit = 0.0;
    for (int i = 0; i < 16; i++) {
        vec2 o = rotate * wbm_disc[i].yx * radius;
        lit += textureLod(map, coord.xy + o / size, 0.0).x < coord.z + dot(slope, o) ? 0.0 : 1.0;
    }
    return lit / 16.0;
}
)";

// The engine's shadow map holds scenery only, drawn once per scene; characters
// get a blurred decal on the ground and shadow nothing else. The backend draws
// a second map every frame with everything visible in it (see k_xl_dyn_sampler)
// covering a window around the camera at much finer resolution, in the same
// space as the engine's, so that the shadow coordinate a shader already has
// indexes both. This samples it: 12 taps on a small rotated disc, each compared
// against the receiving surface's own depth at that tap, and fading to "lit"
// at the window's edge.
const char k_dynamic_shadow_function[] = R"(
uniform sampler2D wbm_dyn_map;
uniform vec4 wbm_dyn;
uniform vec4 wbm_dyn_near;
uniform float wbm_dyn_bias;
const vec2 wbm_dyn_disc[12] = vec2[12](
    vec2(-0.326, -0.406), vec2(-0.840, -0.074), vec2(-0.696, 0.457), vec2(-0.203, 0.621),
    vec2(0.962, -0.195), vec2(0.473, -0.480), vec2(0.519, 0.767), vec2(0.185, -0.893),
    vec2(0.507, 0.064), vec2(0.896, 0.412), vec2(-0.322, -0.933), vec2(-0.792, -0.598));

// One of the two windows, which lie side by side in the map: column 0 is the
// wide one, column 1 the fine one around the camera. `inside` comes back as 0
// outside the window, rising to 1 a little way in.
float wbm_dyn_cascade(vec3 coord, vec3 cdx, vec3 cdy, vec4 window, float column, float spread, float bias_scale, out float inside)
{
    vec3 q = vec3((coord.xy - window.xy) * window.zw + 0.5, coord.z);
    vec2 margin = min(q.xy, 1.0 - q.xy);
    inside = clamp(min(margin.x, margin.y) * 40.0, 0.0, 1.0);
    if (inside <= 0.0) return 1.0;
    const float size = DYN_SIZE;
    vec3 dx = vec3(cdx.xy * window.zw, cdx.z), dy = vec3(cdy.xy * window.zw, cdy.z);
    float det = dx.x * dy.y - dx.y * dy.x;
    vec2 slope = vec2(0.0); // change of the surface's depth per texel of this window
    if (abs(det) > 1e-18) slope = vec2(dy.y * dx.z - dx.y * dy.z, dx.x * dy.z - dy.x * dx.z) / (det * size);
    slope = clamp(slope, vec2(-wbm_dyn_bias), vec2(wbm_dyn_bias));
    // Far away a pixel spans several texels of the map. The filter is widened
    // to the pixel's footprint there, or the few texels it happened to hit
    // would change from frame to frame and the shadow would flicker.
    float footprint = clamp(max(length(dx.xy), length(dy.xy)) * size, 0.0, 12.0);
    // Very close up it is the other way about: a pixel is a small fraction of a
    // texel, the differences the slope is worked out from are a few of the
    // coordinate's last bits, and the slope is noise that shows as speckles.
    slope *= smoothstep(0.04, 0.15, footprint);
    float radius = max(spread, footprint * 0.75);
    float bias = wbm_dyn_bias * bias_scale * (1.0 + 0.25 * footprint);
    float angle = 6.2831853 * fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    mat2 rotate = mat2(cos(angle), sin(angle), -sin(angle), cos(angle));
    float lit = 0.0;
    for (int i = 0; i < 12; i++) {
        vec2 o = rotate * wbm_dyn_disc[i] * radius;
        vec2 t = clamp(q.xy + o / size, vec2(1.0 / size), vec2(1.0 - 1.0 / size)); // stay in this window's half
        // The four texels round the place are each asked and the answers blended by
        // how near each is: seen close, where a texel covers many pixels, the
        // shadow's edge is then a slope and not steps with a dither on them.
        vec2 at = vec2((t.x + column) * 0.5, t.y);
        vec2 f = fract(at * vec2(2.0 * size, size) - 0.5);
        vec4 lights = step(vec4(q.z - bias + dot(slope, o)), textureGather(wbm_dyn_map, at, 0));
        lit += mix(mix(lights.w, lights.z, f.x), mix(lights.x, lights.y, f.x), f.y);
    }
    return lit / 12.0;
}

// How far behind the surface nearest the sun this point lies, in the depth's
// own units, from the fine window; -1 where that window does not reach.
float wbm_dyn_thickness(vec3 coord)
{
    if (wbm_dyn_near.z <= 0.0) return -1.0;
    vec3 q = vec3((coord.xy - wbm_dyn_near.xy) * wbm_dyn_near.zw + 0.5, coord.z);
    vec2 margin = min(q.xy, 1.0 - q.xy);
    if (min(margin.x, margin.y) <= 0.02) return -1.0;
    const float size = DYN_SIZE;
    float sum = 0.0;
    for (int i = 0; i < 4; i++) {
        vec2 t = q.xy + vec2(i < 2 ? -0.7 : 0.7, (i & 1) == 0 ? -0.7 : 0.7) / size;
        float nearest = textureLod(wbm_dyn_map, vec2((t.x + 1.0) * 0.5, t.y), 0.0).x;
        if (nearest > 1.0e8) return -1.0; // nothing was drawn there: no telling
        sum += max(q.z - nearest, 0.0);
    }
    return sum * 0.25;
}

float wbm_dynamic_shadow(vec3 coord)
{
    if (wbm_dyn.z <= 0.0) return 1.0;
    vec3 cdx = dFdx(coord), cdy = dFdy(coord);
    float in_near = 0.0, near = 1.0;
    if (wbm_dyn_near.z > 0.0) near = wbm_dyn_cascade(coord, cdx, cdy, wbm_dyn_near, 1.0, 1.5, 0.4, in_near);
    if (in_near >= 1.0) return near;
    float in_far;
    float far = wbm_dyn_cascade(coord, cdx, cdy, wbm_dyn, 0.0, 1.8, 1.0, in_far);
    return mix(mix(1.0, far, in_far), near, in_near);
}
)";

// Rewrites the engine's shadow filter. The block is generated code and has one
// shape in every shader that has it; a shader where the shape is not found is
// left as it is. filter: 0 keeps the engine's four samples, 1 uses wbm_shadow,
// 2 wbm_shadow_pcss. dynamic: also take the darker of that and wbm_dynamic_shadow.
bool rewrite_shadows(std::string &fs, int filter, bool dynamic)
{
    static const std::regex blend(
        R"((\w+) = clamp \(mix \(mix \(\w+, \w+, \w+\.x\), mix \(\w+, \w+, \w+\.x\), \w+\.y\), 0\.0, 1\.0\);)");
    for (const char *name : {"uniform sampler2D shadowmap_texture;", "uniform float fShadowMapSize;", "ShadowTexCoord"})
        if (fs.find(name) == std::string::npos) return false;
    std::smatch m;
    if (!std::regex_search(fs, m, blend)) return false;
    std::string var = m[1], statement = m.str();
    if (filter == 1) statement = var + " = wbm_shadow(shadowmap_texture, ShadowTexCoord.xyz, fShadowMapSize);";
    if (filter == 2) statement = var + " = wbm_shadow_pcss(shadowmap_texture, ShadowTexCoord.xyz, fShadowMapSize, matSunViewProj);";
    if (dynamic) statement += " " + var + " = min(" + var + ", wbm_dynamic_shadow(ShadowTexCoord.xyz));";
    fs = std::string(m.prefix()) + statement + std::string(m.suffix());
    return true;
}

// The engine cuts shapes out of textures (leaves, hair, fur, fences) with an
// alpha test: the pixel is dropped when alpha is below a threshold. That edge
// is not smoothed by multisampling, because the shader runs once per pixel.
//
// This rewrites each test so that alpha near the threshold covers only some of
// the pixel's samples: how far alpha is from the threshold, measured in pixels
// (fwidth), becomes the covered fraction. With one sample per pixel the result
// is exactly the original test. A threshold of zero means the engine has the
// test switched off, and the pixel is then left fully covered.
//
// Only tests against a positive threshold are rewritten; "alpha < 0.0" tests
// follow a subtraction the pattern does not see, and are left alone.
//
// hash (WBM_ALPHA_HASH): with one sample a pixel, the same covered fraction is
// kept instead by dropping the pixel against a threshold of noise that differs
// from pixel to pixel (and, under temporal upscaling, from frame to frame), so
// that an edge comes out dithered and the temporal scaler, which averages
// frames, turns the dither into a smooth edge. The engine's depth pre-pass
// takes the same rewrite and the same noise, so the two agree pixel for pixel.
const char k_alpha_hash_function[] = R"(
uniform float wbm_hash_frame;
float wbm_hash()
{
    return fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))) + wbm_hash_frame * 0.61803399);
}
)";

bool smooth_alpha_tests(std::string &fs, bool hash)
{
    static const std::regex tests[] = {
        std::regex(R"(if \(\(\(([\w.]+) - (alpha_test_val|\d+\.\d+)\) < 0\.0\)\) \{\s*discard;\s*\})"),
        std::regex(R"(if \(\(([\w.]+) < (\d+\.\d+)\)\) \{\s*discard;\s*\})"),
    };
    bool changed = false;
    for (const std::regex &test : tests) {
        std::string out;
        auto begin = std::sregex_iterator(fs.begin(), fs.end(), test), end = std::sregex_iterator();
        size_t pos = 0;
        for (auto it = begin; it != end; ++it) {
            std::string alpha = (*it)[1], threshold = (*it)[2];
            out.append(fs, pos, (size_t)it->position() - pos);
            pos = (size_t)(it->position() + it->length());
            if (threshold != "alpha_test_val" && atof(threshold.c_str()) <= 0) {
                out += it->str();
                continue;
            }
            if (hash) {
                out += "{ float wbm_c = clamp((" + alpha + " - " + threshold + ") / max(fwidth(" + alpha + "), 0.00001) + 0.5, 0.0, 1.0); if (" + threshold +
                       " > 0.0 && wbm_c <= wbm_hash()) { discard; } }";
                changed = true;
                continue;
            }
            out += "{ float wbm_c = clamp((" + alpha + " - " + threshold + ") / max(fwidth(" + alpha +
                   "), 0.00001) + 0.5, 0.0, 1.0); if (" + threshold + " > 0.0) { int wbm_n = int(wbm_c * " +
                   k_xl_samples_uniform + " + 0.5); if (wbm_n <= 0) { discard; } wbm_cover &= (1 << wbm_n) - 1; } "
                   "gl_SampleMask[0] = wbm_cover; }";
            changed = true;
        }
        out.append(fs, pos, std::string::npos);
        fs = out;
    }
    return changed;
}

// The engine's highlight is Blinn-Phong: strength * pow(dot(N, H), shininess).
// wbm_specular stands in for the pow() factor with a microfacet lobe:
//
//   - GGX distribution, scaled so that its peak is 1 and its energy matches the
//     Blinn-Phong lobe of the same exponent (alpha^2 = 2 / (shininess + 2)).
//     Highlights keep their brightness and size but gain GGX's soft falloff.
//   - Smith shadowing-masking, which fades the highlight as the light or the
//     view grazes the surface instead of letting it show on unlit faces.
//   - Schlick Fresnel relative to an assumed base reflectance, so highlights
//     strengthen towards grazing angles.
//
// The strength in front (specular colour, specular map, light colour) is the
// game's own and is not touched. This is the specular half of physically
// based shading; the game has no roughness or metalness maps to drive more.
const char k_pbr_specular_function[] = R"(
float wbm_specular(vec3 view, vec3 light, vec3 normal, float shininess)
{
    vec3 v = normalize(view), l = normalize(light), n = normalize(normal);
    vec3 h = normalize(v + l);
    float nh = clamp(dot(n, h), 0.0, 1.0), nl = clamp(dot(n, l), 0.0, 1.0);
    float nv = clamp(abs(dot(n, v)), 0.001, 1.0), vh = clamp(dot(v, h), 0.0, 1.0);
    float a2 = wbm_a2_fixed >= 0.0 ? wbm_a2_fixed : clamp(2.0 / (shininess + 2.0) * wbm_rough, 0.0004, 1.0);
    float d = nh * nh * (a2 - 1.0) + 1.0;
    float lobe = (a2 * a2) / (d * d);
    float gl = 2.0 * nl / (nl + sqrt(a2 + (1.0 - a2) * nl * nl));
    float gv = 2.0 * nv / (nv + sqrt(a2 + (1.0 - a2) * nv * nv));
    const float f0 = 0.25;
    float fresnel = (f0 + (1.0 - f0) * pow(1.0 - vh, 5.0)) / f0;
    wbm_n = n;
    wbm_a2 = a2;
    return lobe * gl * gv * fresnel * wbm_gain;
}
)";

// Shared between the pieces below. wbm_specular records the shading normal and
// roughness it was called with, which the environment term reuses.
const char k_pbr_globals[] = R"(
vec3 wbm_n = vec3(0.0, 0.0, 1.0);
float wbm_a2 = 0.25;
vec3 wbm_albedo = vec3(0.5);
float wbm_mask = 0.0;
float wbm_metal = 0.0;
float wbm_debug_mask = 0.0, wbm_debug_gloss = 0.0;
float wbm_metal_map = -1.0; // metalness read from a specular map, when the shader has one
float wbm_a2_fixed = -1.0;  // squared roughness from a metalness map (k_xl_pbr_sampler), when the surface has one
float wbm_dim = 0.35;  // how far a metal loses its diffuse colour
float wbm_rough = 1.0; // factors from the material class, see k_xl_material_uniform
float wbm_gain = 1.0;
)";

// What a surface reflects besides the sun. The engine renders no environment
// map for objects, so one is made from what every lit shader is given anyway:
// the sky's light colour above the horizon and the ground's below it, blended
// more softly on rougher surfaces. `up` may be a zero vector when the shader
// has no sky direction, in which case the two are averaged.
//
// The reflection is weighted by Fresnel, with a base reflectance that goes
// from 4% (non-metals) to the surface's own colour (metals), and by the
// material's specular mask, so that matt surfaces keep their look.
const char k_pbr_environment_function[] = R"(
vec3 wbm_environment(vec3 view, vec3 up, vec3 sky, vec3 ground)
{
    vec3 v = normalize(view), n = wbm_n;
    float nv = clamp(dot(n, v), 0.0, 1.0);
    vec3 r = reflect(-v, n);
    float ul = length(up);
    float t = ul > 0.001 ? dot(r, up / ul) * 0.5 + 0.5 : 0.5;
    float blur = 0.08 + 0.42 * sqrt(wbm_a2);
    vec3 env = mix(ground, sky, smoothstep(0.5 - blur, 0.5 + blur, t));
    vec3 f0 = mix(vec3(0.04), wbm_albedo, wbm_metal);
    vec3 fresnel = f0 + (1.0 - f0) * pow(1.0 - nv, 5.0);
    return env * fresnel * mix(wbm_mask, 1.0, wbm_metal * wbm_metal); // a metal reflects whatever its highlight map says
}
)";

// The same with a real sky. The backend redraws the scene's sky dome into a
// small atlas of six faces each frame (see k_xl_sky_sampler); a reflection
// pointing upwards samples it, blurrier for rougher surfaces through the
// atlas's mip levels, and one pointing down gets the ground colour as before.
//
// The reflection has to be in world space to index the atlas. Shaders that
// light with a world-space normal have it directly. Normal-mapped shaders work
// in tangent space and are never told how that relates to the world, but they
// are given two directions in both spaces: to the sun and to the sky light.
// wbm_to_world builds a frame from that pair in each space and carries the
// vector across. (Where a mesh's texture mapping is mirrored the tangent frame
// is left-handed and the result is mirrored too; the sky is smooth enough for
// that to pass.)
const char k_pbr_sky_functions[] = R"(
uniform sampler2D wbm_sky;

vec3 wbm_to_world(vec3 v, vec3 ts, vec3 tk, vec3 ws, vec3 wk)
{
    vec3 t1 = normalize(ts), t2 = normalize(cross(ts, tk)), t3 = cross(t1, t2);
    vec3 w1 = normalize(ws), w2 = normalize(cross(ws, wk)), w3 = cross(w1, w2);
    return dot(v, t1) * w1 + dot(v, t2) * w2 + dot(v, t3) * w3;
}

vec3 wbm_sky_sample(vec3 d, float lod)
{
    vec3 a = abs(d), forward, up;
    float face;
    if (a.x >= a.y && a.x >= a.z) {
        forward = vec3(sign(d.x), 0.0, 0.0), up = vec3(0.0, 0.0, 1.0), face = d.x > 0.0 ? 0.0 : 1.0;
    } else if (a.y >= a.z) {
        forward = vec3(0.0, sign(d.y), 0.0), up = vec3(0.0, 0.0, 1.0), face = d.y > 0.0 ? 2.0 : 3.0;
    } else {
        forward = vec3(0.0, 0.0, sign(d.z)), up = vec3(0.0, 1.0, 0.0), face = d.z > 0.0 ? 4.0 : 5.0;
    }
    vec2 uv = vec2(dot(d, cross(forward, up)), dot(d, up)) / dot(d, forward) * 0.5 + 0.5;
    float inset = 0.5 * exp2(ceil(lod)) / FACE_SIZE; // keep the filter inside this face
    uv = clamp(uv, vec2(inset), vec2(1.0 - inset));
    return textureLod(wbm_sky, vec2((uv.x + face) / 6.0, uv.y), lod).xyz;
}

vec3 wbm_environment_sky(vec3 view, vec3 reflection, vec3 ground)
{
    vec3 v = normalize(view), n = wbm_n;
    float nv = clamp(dot(n, v), 0.0, 1.0);
    vec3 d = normalize(reflection);
    float rough = sqrt(wbm_a2);
    vec3 above = normalize(vec3(d.xy, max(d.z, 0.04)));
    // Indoors the atlas holds the room all round, floor included, and says so in its fourth channel (a quarter; a sky's is 1):
    // the guess at the ground, made from the fog's colour, which indoors is the wrong blue, is not wanted there.
    bool room = abs(textureLod(wbm_sky, vec2(0.5 / 6.0, 0.5), 7.0).w - 0.25) < 0.1;
    vec3 sky = pow(max(wbm_sky_sample(room ? d : above, clamp(rough * 7.0, 0.0, 5.0)), vec3(0.0)), vec3(2.2));
    float blur = 0.04 + 0.3 * rough;
    vec3 env = room ? sky : mix(ground, sky, smoothstep(-blur, blur, d.z));
    vec3 f0 = mix(vec3(0.04), wbm_albedo, wbm_metal);
    vec3 fresnel = f0 + (1.0 - f0) * pow(1.0 - nv, 5.0);
    return env * fresnel * mix(wbm_mask, 1.0, wbm_metal * wbm_metal); // a metal reflects whatever its highlight map says
}
)";

// Ambient light with a direction to it. The engine lights what the sun does
// not reach with one flat colour plus a "sky light" from straight above, both
// set per scene. Two things are added (see k_xl_ambient_uniform):
//
//   - the sum is tinted by the colour of the captured sky on the side the
//     surface faces, relative to the sky's mean, so that a wall facing a
//     sunset is lit warmer than one facing away from it;
//   - surfaces that face sideways or down receive sunlight bounced off the
//     ground, in proportion to how much of the ground they see.
//
// Shaders that light per vertex have the world-space normal. The normal-mapped
// ones work in tangent space and know only which way is up there, so they get
// the tint by elevation alone.
//
// That bounce is sunlight, so it has to stop where the sun does: under a gate
// or a roof the ground around is in shadow and throws nothing up. A fragment
// shader that has the engine's shadow map (scenery only, so a figure's own
// shaded side still counts as standing in the sun) asks it through
// wbm_ground_sun: see k_ground_function. It is asked only where there is bounce
// to scale: the lookup is not cheap, and the ground itself, which faces up and
// fills most pictures, gets none. Elsewhere the answer is 1.
const char k_ambient_functions[] = R"(
uniform vec4 wbm_amb[6];
float wbm_ground_sun();

vec3 wbm_amb_horizon()
{
    return 0.25 * (wbm_amb[0].xyz + wbm_amb[1].xyz + wbm_amb[2].xyz + wbm_amb[3].xyz);
}

vec3 wbm_ambient(vec3 light, vec3 normal)
{
    float k = wbm_amb[5].w;
    if (k <= 0.0) return light;
    vec3 n = normalize(normal), q = n * n;
    vec3 tint = q.x * (n.x > 0.0 ? wbm_amb[0].xyz : wbm_amb[1].xyz) + q.y * (n.y > 0.0 ? wbm_amb[2].xyz : wbm_amb[3].xyz) +
                q.z * (n.z > 0.0 ? wbm_amb[4].xyz : wbm_amb_horizon());
    return light * mix(vec3(1.0), tint, k) + wbm_amb[5].xyz * (0.5 - 0.5 * n.z);
}

// sky: the engine's sky-light term; ambient: its flat ambient colour, added
// elsewhere; up: the normal's component along the world's up axis.
vec4 wbm_ambient_up(vec4 sky, vec4 ambient, float up)
{
    float k = wbm_amb[5].w;
    if (k <= 0.0) return sky;
    up = clamp(up, -1.0, 1.0);
    vec3 horizon = wbm_amb_horizon();
    vec3 tint = mix(horizon, up > 0.0 ? wbm_amb[4].xyz : horizon, up * up);
    float under = 0.5 - 0.5 * up;
    if (under > 0.12 && dot(wbm_amb[5].xyz, vec3(1.0)) > 0.003) under *= mix(1.0, wbm_ground_sun(), smoothstep(0.12, 0.25, under));
    return vec4(sky.xyz + (ambient.xyz + sky.xyz) * (mix(vec3(1.0), tint, k) - 1.0) + wbm_amb[5].xyz * under, sky.w);
}
)";

// How much of the ground about a point is in the sun: eight points level with
// it and a little above, 1.5 and 3.5 metres off, looked up in the engine's
// shadow map. The sun's matrix says what a metre east, north and up comes to
// in the map's coordinates and depth. Outside the map counts as lit. A little
// is always left: light comes round corners.
const char k_ground_function[] = R"(
const vec2 wbm_ground_ring[8] = vec2[8](
    vec2(1.5, 0.0), vec2(-1.5, 0.0), vec2(0.0, 1.5), vec2(0.0, -1.5),
    vec2(2.5, 2.5), vec2(-2.5, 2.5), vec2(2.5, -2.5), vec2(-2.5, -2.5));

float wbm_ground_lit(sampler2D map, vec3 coord, float size, mat4 sun)
{
    vec3 base = coord + sun[2].xyz * 0.7;
    float metre = length(vec3(sun[0][2], sun[1][2], sun[2][2])); // of depth
    float lit = 0.0;
    for (int i = 0; i < 8; i++) {
        vec3 q = base + sun[0].xyz * wbm_ground_ring[i].x + sun[1].xyz * wbm_ground_ring[i].y;
        bool outside = any(lessThan(q.xy, vec2(0.0))) || any(greaterThan(q.xy, vec2(1.0)));
        if (outside) { lit += 1.0; continue; }
        // By degrees in depth and between the map's texels, or it shows in patches.
        vec2 at = q.xy * size - 0.5, f = fract(at);
        vec4 s = smoothstep(-1.5 * metre, -0.2 * metre, textureGather(map, (floor(at) + 1.0) / size, 0) - q.z);
        lit += mix(mix(s.w, s.z, f.x), mix(s.x, s.y, f.x), f.y);
    }
    return mix(0.15, 1.0, lit / 8.0);
}
)";

bool add_sky_ambient(std::string &glsl, bool vertex)
{
    if (vertex) {
        static const std::regex term(
            R"(\(([\w.]+) \+ \(clamp \(\s*dot \(([\w.]+), -\(vSkyLightDir\.xyz\)\)\s*, 0\.0, 1\.0\) \* vSkyLightColor\.xyz\)\))");
        if (!std::regex_search(glsl, term)) return false;
        glsl = std::regex_replace(glsl, term, "wbm_ambient($&, vec3($2))");
        return true;
    }
    static const std::regex term(R"(\(clamp \(\s*dot \(SkyLightDir, ([\w.]+)\)\s*, 0\.0, 1\.0\) \* vSkyLightColor\))");
    // The engine's face shader lights with the flat ambient colour alone: no
    // light from the sky at all, where the hat on the same head gets it. A
    // face in shadow is then far darker than what is round it. Its vertex
    // shader does hand over the sky's direction; the term is put in as the
    // other shaders have it, and then treated like theirs.
    static const std::regex flat(R"(= \(vAmbientColor \+ \()");
    static const std::regex mapped(R"(dot \(SunLightDir, (normal_\d+)\))");
    std::smatch m;
    if (glsl.find("SkyLightDir") == std::string::npos && glsl.find("max ((0.2 * (") != std::string::npos && glsl.find("varying vec3 SunLightDir;") != std::string::npos &&
        std::regex_search(glsl, flat) && std::regex_search(glsl, m, mapped)) {
        const std::string n = m[1];
        glsl = std::regex_replace(glsl, flat,
                                  "= ((vAmbientColor + wbm_ambient_up(clamp(dot(SkyLightDir, " + n + "), 0.0, 1.0) * vSkyLightColor, vAmbientColor, dot(normalize(SkyLightDir), " + n + "))) + (",
                                  std::regex_constants::format_first_only);
        glsl.insert(0, std::string("varying vec3 SkyLightDir;\n") + (glsl.find("uniform vec4 vSkyLightColor;") == std::string::npos ? "uniform vec4 vSkyLightColor;\n" : ""));
        return true;
    }
    // The face shader with a specular map has the term with the dot product worked out a line before, and wrapped as its sunlight is.
    static const std::regex apart(R"(\(clamp \(\s*max \(\(0\.2 \* \((tmpvar_\d+) \+ 0\.9\)\), \1\)\s*, 0\.0, 1\.0\) \* vSkyLightColor\))");
    if (glsl.find("uniform vec4 vAmbientColor;") != std::string::npos && !std::regex_search(glsl, term) && std::regex_search(glsl, m, apart) &&
        std::regex_search(glsl, std::regex(m[1].str() + R"( = dot \(SkyLightDir, )"))) {
        glsl = std::regex_replace(glsl, apart, "wbm_ambient_up($&, vAmbientColor, $1 / max(length(SkyLightDir), 1e-4))", std::regex_constants::format_first_only);
        return true;
    }
    if (glsl.find("uniform vec4 vAmbientColor;") == std::string::npos || !std::regex_search(glsl, term)) return false;
    glsl = std::regex_replace(glsl, term, "wbm_ambient_up($&, vAmbientColor, dot(normalize(SkyLightDir), vec3($1)))");
    return true;
}

// Subsurface scattering for skin, done where the skin is lit and not as a blur
// of the finished picture: light entering skin comes out a little way off, red
// travelling farthest. Seen from outside, the lit side reaches round the
// terminator, reddest at its edge; fine relief shows in blue and green but is
// smoothed away in red; and the edge of a cast shadow glows red.
//
// Skin has no shader of its own in this engine (two face shaders apart, which
// only the player's kind of face uses): troops' faces, hands and bodies are
// drawn with the general per-pixel shaders. So the sun's term is rewritten in
// all of those and switched per draw by wbm_skin, which the backend sets from
// what the base texture shows. They light in tangent space, where the smooth
// surface's normal is simply +Z, so the sun's direction gives the unmapped
// lighting for nothing.
const char k_skin_function[] = R"(
uniform vec4 wbm_skin;
// How far behind the surface nearest the sun this point is, in the shadow
// map's depth units (wbm_skin.y makes metres of it); below 0 when unknown.
float wbm_skin_gap = -1.0;

// mapped and smooth_: the cosine of the sun's angle to the normal-mapped
// surface and to the smooth one.
vec3 wbm_skin_core(float mapped, float smooth_, float shadow)
{
    // Where the light only grazes, a sharp shadow map draws the head's own
    // polygons on it as steps. Light there is nearly gone anyway; the shadow is let go.
    // That is for the head's shadow on itself, which lies millimetres behind
    // the lit surface. A shadow cast from farther off, a hat's brim over the
    // eyes, stays: let go, it left lit patches inside the shadow.
    float own = wbm_skin_gap >= 0.0 && wbm_skin.y > 0.0 ? 1.0 - smoothstep(0.012, 0.035, wbm_skin_gap * wbm_skin.y) : 1.0;
    shadow = mix(shadow, mix(1.0, shadow, smoothstep(0.0, 0.3, smooth_)), own);
    const vec3 wrap = vec3(0.38, 0.16, 0.07);       // how far round each colour reaches
    const vec3 relief = vec3(0.25, 0.7, 1.0);       // how much of the normal map each colour sees
    vec3 facing = mix(vec3(smooth_), vec3(mapped), relief);
    vec3 lit = clamp((facing + wrap) / (1.0 + wrap), 0.0, 1.0);
    lit *= 1.0 / (1.0 + 0.5 * wrap);                // what is gained round the edge is given up at the front
    return lit * pow(vec3(shadow), vec3(0.7, 1.0, 1.25));
}

// Sunlight coming through thin parts (ears, the wings of a nose, fingers) when
// the sun is behind them. depth: how far behind the sunward surface this
// point is (wbm_dyn_thickness), which wbm_skin.y turns into metres; without
// that there is no telling an ear from the far side of a head, and no light.
// Red gets through about two centimetres, blue a few millimetres.
vec3 wbm_skin_back(vec3 view, vec3 to_sun, float depth)
{
    if (depth < 0.0 || wbm_skin.y <= 0.0) return vec3(0.0);
    vec3 v = normalize(view), l = normalize(to_sun);
    float towards = pow(clamp(dot(v, -normalize(l + vec3(0.0, 0.0, 0.35))), 0.0, 1.0), 2.0); // looking into the light
    float away = smoothstep(0.25, -0.15, l.z);                                                // on the side turned from it
    // Even the thinnest path is through some skin: half a centimetre is added,
    // so that what comes through is always warm and never plain white.
    return towards * away * wbm_skin.z * exp(-(depth * wbm_skin.y + 0.005) * vec3(45.0, 130.0, 220.0));
}

// For the general shaders: skin or not is the draw's business.
vec4 wbm_skin_sun(vec3 to_sun, vec3 normal, float shadow, vec3 back)
{
    float mapped = dot(to_sun, normal);
    float plain = clamp(mapped, 0.0, 1.0) * shadow;
    if (wbm_skin.w > 0.0) { // a leaf: what falls on its far side comes through, yellower
        float through = clamp(-mapped, 0.0, 1.0) * wbm_skin.w * shadow;
        return vec4(max(vec3(plain), through * vec3(0.85, 1.0, 0.35)), plain);
    }
    if (wbm_skin.x < 0.5) return vec4(vec3(plain), plain);
    return vec4(wbm_skin_core(mapped, normalize(to_sun).z, shadow) + back, plain);
}

// For the engine's own skin shader, which is skin whatever it draws.
vec4 wbm_skin_lit(float mapped, float smooth_, float shadow, vec3 back)
{
    return vec4(wbm_skin_core(mapped, smooth_, shadow) + back, clamp(max(0.2 * (mapped + 0.9), mapped), 0.0, 1.0) * shadow);
}
)";

// Wind. Foliage is moved sideways before it is projected: slowly to and fro
// with a quicker flutter on top, by an amount that grows with height, so that
// a tree's crown sways while its foot stays put. Grass comes in batches already
// placed in the world, where height says nothing; there the texture coordinate
// does: a blade's tip is at one end of the texture and its root at the other.
// Only the position on screen moves: lighting and shadows are worked out for
// the plant at rest.
const char k_wind_function[] = R"(
uniform vec4 wbm_wind;

// height: for batches, the vertex colour's alpha, which the backend has made
// the height above the plant's foot in units of 25.5 m; below 0 when the
// shader has no vertex colour, and then a batch is left still.
vec4 wbm_sway_at(vec4 p, float height, float t)
{
    if (wbm_wind.y <= 0.0) return p;
    float metres, phase;
    if (wbm_wind.z < 0.5) {
        metres = p.z;
        phase = wbm_wind.w + dot(p.xyz, vec3(0.31, 0.27, 0.19));
    } else {
        if (height < 0.0) return p;
        metres = height * 25.5;
        phase = dot(p.xy, vec2(0.23, 0.17)) + p.z * 0.19;
    }
    float h = clamp(metres / 6.0, 0.0, 1.5);
    float reach = 0.11 * h * h + 0.035 * smoothstep(0.0, 1.5, metres); // the second part is for bushes and grass
    float gust = 0.6 + 0.4 * sin(t * 0.5 + phase * 0.11);
    vec2 push = vec2(sin(t * 1.3 + phase) + 0.35 * sin(t * 3.7 + phase * 2.3), 0.6 * cos(t * 1.1 + phase * 0.8) + 0.35 * sin(t * 4.3 + phase * 1.9));
    p.xy += push * (reach * gust * wbm_wind.y);
    return p;
}

vec4 wbm_sway(vec4 p, float height)
{
    return wbm_sway_at(p, height, wbm_wind.x);
}

// The vertex colour as the shader should see it: with its alpha full where
// that alpha has been taken over for heights.
vec4 wbm_tint(vec4 colour)
{
    return wbm_wind.z > 0.5 ? vec4(colour.xyz, 1.0) : colour;
}
)";

// For temporal upscaling every pixel's place a frame ago is wanted (see
// apply_temporal_upscale in src/backend/upscale.inc). Depth and the two cameras give
// it for what stands still, but not for what the wind moves. The shaders that
// sway therefore say it themselves: the vertex shader works the vertex out a
// second time as it was a frame ago (wbm_wind_before is that frame's time for
// the wind, wbm_reproject takes this frame's clip space to that one's), and the
// fragment shader writes the step between the two into a second target, with
// its distance, in the layout the moving objects' texture has.
const char k_motion_vertex_uniforms[] = "uniform mat4 wbm_reproject;\nuniform float wbm_wind_before;\n";

bool add_wind(std::string &vs, bool motion)
{
    // Rigid geometry only: what is skinned moves by its bones.
    if (vs.find("matWorldArray") != std::string::npos || vs.find("uniform mat4 matWorldViewProj;") == std::string::npos) return false;
    static const std::regex project(R"(\(matWorldViewProj \* (\w+)\))");
    if (!std::regex_search(vs, project)) return false;
    static const std::regex declaration(R"(attribute vec4 inColor0;)");
    bool coloured = std::regex_search(vs, declaration);
    if (coloured) {
        vs = std::regex_replace(vs, std::regex(R"(\binColor0\b)"), "wbm_tint(inColor0)");
        vs = std::regex_replace(vs, std::regex(R"(attribute vec4 wbm_tint\(inColor0\);)"), "attribute vec4 inColor0;");
    }
    std::smatch m;
    std::regex_search(vs, m, project);
    const std::string point = m[1], height = coloured ? "inColor0.w" : "-1.0";
    vs = std::regex_replace(vs, project, "(matWorldViewProj * wbm_sway($1, " + height + "))");
    size_t end = vs.rfind('}');
    if (motion && end != std::string::npos) {
        vs.insert(end, "  wbm_mnow = (matWorldViewProj * wbm_sway(" + point + ", " + height + "));\n  wbm_mbefore = (wbm_reproject * (matWorldViewProj * wbm_sway_at(" + point + ", " + height +
                           ", wbm_wind_before)));\n");
        vs.insert(0, "varying vec4 wbm_mnow;\nvarying vec4 wbm_mbefore;\n");
    }
    return true;
}

// The engine's own foliage shaders (grass, bushes, leaves) light without a
// normal: a fixed share of the sun's colour, whichever way a leaf faces. For
// those the light coming through is judged by where the viewer stands: looking
// towards the sun, one sees the far side of the leaves, lit through.
// wbm_leaf_sun: xyz the direction to the sun in view space (the space
// matWorldView leads to), w the strength, 0 for what is not foliage.
const char k_leaf_function[] = R"(
uniform vec4 wbm_leaf_sun;

vec4 wbm_leaf_glow(vec3 seen)
{
    if (wbm_leaf_sun.w <= 0.0) return vec4(1.0);
    float into = clamp(dot(normalize(seen), wbm_leaf_sun.xyz), 0.0, 1.0);
    return vec4(1.0) + wbm_leaf_sun.w * 2.5 * into * into * vec4(0.85, 1.0, 0.35, 0.0);
}
)";

bool add_leaf_glow(std::string &vs)
{
    // SunLight = ((sun colour * 0.34) * (vMaterialColor * inColor0.zyxw)), and the view-space position from matWorldView * P
    static const std::regex flat(R"(SunLight = \(\((tmpvar_\d+) \* (0\.\d+)\) \* \(vMaterialColor \* inColor0\.zyxw\)\);)");
    static const std::regex seen(R"(\(matWorldView \* (tmpvar_\d+)\))");
    std::smatch m;
    if (!std::regex_search(vs, flat) || !std::regex_search(vs, m, seen)) return false;
    vs = std::regex_replace(vs, flat, "SunLight = ((($1 * $2) * wbm_leaf_glow((matWorldView * " + std::string(m[1]) + ").xyz)) * (vMaterialColor * inColor0.zyxw));");
    return true;
}

// Shells of fur: the backend draws a coat again a few times over, each time a
// little farther out along the surface's normal (wbm_shell.x, in metres), and
// the fragment shader lets through only the strands that reach that far (see
// k_fur_function). Every skinned vertex shader can do the pushing; it is zero
// for all but those draws.
bool add_shell_push(std::string &vs)
{
    static const std::regex skinned(
        R"((\w+) = \(\(\(\s*\(\(matWorldArray\[(\w+)\] \* (\w+)\)\s*\* inBlendWeight\.x\)\s*\+\s*\(\(matWorldArray\[(\w+)\] \* \3\)\s*\* inBlendWeight\.y\)\s*\)\s*\+\s*\(\s*\(matWorldArray\[(\w+)\] \* \3\)\s*\* inBlendWeight\.z\)\)\s*\+\s*\(\(matWorldArray\[(\w+)\] \* \3\)\s*\* inBlendWeight\.w\)\);)");
    if (vs.find("attribute vec3 inNormal;") == std::string::npos || !std::regex_search(vs, skinned)) return false;
    // Layers of long hair (wbm_shell.z below 0) do not stand straight out: the
    // farther along the hair, the less it goes outwards and the more it goes
    // down, whichever way the horse is turned. Down is the world's, brought
    // into the space the bones leave the vertex in.
    bool placed = vs.find("uniform mat4 matWorld;") != std::string::npos;
    const std::string normal = "normalize(mat3(matWorldArray[$2]) * inNormal * inBlendWeight.x + mat3(matWorldArray[$4]) * inNormal * inBlendWeight.y + "
                               "mat3(matWorldArray[$5]) * inNormal * inBlendWeight.z + mat3(matWorldArray[$6]) * inNormal * inBlendWeight.w)";
    const std::string down = placed ? "normalize(vec3(0.0, 0.0, -1.0) * mat3(matWorld))" : "vec3(0.0)";
    // Strands (wbm_shell.x below 0): the backend draws ribbons whose vertices all
    // sit at a root on the skin and say in their colour which point of which
    // strand they are (see build_strands in the backend). The strand is laid
    // out here: it leaves the skin the way the hair map has it combed, bends
    // over to hang, and is widened across the line of sight.
    //   colour x: segment * 2 + side, plus 32 times how far down the tail the
    //   root is (0 to 7); y: a chance number (0 to 63), plus 64 for a mane that
    //   falls to the model's +x side, plus 128 for the tail; z, w: the hair
    //   map's red and green at the root.
    // The mane grows from the crest of the neck and falls down one side of it;
    // on the head (marked like the far end of a tail) it is the forelock, shorter, falling forwards.
    // The tail's hair runs on along the tail (which at rest hangs straight down
    // the model, and is turned with its bones), past its end, and droops: long
    // from the dock, shorter from farther down, so that it ends much together.
    // wbm_shell.x: -(the mane's length), y: the tail's, metres.
    std::string strand;
    if (placed && vs.find("attribute vec4 inColor0;") != std::string::npos && vs.find("attribute vec3 inTangent;") != std::string::npos) {
        vs = std::regex_replace(vs, std::regex(R"(\binColor0\b(?!;))"), "wbm_colour");
        size_t body = vs.find("void main ()");
        if (body != std::string::npos) body = vs.find('{', body);
        if (body == std::string::npos) return false;
        vs.insert(body + 1, "\n  wbm_colour = wbm_shell.x < 0.0 ? vec4(1.0) : inColor0;\n  wbm_strand_v = vec4(0.0);");
        vs.insert(0, "varying vec4 wbm_strand_v;\n");
        const std::string handed = vs.find("attribute vec3 inBinormal;") != std::string::npos ? "(dot(cross(inNormal, inTangent), inBinormal) < 0.0 ? -1.0 : 1.0)" : "1.0";
        const std::string eye = vs.find("uniform vec4 vCameraPos;") != std::string::npos ? "((vCameraPos.xyz - matWorld[3].xyz) * mat3(matWorld) - $1.xyz)" : "wbm_n";
        strand = "\n  if (wbm_shell.x < 0.0) {\n"
                 "    float wbm_v = floor(inColor0.x * 255.0 + 0.5), wbm_r = floor(inColor0.y * 255.0 + 0.5);\n"
                 "    float wbm_far = floor(wbm_v / 32.0) / 7.0;\n"
                 "    wbm_v -= floor(wbm_v / 32.0) * 32.0;\n"
                 "    float wbm_side = mod(wbm_v, 2.0) * 2.0 - 1.0, wbm_t = floor(wbm_v / 2.0) / 8.0;\n"
                 "    float wbm_tail = step(128.0, wbm_r);\n"
                 "    wbm_r -= 128.0 * wbm_tail;\n"
                 "    float wbm_fall = step(64.0, wbm_r) * 2.0 - 1.0, wbm_chance = (wbm_r - 64.0 * step(64.0, wbm_r)) / 63.0;\n"
                 "    vec2 wbm_d = inColor0.zw * 2.0 - 1.0;\n"
                 "    mat3 wbm_m = mat3(matWorldArray[$2]) * inBlendWeight.x + mat3(matWorldArray[$4]) * inBlendWeight.y + mat3(matWorldArray[$5]) * inBlendWeight.z + mat3(matWorldArray[$6]) * inBlendWeight.w;\n"
                 "    vec3 wbm_n = normalize(wbm_m * inNormal), wbm_along = wbm_m * inTangent;\n"
                 "    vec3 wbm_down = " + down + ";\n"
                 "    vec3 wbm_c = wbm_along * wbm_d.x - cross(wbm_n, wbm_along) * " + handed + " * wbm_d.y;\n"
                 "    wbm_c = dot(wbm_c, wbm_c) > 0.0004 ? normalize(wbm_c) : wbm_down;\n"
                 "    vec3 wbm_out = wbm_tail > 0.5 ? normalize(normalize(wbm_m * vec3(0.0, -0.12, -1.0)) + wbm_n * 0.12) : normalize(normalize(wbm_m * (wbm_far > 0.5 ? vec3(wbm_fall * 0.25, 1.0, 0.0) : vec3(wbm_fall, (fract(wbm_chance * 5.7) - 0.5) * 0.9, 0.0))) * (0.3 + 0.35 * fract(wbm_chance * 3.1)) + wbm_n * 0.2 + wbm_down * 0.35);\n"
                 "    float wbm_len = (wbm_tail > 0.5 ? wbm_shell.y * (1.0 - 0.6 * wbm_far) : -wbm_shell.x * (1.0 - 0.45 * wbm_far)) * (wbm_tail > 0.5 ? 0.7 + 0.3 * wbm_chance : 0.45 + 0.55 * wbm_chance);\n"
                 "    float wbm_k = (wbm_tail > 0.5 ? 0.85 : 1.0) * (1.0 - 0.9 * min(length(wbm_drift.xyz), 1.0));\n" // streaming behind, it does not hang as well
                 "    vec3 wbm_dir = wbm_out * (1.0 - wbm_k * wbm_t) + wbm_down * (wbm_k * wbm_t);\n"
                 "    vec3 wbm_off = wbm_len * (wbm_out * (wbm_t - 0.5 * wbm_k * wbm_t * wbm_t) + wbm_down * (0.5 * wbm_k * wbm_t * wbm_t));\n"
                 // They swing a little, each out of step with the next, the tips most; the tail more than the mane.
                 "    float wbm_time = wbm_shell.w - 1.0, wbm_place = dot(inPosition, vec3(5.0, 3.0, 2.0));\n"
                 "    vec3 wbm_blow = vec3(sin(wbm_time * 1.9 + wbm_place + wbm_chance * 2.0) + 0.4 * sin(wbm_time * 4.7 + wbm_chance * 40.0), cos(wbm_time * 1.4 + wbm_place * 1.3 + wbm_chance * 2.0) + 0.4 * cos(wbm_time * 3.9 + wbm_chance * 31.0), 0.0);\n"
                 "    wbm_off += (wbm_blow * mat3(matWorld)) * ((wbm_tail > 0.5 ? 0.05 : 0.08) * wbm_len * wbm_t * wbm_t);\n"
                 // Left behind a little when the horse moves off (see drift_of in the backend).
                 "    vec3 wbm_trail = (wbm_drift.xyz * mat3(matWorld)) * wbm_len;\n"
                 "    wbm_off += wbm_trail * (wbm_t * wbm_t);\n"
                 // However it is blown about, a strand is no longer than it is.
                 "    wbm_off *= min(1.0, wbm_len * wbm_t / max(length(wbm_off), 1e-6));\n"
                 "    $1.xyz += wbm_off;\n"
                 "    wbm_dir = normalize(wbm_dir * wbm_len + wbm_trail * (2.0 * wbm_t));\n"
                 "    vec3 wbm_eye = " + eye + ";\n"
                 "    vec3 wbm_across = cross(wbm_dir, wbm_eye);\n"
                 "    wbm_across = dot(wbm_across, wbm_across) > 1e-12 ? normalize(wbm_across) : wbm_n;\n"
                 "    float wbm_w = (wbm_tail > 0.5 ? 0.005 : 0.003) * (0.6 + 0.8 * fract(wbm_chance * 7.3)) * (1.0 - 0.6 * wbm_t);\n" // half the width of a lock, metres
                 "    float wbm_used = max(wbm_w, 0.0012 * length(wbm_eye));\n" // not much under a pixel wide: the rest is made up in how much it covers
                 "    $1.xyz += wbm_across * (wbm_side * wbm_used);\n"
                 "    wbm_strand_v = vec4(wbm_side, wbm_t, wbm_chance, wbm_w / wbm_used);\n"
                 "  }";
    }
    vs = std::regex_replace(vs, skinned,
                            "$&" + strand + "\n  if (wbm_shell.x > 0.0) {\n    vec3 wbm_n = " + normal + ";\n    if (wbm_shell.z < 0.0) $1.xyz += wbm_shell.x * (wbm_n * (1.0 - 0.55 * wbm_shell.y) + " + down +
                                " * (2.2 * wbm_shell.y));\n    else $1.xyz += wbm_shell.x * wbm_n;\n  }",
                            std::regex_constants::format_first_only);
    return true;
}

// How people's hair takes the light, after Marschner as the horses' manes do
// (see wbm_fur_lit): a narrow white band off the strands' surface, shifted
// towards the root; a wider band in the hair's colour from inside them,
// shifted towards the tip and broken up from strand to strand; light coming
// straight through when the sun is behind; and a broad sheen from the sky.
// All vectors in the world's space; t runs along the hair. x, y, z: per unit
// of sunlight; w: the sky's sheen, per unit of sky light.
const char k_people_hair_function[] = R"(
vec4 wbm_people_hair(vec3 normal_, vec3 tangent_, vec3 view, vec3 to_sun, vec3 colour, vec2 uv)
{
    vec3 n = normalize(normal_), t = normalize(tangent_), v = normalize(view), l = normalize(to_sun);
    vec3 h = normalize(v + l);
    float jitter = fract(sin(dot(floor(uv * vec2(700.0, 40.0)), vec2(12.9898, 78.233))) * 43758.5453) - 0.5;
    vec3 t_surface = normalize(t - n * 0.10), t_inside = normalize(t + n * (0.16 + 0.12 * jitter));
    float a = dot(t_surface, h), b = dot(t_inside, h);
    float off_surface = pow(sqrt(max(1.0 - a * a, 0.0)), 200.0), from_inside = pow(sqrt(max(1.0 - b * b, 0.0)), 32.0);
    float bright = max(colour.x, max(colour.y, colour.z));
    vec3 hue = mix(vec3(1.0), colour / max(bright, 0.02), 0.8) * mix(0.25, 1.0, smoothstep(0.05, 0.4, bright)); // dark hair swallows what goes into it
    float lit_side = clamp(dot(n, l) * 0.7 + 0.5, 0.0, 1.0);
    float through = pow(clamp(dot(-v, l), 0.0, 1.0), 7.0) * (1.0 - abs(dot(t, l)));
    float c = dot(t, normalize(v + vec3(0.0, 0.0, 1.0)));
    float sheen = pow(sqrt(max(1.0 - c * c, 0.0)), 14.0) * (0.25 + 0.75 * pow(1.0 - clamp(dot(v, n), 0.0, 1.0), 3.0));
    return vec4(lit_side * (0.14 * off_surface + 0.30 * from_inside * hue) + 0.30 * through * hue, 0.12 * sheen);
}
)";

// Water. The engine's water reflects a picture of the surroundings by how
// steeply it is seen, and fades out where the bed is near. It takes next to
// nothing from the sun (a hundredth of its light). Added here:
//   the sun's glitter, a sharp highlight off each ripple of the normal map;
//   where the water is shallow, the bed's light coming back up through it,
//   pale and green, fading as it deepens;
//   foam in a ragged band along the shore, broken up by the ripples.
// The last two need the depth the engine's own fade is worked out from, which
// is a share of the far plane's distance: 1250 m in a battle, where 0.0008 is a metre.
// The shallows fade over about a metre of water, the foam lies where it is under 20 cm.
// The sky as the renderer has captured it (see k_xl_sky_sampler), for water to reflect.
const char k_water_sky_function[] = R"(
uniform sampler2D wbm_sky;
vec3 wbm_water_sky(vec3 d)
{
    vec3 a = abs(d), forward, up;
    float face;
    if (a.x >= a.y && a.x >= a.z) {
        forward = vec3(sign(d.x), 0.0, 0.0), up = vec3(0.0, 0.0, 1.0), face = d.x > 0.0 ? 0.0 : 1.0;
    } else if (a.y >= a.z) {
        forward = vec3(0.0, sign(d.y), 0.0), up = vec3(0.0, 0.0, 1.0), face = d.y > 0.0 ? 2.0 : 3.0;
    } else {
        forward = vec3(0.0, 0.0, sign(d.z)), up = vec3(0.0, 1.0, 0.0), face = d.z > 0.0 ? 4.0 : 5.0;
    }
    vec2 uv = vec2(dot(d, cross(forward, up)), dot(d, up)) / dot(d, forward) * 0.5 + 0.5;
    uv = clamp(uv, vec2(1.0 / FACE_SIZE), vec2(1.0 - 1.0 / FACE_SIZE));
    return pow(max(textureLod(wbm_sky, vec2((uv.x + face) / 6.0, uv.y), 1.0).xyz, vec3(0.0)), vec3(2.2));
}
)";

// sky: the captured sky is at hand (XlOptions::pbr_sky), and the water reflects
// it; without it the engine's own picture of the surroundings stays.
// The surface is put together afresh:
//   two layers of ripples, so that the pattern does not repeat so plainly;
//   the water's own colour, a dark blue-green lit by the sun, a little paler in the shallows;
//   over it the sky, by how steeply the water is seen (little straight down, all of it along the surface);
//   how much of the bed shows through: all of it at the edge, less with every metre of water over it;
//   foam at the shore and round whatever stands in the water, and the sun's glitter.
bool add_water(std::string &fs, bool sky)
{
    // With the depth effects' inputs the water can look into a copy of the scene (wbm_scene, wbm_water): see below.
    const bool looks = fs.find("varying vec4 projCoord;") != std::string::npos && fs.find("varying float Depth;") != std::string::npos && fs.find("uniform sampler2D depth_texture;") != std::string::npos;
    static const std::regex sun(R"((\w+) = \(\(0\.01 \* clamp \(\s*dot \((normal_\d+), LightDir_Alpha\.xyz\)\s*, 0\.0, 1\.0\)\) \* LightDif\);)");
    static const std::regex deep(R"((\w+)\.w = \(\1\.w \+ clamp \(\(\s*\((tmpvar_\d+)\.x - Depth\)\s*\* 32\.0\), 0\.0, 1\.0\)\);)");
    static const std::regex encode(R"((\w+)\.xyz = pow \(\1\.xyz, output_gamma_inv\.xyz\);)");
    std::smatch m;
    if (fs.find("varying vec4 LightDir_Alpha;") == std::string::npos || fs.find("varying vec3 CameraDir;") == std::string::npos || fs.find("varying vec4 LightDif;") == std::string::npos ||
        !std::regex_search(fs, m, sun) || !std::regex_search(fs, encode))
        return false;
    const std::string out = m[1], normal = m[2];
    size_t body = fs.find("void main ()");
    if (body != std::string::npos) body = fs.find('{', body);
    if (body == std::string::npos) return false;
    // The depth of water over the bed, in the units of the engine's depth picture (0.0008 is a metre in a battle); below 0 when not known.
    if (std::regex_search(fs, m, deep) && std::string(m[1]) == out) fs = std::regex_replace(fs, deep, "$&\n    wbm_deep = max($2.x - Depth, 0.0);", std::regex_constants::format_first_only);
    fs = std::regex_replace(fs, encode,
                            "{\n    vec3 wbm_n = normalize(" + normal + " + vec3((2.0 * texture2D(normal_texture, Tex0.yx * 2.7 + 0.31).wy - 1.0) * 0.6, 0.0));\n"
                            "    vec3 wbm_v = normalize(CameraDir), wbm_l = normalize(LightDir_Alpha.xyz);\n"
                            "    float wbm_fresnel = 0.02 + 0.98 * pow(1.0 - clamp(dot(wbm_n, wbm_v), 0.0, 1.0), 5.0);\n"
                            "    vec3 wbm_r = reflect(-wbm_v, wbm_n);\n"
                            "    float wbm_shallow = wbm_deep >= 0.0 ? exp(-wbm_deep * 1200.0) : 0.0;\n"
                            "    vec3 wbm_body = (vec3(0.010, 0.045, 0.05) + vec3(0.014, 0.04, 0.032) * wbm_shallow) * (LightDif.xyz * (0.35 + 0.65 * clamp(wbm_l.z, 0.0, 1.0)) + 0.15);\n" +
                            (sky ? "    vec3 wbm_above = wbm_water_sky(normalize(vec3(wbm_r.xy, max(wbm_r.z, 0.03))));\n"
                                   "    " + out + ".xyz = mix(wbm_body, wbm_above, wbm_fresnel);\n"
                                 : "    " + out + ".xyz += wbm_body * (1.0 - wbm_fresnel);\n") +
                            "    float wbm_thick = wbm_deep >= 0.0 ? 1.0 - exp(-wbm_deep * 900.0) : 0.85;\n" // over a metre of water little of the bed is left
                            "    float wbm_alpha = max(wbm_fresnel, wbm_thick) * LightDir_Alpha.w, wbm_soft = 1.0;\n" +
                            (looks ?
                            // The scene as it was before the water: seen through the surface, shifted by the ripples
                            // (the bed, refracted), and searched along the mirrored line of sight for what the surface
                            // reflects: banks, trees, people. Where the search finds nothing on screen, the sky stays.
                            "    if (wbm_water[2].x > 0.5 && wbm_deep >= 0.0) {\n"
                            "      vec2 wbm_suv = vec2(projCoord.x, projCoord.w - projCoord.y) / projCoord.w;\n"
                            "      float wbm_far = wbm_water[1].w;\n"
                            "      vec2 wbm_ruv = clamp(wbm_suv + wbm_n.xy * (0.025 * clamp(wbm_deep * 1500.0, 0.0, 1.0)), vec2(0.001), vec2(0.999));\n"
                            "      if (texture2D(depth_texture, wbm_ruv).x < Depth) wbm_ruv = wbm_suv;\n" // not what stands in front of the water
                            "      vec3 wbm_under = pow(max(textureLod(wbm_scene, wbm_ruv, 0.0).xyz, vec3(0.0)), vec3(2.2));\n"
                            "      float wbm_w0 = ((2.0 * Depth * wbm_far - 1.0) - wbm_water[0].w) / wbm_water[0].z;\n"
                            "      vec3 wbm_p = vec3((wbm_suv * 2.0 - 1.0) * wbm_w0 / wbm_water[0].xy, wbm_w0);\n"
                            "      vec3 wbm_up = normalize(wbm_water[1].xyz);\n"
                            "      vec3 wbm_dir = reflect(normalize(wbm_p), wbm_up);\n"
                            "      float wbm_stride = max(0.15, wbm_w0 * 0.03);\n"
                            "      float wbm_jitter = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));\n"
                            "      vec3 wbm_s = wbm_p + wbm_dir * (wbm_stride * wbm_jitter);\n"
                            "      float wbm_hit = 0.0;\n"
                            "      vec2 wbm_huv = wbm_suv;\n"
                            "      for (int wbm_i = 0; wbm_i < 64 && float(wbm_i) < wbm_water[2].y; wbm_i++) {\n"
                            "        wbm_s += wbm_dir * wbm_stride;\n"
                            "        if (wbm_s.z <= 0.1) break;\n"
                            "        vec2 wbm_u = wbm_s.xy * wbm_water[0].xy / wbm_s.z * 0.5 + 0.5;\n"
                            "        if (wbm_u.x < 0.0 || wbm_u.x > 1.0 || wbm_u.y < 0.0 || wbm_u.y > 1.0) break;\n"
                            "        float wbm_ws = ((2.0 * texture2D(depth_texture, wbm_u).x * wbm_far - 1.0) - wbm_water[0].w) / wbm_water[0].z;\n"
                            "        if (wbm_s.z > wbm_ws && wbm_s.z - wbm_ws < wbm_stride * 2.5 + 0.3) {\n"
                            "          vec2 wbm_edge = min(wbm_u, 1.0 - wbm_u);\n"
                            "          wbm_hit = clamp(min(wbm_edge.x, wbm_edge.y) * 12.0, 0.0, 1.0), wbm_huv = wbm_u;\n"
                            "          break;\n"
                            "        }\n"
                            "        wbm_stride *= 1.18;\n"
                            "      }\n"
                            "      vec3 wbm_mirror = " + std::string(sky ? "wbm_above" : out + ".xyz") + ";\n"
                            "      if (wbm_hit > 0.0) wbm_mirror = mix(wbm_mirror, pow(max(textureLod(wbm_scene, clamp(wbm_huv + wbm_n.xy * 0.012, vec2(0.001), vec2(0.999)), 0.0).xyz, vec3(0.0)), vec3(2.2)), wbm_hit);\n"
                            "      " + out + ".xyz = mix(mix(wbm_under, wbm_body, wbm_thick), wbm_mirror, wbm_fresnel);\n"
                            "      wbm_alpha = 1.0, wbm_soft = 0.5;\n" // what was blended in thinly before is now mixed here: foam and glitter are held back to match
                            "    }\n" : std::string()) +
                            "    float wbm_ragged = 0.6 * texture2D(normal_texture, Tex0 * 9.0).y + 0.4 * texture2D(normal_texture, Tex0 * 23.0 + 0.37).w;\n"
                            // Foam: a narrow line right at the edge, and beyond it patches that thin out over some 20 cm of depth.
                            "    float wbm_foam = wbm_deep > 1e-7 ? max(0.7 * smoothstep(0.00005, 0.00001, wbm_deep), smoothstep(0.00018, 0.00004, wbm_deep) * smoothstep(0.42, 0.62, wbm_ragged)) : 0.0;\n"
                            "    " + out + ".xyz = mix(" + out + ".xyz, LightDif.xyz * 0.55 + vec3(0.12), wbm_foam * wbm_soft);\n"
                            "    wbm_alpha = max(wbm_alpha, wbm_foam * 0.7);\n"
                            "    float wbm_glint = pow(clamp(dot(wbm_n, normalize(wbm_l + wbm_v)), 0.0, 1.0), 700.0) * step(0.0, wbm_l.z);\n"
                            "    " + out + ".xyz += LightDif.xyz * (1.1 * wbm_glint * wbm_soft);\n"
                            "    " + out + ".w = max(wbm_alpha, clamp(wbm_glint * 0.6, 0.0, 1.0) * LightDir_Alpha.w);\n  }\n  $&",
                            std::regex_constants::format_first_only);
    fs.insert(body + 1, "\n  float wbm_deep = -1.0;");
    if (looks) fs.insert(0, "uniform sampler2D wbm_scene;\nuniform vec4 wbm_water[3];\n");
    return true;
}

// Parallax. See XlOptions::parallax. The surface is walked down along the
// line of sight in eight steps until the texture's height is above the line,
// and the last step is split by where the two cross. Far off, where a texel is
// smaller than a pixel, there is nothing to see of it and it fades out.
// Every later use of the texture coordinate in the shader takes the moved one.
bool add_parallax(std::string &fs)
{
    const char *needed[] = {"uniform sampler2D diffuse_texture;", "uniform sampler2D normal_texture;", "varying vec2 Tex0;", "varying vec3 SunLightDir;"};
    for (const char *n : needed)
        if (fs.find(n) == std::string::npos) return false;
    // Not faces, hair or coats, whose shaders have the same inputs.
    if (fs.find("max ((0.2 * (") != std::string::npos || fs.find("0.800118") != std::string::npos || fs.find("wbm_skin_lit") != std::string::npos) return false;
    size_t body = fs.find("void main ()");
    if (body != std::string::npos) body = fs.find('{', body);
    if (body == std::string::npos) return false;
    std::string head = fs.substr(0, body + 1), rest = fs.substr(body + 1);
    rest = std::regex_replace(rest, std::regex(R"(\bTex0\b)"), "wbm_uv");
    fs = head +
         "\n  vec2 wbm_uv = Tex0;\n"
         "  if (wbm_parallax.x > 0.0) {\n"
         "    vec2 wbm_dx = dFdx(Tex0), wbm_dy = dFdy(Tex0);\n"
         "    float wbm_near = 1.0 - smoothstep(0.004, 0.012, length(dFdx(wbm_tview)));\n" // metres to the pixel
         "    vec3 wbm_v = normalize(wbm_tview);\n"         // The depth is in metres: how much of the texture a metre of this surface covers is had from how fast each changes across the screen.
         "    float wbm_per_metre = min(length(wbm_dx) / max(length(dFdx(wbm_tview)), 1e-6), 4.0);\n"
         "    float wbm_blur = 0.03 * wbm_per_metre / max(max(length(wbm_dx), length(wbm_dy)), 1e-6);\n"
         "    vec2 wbm_bx = wbm_dx * max(wbm_blur, 1.0), wbm_by = wbm_dy * max(wbm_blur, 1.0);\n"
         "    if (wbm_near > 0.0 && wbm_v.z > 0.0) {\n"
         "      vec2 wbm_step = wbm_v.xy * vec2(wbm_parallax.y, wbm_parallax.z) / max(wbm_v.z, 0.4) * (wbm_parallax.x * wbm_per_metre * wbm_near / 8.0);\n"
         "      float wbm_layer = 1.0, wbm_h = 0.0, wbm_was = 1.0;\n"
         "      for (int wbm_i = 0; wbm_i < 8; wbm_i++) {\n"
         // Read well blurred: it is the stones and the joints between them that have depth, not the grain painted on them.
         // (Unless a pack has the surface's heights as a map of their own: wbm_parallax.w says so, and they are read as they are.)
         "        wbm_h = wbm_parallax.w > 0.0 ? textureGrad(wbm_depth_map, wbm_uv, wbm_dx, wbm_dy).x\n"
         "                                     : smoothstep(0.1, 0.5, dot(textureGrad(diffuse_texture, wbm_uv, wbm_bx, wbm_by).xyz, vec3(0.3333)));\n"
         "        if (wbm_h >= wbm_layer) break;\n"
         "        wbm_was = wbm_layer - wbm_h;\n"
         "        wbm_uv -= wbm_step;\n"
         "        wbm_layer -= 0.125;\n"
         "      }\n"
         "      float wbm_over = wbm_h - wbm_layer;\n" // how far past the surface the last step went
         "      if (wbm_over > 0.0) wbm_uv += wbm_step * (wbm_over / max(wbm_over + wbm_was, 1e-4));\n"
         "    }\n"
         "  }" +
         rest;
    fs.insert(0, "varying vec3 wbm_tview;\nuniform sampler2D wbm_depth_map;\n");
    return true;
}

// The vertex shader's part: the direction to the eye in the surface's own
// space (along the texture, across it, and out), which the engine's shaders
// work out for the sun but not for the eye. For meshes without bones.
bool add_parallax_vertex(std::string &vs)
{
    static const std::regex basis(R"(\((tmpvar_\d+) \* -\(vSunDir\.xyz\)\))");
    std::smatch m;
    const char *needed[] = {"uniform vec4 vCameraPos;", "uniform mat4 matWorld;", "attribute vec3 inPosition;", "varying vec3 SunLightDir;"};
    for (const char *n : needed)
        if (vs.find(n) == std::string::npos) return false;
    size_t end = vs.rfind('}');
    if (vs.find("matWorldArray") != std::string::npos || !std::regex_search(vs, m, basis) || end == std::string::npos) return false;
    vs.insert(end, "  wbm_tview = " + std::string(m[1]) + " * (vCameraPos.xyz - (matWorld * vec4(inPosition, 1.0)).xyz);\n");
    vs.insert(0, "varying vec3 wbm_tview;\n");
    return true;
}

// People's hair as strands. The engine's hair shader draws a rigid mesh whose
// binormal is the way the hair runs. For characters near the camera the backend
// draws ribbons over it (build_strands), all of whose vertices sit at a root on
// that mesh; here each is laid out along the hair's run, lifted a little off
// the mesh and drooping a little, and widened across the line of sight.
//   colour x: segment * 2 + side; y: a chance number (0 to 255); z: how long
//   strands are at this root, as a share of the longest.
// wbm_shell.x: -(the longest strand's length), metres.
bool add_hair_strands(std::string &glsl, bool vertex)
{
    if (glsl.find("varying vec3 tangent;") == std::string::npos || glsl.find("varying vec3 viewVec;") == std::string::npos) return false;
    size_t body = glsl.find("void main ()");
    if (body != std::string::npos) body = glsl.find('{', body);
    if (body == std::string::npos) return false;
    if (!vertex) {
        static const std::regex result(R"(gl_FragColor = (\w+);)");
        static const std::regex encode(R"((\w+)\.xyz = pow \(\1\.xyz, output_gamma_inv\.xyz\);)");
        if (glsl.find("0.800118") == std::string::npos || !std::regex_search(glsl, result)) return false;
        // The engine's own two streaks of highlight make way for the bands above, on the mesh and on the strands alike.
        std::smatch m;
        std::string sun = std::regex_search(glsl, m, std::regex(R"(\bsun_amount_\d+\b)")) ? m.str() : "1.0";
        std::string colour = std::regex_search(glsl, m, std::regex(R"(\bfinal_col_\d+\b)")) ? m.str() + ".xyz" : "vec3(0.5)";
        if (std::regex_search(glsl, encode) && glsl.find("uniform vec4 vSunDir;") != std::string::npos && glsl.find("uniform vec4 vSunColor;") != std::string::npos &&
            glsl.find("uniform vec4 vAmbientColor;") != std::string::npos) {
            glsl = std::regex_replace(glsl, std::regex(R"(vec3\(0\.800118, 0\.88902, 0\.88902\))"), "vec3(0.2, 0.22, 0.22)");
            glsl = std::regex_replace(glsl, std::regex(R"(vec3\(0\.7326, 0\.65934, 0\.7326\))"), "vec3(0.18, 0.16, 0.18)");
            glsl = std::regex_replace(glsl, encode,
                                      "vec4 wbm_shine = wbm_people_hair(normal, tangent, viewVec, -vSunDir.xyz, " + colour + ", Tex0);\n"
                                      "  $1.xyz += wbm_shine.xyz * vSunColor.xyz * " + sun + " + wbm_shine.w * vAmbientColor.xyz;\n"
                                      // Hair shades hair: a strand is darker at its root, down among the others, than at its tip.
                                      "  if (wbm_shell.x < 0.0) $1.xyz *= mix(0.7, 1.0, wbm_strand.y);\n  $&",
                                      std::regex_constants::format_first_only);
        }
        // A strand has the texture's colour where it is rooted, in a tone of its
        // own; where the texture is clear there is no hair to root one in.
        glsl = std::regex_replace(glsl, result,
                                  "if (wbm_shell.x < 0.0) {\n    wbm_strand_soft = step(0.0, wbm_shell.z);\n    $1.w = wbm_strand_pass(wbm_strand_edge(false) * smoothstep(0.3, 0.6, $1.w));\n"
                                  "    if ($1.w < 0.02) discard;\n    $1.xyz *= 0.75 + 0.45 * fract(wbm_strand.z * 9.7);\n  }\n  $&");
        glsl.insert(body + 1, "\n  wbm_strand = wbm_strand_v;");
        glsl.insert(0, "varying vec4 wbm_strand_v;\n");
        return true;
    }
    static const std::regex placed(R"((\w+)\.xyz = inPosition;)");
    const char *needed[] = {"attribute vec3 inBinormal;", "attribute vec4 inColor0;", "attribute vec3 inNormal;", "uniform mat4 matWorld;", "uniform vec4 vCameraPos;"};
    for (const char *n : needed)
        if (glsl.find(n) == std::string::npos) return false;
    if (glsl.find("matWorldArray") != std::string::npos || !std::regex_search(glsl, placed)) return false;
    glsl = std::regex_replace(glsl, std::regex(R"(\binColor0\b(?!;))"), "wbm_colour");
    if (glsl.find("Tex0 = inTexCoord;") == std::string::npos) return false;
    glsl = std::regex_replace(glsl, std::regex(R"(Tex0 = inTexCoord;)"), "Tex0 = wbm_uv;");
    body = glsl.find('{', glsl.find("void main ()"));
    glsl.insert(body + 1, "\n  wbm_colour = wbm_shell.x < 0.0 ? vec4(1.0) : inColor0;\n  wbm_uv = inTexCoord;\n  wbm_strand_v = vec4(0.0);");
    glsl = std::regex_replace(glsl, placed,
                              "$&\n  if (wbm_shell.x < 0.0) {\n"
                              "    float wbm_v = floor(inColor0.x * 255.0 + 0.5), wbm_chance = inColor0.y;\n"
                              "    float wbm_side = mod(wbm_v, 2.0) * 2.0 - 1.0, wbm_t = floor(wbm_v / 2.0) / 8.0;\n"
                              "    vec3 wbm_n = normalize(inNormal), wbm_down = normalize(vec3(0.0, 0.0, -1.0) * mat3(matWorld));\n"
                              "    vec3 wbm_c = dot(inBinormal, inBinormal) > 0.0004 ? normalize(inBinormal) : wbm_down;\n"
                              "    vec3 wbm_out = normalize(wbm_c + wbm_n * (fract(wbm_chance * 3.1) > 0.95 ? 0.45 : 0.02 + 0.14 * fract(wbm_chance * 3.1)));\n"
                              "    float wbm_len = -wbm_shell.x * inColor0.z * (0.4 + 0.6 * wbm_chance), wbm_k = 0.4 * (1.0 - 0.9 * min(length(wbm_drift.xyz), 1.0));\n"
                              "    vec3 wbm_dir = wbm_out * (1.0 - wbm_k * wbm_t) + wbm_down * (wbm_k * wbm_t);\n"
                              "    vec3 wbm_off = wbm_len * (wbm_out * (wbm_t - 0.5 * wbm_k * wbm_t * wbm_t) + wbm_down * (0.5 * wbm_k * wbm_t * wbm_t));\n"
                              // The wind: gusts that pass over the head, each strand a little out of step, the tip moving most.
                              "    float wbm_time = wbm_shell.w - 1.0, wbm_place = dot(inPosition, vec3(7.0, 5.0, 3.0));\n"
                              "    vec3 wbm_blow = vec3(sin(wbm_time * 2.3 + wbm_place + wbm_chance * 2.0) + 0.5 * sin(wbm_time * 5.1 + wbm_chance * 40.0), cos(wbm_time * 1.7 + wbm_place * 1.3 + wbm_chance * 2.0) + 0.5 * cos(wbm_time * 4.3 + wbm_chance * 31.0), 0.0);\n"
                              "    vec3 wbm_sway = (wbm_blow * mat3(matWorld)) * (0.16 * wbm_shell.y * wbm_len * wbm_t * wbm_t);\n"
                              "    wbm_off += wbm_sway + (wbm_drift.xyz * mat3(matWorld)) * (wbm_len * wbm_t * wbm_t);\n"
                              "    wbm_off *= min(1.0, wbm_len * wbm_t / max(length(wbm_off), 1e-6));\n"
                              "    $1.xyz += wbm_off;\n"
                              "    wbm_dir = normalize(wbm_dir * wbm_len + wbm_sway * 2.0 + (wbm_drift.xyz * mat3(matWorld)) * (2.0 * wbm_len * wbm_t));\n"
                              "    vec3 wbm_eye = (vCameraPos.xyz - matWorld[3].xyz) * mat3(matWorld) - $1.xyz;\n"
                              "    vec3 wbm_across = cross(wbm_dir, wbm_eye);\n"
                              "    wbm_across = dot(wbm_across, wbm_across) > 1e-12 ? normalize(wbm_across) : wbm_n;\n"
                              "    float wbm_w = 0.0004 * (0.6 + 0.8 * fract(wbm_chance * 7.3)) * (1.0 - 0.5 * wbm_t);\n" // half the width of a lock, metres
                              "    float wbm_used = max(wbm_w, 0.0012 * length(wbm_eye));\n"
                              "    $1.xyz += wbm_across * (wbm_side * wbm_used);\n"
                              "    wbm_strand_v = vec4(wbm_side, wbm_t, wbm_chance, wbm_w / wbm_used);\n"
                              "  }",
                              std::regex_constants::format_first_only);
    glsl.insert(0, "varying vec4 wbm_strand_v;\n");
    return true;
}

// Shaders that light per vertex get the wrap alone, from the vertex normal.
const char k_skin_vertex_function[] = R"(
uniform vec4 wbm_skin;

vec4 wbm_skin_wrap(float facing)
{
    float plain = clamp(facing, 0.0, 1.0);
    if (wbm_skin.w > 0.0) return vec4(max(vec3(plain), clamp(-facing, 0.0, 1.0) * wbm_skin.w * vec3(0.85, 1.0, 0.35)), plain);
    if (wbm_skin.x < 0.5) return vec4(plain);
    const vec3 wrap = vec3(0.38, 0.16, 0.07);
    vec3 lit = clamp((vec3(facing) + wrap) / (1.0 + wrap), 0.0, 1.0) / (1.0 + 0.5 * wrap);
    return vec4(lit, plain);
}
)";

bool add_skin_scattering_vertex(std::string &vs)
{
    // SunLight = (((clamp(dot(N, -vSunDir), 0, 1) * sun colour) * material) * vertex colour)
    static const std::regex term(R"(\(\s*clamp \(dot \(([\w.]+), -\(vSunDir\.xyz\)\), 0\.0, 1\.0\)\s*\* (tmpvar_\d+)\))");
    if (vs.find("varying vec4 SunLight;") == std::string::npos || !std::regex_search(vs, term)) return false;
    vs = std::regex_replace(vs, term, "(wbm_skin_wrap(dot($1, -(vSunDir.xyz))) * $2)", std::regex_constants::format_first_only);
    return true;
}

// Strands: hair drawn as ribbons that the vertex shader lays out (see
// add_shell_push and add_hair_strands). wbm_strand says where on its ribbon a
// pixel is: x -1 to 1 across, y 0 at the root to 1 at the tip, z a chance
// number of the strand's own, w how much of the ribbon's drawn width is hair.
const char k_strand_function[] = R"(
vec4 wbm_strand = vec4(0.0);
// Strands are drawn twice: first what is solid, which leaves depth; then the
// soft edges and thin tips, blended, which leave none and so hide nothing behind them.
float wbm_strand_soft = 0.0; // 1 in the second

// How much of the pixel the strand covers: its edges are soft, and it thins out to nothing at the tip.
// How much of the pixel the ribbon covers, 0 to 1. fibres: whether it is a lock of separate hairs with gaps between.
float wbm_strand_edge(bool fibres)
{
    float pixel = max(fwidth(wbm_strand.x), 1e-4);
    // A ribbon is a lock: some hairs side by side with gaps between, more gaps towards the tip.
    float hairs = wbm_strand.x * 2.0 + wbm_strand.z * 31.0;
    float gap = abs(fract(hairs) - 0.5) * 2.0; // 0 on a hair, 1 between two
    float thin = mix(0.25, 0.9, wbm_strand.y) * (0.6 + 0.4 * fract(sin(floor(hairs) * 12.9898 + wbm_strand.z * 78.233) * 43758.5453));
    float lock = mix(1.0, 1.0 - thin * smoothstep(0.35, 0.9, gap), clamp(0.6 / (pixel * 2.0), 0.0, 1.0)); // too fine to see, they even out
    float cover = clamp((1.0 - abs(wbm_strand.x)) / pixel + 0.5, 0.0, 1.0) * wbm_strand.w * smoothstep(1.0, 0.8, wbm_strand.y) * (fibres ? lock : 1.0);
    return cover;
}

// The same, as each of the two draws wants it: all or nothing in the first, the rest in the second.
float wbm_strand_pass(float cover)
{
    if (wbm_strand_soft > 0.5) return cover < 0.7 ? cover : 0.0;
    return cover < 0.7 ? 0.0 : 1.0;
}

float wbm_strand_cover()
{
    return wbm_strand_pass(wbm_strand_edge(true));
}
)";

// Fur. See XlOptions::fur.
const char k_fur_function[] = R"(
uniform sampler2D wbm_fur_map;  // blue: how much of the surface is coat (a texture's _pbr map)
uniform sampler2D wbm_hair_map; // a texture's _hair map: red and green the way the hair is combed, in the picture; blue 1 long hair, about 0.5 coat, 0 none
uniform vec4 wbm_fur;       // x: strength of the coat, 0 without a map; y: metres per unit of shadow-map depth, or 0; z: light through thin parts; w: how much of the texture a metre of the surface covers, negated without a hair map
uniform vec4 wbm_hair_colour; // the long hair's own colour; w 0 when it is not known
// The coat lit as glTF's KHR_materials_anisotropy and KHR_materials_sheen have it, where the texture pack
// gives a horse a hair map made from an anisotropy map, a sheen map and a _pbr map (texture_tool import).
// x: 1 when it has all three; y: the anisotropy strength (times the hair map's blue); z: the sheen's strength.
uniform vec4 wbm_sheen;
uniform sampler2D wbm_sheen_map; // colour (sRGB) and, in alpha, the sheen's roughness
float wbm_hair_cover = 1.0; // of the pixel, by this layer's hair (wbm_shell_cover)
vec3 wbm_hair_shine = vec3(0.0); // what long hair reflects and lets through, per unit of sunlight; added to the finished colour
uniform vec4 wbm_horse_eye; // xy: the eye's place in the texture; z: its size there; w: strength, 0 for none
// wbm_shell. x: how far out this layer is, metres (vertex stage); y: the same
// as a share of the longest hair, 0 for the surface itself; z: hairs per unit
// of texture coordinate; w: 1 on the surface when layers will follow.
uniform vec4 wbm_shell;
// A strand drawn as a ribbon (wbm_shell.x below 0, see add_shell_push) says
// where on it this is: x -1 to 1 across, y 0 at the root to 1 at the tip, z a
// chance number of its own, w how much of its drawn width is really hair. It
// stands for a layer of long hair as far along as it has got.
vec4 wbm_shell_given() { return wbm_shell; }
vec4 wbm_layer;
#define wbm_shell wbm_layer

// What grows at a place: x how much it is coat or hair at all, y how much of that is long hair.
vec2 wbm_grows(vec2 uv)
{
    if (wbm_fur.w > 0.0) {
        float kind = texture2D(wbm_hair_map, uv).z;
        return vec2(smoothstep(0.15, 0.35, kind), smoothstep(0.65, 0.85, kind));
    }
    return vec2(texture2D(wbm_fur_map, uv).z, 0.0);
}

// The way the hair is combed, along the surface, in the space the shader
// lights in (x along the texture, y up it). Zero where that is not known.
vec3 wbm_combed(vec2 uv)
{
    if (wbm_fur.w <= 0.0) return vec3(0.0);
    vec2 d = texture2D(wbm_hair_map, uv).xy * 2.0 - 1.0;
    return vec3(d.x, -d.y, 0.0); // the picture's rows run downwards
}

// The glTF way (wbm_sheen.x): what the surface sends back of the sun over and above its diffuse light, per
// unit of sunlight, untinted by the base colour. A microfacet highlight (GGX, height-correlated Smith), its
// roughness from the _pbr map, stretched along the hair as KHR_materials_anisotropy stretches it
// (alpha along = mix(alpha, 1, strength^2)), Fresnel from the metalness; and KHR_materials_sheen's
// sheen: the "Charlie" distribution and Neubelt's visibility, in the sheen map's colour and roughness.
// Everything here is in the space the shader lights in, with x along the texture and y up it.
vec3 wbm_gltf_shine(vec3 view, vec3 normal, vec3 to_sun, vec2 uv)
{
    vec3 n = normalize(normal), v = normalize(view), l = normalize(to_sun), h = normalize(v + l);
    float nl = clamp(dot(n, l), 0.0, 1.0), nv = clamp(dot(n, v), 1e-3, 1.0), nh = clamp(dot(n, h), 0.0, 1.0);
    if (nl <= 0.0) return vec3(0.0);
    vec4 maps = texture2D(wbm_fur_map, uv);
    float metal = maps.x, rough = max(maps.y, 0.08), alpha = rough * rough;
    vec4 flow = texture2D(wbm_hair_map, uv);
    vec2 d = flow.xy * 2.0 - 1.0;
    vec3 t = vec3(d.x, -d.y, 0.0); // the picture's rows run downwards
    t -= n * dot(n, t);
    float strength = dot(t, t) > 1e-4 ? flow.z * wbm_sheen.y : 0.0;
    t = strength > 0.0 ? normalize(t) : normalize(abs(n.x) < 0.9 ? cross(n, vec3(1.0, 0.0, 0.0)) : cross(n, vec3(0.0, 1.0, 0.0)));
    vec3 b = cross(n, t);
    float at = mix(alpha, 1.0, strength * strength), ab = alpha;
    float th = dot(t, h), bh = dot(b, h), tv = dot(t, v), bv = dot(b, v), tl = dot(t, l), bl = dot(b, l);
    float f = th * th / (at * at) + bh * bh / (ab * ab) + nh * nh;
    float D = 1.0 / (3.14159265 * at * ab * f * f);
    float V = 0.5 / (nl * length(vec3(at * tv, ab * bv, nv)) + nv * length(vec3(at * tl, ab * bl, nl)));
    vec3 f0 = mix(vec3(0.04), vec3(0.6), metal);
    vec3 F = f0 + (1.0 - f0) * pow(1.0 - clamp(dot(v, h), 0.0, 1.0), 5.0);
    vec3 shine = D * V * F * nl;
    // The sheen.
    vec4 s = texture2D(wbm_sheen_map, uv);
    vec3 colour = pow(max(s.xyz, vec3(0.0)), vec3(2.2)) * wbm_sheen.z;
    if (max(colour.x, max(colour.y, colour.z)) > 0.0) {
        float r = max(s.w * s.w, 0.07), sin_h = sqrt(max(1.0 - nh * nh, 0.0));
        float charlie = (2.0 + 1.0 / r) * pow(sin_h, 1.0 / r) / (2.0 * 3.14159265);
        float neubelt = 1.0 / (4.0 * (nl + nv - nl * nv));
        shine += colour * charlie * neubelt * nl;
    }
    return shine;
}

// facing: the cosine of the sun's angle to the surface; highlight: the engine's
// own; depth: how far behind the sunward surface this point is
// (wbm_dyn_thickness), below 0 when unknown.
vec4 wbm_fur_lit(float facing, vec4 highlight, vec3 view, vec3 normal, vec3 to_sun, vec2 uv, float depth)
{
    vec4 plain = vec4(clamp(facing, 0.0, 1.0)) + highlight;
    if (wbm_sheen.x > 0.5 && wbm_shell.x >= 0.0 && wbm_shell.y <= 0.0) {
        // The glTF way, on the surface itself (layers of fur and strands keep the hair's own way, below): the
        // diffuse light alone goes through the base colour; the highlight and the sheen are added to the
        // finished colour (wbm_hair_shine), as light reflected off the hair, not through it. The engine's own
        // highlight and the fur's stand-ins for these (rim, strands) are left out.
        wbm_hair_shine = wbm_gltf_shine(view, normal, to_sun, uv);
        return vec4(clamp(facing, 0.0, 1.0), clamp(facing, 0.0, 1.0), clamp(facing, 0.0, 1.0), plain.w);
    }
    vec2 grows = wbm_grows(uv);
    float fur = grows.x * wbm_fur.x;
    if (fur <= 0.0) return plain;
    float wrapped = clamp((facing + 0.3) / 1.3, 0.0, 1.0) / 1.12;
    // Hairs standing off the surface catch light when seen along it: brightest
    // at the outline, on the side the sun is on.
    vec3 v = normalize(view), l = normalize(to_sun);
    float edge = 1.0 - clamp(dot(v, normalize(normal)), 0.0, 1.0);
    float rim = edge * edge * edge * clamp(facing * 0.5 + 0.5, 0.0, 1.0) * 0.55;
    // Sunlight through what is thin, an ear above all, seen against the sun;
    // hide and hair let less through than a person's skin.
    vec3 back = vec3(0.0);
    if (depth >= 0.0 && wbm_fur.y > 0.0) {
        float towards = pow(clamp(dot(v, -normalize(l + vec3(0.0, 0.0, 0.35))), 0.0, 1.0), 2.0);
        back = towards * smoothstep(0.25, -0.15, l.z) * wbm_fur.z * exp(-(depth * wbm_fur.y + 0.004) * vec3(70.0, 190.0, 300.0));
    }
    if (wbm_shell.x < 0.0) back = vec3(0.0); // a strand hangs clear of the body: what the map knows of thickness there is not its own

    // Long hair, after Marschner: a strand sends light back three ways. Off
    // its surface, white, in a narrow band shifted towards the root. From
    // inside, having passed through the hair twice, in its colour, in a wider
    // band shifted towards the tip and broken up from strand to strand. And
    // straight through, when the sun is behind it, in its colour again. None
    // of that depends on what is painted under the hair, so it is kept apart
    // (wbm_hair_shine) and added to the finished colour.
    float strands = 0.0;
    vec3 combed = wbm_combed(uv);
    if (grows.y > 0.0 && dot(combed, combed) > 0.04) {
        vec3 n = normalize(normal), t = normalize(combed);
        vec3 h = normalize(v + l);
        float jitter = fract(sin(dot(floor(uv * 900.0), vec2(12.9898, 78.233))) * 43758.5453) - 0.5;
        vec3 t_surface = normalize(t - n * 0.10), t_inside = normalize(t + n * (0.16 + 0.10 * jitter));
        float a = dot(t_surface, h), b = dot(t_inside, h);
        float off_surface = pow(sqrt(max(1.0 - a * a, 0.0)), 220.0), from_inside = pow(sqrt(max(1.0 - b * b, 0.0)), 36.0);
        vec3 hue = wbm_hair_colour.w > 0.0 ? wbm_hair_colour.xyz / max(max(wbm_hair_colour.x, max(wbm_hair_colour.y, wbm_hair_colour.z)), 0.02) : vec3(1.0, 0.85, 0.7);
        hue = mix(vec3(1.0), hue, 0.75);
        // Dark hair swallows what goes into it: little comes back from inside or passes through.
        float fair = wbm_hair_colour.w > 0.0 ? smoothstep(0.08, 0.45, max(wbm_hair_colour.x, max(wbm_hair_colour.y, wbm_hair_colour.z))) : 0.5;
        hue *= mix(0.2, 1.0, fair);
        float lit_side = clamp(facing * 0.7 + 0.5, 0.0, 1.0);
        float through = pow(clamp(dot(-v, l), 0.0, 1.0), 7.0) * (1.0 - abs(dot(t, l))) ;
        // Hair shadows hair: sunlight reaching a layer has come through those
        // above it, and through more of them the lower the sun stands over the
        // surface. (The shadow map adds what the mane casts on the body and on itself as a whole.)
        float outer = wbm_shell.z < 0.0 ? wbm_shell.y : 0.0; // 1 at the tips, 0 on the skin under the hair
        float through_hair = exp(-0.9 * (1.0 - outer) / max(normalize(l).z, 0.35));
        wrapped *= mix(1.0, through_hair, grows.y);
        rim *= mix(1.0, through_hair, grows.y);
        wbm_hair_shine = through_hair * grows.y * wbm_fur.x * (lit_side * (0.10 * off_surface + 0.16 * from_inside * hue) + 0.22 * through * hue);
        // A strand is lit by how far it lies across the light, not by which way the skin under it faces.
        float c = dot(t, l);
        strands = grows.y * (0.8 * sqrt(max(1.0 - c * c, 0.0)) * clamp(facing * 0.5 + 0.6, 0.0, 1.0) - wrapped) * 0.25 * through_hair;
    }
    vec4 coat = vec4(vec3(wrapped + rim + strands) + back, plain.w) + vec4(highlight.xyz * mix(0.45, 0.0, grows.y), 0.0);
    return mix(plain, coat, fur);
}

// Whether this layer has a hair here: each cell of a fine grid over the
// texture holds one, of a length of its own, thinner towards its tip. The
// hair leans the way it is combed and droops (down: which way that is along
// the surface), so a layer farther out finds its hairs a little way along.
//
// Long hair (mane, tail) gets layers of its own, marked by a negative
// wbm_shell.z: fewer, thicker strands that reach several centimetres out,
// hang much farther along the surface, and swing a little (wbm_shell.w
// carries the time, as 1 + seconds).
// Long hair has one colour of its own (wbm_hair_colour, when w is above 0),
// lighter and darker as the texture is where the hair is rooted: what else is
// painted across the mane there, a strap or a patch of coat, keeps only its
// brightness.
vec4 wbm_hair_tint(vec4 painted)
{
    if (wbm_shell.z >= 0.0 || wbm_shell.y <= 0.0 || wbm_hair_colour.w <= 0.0) return painted;
    float tone = dot(painted.xyz, vec3(0.3333)) / max(dot(wbm_hair_colour.xyz, vec3(0.3333)), 0.004);
    // A strand is of one tone from end to end, its own; what is painted at its root (a highlight on the crest) is not it.
    if (wbm_shell.x < 0.0) tone = (0.55 + 0.75 * fract(wbm_strand.z * 9.7)) * mix(0.75, 1.0, wbm_strand.y);
    return vec4(wbm_hair_colour.xyz * clamp(tone, 0.55, 1.7), painted.w);
}

// Noise that changes smoothly from place to place, 0 to 1.
float wbm_noise(vec2 p)
{
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = fract(sin(dot(i, vec2(127.1, 311.7))) * 43758.5453), b = fract(sin(dot(i + vec2(1.0, 0.0), vec2(127.1, 311.7))) * 43758.5453);
    float c = fract(sin(dot(i + vec2(0.0, 1.0), vec2(127.1, 311.7))) * 43758.5453), d = fract(sin(dot(i + vec2(1.0, 1.0), vec2(127.1, 311.7))) * 43758.5453);
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Where on the texture the hair passing over uv in this layer is rooted.
vec2 wbm_shell_root(vec2 uv, vec3 down)
{
    if (wbm_shell.y <= 0.0 || wbm_shell.x < 0.0) return uv;
    // Lengths are in metres; how much of the texture a metre covers differs from one horse to the next.
    float per_metre = max(abs(wbm_fur.w), 0.02), share = wbm_shell.y;
    vec2 lean;
    if (wbm_shell.z < 0.0) {
        float t = wbm_shell.w - 1.0;
        vec2 swing = vec2(sin(t * 1.7 + uv.x * 40.0), cos(t * 1.3 + uv.y * 31.0)) * 0.012;
        lean = (wbm_combed(uv).xy * 0.02 + swing) * share * share + wbm_combed(uv).xy * 0.01 * share;
    } else {
        lean = wbm_combed(uv).xy * 0.007 * share + down.xy * 0.005 * share * share;
    }
    return uv - lean * per_metre;
}

// How much of the pixel a layer's hair covers here: 1 on the surface itself, 0
// where the layer is bare. A hair's edge, and a hair thinner than a pixel,
// cover part of it and are blended in; cut out whole or not at all they dither.
float wbm_shell_cover(vec2 uv, vec3 down)
{
    if (wbm_shell.x < 0.0) return wbm_strand_cover();
    if (wbm_shell.y <= 0.0) return 1.0;
    bool long_ = wbm_shell.z < 0.0;
    float share = wbm_shell.y, density = abs(wbm_shell.z) / max(abs(wbm_fur.w), 0.02) * (long_ ? 0.17 : 1.0); // long hair is coarser than the coat
    vec2 grows;
    // What grows is asked where the hair is rooted, not where it has got to.
    vec2 root = wbm_shell_root(uv, down);
    grows = wbm_grows(root);
    if (long_ ? grows.y < 0.5 : grows.x < 0.5) return 0.0;
    // Long hair is drawn only over what the map itself marks as long hair. Its
    // layers hang by being moved (see add_shell_push), not by reaching across
    // the texture, where a leg may be painted beside the tail.
    if (long_ && wbm_grows(uv).y < 0.5) return 0.0;
    if (long_) {
        // Seen close, a hair cut through by a layer must not show as a dot: in
        // each layer a long hair is a streak lying the way it is combed, many
        // times longer than it is wide, so that the layers together make strands.
        vec2 along = wbm_combed(root).xy;
        along = dot(along, along) > 0.04 ? normalize(along) : vec2(0.0, -1.0);
        float across = dot(root, vec2(-along.y, along.x)) * density;
        float strand = floor(across);
        float run = dot(root, along) * density / 16.0 + fract(sin(strand * 12.9898) * 43758.5453) * 9.0;
        float chance = fract(sin(dot(vec2(strand, floor(run)), vec2(12.9898, 78.233))) * 43758.5453);
        // Hair gathers into locks: several strands side by side share a length
        // and a thickness, with thin gaps between one lock and the next that
        // widen towards the tips, and a stray thin hair here and there.
        float lock = floor(across / 7.0 + 0.6 * wbm_noise(vec2(across / 23.0, run * 0.08)));
        float lock_chance = fract(sin(lock * 78.233) * 43758.5453);
        float edge = abs(fract(across / 7.0 + 0.6 * wbm_noise(vec2(across / 23.0, run * 0.08))) - 0.5) * 2.0; // 0 mid-lock, 1 at its edge
        float length_ = (0.45 + 0.55 * lock_chance) * (0.85 + 0.15 * chance);
        if (share > length_) return 0.0;
        if (edge > 1.0 - 0.35 * share / length_ && chance > 0.12) return 0.0; // the gap, but for a stray hair
        float thick = mix(0.25, 0.5, fract(lock_chance * 13.7)) * (1.0 - 0.5 * share / length_);
        float wide = max(fwidth(across), 1e-4); // strands to the pixel
        return clamp(0.5 + (thick - abs(fract(across) - 0.5)) / wide, 0.0, 1.0) * min(1.0, 2.0 * thick / wide + 0.35);
    }
    // Hairs on a plain grid make stripes with the layers and with the mesh's
    // own triangles. So each hair stands off the middle of its cell by a
    // chance amount, and how long the hairs are varies smoothly from place to
    // place, in tufts, as well as from hair to hair.
    // A hair smaller than a pixel is haze. Farther off, where the pixels are
    // larger on the horse, there are fewer hairs and they are thicker: the
    // count is halved as often as it takes for a hair's cell to span two
    // pixels, the two nearest counts being blended by which hairs are let through.
    float pixel = max(length(dFdx(root)), length(dFdy(root))) * density; // cells to the pixel
    float halvings = max(0.0, log2(max(pixel * 2.0, 1e-6)));
    float coarser = floor(halvings), between = halvings - coarser;
    density /= exp2(coarser);
    vec2 at = root * density;
    vec2 cell = floor(at);
    float chance = fract(sin(dot(cell, vec2(12.9898, 78.233))) * 43758.5453);
    vec2 centre = 0.5 + 0.28 * (vec2(fract(chance * 37.17), fract(chance * 91.73)) - 0.5) * 2.0;
    vec2 within = fract(at) - centre;
    // A hair of the coat lies along the way the coat is combed, where the map
    // knows it: seen from above it is longer that way than it is across.
    vec2 lie = wbm_combed(root).xy;
    float known = dot(lie, lie);
    if (known > 0.01) {
        lie /= sqrt(known);
        within = vec2(dot(within, lie) / (1.0 + 1.6 * min(sqrt(known), 1.0)), dot(within, vec2(-lie.y, lie.x)));
    }
    float tufts = 0.6 * wbm_noise(at / 9.0) + 0.4 * wbm_noise(at / 3.5 + 17.0);
    float length_ = (0.45 + 0.15 * chance + 0.4 * tufts) * mix(0.6, 1.0, grows.y); // neighbours are much of a length; it is the tufts that differ
    if (share > length_) return 0.0;
    // Towards the next halving, the hairs that will not survive it go, and the rest grow to fill in.
    if (fract(chance * 5.31) < between * 0.75 && (mod(cell.x, 2.0) > 0.5 || mod(cell.y, 2.0) > 0.5)) return 0.0;
    float thickness = (0.36 + 0.1 * between) * (1.0 - 0.75 * share / length_);
    return clamp(0.5 + (thickness - length(within)) / max(pixel / exp2(coarser), 1e-4), 0.0, 1.0);
}

// Hairs shade one another: dark at the roots, full at the tips.
float wbm_shell_shade(vec2 uv)
{
    // The coat's roots are only a little darker than its tips: more, and the
    // gaps between hairs a pixel wide show as dark specks all over the horse.
    if (wbm_shell.y > 0.0) return (wbm_shell.z < 0.0 ? 0.72 : 0.82) + (wbm_shell.z < 0.0 ? 0.28 : 0.18) * wbm_shell.y; // long hair: the sun's part is in wbm_fur_lit
    return wbm_shell.w > 0.5 && wbm_grows(uv).x >= 0.5 ? 0.82 : 1.0;
}

// What long hair reflects of the sky: a broad sheen where the strands lie
// across the line halfway between the eye and straight up, strongest seen
// edge-on, white with a little of the hair's colour. Per unit of sky light.
vec3 wbm_hair_sky(vec2 uv, vec3 view, vec3 normal, vec3 up)
{
    vec2 grows = wbm_grows(uv);
    vec3 combed = wbm_combed(uv);
    if (grows.y <= 0.0 || dot(combed, combed) <= 0.04 || dot(up, up) < 0.0001) return vec3(0.0);
    vec3 v = normalize(view), n = normalize(normal), t = normalize(normalize(combed) - n * 0.05);
    float along = dot(t, normalize(v + normalize(up)));
    float sheen = pow(sqrt(max(1.0 - along * along, 0.0)), 14.0);
    float edge = 0.25 + 0.75 * pow(1.0 - clamp(dot(v, n), 0.0, 1.0), 3.0);
    vec3 hue = wbm_hair_colour.w > 0.0 ? wbm_hair_colour.xyz / max(max(wbm_hair_colour.x, max(wbm_hair_colour.y, wbm_hair_colour.z)), 0.02) : vec3(1.0);
    float outer = wbm_shell.z < 0.0 ? wbm_shell.y : 0.0;
    return grows.y * wbm_fur.x * 0.5 * sheen * edge * (0.35 + 0.65 * outer) * mix(vec3(1.0), hue, 0.35);
}

// A wet eye: a sharp highlight from the sun and a little of the sky, on top of
// whatever glint is painted there.
vec3 wbm_horse_glint(vec2 uv, vec3 view, vec3 normal, vec3 to_sun, float shadow, vec3 sun, vec3 sky)
{
    if (wbm_horse_eye.w <= 0.0) return vec3(0.0);
    vec2 d = (uv - wbm_horse_eye.xy) / wbm_horse_eye.z;
    float inside = 1.0 - smoothstep(0.5, 1.0, dot(d, d));
    if (inside <= 0.0) return vec3(0.0);
    vec3 n = normalize(normalize(normal) + vec3(d * 0.7, 0.0)); // the ball bulges
    vec3 v = normalize(view), h = normalize(v + normalize(to_sun));
    float glint = pow(max(dot(n, h), 0.0), 160.0) * 2.5 * shadow;
    // A wet eye reflects the sky the way water does: a few hundredths head-on, more only towards its rim.
    // A flat sheet of sky over the whole ball reads as a cataract, the dark of the eye greyed out.
    float grazing = min(0.02 + 0.98 * pow(1.0 - max(dot(n, v), 0.0), 5.0), 0.12);
    return inside * wbm_horse_eye.w * (glint * sun + grazing * sky * 0.35);
}
)";

bool add_fur(std::string &fs)
{
    // total = (... + (((clamp(dot(SunLightDir, N), 0, 1) + fSpecular) * sun_amount) * colour)), shadowed or not
    static const std::regex shadowed(
        R"(\(\(\s*\(clamp \(dot \(SunLightDir, ([\w.]+)\), 0\.0, 1\.0\) \+ (fSpecular_\d+)\)\s*\* (sun_amount_\d+)\) \* (tmpvar_\d+)\))");
    static const std::regex plain(R"(\(\(\s*clamp \(dot \(SunLightDir, ([\w.]+)\), 0\.0, 1\.0\)\s*\+ (fSpecular_\d+)\) \* (tmpvar_\d+)\))");
    static const std::regex encode(R"((\w+)\.xyz = pow \(\1\.xyz, output_gamma_inv\.xyz\);)");
    if (fs.find("0.800118") == std::string::npos || fs.find("varying vec3 ViewDir;") == std::string::npos) return false; // the engine's hair-like highlight marks the shaders
    const std::string fur = "Tex0";
    // Which way is down along the surface, for hair to droop: these shaders carry the world's up as the sky light's direction.
    const std::string down = fs.find("varying vec3 SkyLightDir;") != std::string::npos ? "-SkyLightDir" : "vec3(0.0)";
    const std::string depth = fs.find("wbm_dynamic_shadow(") != std::string::npos ? "wbm_dyn_thickness(ShadowTexCoord.xyz)" : "-1.0";
    std::smatch m;
    std::string normal, lit = "1.0";
    if (std::regex_search(fs, m, shadowed)) {
        normal = m[1], lit = m[3];
        fs = std::regex_replace(fs, shadowed, "((wbm_fur_lit(dot(SunLightDir, $1), $2, ViewDir, $1, SunLightDir, " + fur + ", " + depth + ") * $3) * $4)",
                                std::regex_constants::format_first_only);
    } else if (std::regex_search(fs, m, plain)) {
        normal = m[1];
        fs = std::regex_replace(fs, plain, "(wbm_fur_lit(dot(SunLightDir, $1), $2, ViewDir, $1, SunLightDir, " + fur + ", " + depth + ") * $3)", std::regex_constants::format_first_only);
    } else {
        return false;
    }
    fs = std::regex_replace(fs, encode, "$1.xyz *= wbm_shell_shade(Tex0);\n  $&", std::regex_constants::format_first_only);
    fs = std::regex_replace(fs, encode, "$1.xyz += wbm_hair_shine * vSunColor.xyz * (" + lit + " * wbm_shell_shade(Tex0));\n  $&", std::regex_constants::format_first_only);
    fs.insert(0, "varying vec4 wbm_strand_v;\n");
    size_t body = fs.find("void main ()");
    if (body != std::string::npos) body = fs.find('{', body);
    if (body != std::string::npos) fs.insert(body + 1, "\n  wbm_strand = wbm_strand_v;\n  wbm_layer = wbm_shell_given();\n  if (wbm_layer.x < 0.0) wbm_strand_soft = step(0.0, wbm_layer.z), wbm_layer = vec4(-1.0, max(wbm_strand.y, 0.02), -abs(wbm_layer.z), wbm_layer.w);\n  wbm_hair_cover = wbm_shell_cover(Tex0, " + down + ");\n  if (wbm_hair_cover < 0.02) discard;");
    // A layer is blended in by how much of the pixel its hair covers (the renderer turns blending on for layers).
    fs = std::regex_replace(fs, std::regex(R"(gl_FragColor = (\w+);)"), "$1.w = wbm_shell.y > 0.0 ? wbm_hair_cover : $1.w;\n  $&");
    // A hair has the colour of the place it grows from, not of what it hangs over.
    fs = std::regex_replace(fs, std::regex(R"(texture2D \(diffuse_texture, Tex0\))"), "wbm_hair_tint(texture2D (diffuse_texture, wbm_shell_root(Tex0, " + down + ")))");
    if (fs.find("uniform vec4 vAmbientColor;") != std::string::npos && fs.find("uniform vec4 vSkyLightColor;") != std::string::npos &&
        fs.find("varying vec3 SkyLightDir;") != std::string::npos)
        fs = std::regex_replace(fs, encode, "$1.xyz += wbm_hair_sky(Tex0, ViewDir, " + normal + ", SkyLightDir) * (vAmbientColor.xyz + vSkyLightColor.xyz);\n  $&",
                                std::regex_constants::format_first_only);
    if (fs.find("uniform vec4 vAmbientColor;") != std::string::npos)
        fs = std::regex_replace(fs, encode, "$1.xyz += wbm_horse_glint(Tex0, ViewDir, " + normal + ", SunLightDir, " + lit + ", vSunColor.xyz, vAmbientColor.xyz);\n  $&",
                                std::regex_constants::format_first_only);
    return true;
}

// Eyes. Where they are in a face texture is the same for every face of a sex
// (see wbm_eye_place), in the coordinates the head's mesh uses, which run upwards.
const char k_eye_function[] = R"(
uniform vec4 wbm_gaze; // xy: how far the eyes are turned, in the texture's coordinates; z: 1 for a woman's face

// Where uv is relative to the nearer eye: 0 at the middle of the iris, length
// 1 at the lids. Men's and women's heads are laid out differently in their
// textures (measured on manface_young and womanface_young).
vec2 wbm_eye_place(vec2 uv)
{
    bool woman = wbm_gaze.z > 0.5;
    vec2 left = woman ? vec2(0.375, 0.645) : vec2(0.388, 0.558), right = woman ? vec2(0.625, 0.645) : vec2(0.610, 0.558);
    vec2 reach = woman ? vec2(0.046, 0.023) : vec2(0.040, 0.019);
    vec2 a = (uv - left) / reach, b = (uv - right) / reach;
    return dot(a, a) < dot(b, b) ? a : b;
}

// Where to read the face's texture so that the iris sits wbm_gaze.xy off its
// painted place: all of the shift at the middle of the eye, none at the lids.
vec2 wbm_eye_uv(vec2 uv)
{
    vec2 d = wbm_eye_place(uv);
    float hold = 1.0 - smoothstep(0.15, 1.0, dot(d, d));
    return uv - wbm_gaze.xy * hold;
}

// painted: the face texture's colour here. The shine belongs to the ball, not
// to the lids and the skin about it, which share the eye's place in the
// texture and, on a face with heavy brows, most of what is seen of it. The
// ball is told from skin by its paint: the white is grey, the pupil and the
// lashes dark, and skin is neither.
vec3 wbm_eye(vec2 uv, vec3 view, vec3 to_sun, float shadow, vec3 sun, vec3 sky, vec3 painted)
{
    vec2 d = wbm_eye_place(uv);
    float inside = 1.0 - smoothstep(0.45, 0.9, dot(d, d));
    if (inside <= 0.0) return vec3(0.0);
    float most = max(max(painted.r, painted.g), painted.b), least = min(min(painted.r, painted.g), painted.b);
    float grey = 1.0 - smoothstep(0.2, 0.34, (most - least) / max(most, 0.05));
    float dark = 1.0 - smoothstep(0.22, 0.36, most);
    inside *= max(grey, dark);
    vec3 n = normalize(vec3(d * vec2(0.8, 0.45), 1.0)); // the ball bulges out of the flat it is painted on
    vec3 v = normalize(view), h = normalize(v + normalize(to_sun));
    float glint = pow(max(dot(n, h), 0.0), 220.0) * 2.5 * shadow;
    float grazing = min(0.03 + 0.97 * pow(1.0 - max(dot(n, v), 0.0), 5.0), 0.2); // seen edge-on, an eye is behind its lids
    return inside * (glint * sun + grazing * sky * 0.8);
    // (The highlight stays put while the iris moves under it, as it does on a real eye.)
}
)";

bool add_eye_glint(std::string &fs)
{
    // The engine's skin shader only (see add_skin_scattering), which has all that is needed.
    if (fs.find("max ((0.2 * (") == std::string::npos || fs.find("varying vec3 ViewDir;") == std::string::npos ||
        fs.find("varying vec3 SunLightDir;") == std::string::npos || fs.find("varying vec2 Tex0;") == std::string::npos ||
        fs.find("uniform vec4 vSkyLightColor;") == std::string::npos || fs.find("uniform vec4 vAmbientColor;") == std::string::npos)
        return false;
    static const std::regex encode(R"((\w+)\.xyz = clamp \(pow \(\1\.xyz, output_gamma_inv\.xyz\), 0\.0, 1\.0\);)");
    static const std::regex shadow(R"(\b(sun_amount_\d+)\b)");
    if (!std::regex_search(fs, encode)) return false;
    std::smatch m;
    std::string lit = std::regex_search(fs, m, shadow) ? std::string(m[1]) : "1.0";
    // The base colour (both layers: young and aged) is read through the moving eye.
    fs = std::regex_replace(fs, std::regex(R"(texture2D \((diffuse_texture(_2)?), Tex0\))"), "texture2D ($1, wbm_eye_uv(Tex0))");
    fs = std::regex_replace(fs, encode, "$1.xyz += wbm_eye(Tex0, ViewDir, SunLightDir, " + lit + ", vSunColor.xyz, vAmbientColor.xyz + vSkyLightColor.xyz, texture2D(diffuse_texture, wbm_eye_uv(Tex0)).xyz);\n  $&",
                            std::regex_constants::format_first_only);
    return true;
}

// skin: scattering in skin is wanted, and not just the light through leaves that shares the rewrite.
bool add_skin_scattering(std::string &fs, bool skin)
{
    // The back-light needs the view direction and, for thickness, the per-frame shadow map.
    bool can_back = skin && fs.find("varying vec3 ViewDir;") != std::string::npos && fs.find("wbm_dynamic_shadow(") != std::string::npos;
    const std::string back = can_back ? "wbm_skin_back(ViewDir, SunLightDir, wbm_dyn_thickness(ShadowTexCoord.xyz))" : "vec3(0.0)";
    // The sun's diffuse term, in the four shapes the engine's shaders have it:
    // with and without a shadow, as a vec3 times the sun's colour or (the two
    // face shaders) as a scalar times a vec4.
    static const std::regex shapes[4] = {
        std::regex(R"(\(\(vec3\(\s*clamp \(dot \(SunLightDir, ([\w.]+)\), 0\.0, 1\.0\)\s*\) \* (sun_amount_\d+)\) \* vSunColor\.xyz\))"),
        std::regex(R"(\(vec3\(clamp \(\s*dot \(SunLightDir, ([\w.]+)\)\s*, 0\.0, 1\.0\)\) \* vSunColor\.xyz\))"),
        std::regex(R"(\(\(\s*clamp \(dot \(SunLightDir, ([\w.]+)\), 0\.0, 1\.0\)\s*\* (sun_amount_\d+)\) \* (tmpvar_\d+)\))"),
        std::regex(R"(\(clamp \(\s*dot \(SunLightDir, ([\w.]+)\)\s*, 0\.0, 1\.0\) \* (tmpvar_\d+)\))"),
    };
    // The engine's skin shader, which troops' faces are drawn with, has a wrap
    // of its own: clamp(max(0.2 * (d + 0.9), d), 0, 1), the same in every colour.
    static const std::regex own_shadowed(
        R"(\(\(\s*clamp \(max \(\(0\.2 \* \((\w+) \+ 0\.9\)\), \1\), 0\.0, 1\.0\)\s*\* (sun_amount_\d+)\) \* (tmpvar_\d+)\))");
    static const std::regex own_plain(R"(\(clamp \(\s*max \(\(0\.2 \* \((\w+) \+ 0\.9\)\), \1\)\s*, 0\.0, 1\.0\) \* (tmpvar_\d+)\))");
    if (skin && fs.find("varying vec3 SunLightDir;") != std::string::npos) {
        if (std::regex_search(fs, own_shadowed)) {
            fs = std::regex_replace(fs, own_shadowed, "(wbm_skin_lit($1, normalize(SunLightDir).z, $2, " + back + ") * $3)", std::regex_constants::format_first_only);
            return true;
        }
        if (std::regex_search(fs, own_plain)) {
            fs = std::regex_replace(fs, own_plain, "(wbm_skin_lit($1, normalize(SunLightDir).z, 1.0, " + back + ") * $2)", std::regex_constants::format_first_only);
            return true;
        }
    }
    const std::string with[4] = {
        "(wbm_skin_sun(SunLightDir, vec3($1), $2, " + back + ").xyz * vSunColor.xyz)",
        "(wbm_skin_sun(SunLightDir, vec3($1), 1.0, " + back + ").xyz * vSunColor.xyz)",
        "(wbm_skin_sun(SunLightDir, vec3($1), $2, " + back + ") * $3)",
        "(wbm_skin_sun(SunLightDir, vec3($1), 1.0, " + back + ") * $2)",
    };
    if (fs.find("varying vec3 SunLightDir;") == std::string::npos) return false;
    // The leaf shader squares its shadow.
    static const std::regex squared(
        R"(\(\(\s*clamp \(dot \(SunLightDir, ([\w.]+)\), 0\.0, 1\.0\)\s*\*\s*\((sun_amount_\d+) \* \2\)\s*\) \* (tmpvar_\d+)\))");
    if (std::regex_search(fs, squared)) {
        fs = std::regex_replace(fs, squared, "(wbm_skin_sun(SunLightDir, vec3($1), ($2 * $2), " + back + ") * $3)", std::regex_constants::format_first_only);
        return true;
    }
    for (int i = 0; i < 4; i++) {
        if (!std::regex_search(fs, shapes[i])) continue;
        fs = std::regex_replace(fs, shapes[i], with[i], std::regex_constants::format_first_only);
        return true;
    }
    return false;
}

// Finds pow(clamp(dot(normalize(V +/- L), N), 0, 1), fMaterialPower) and
// replaces it with wbm_specular(V, +/-L, N, fMaterialPower).
bool use_pbr_specular(std::string &fs)
{
    static const std::regex lobe(
        R"(pow \(\s*clamp \(\s*dot \(normalize\(\s*\(([\w.]+) ([+-]) ([\w.]+)\)\s*\), ([\w.]+)\)\s*, 0\.0, 1\.0\)\s*, fMaterialPower\))");
    if (!std::regex_search(fs, lobe)) return false;
    fs = std::regex_replace(fs, lobe, "wbm_specular(vec3($1), $2vec3($3), vec3($4), fMaterialPower)");
    return true;
}

// Adds estimated metalness and the environment reflection to a shader whose
// highlight was already rewritten. Three places are touched, and the shader is
// left alone unless all of them are found:
//
//   - the start of main: read the specular mask (a specular map, or the base
//     texture's alpha, whichever the shader uses) and derive metalness from it;
//   - where the base colour is decoded: remember it, and darken it for metals,
//     which in this model show reflection instead of diffuse colour;
//   - just before the colour is encoded for display: add the reflection.
// `sky` asks for the captured sky where the shader allows it; `extra` receives
// declarations the shader then needs in front of it.
bool add_pbr_environment(std::string &fs, bool sky, bool materials, bool debug, std::string &extra)
{
    static const std::regex decode(R"((\w+)\.xyz = pow \((\w+)\.xyz, vec3\(2\.2, 2\.2, 2\.2\)\);)");
    static const std::regex encode(R"((\w+)\.xyz = pow \(\1\.xyz, output_gamma_inv\.xyz\);)");
    static const std::regex specular_map(R"(texture2D \(specular_texture, ([\w.]+)\))");
    static const std::regex alpha_mask(R"(spec_coef\) \* \w+\) \* (\w+)\.w\))");
    auto has = [&](const char *text) { return fs.find(text) != std::string::npos; };

    // The engine's skin shader (recognised by its own wrap lighting, see
    // add_skin_scattering) draws faces and nothing else. A face's specular map is
    // bright where skin is oily, which the reading below would take for metal.
    if (fs.find("max ((0.2 * (") != std::string::npos && fs.find(" + 0.9)), ") != std::string::npos) return false;
    if (std::distance(std::sregex_iterator(fs.begin(), fs.end(), decode), std::sregex_iterator()) != 1) return false;
    if (!std::regex_search(fs, encode) || !has("varying vec3 ViewDir;")) return false;
    // The opening brace of main. The engine's shader files have CRLF line ends,
    // so the brace is looked for separately from the signature.
    size_t body = fs.find("void main ()");
    if (body != std::string::npos) body = fs.find('{', body);
    if (body == std::string::npos) return false;

    std::string mask;
    std::smatch m;
    bool has_map = false;
    if (std::regex_search(fs, m, specular_map)) {
        has_map = true;
        mask = "dot(texture2D(specular_texture, " + std::string(m[1]) + ").xyz, vec3(0.33))";
    } else if (std::regex_search(fs, m, alpha_mask)) {
        std::smatch uv;
        std::regex sample("\\b" + std::string(m[1]) + " = texture2D \\(diffuse_texture, ([\\w.]+)\\);");
        if (!std::regex_search(fs, uv, sample)) return false;
        mask = "texture2D(diffuse_texture, " + std::string(uv[1]) + ").w";
    } else {
        return false;
    }

    // Where the base texture is sampled, for a metalness map laid out like it.
    std::string map_uv = "vec2(0.0)";
    {
        std::smatch d, uv;
        if (std::regex_search(fs, d, decode)) {
            std::regex sample("\\b" + std::string(d[2]) + " = texture2D \\(diffuse_texture, ([\\w.]+)\\);");
            if (std::regex_search(fs, uv, sample)) map_uv = uv[1];
        }
    }
    // Up, in whatever space the shader's normal is in. Shaders that light with a
    // world-space normal can use the world's up axis (Z in this engine); the
    // normal-mapped ones work in tangent space and only have the direction of
    // the sky light there, which leans off vertical but is the best available.
    std::string up = has("uniform vec4 vSkyLightDir;") ? "vec3(0.0, 0.0, 1.0)" : has("varying vec3 SkyLightDir;") ? "SkyLightDir" : "vec3(0.0)";
    // The engine's ambient colours are far too dim to stand in for what a
    // mirror would show: measured in a battle, sky ambient was about 0.1 and
    // ground ambient a saturated green. The fog colour is the colour of the
    // horizon, so the environment is built from that, with the ground a darker
    // blend of it and the ground ambient.
    if (!has("uniform vec4 vFogColor;") || !has("uniform vec4 vSpecularColor;")) return false;
    std::string horizon = "pow(vFogColor.xyz, vec3(2.2))";
    std::string sky_colour = "(" + horizon + " * 1.3)";
    std::string ground = has("uniform vec4 vGroundAmbientColor;") ? "(mix(vGroundAmbientColor.xyz, " + horizon + ", 0.4) * 0.6)"
                                                                  : "(" + horizon + " * 0.35)";
    std::string call = "wbm_environment(ViewDir, " + up + ", " + sky_colour + ", " + ground + ")";
    if (sky) {
        std::string reflection = "reflect(-normalize(ViewDir), wbm_n)";
        if (has("uniform vec4 vSkyLightDir;")) { // world-space lighting
            call = "wbm_environment_sky(ViewDir, " + reflection + ", " + ground + ")";
            extra += k_pbr_sky_functions;
        } else if (has("varying vec3 SunLightDir;") && has("varying vec3 SkyLightDir;")) { // tangent-space lighting
            if (!has("uniform vec4 vSunDir;")) extra += "uniform vec4 vSunDir;\n";
            extra += "uniform vec4 vSkyLightDir;\n";
            call = "wbm_environment_sky(ViewDir, wbm_to_world(" + reflection + ", SunLightDir, SkyLightDir, -vSunDir.xyz, -vSkyLightDir.xyz), " + ground + ")";
            extra += k_pbr_sky_functions;
        }
    }
    fs = std::regex_replace(fs, encode, "$1.xyz += " + call + ";\n  $&" + (debug ? "\n  $1.xyz = vec3(wbm_metal);" : ""),
                            std::regex_constants::format_first_only);
    fs = std::regex_replace(fs, decode,
                            std::string("wbm_albedo = pow ($2.xyz, vec3(2.2, 2.2, 2.2));\n  ") + (materials ? "wbm_classify(" + map_uv + ");\n  " : "") +
                                "$1.xyz = (wbm_albedo * (1.0 - (wbm_dim * wbm_metal)));");
    body = fs.find('{', fs.find("void main ()")) + 1; // again: the replacements above moved it
    // The mask alone does not say how reflective a material is: on many opaque
    // textures it is simply 1. The engine scales it by the material's specular
    // colour, whose brightness runs from under 1 (stone, cloth) through 3-15
    // (leather, wood, mail) to 50 and more (plate), so that is what separates
    // metal from non-metal here.
    std::string start = "\n  float wbm_gloss = dot(vSpecularColor.xyz, vec3(0.3333));"
                        "\n  float wbm_raw_mask = clamp(" + mask + ", 0.0, 1.0);"
                        "\n  wbm_mask = wbm_raw_mask * clamp(wbm_gloss / 40.0, 0.0, 1.0);"
                        // A specular map is grey-scale and works as a mask of materials:
                        // measured on a helmet texture, fur and leather are below 0.08 and
                        // steel from 0.3 up. Without one the mask is the base texture's
                        // alpha, which most armour textures do not have (it reads as 1).
                        + (has_map ? "\n  wbm_metal_map = smoothstep(0.12, 0.3, wbm_raw_mask);"
                                     "\n  wbm_metal = wbm_metal_map * smoothstep(3.0, 10.0, wbm_gloss);"
                                   : "\n  wbm_metal = smoothstep(0.2, 0.6, wbm_raw_mask) * smoothstep(6.0, 30.0, wbm_gloss);") +
                        "\n  wbm_debug_mask = wbm_raw_mask;"
                        "\n  wbm_debug_gloss = clamp(wbm_gloss / 40.0, 0.0, 1.0);";
    if (materials) {
        // What the texture is says more than the two numbers above can.
        // A specular map gives metalness where it is bright (see above), but not
        // every map marks all the metal: a shield's leaves its boss dark. Two
        // more things can tell, map or no map: the material's specular brightness,
        // which the engine sets high for whatever should shine, and colour,
        // since steel is grey where the leather, cloth and painted wood that
        // share its textures are not. The texture's class limits how far that
        // guess is believed (see k_xl_material_uniform); an unknown texture is
        // believed most of the way.
        extra += std::string("uniform vec4 ") + k_xl_material_uniform + ";\n" +
                 "uniform sampler2D " + k_xl_pbr_sampler + ";\nuniform vec4 " + k_xl_pbr_uniform + ";\n"
                 "void wbm_classify(vec2 uv)\n{\n"
                 "    if (" + k_xl_pbr_uniform + ".x > 0.5) {\n" // a map made for this texture: no guessing
                 "        vec4 m = texture2D(" + k_xl_pbr_sampler + ", uv);\n"
                 "        wbm_metal = m.x;\n"
                 "        wbm_a2_fixed = clamp(m.y * m.y * m.y * m.y, 0.0004, 1.0);\n"
                 "        wbm_gain = 1.0;\n"
                 "        wbm_dim = 0.8;\n"
                 "        return;\n"
                 "    }\n"
                 "    float known = " + k_xl_material_uniform + ".w;\n"
                 "    float hi = max(max(wbm_albedo.r, wbm_albedo.g), wbm_albedo.b), lo = min(min(wbm_albedo.r, wbm_albedo.g), wbm_albedo.b);\n"
                 "    float grey = 1.0 - smoothstep(0.1, 0.28, (hi - lo) / max(hi, 0.08));\n"
                 "    float shiny = smoothstep(0.12, 0.3, wbm_debug_gloss);\n"
                 "    float most = known > 0.0 ? known - 1.0 : 0.8;\n"
                 // A texture known to show skin, cloth, hair or the like has no metal on
                 // it whatever its specular map says; leather and wood may have some.
                 "    wbm_metal = clamp(max(wbm_metal, most * grey * shiny), 0.0, known > 0.0 ? most : 1.0);\n"
                 "    wbm_dim = 0.8;\n"
                 "}\n";
        start += std::string("\n  if (") + k_xl_material_uniform + ".w > 0.0) {"
                 "\n    wbm_rough = " + k_xl_material_uniform + ".y;"
                 "\n    wbm_gain = " + k_xl_material_uniform + ".z;"
                 "\n    wbm_mask *= " + k_xl_material_uniform + ".z;"
                 "\n  }";
    }
    fs.insert(body, start);
    return true;
}

// The engine's last post-processing shader adds bloom to the scene, applies
// the exposure it measured, runs one of four tone-mapping operators chosen by
// the module (none, exponential, Reinhard, extended Reinhard), and encodes the
// result for display. Two optional changes are made just before the encode:
//
//   - filmic: ignore the engine's operator and keep the exposed colour as it is
//     up to 0.6, then roll the rest off smoothly towards 1. The Native module
//     uses no operator at all (measured: postfx_editor_vector[0].x = 0.1), so
//     its highlights clip hard at white; this only changes that. An ACES curve
//     was tried first and made the whole image darker, because it also
//     compresses shadows and mid-tones that the module leaves linear.
//   - HDR: the operators all end at 1.0, display white. For an extended-range
//     drawable the top of that range is stretched towards the display's
//     headroom: colour * (1 + (headroom - 1) * peak^6). Mid-tones barely move
//     (peak 0.5 gains 5% at headroom 4) while values near white reach the full
//     headroom, so the sun, fire and bright sky go brighter than the UI's white.
const char k_tonemap_functions[] = R"(
vec3 wbm_filmic(vec3 x)
{
    const float knee = 0.6;
    x = max(x, vec3(0.0)); // below black there is nothing: left negative, it comes out of the encoding that follows as white
    vec3 over = max(x - knee, vec3(0.0));
    return min(x, vec3(knee)) + (1.0 - knee) * (1.0 - exp(-over / (1.0 - knee)));
}
// Grading, on the picture as it will be shown (0 to 1): contrast is an
// S-curve about middle grey, which leaves black and white where they are;
// saturation moves each colour away from its own grey; warmth trades blue for
// red. Each is 0 for no change, and about -1 to 1.
uniform vec4 wbm_grade;
vec3 wbm_graded(vec3 c)
{
    c = clamp(c, 0.0, 1.0);
    vec3 curved = c * c * (3.0 - 2.0 * c);
    vec3 flat_ = mix(c, vec3(0.5), 0.5);
    c = wbm_grade.x >= 0.0 ? mix(c, curved, wbm_grade.x) : mix(c, flat_, -wbm_grade.x);
    float grey = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c = max(mix(vec3(grey), c, 1.0 + wbm_grade.y), vec3(0.0));
    c *= vec3(1.0 + 0.12 * wbm_grade.z, 1.0, 1.0 - 0.12 * wbm_grade.z);
    return c * (grey / max(dot(c, vec3(0.2126, 0.7152, 0.0722)), 1e-4)); // warmth changes the colour, not the brightness
}
vec3 wbm_expand(vec3 c, float headroom)
{
    vec3 g = clamp(c, 0.0, 1.0);
    float peak = max(g.r, max(g.g, g.b));
    return g * (1.0 + (max(headroom, 1.0) - 1.0) * pow(peak, 6.0));
}
)";

// Returns true if this is the final shader and it was changed.
bool rewrite_final_pass(std::string &fs, const XlOptions &options)
{
    static const std::regex first_branch(R"(if \(\(tonemapOp_\d+ < 1\.0\)\) \{\s*(final_color_\d+) = (tmpvar_\d+);)");
    static const std::regex encode(R"(color_\d+\.xyz = pow \((final_color_\d+), output_gamma_inv\.xyz\);)");
    std::smatch branch, enc;
    if (!std::regex_search(fs, branch, first_branch) || !std::regex_search(fs, enc, encode)) return false;
    std::string result = enc[1], exposed = branch[2];
    if (result != std::string(branch[1])) return false;
    std::string extra;
    if (options.filmic_tonemap) extra += result + " = wbm_filmic(" + exposed + ");\n  ";
    if (options.grade) extra += result + " = wbm_graded(" + result + ");\n  ";
    if (options.hdr_output) extra += result + " = wbm_expand(" + result + ", " + k_xl_headroom_uniform + ");\n  ";
    fs.insert((size_t)enc.position(), extra);
    return true;
}

// The engine decodes texture colour itself, pow(sample.rgb, 2.2), after the
// sample has been filtered. An sRGB texture format does it per texel before
// filtering and with the exact curve. This removes the pow for every sampler
// whose samples are all decoded that way and used in no other way, and
// reports those samplers, which must then be given sRGB views.
void decode_srgb_in_hardware(std::string &fs, std::vector<std::string> &samplers)
{
    static const std::regex decode(R"(pow \((\w+)\.xyz, vec3\(2\.2, 2\.2, 2\.2\)\))");
    auto count = [&](const std::string &pattern) {
        std::regex re(pattern);
        return (int)std::distance(std::sregex_iterator(fs.begin(), fs.end(), re), std::sregex_iterator());
    };
    std::map<std::string, std::vector<std::string>> decoded; // sampler -> variables holding its decoded samples
    std::set<std::string> rejected;
    for (auto it = std::sregex_iterator(fs.begin(), fs.end(), decode); it != std::sregex_iterator(); ++it) {
        std::string var = (*it)[1];
        std::smatch m;
        std::regex assign("\\b" + var + " = texture2D \\((\\w+), ");
        if (!std::regex_search(fs, m, assign)) continue;
        std::string sampler = m[1];
        // The variable must be declared and assigned once, and read only as .w or inside the pow.
        int uses = count("\\b" + var + "\\b");
        int expected = 1 + count("\\bvec4 " + var + ";") + count("pow \\(" + var + "\\.xyz, vec3\\(2\\.2, 2\\.2, 2\\.2\\)\\)") +
                       count("\\b" + var + "\\.w\\b");
        if (uses != expected) rejected.insert(sampler);
        auto &vars = decoded[sampler];
        if (std::find(vars.begin(), vars.end(), var) == vars.end()) vars.push_back(var);
    }
    for (auto &entry : decoded) {
        const std::string &sampler = entry.first;
        if (rejected.count(sampler)) continue;
        if (count("texture2D\\w* \\(" + sampler + ",") != (int)entry.second.size()) continue; // sampled some other way too
        for (const std::string &var : entry.second)
            fs = std::regex_replace(fs, std::regex("pow \\(" + var + "\\.xyz, vec3\\(2\\.2, 2\\.2, 2\\.2\\)\\)"), var + ".xyz");
        samplers.push_back(sampler);
    }
}

// Shadows from the engine's point lights (torches, candles, fires), which in the game cast none. The backend
// draws the scene from each light into a cube of depths (see render_torch_shadows); here the shaders learn to
// look their own surface up in them.
//
// The engine weighs most of these lights at the vertices and hands the fragment shader their sum, so a light
// cannot simply be switched off for a pixel. The vertex shader therefore also passes on how much each light
// contributed (wbm_torch_w) and where the surface is (wbm_torch_at); the fragment shader works out what share
// of the sum the lights it can see account for, and scales the sum by that. Lights the fragment shader weighs
// itself are scaled one by one.
const char *const k_torch_function = R"(
uniform samplerCubeArrayShadow wbm_torch_map;
uniform vec4 wbm_torch_light[4]; // xyz: where the engine's light of that number is; w: 1 + the number of its cube, 0 for none
uniform vec4 wbm_torch;          // x: strength, 0 for no maps; y, z: a cube's depth is y + z / distance along the face's axis; w: a texel's width at distance 1
vec4 wbm_torch_s = vec4(1.0);    // how much of each light reaches this pixel
vec3 wbm_torch_ratio = vec3(1.0);
float wbm_torch_gloss = 1.0;
float wbm_torch_seen(int i, vec3 n, float turn)
{
    vec4 light = wbm_torch_light[i];
    if (light.w <= 0.0) return 1.0;
    vec3 v = wbm_torch_at - light.xyz;
    if (dot(n, v) > 0.0) n = -n; // towards the light
    vec3 a = abs(v);
    float texel = max(a.x, max(a.y, a.z)) * wbm_torch.w;
    float len = max(length(v), 1.0e-4);
    // The surface is moved off itself before the comparison, farther the more it is seen edge-on from the light.
    float slant = sqrt(clamp(1.0 - pow(dot(n, v) / len, 2.0), 0.0, 1.0));
    v += n * (texel * (1.0 + 2.5 * slant) + 0.012);
    a = abs(v);
    float depth = wbm_torch.y + wbm_torch.z / max(max(a.x, max(a.y, a.z)), 1.0e-3);
    vec3 d = v / len;
    vec3 side = normalize(cross(d, abs(d.z) < 0.9 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0)));
    vec3 up = cross(d, side);
    vec2 k = vec2(cos(turn), sin(turn)) * texel * 1.5;
    float lit = texture(wbm_torch_map, vec4(v + side * k.x + up * k.y, light.w - 1.0), depth);
    lit += texture(wbm_torch_map, vec4(v - side * k.y + up * k.x, light.w - 1.0), depth);
    lit += texture(wbm_torch_map, vec4(v - side * k.x - up * k.y, light.w - 1.0), depth);
    lit += texture(wbm_torch_map, vec4(v + side * k.y - up * k.x, light.w - 1.0), depth);
    return lit * 0.25;
}
void wbm_torch_setup()
{
    if (wbm_torch.x <= 0.0) return;
    vec3 n = cross(dFdx(wbm_torch_at), dFdy(wbm_torch_at));
    n = dot(n, n) > 0.0 ? normalize(n) : vec3(0.0, 0.0, 1.0);
    float turn = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715)))) * 6.2831853;
    vec3 all = vec3(0.0), kept = vec3(0.0);
    float gloss = 0.0, gloss_kept = 0.0;
    for (int i = 0; i < 4; i++) {
        if (wbm_torch_light[i].w <= 0.0) continue;
        float s = mix(1.0, wbm_torch_seen(i, n, turn), wbm_torch.x);
        wbm_torch_s[i] = s;
        vec3 c = vLightDiffuse[i].xyz * wbm_torch_w[i];
        all += c;
        kept += c * s;
        vec3 to = wbm_torch_light[i].xyz - wbm_torch_at;
        float g = dot(vLightDiffuse[i].xyz, vec3(0.3333)) / max(dot(to, to), 1.0e-4); // highlights go by nearness alone
        gloss += g;
        gloss_kept += g * s;
    }
    wbm_torch_ratio = (kept + 1.0e-6) / (all + 1.0e-6);
    wbm_torch_gloss = (gloss_kept + 1.0e-6) / (gloss + 1.0e-6);
}
)";

// The vertex shader's part. Where the engine sums its point lights:
//   total = (total + ((facing * vLightDiffuse[i]) * (1.0/(dot (to_light, to_light)))));
// each light's weight is kept as well. What becomes of the sum is noted for the fragment shader's part: it is
// either passed on by itself, or added to the rest of the light at once, and then its share of the whole goes along.
bool add_torch_vertex(std::string &vs, Varyings &varyings, bool lighting)
{
    static const std::regex to_light(R"(vLightPosDir\[[^;]*?\]\.xyz - (\w+)(\.xyz)?\))");
    static const std::regex term(R"((total_\d+) = \(\1 \+ \(\(\s*([^;]*?)\s*\* vLightDiffuse\[(\w+)\]\) \* \(1\.0/\(\s*dot \((\w+), \4\)\s*\)\)\)\);)");
    static const std::regex gloss_term(R"((total_\d+) = \(\1 \+ \(\(\s*\(1\.0/\(dot \((\w+), \2\)\)\)\s*\* vLightDiffuse\[\w+\]\) \* pow \()");
    std::smatch m;
    size_t body = vs.find("void main ()");
    if (body != std::string::npos) body = vs.find('{', body);
    if (body == std::string::npos || !std::regex_search(vs, m, to_light)) return false;
    const std::string at = m[1];
    auto varying = [&](const std::string &name) { return std::regex_search(vs, std::regex("varying \\w+ " + name + ";")); };
    // Where a sum ends up: in a varying, at once or by way of one temporary.
    auto destination = [&](const std::string &total) -> std::string {
        std::smatch d;
        if (!std::regex_search(vs, d, std::regex("\\n\\s*(\\w+) = " + total + "(\\.xyz)?;"))) return "";
        std::string to = d[1];
        if (!varying(to) && std::regex_search(vs, d, std::regex("\\n\\s*(\\w+) = " + to + ";"))) to = d[1];
        return varying(to) ? to : "";
    };
    if (std::regex_search(vs, m, gloss_term)) {
        std::string to = destination(m[1]);
        if (!to.empty()) varyings.torch_gloss.push_back(to);
    }
    if (std::regex_search(vs, m, term)) {
        const std::string total = m[1];
        vs = std::regex_replace(vs, term, "$&\n      wbm_torch_w[$3] += ($2) / dot($4, $4);");
        varyings.torch_total = destination(total);
        if (varyings.torch_total.empty() && std::regex_search(vs, m, std::regex("\\n\\s*(\\w+) = \\(\\1 \\+ " + total + "\\);"))) {
            const std::string sum = m[1];
            size_t after = (size_t)m.position(0) + (size_t)m.length(0);
            vs.insert(after, "\n  wbm_torch_share = " + total + ".xyz / max(" + sum + ".xyz, vec3(1.0e-5));");
            // The whole may pass through other variables on its way to a varying.
            static const std::regex assignment(R"(\n\s*(\w+)(\.\w+)? = ([^;]*);)");
            std::vector<std::string> carriers = {sum};
            for (std::sregex_iterator it(vs.begin() + (std::ptrdiff_t)after, vs.end(), assignment), none; it != none; ++it) {
                const std::string to = (*it)[1], from = (*it)[3];
                bool carried = false;
                for (const std::string &name : carriers) carried = carried || std::regex_search(from, std::regex("\\b" + name + "\\b"));
                if (!carried) continue;
                if (!varying(to)) {
                    carriers.push_back(to);
                    continue;
                }
                varyings.torch_total = to;
                varyings.torch_shared = true;
                break;
            }
        }
    }
    body = vs.find('{', vs.find("void main ()"));
    vs.insert(body + 1, "\n  wbm_torch_w = vec4(0.0);\n  wbm_torch_share = vec3(1.0);");
    vs.insert(vs.rfind('}'), "  wbm_torch_at = " + at + ".xyz;\n");
    vs.insert(0, "varying vec3 wbm_torch_at;\nvarying vec4 wbm_torch_w;\nvarying vec3 wbm_torch_share;\n");
    // For lights weighed again at each pixel (see k_torch_lighting_function): the engine's standard shaders
    // either light in the surface's own space, and then have the matrix that leads there from the world, or
    // light in the world's.
    if (lighting && varyings.torch_total == "VertexLighting" && !varyings.torch_shared && vs.find("varying vec3 ViewDir;") != std::string::npos) {
        static const std::regex basis(R"(\((tmpvar_\d+) \* -\(vSunDir\.xyz\)\))");
        bool own = std::regex_search(vs, m, basis);
        vs.insert(vs.rfind('}'), "  wbm_torch_tbn = " + (own ? std::string(m[1]) : std::string("mat3(1.0)")) + ";\n");
        vs.insert(0, "varying mat3 wbm_torch_tbn;\n");
        varyings.torch_space = own ? 1 : 2;
    }
    return true;
}

// With torch_lighting, the lights the engine weighs at the vertices are weighed at each pixel instead, in its
// standard shaders: with the normal the shader has there (from its normal map, when it has one), and with a
// highlight per light of the kind the sun's has. LOBE is that highlight; `first` is the light a shader with
// a normal map already weighs per pixel by itself, -1 for none.
const char *const k_torch_lighting_function = R"(
vec3 wbm_torch_diffuse(vec3 n, int first)
{
    vec3 sum = vec3(0.0);
    n = normalize(n);
    for (int i = 0; i < 4 && i < iLightPointCount; i++) {
        if (i == first) continue;
        vec3 to = vLightPosDir[i].xyz - wbm_torch_at;
        sum += vLightDiffuse[i].xyz * (clamp(dot(n, normalize(wbm_torch_tbn * to)), 0.0, 1.0) / max(dot(to, to), 1.0e-4) * wbm_torch_s[i]);
    }
    return sum;
}
vec3 wbm_torch_shine(vec3 n, vec3 view, int first)
{
    vec3 sum = vec3(0.0);
    for (int i = 0; i < 4 && i < iLightPointCount; i++) {
        if (i == first) continue;
        vec3 to = vLightPosDir[i].xyz - wbm_torch_at;
        vec3 l = normalize(wbm_torch_tbn * to);
        sum += vLightDiffuse[i].xyz * (LOBE / max(dot(to, to), 1.0e-4) * wbm_torch_s[i]);
    }
    return sum;
}
)";

// The fragment shader's part, where the program's vertex shader has done its own.
bool add_torch_fragment(std::string &fs, const Varyings &varyings, bool room_fill)
{
    static const std::regex colour(R"(vLightDiffuse\[((?:[^\[\]]|\[[^\[\]]*\])*)\])");
    size_t main_at = fs.find("void main ()");
    size_t body = main_at != std::string::npos ? fs.find('{', main_at) : std::string::npos;
    if (body == std::string::npos || !varyings.location.count("wbm_torch_at")) return false;
    std::string head = fs.substr(0, main_at), open = fs.substr(main_at, body + 1 - main_at), rest = fs.substr(body + 1);
    std::string start = "\n  wbm_torch_setup();";
    bool any = false;
    if (std::regex_search(rest, colour)) { // lights weighed here, each by itself
        rest = std::regex_replace(rest, colour, "(vLightDiffuse[$1] * wbm_torch_s[$1])");
        any = true;
    }
    auto scale = [&](const std::string &name, const std::string &by) {
        std::smatch m;
        if (!std::regex_search(head, m, std::regex("varying (vec[34]) " + name + ";"))) return;
        rest = std::regex_replace(rest, std::regex("\\b" + name + "\\b"), "wbm_torch_" + name);
        start += "\n  " + std::string(m[1]) + " wbm_torch_" + name + " = " + name + ";\n  wbm_torch_" + name + ".xyz *= " + by + ";";
        any = true;
    };
    // Lights weighed per pixel, where this shader and the program's vertex shader work in the same space.
    std::string normal, first = "-1", lighting;
    std::smatch m;
    static const std::regex mapped(R"(\n\s*(normal_\d+)(\.\w+)? = )");
    bool has_map = head.find("uniform sampler2D normal_texture;") != std::string::npos;
    if (varyings.torch_space == 1 && has_map && head.find("varying vec4 PointLightDir;") != std::string::npos && std::regex_search(rest, m, mapped))
        normal = m[1], first = "iLightIndices[0]";
    if (varyings.torch_space == 2 && !has_map && head.find("varying vec3 SunLightDir;") != std::string::npos) normal = "SunLightDir";
    if (!normal.empty() && head.find("varying vec3 ViewDir;") != std::string::npos) {
        // The normal has to be there by the time the light is asked for.
        size_t ready = normal == "SunLightDir" ? 0 : rest.find(';', rest.find(normal + ".z = ") != std::string::npos ? rest.find(normal + ".z = ") : (size_t)m.position(0) + 1);
        auto replace = [&](const std::string &name, const std::string &with) {
            std::smatch use;
            if (!std::regex_search(head, std::regex("varying vec[34] " + name + ";")) || !std::regex_search(rest, use, std::regex("\\b" + name + "\\b(\\.xyz)?"))) return false;
            if (ready == std::string::npos || (size_t)use.position(0) < ready) return false;
            rest = std::regex_replace(rest, std::regex("\\b" + name + "\\b(\\.xyz)?"), with);
            return true;
        };
        bool lit = replace(varyings.torch_total, "wbm_torch_diffuse(" + normal + ", " + first + ")");
        bool shone = false;
        for (const std::string &name : varyings.torch_gloss) shone = replace(name, "wbm_torch_shine(" + normal + ", ViewDir, " + first + ")") || shone;
        if (lit || shone) {
            lighting = std::regex_replace(k_torch_lighting_function, std::regex("LOBE"),
                                          fs.find("wbm_specular(") != std::string::npos ? "wbm_specular(view, l, n, fMaterialPower)"
                                                                                       : "pow(clamp(dot(normalize(normalize(view) + l), normalize(n)), 0.0, 1.0), fMaterialPower)");
            any = true;
        }
    }
    // A room's walls and floors add their ambient light as one flat colour: where no torch reaches, in a
    // torch's shadow for one, a wall is its picture and nothing more, whatever its normal map says. But a
    // room's ambient light is its torches' light come back off the other walls, and more of it comes from
    // the side they are on. So it is weighed by how far the surface, by its map, is turned towards the
    // lights: half as much turned away, half as much again turned towards them, the same on the whole.
    static const std::regex room_normal(R"(dot \((\w+), normalize\(vec_to_light_0\)\))");
    static const std::regex room_ambient(R"(= \(vAmbientColor \+ )");
    std::string fill;
    if (room_fill && head.find("varying vec3 vec_to_light_2;") != std::string::npos && head.find("uniform int iLightIndices[4];") != std::string::npos &&
        std::regex_search(rest, m, room_normal) && std::regex_search(rest, room_ambient)) {
        rest = std::regex_replace(rest, room_ambient, "= ((vAmbientColor * wbm_torch_fill(" + std::string(m[1]) + ")) + ", std::regex_constants::format_first_only);
        fill = R"(
float wbm_torch_fill(vec3 n)
{
    vec3 to[3] = vec3[3](vec_to_light_0, vec_to_light_1, vec_to_light_2);
    float all = 0.0, towards = 0.0;
    for (int k = 0; k < 3; k++) {
        float w = dot(vLightDiffuse[iLightIndices[k]].xyz, vec3(0.3333)) / max(dot(to[k], to[k]), 1.0e-4);
        all += w;
        towards += w * (0.5 + 0.5 * dot(normalize(n), normalize(to[k])));
    }
    return all > 1.0e-6 ? 0.5 + towards / all : 1.0;
}
)";
        any = true;
    }
    if (!varyings.torch_total.empty()) scale(varyings.torch_total, varyings.torch_shared ? "1.0 - wbm_torch_share * (1.0 - wbm_torch_ratio)" : "wbm_torch_ratio");
    for (const std::string &name : varyings.torch_gloss) scale(name, "wbm_torch_gloss");
    if (!any) return false;
    std::string declared = "varying vec3 wbm_torch_at;\nvarying vec4 wbm_torch_w;\nvarying vec3 wbm_torch_share;\n";
    if (head.find("uniform vec4 vLightDiffuse[4];") == std::string::npos) declared += "uniform vec4 vLightDiffuse[4];\n";
    if (!lighting.empty()) {
        declared += "varying mat3 wbm_torch_tbn;\n";
        for (const char *needed : {"uniform vec4 vLightPosDir[4];", "uniform int iLightPointCount;", "uniform int iLightIndices[4];", "uniform float fMaterialPower;"})
            if (head.find(needed) == std::string::npos) declared += std::string(needed) + "\n";
    }
    fs = declared + head + k_torch_function + lighting + fill + open + start + rest;
    return true;
}

// A pack's map of a surface (k_xl_pbr_sampler) in the shaders that have no highlight of their own to steer
// with it, which is most of what buildings are drawn with:
//   - its alpha is how much of the light from all round reaches the spot (ambient occlusion): the ambient
//     light is scaled by it where the shader adds that itself, and the whole by half as much where the
//     vertex shader has already summed the light;
//   - its roughness and metalness give the sun a highlight where the shader lights by a normal map and has
//     none: next to nothing on plaster, a sheen on planed wood or wet stone.
// `declared`: the sampler and its uniform are in the shader already.
const char *const k_surface_function = R"(
float wbm_ao_at(vec2 uv)
{
    return wbm_pbr.x > 0.0 ? texture(wbm_pbr_map, uv).a : 1.0;
}
vec3 wbm_map_shine(vec3 n, vec3 view, vec3 light, vec3 colour, vec2 uv)
{
    if (wbm_pbr.x <= 0.0) return vec3(0.0);
    vec4 m = texture(wbm_pbr_map, uv);
    vec3 v = normalize(view), l = normalize(light);
    n = normalize(n);
    vec3 h = normalize(v + l);
    float nl = clamp(dot(n, l), 0.0, 1.0), nh = clamp(dot(n, h), 0.0, 1.0), lh = clamp(dot(l, h), 0.0, 1.0);
    float a = max(m.g * m.g, 0.01), a2 = a * a;
    float d = nh * nh * (a2 - 1.0) + 1.0;
    float seen = 0.25 / (max(lh * lh, 0.1) * (m.g + 0.5));
    vec3 f0 = mix(vec3(0.04), vec3(0.7), m.r);
    vec3 fresnel = f0 + (1.0 - f0) * pow(1.0 - lh, 5.0);
    return colour * fresnel * (a2 / (d * d) * seen * nl * m.a);
}
)";

bool add_surface_maps(std::string &fs, const Varyings &varyings, bool declared)
{
    size_t main_at = fs.find("void main ()");
    size_t body = main_at != std::string::npos ? fs.find('{', main_at) : std::string::npos;
    if (body == std::string::npos || fs.find("uniform sampler2D diffuse_texture;") == std::string::npos || fs.find("varying vec2 Tex0;") == std::string::npos) return false;
    std::string head = fs.substr(0, main_at), open = fs.substr(main_at, body + 1 - main_at), rest = fs.substr(body + 1);
    const std::string uv = rest.find("vec2 wbm_uv") != std::string::npos ? "wbm_uv" : "Tex0";
    bool any = false;
    static const std::regex ambient(R"(\b(vAmbientColor|vGroundAmbientColor)\b)");
    if (std::regex_search(rest, ambient)) {
        rest = std::regex_replace(rest, ambient, "($1 * wbm_ao_at(" + uv + "))");
        any = true;
    } else if (head.find("varying vec4 Color;") != std::string::npos && head.find("varying vec4 SunLight;") != std::string::npos) {
        rest.insert(rest.rfind('}'), "  gl_FragColor.xyz *= mix(1.0, wbm_ao_at(" + uv + "), 0.5);\n");
        any = true;
    }
    static const std::regex lit(R"(dot \(SunLightDir, (normal_\d+)\))");
    static const std::regex encode(R"(\n(\s*)(\w+)\.xyz = pow \(\2\.xyz, output_gamma_inv\.xyz\);)");
    static const std::regex shadow(R"(\b(sun_amount_\d+)\b)");
    std::smatch m, e, sh;
    if (rest.find("wbm_specular(") == std::string::npos && rest.find("fSpecular") == std::string::npos && head.find("uniform vec4 vSunColor;") != std::string::npos &&
        varyings.location.count("ViewDir") && std::regex_search(rest, m, lit) && std::regex_search(rest, e, encode)) {
        size_t at = (size_t)e.position(0);
        bool shaded = std::regex_search(rest, sh, shadow) && (size_t)sh.position(0) < at;
        rest.insert(at, "\n  " + std::string(e[2]) + ".xyz += wbm_map_shine(" + std::string(m[1]) + ", ViewDir, SunLightDir, vSunColor.xyz" + (shaded ? " * " + std::string(sh[1]) : std::string()) + ", " + uv + ");");
        if (head.find("varying vec3 ViewDir;") == std::string::npos) head.insert(0, "varying vec3 ViewDir;\n");
        any = true;
    }
    if (!any) return false;
    if (!declared) head.insert(0, std::string("uniform sampler2D ") + k_xl_pbr_sampler + ";\nuniform vec4 " + k_xl_pbr_uniform + ";\n");
    fs = head + k_surface_function + open + rest;
    return true;
}

// Rewrites the declarations and built-ins that changed between GLSL 1.20 and
// 4.50. Attributes and varyings get explicit locations so that the two stages
// and the vertex descriptor agree without relying on declaration order.
std::string modernise(const std::string &source, bool vertex, const std::map<std::string, int> &attribs, Varyings &varyings,
                      const XlOptions &options, std::vector<std::string> &srgb_samplers, std::string &log)
{
    std::string glsl = source;
    if (vertex && options.torch_shadows) add_torch_vertex(glsl, varyings, options.torch_lighting); // first: it goes by the shapes the engine's own code has
    int shadow_filter = options.pcss_shadows ? 2 : options.soft_shadows ? 1 : 0;
    bool shadows = !vertex && (shadow_filter || options.dynamic_shadows) && rewrite_shadows(glsl, shadow_filter, options.dynamic_shadows);
    bool soft_shadows = shadows && shadow_filter != 0;
    // The vertex shader has the sun's matrix; the fragment shader needs it too now.
    bool needs_sun_matrix = soft_shadows && options.pcss_shadows && glsl.find("uniform mat4 matSunViewProj;") == std::string::npos;
    bool smooth_alpha = !vertex && (options.smooth_alpha_test || options.alpha_hash) && smooth_alpha_tests(glsl, options.alpha_hash);
    bool pbr_specular = !vertex && (options.pbr_specular || options.pbr_environment || options.pbr_sky) && use_pbr_specular(glsl);
    std::string pbr_extra;
    bool pbr_environment = pbr_specular && (options.pbr_environment || options.pbr_sky) && add_pbr_environment(glsl, options.pbr_sky, options.materials, options.debug_materials, pbr_extra);
    if (!vertex && options.srgb_textures) decode_srgb_in_hardware(glsl, srgb_samplers); // after the above: it removes a pow they look for
    // After the shadow rewrite, which looks for the sun term's shadow variable by its original shape.
    bool skin = (options.skin_scattering || options.foliage_backlight) && (vertex ? add_skin_scattering_vertex(glsl) : add_skin_scattering(glsl, options.skin_scattering));
    bool eyes = !vertex && options.eyes && add_eye_glint(glsl);
    bool fur = !vertex && options.fur && add_fur(glsl);
    bool leaf = vertex && options.foliage_backlight && add_leaf_glow(glsl); // before the wind wraps the position
    bool wind = vertex && options.wind && add_wind(glsl, options.motion_vectors);
    bool shell_push = vertex && options.fur && add_shell_push(glsl);
    bool hair_strands = options.hair_strands && add_hair_strands(glsl, vertex);
    bool water = !vertex && options.water && add_water(glsl, options.pbr_sky);
    bool sky_ambient = options.sky_ambient && add_sky_ambient(glsl, vertex);
    // The bounce off the ground only where the ground about is in the sun. The function that asks is defined after
    // the shader's own text, where the map and the sun's matrix are declared.
    bool ground_sun = sky_ambient && !vertex && glsl.find("uniform sampler2D shadowmap_texture;") != std::string::npos &&
                      glsl.find("varying vec4 ShadowTexCoord;") != std::string::npos && glsl.find("uniform float fShadowMapSize;") != std::string::npos;
    if (ground_sun && glsl.find("uniform mat4 matSunViewProj;") == std::string::npos) needs_sun_matrix = true;
    // The fragment's half only where the vertex shader of this program has supplied the eye's direction.
    bool parallax = options.parallax && (vertex ? add_parallax_vertex(glsl) : varyings.location.count("wbm_tview") && add_parallax(glsl)); // last: it takes over the texture coordinate wherever the others left it
    if (!vertex && options.torch_shadows) add_torch_fragment(glsl, varyings, options.torch_lighting); // after the rest, which look for the names it replaces
    if (!vertex && options.materials) add_surface_maps(glsl, varyings, pbr_extra.find(k_xl_pbr_sampler) != std::string::npos); // after the parallax: it reads where that looks
    bool final_pass = !vertex && (options.filmic_tonemap || options.hdr_output || options.grade) && rewrite_final_pass(glsl, options);
    // The movement of what its vertex shader sways (see k_motion_vertex_uniforms), written first thing: a pixel dropped later takes it along.
    bool motion_out = false;
    if (!vertex && options.motion_vectors && varyings.location.count("wbm_mnow")) {
        size_t body = glsl.find("void main ()");
        if (body != std::string::npos) body = glsl.find('{', body);
        if (body != std::string::npos) {
            glsl.insert(body + 1, "\n  wbm_Motion = vec4(wbm_mnow.xy / wbm_mnow.w - wbm_mbefore.xy / max(wbm_mbefore.w, 1.0e-4), wbm_mnow.w, 1.0);");
            glsl.insert(0, "varying vec4 wbm_mnow;\nvarying vec4 wbm_mbefore;\n");
            motion_out = true;
        }
    }
    // Put before main, which has to stay the last function (the lines appended to it are found by its closing
    // brace), and after the shader's own declarations, which the lookup needs.
    if (sky_ambient) {
        size_t body = glsl.find("void main ()");
        const std::string ask = ground_sun ? "float wbm_ground_sun() { return wbm_ground_lit(shadowmap_texture, ShadowTexCoord.xyz, fShadowMapSize, matSunViewProj); }\n"
                                           : "float wbm_ground_sun() { return 1.0; }\n";
        glsl.insert(body == std::string::npos ? 0 : body, ask);
    }

    static const std::regex decl(R"(^\s*(attribute|varying)\s+(\w+)\s+(\w+)\s*(\[\s*\d+\s*\])?\s*;\s*$)");
    std::set<int> used_attribs;
    for (auto &a : attribs) used_attribs.insert(a.second);

    std::ostringstream out;
    out << "#version 450\n";
    if (vertex) out << "uniform float " << k_xl_yflip_uniform << ";\nuniform vec2 " << k_xl_jitter_uniform << ";\n";
    else out << "layout(location = 0) out vec4 wbm_FragColor;\n";
    if (motion_out) out << "layout(location = 1) out vec4 wbm_Motion;\n";
    if (needs_sun_matrix) out << "uniform mat4 matSunViewProj;\n";
    if (soft_shadows) out << (options.pcss_shadows ? k_pcss_shadow_function : k_soft_shadow_function);
    if (shadows && options.dynamic_shadows)
        out << std::regex_replace(k_dynamic_shadow_function, std::regex("DYN_SIZE"), std::to_string((int)k_xl_dyn_size) + ".0");
    if (pbr_specular) out << k_pbr_globals << k_pbr_specular_function;
    if (pbr_environment) out << k_pbr_environment_function;
    if (!pbr_extra.empty()) out << std::regex_replace(pbr_extra, std::regex("FACE_SIZE"), std::to_string((int)k_xl_sky_face_size) + ".0");
    if (eyes) out << k_eye_function;
    if (!vertex && (fur || hair_strands)) out << k_strand_function;
    if (fur) out << k_fur_function;
    if (parallax && !vertex) out << "uniform vec4 wbm_parallax;\n";
    if (water && options.pbr_sky) out << std::regex_replace(k_water_sky_function, std::regex("FACE_SIZE"), std::to_string((int)k_xl_sky_face_size) + ".0");
    if (hair_strands && !vertex) out << k_people_hair_function;
    if (hair_strands) out << "uniform vec4 wbm_shell;\n" << (vertex ? "vec4 wbm_colour;\nvec2 wbm_uv;\nuniform vec4 wbm_drift;\n" : "");
    if (leaf) out << k_leaf_function;
    if (wind) out << k_wind_function;
    if (wind && options.motion_vectors) out << k_motion_vertex_uniforms;
    if (shell_push) out << "uniform vec4 wbm_shell;\n";
    if (shell_push && glsl.find("wbm_colour = ") != std::string::npos) out << "vec4 wbm_colour;\nuniform vec4 wbm_drift;\n";
    if (skin && vertex) out << k_skin_vertex_function;
    if (skin && !vertex && glsl.find("wbm_dynamic_shadow(") != std::string::npos && glsl.find("ShadowTexCoord") != std::string::npos) {
        size_t body = glsl.find("void main ()");
        if (body != std::string::npos) body = glsl.find('{', body);
        if (body != std::string::npos) glsl.insert(body + 1, "\n  wbm_skin_gap = wbm_dyn_thickness(ShadowTexCoord.xyz);");
    }
    if (skin && !vertex) { // with the materials debug view, skin that scatters is painted red
        std::string f = k_skin_function;
        if (options.debug_materials) f = std::regex_replace(f, std::regex("return lit \\* pow\\(vec3\\(shadow\\), vec3\\(0\\.7, 1\\.0, 1\\.25\\)\\);"), "return vec3(4.0, 0.0, 0.0);");
        out << f;
    }
    if (sky_ambient) out << k_ambient_functions;
    if (ground_sun) out << k_ground_function;
    if (final_pass) out << k_tonemap_functions;
    if (final_pass && options.hdr_output) out << "uniform float " << k_xl_headroom_uniform << ";\n";
    if (!vertex && !final_pass && options.hdr_output) out << "uniform float " << k_xl_limit_uniform << ";\n";
    if (smooth_alpha && options.alpha_hash) out << k_alpha_hash_function;
    else if (smooth_alpha) out << "uniform float " << k_xl_samples_uniform << ";\nint wbm_cover = -1;\n";

    std::istringstream in(glsl);
    std::string line;
    while (std::getline(in, line)) {
        if (line.compare(0, 8, "#version") == 0) continue;
        std::smatch m;
        if (!std::regex_match(line, m, decl)) {
            out << line << "\n";
            continue;
        }
        std::string kind = m[1], type = m[2], name = m[3], array = m[4];
        int loc;
        if (kind == "attribute") {
            auto it = attribs.find(name);
            if (it != attribs.end()) {
                loc = it->second;
            } else { // never bound by the game: take the lowest free index
                loc = 0;
                while (used_attribs.count(loc)) loc++;
                used_attribs.insert(loc);
            }
            out << "layout(location = " << loc << ") in " << type << " " << name << array << ";\n";
            continue;
        }
        auto it = varyings.location.find(name);
        if (it != varyings.location.end()) {
            loc = it->second;
        } else {
            if (!vertex) log += "fragment shader reads varying '" + name + "' that the vertex shader does not write\n";
            loc = varyings.next;
            varyings.location[name] = loc;
            varyings.next += location_count(type, array);
        }
        out << "layout(location = " << loc << ") " << (vertex ? "out " : "in ") << type << " " << name << array << ";\n";
    }

    std::string s = out.str();
    s = replace_word(s, "texture2DProj", "textureProj");
    s = replace_word(s, "texture2D", "texture");
    if (vertex) {
        // Appended to main, the last function the optimiser emits: the Y flip, the
        // frame's shift of a fraction of a pixel (zero but for the scene under
        // temporal upscaling, see k_xl_jitter_uniform), and the change from
        // OpenGL's clip-space depth (-w..w) to Metal's (0..w), reversed if asked.
        // SPIRV-Cross's own depth fix-up is turned off.
        size_t end = s.rfind('}');
        if (end != std::string::npos)
            s.insert(end, std::string("  gl_Position.y *= ") + k_xl_yflip_uniform + ";\n  gl_Position.xy += " + k_xl_jitter_uniform + " * gl_Position.w;\n  gl_Position.z = (" +
                              (options.reverse_depth ? "gl_Position.w - gl_Position.z" : "gl_Position.z + gl_Position.w") +
                              ") * 0.5;\n");
    } else {
        if (options.hdr_output) {
            // An 8-bit target clamps what a shader writes to 0..1 before blending,
            // and the engine's UI relies on it. Text is drawn in two passes, and the
            // first multiplies the screen by (font texture + a fade offset), which is
            // above 1 over the whole glyph rectangle while a line fades: harmless
            // when clamped, a visible rectangle on a float target.
            //
            // So the clamp is done here. Alpha is always limited to 0..1. Colour is
            // limited to 0..wbm_limit, which the backend sets to 1 for the screen and
            // leaves open for the engine's HDR buffers; the final shader, whose job
            // is to write colours above 1 to the screen, is not limited.
            size_t end = s.rfind('}');
            std::string top = final_pass ? "max(gl_FragColor.xyz, vec3(0.0))"
                                         : std::string("clamp(gl_FragColor.xyz, vec3(0.0), vec3(") + k_xl_limit_uniform + "))";
            if (end != std::string::npos) s.insert(end, "  gl_FragColor = vec4(" + top + ", clamp(gl_FragColor.w, 0.0, 1.0));\n");
        }
        s = replace_word(s, "gl_FragColor", "wbm_FragColor");
    }
    return s;
}

// Vulkan rules, relaxed so that loose uniforms are collected into one block.
bool to_spirv(EShLanguage lang, const std::string &glsl, std::vector<uint32_t> &spirv, std::string &log)
{
    static bool initialised = glslang::InitializeProcess();
    (void)initialised;
    glslang::TShader shader(lang);
    const char *text = glsl.c_str();
    shader.setStrings(&text, 1);
    shader.setEnvInput(glslang::EShSourceGlsl, lang, glslang::EShClientVulkan, 100);
    shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_0);
    shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_0);
    shader.setEnvInputVulkanRulesRelaxed();
    shader.setGlobalUniformBlockName("WbmUniforms");
    shader.setGlobalUniformSet(0);
    shader.setGlobalUniformBinding(0);
    shader.setAutoMapBindings(true);
    shader.setAutoMapLocations(true);
    EShMessages messages = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules);
    if (!shader.parse(GetDefaultResources(), 450, false, messages)) {
        log += shader.getInfoLog();
        return false;
    }
    glslang::TProgram program;
    program.addShader(&shader);
    if (!program.link(messages) || !program.mapIO()) {
        log += program.getInfoLog();
        return false;
    }
    glslang::GlslangToSpv(*program.getIntermediate(lang), spirv);
    return true;
}

bool to_msl(const std::vector<uint32_t> &spirv, bool vertex, XlStage &stage, std::string &log)
{
    try {
        spirv_cross::CompilerMSL msl(spirv);
        auto options = msl.get_msl_options();
        options.platform = spirv_cross::CompilerMSL::Options::macOS;
        options.set_msl_version(2, 3);
        msl.set_msl_options(options);

        stage.msl = msl.compile();
        stage.entry = msl.get_cleansed_entry_point_name("main", vertex ? spv::ExecutionModelVertex : spv::ExecutionModelFragment);

        auto resources = msl.get_shader_resources();
        for (auto &ubo : resources.uniform_buffers) {
            uint32_t index = msl.get_automatic_msl_resource_binding(ubo.id);
            if (index == ~0u) continue; // declared but unused
            const spirv_cross::SPIRType &type = msl.get_type(ubo.base_type_id);
            stage.uniform_buffer_index = (int)index;
            stage.uniform_buffer_size = (int)msl.get_declared_struct_size(type);
            for (uint32_t i = 0; i < type.member_types.size(); i++) {
                const spirv_cross::SPIRType &member = msl.get_type(type.member_types[i]);
                XlUniform u;
                u.name = msl.get_member_name(ubo.base_type_id, i);
                u.offset = (int)msl.type_struct_member_offset(type, i);
                u.is_int = member.basetype != spirv_cross::SPIRType::Float;
                u.components = (int)member.vecsize;
                u.columns = (int)member.columns;
                if (!member.array.empty()) {
                    u.array_size = (int)member.array[0];
                    u.array_stride = (int)msl.type_struct_member_array_stride(type, i);
                }
                stage.uniforms.push_back(u);
            }
        }
        for (auto &image : resources.sampled_images) {
            uint32_t index = msl.get_automatic_msl_resource_binding(image.id);
            if (index == ~0u) continue;
            stage.samplers.push_back({msl.get_name(image.id), (int)index});
        }
        if (vertex)
            for (auto &input : resources.stage_inputs)
                stage.input_locations.push_back((int)msl.get_decoration(input.id, spv::DecorationLocation));
        return true;
    } catch (const std::exception &e) {
        log += std::string("SPIRV-Cross: ") + e.what() + "\n";
        return false;
    }
}

} // namespace

XlProgram xl_translate(const std::string &vs_glsl, const std::string &fs_glsl,
                       const std::map<std::string, int> &attrib_locations, const XlOptions &options)
{
    XlProgram p;
    Varyings varyings;
    std::string vs = modernise(vs_glsl, true, attrib_locations, varyings, options, p.vs.srgb_samplers, p.log);
    std::string fs = modernise(fs_glsl, false, attrib_locations, varyings, options, p.fs.srgb_samplers, p.log);
    std::vector<uint32_t> vs_spirv, fs_spirv;
    p.ok = to_spirv(EShLangVertex, vs, vs_spirv, p.log) && to_spirv(EShLangFragment, fs, fs_spirv, p.log) &&
           to_msl(vs_spirv, true, p.vs, p.log) && to_msl(fs_spirv, false, p.fs, p.log);
    return p;
}
