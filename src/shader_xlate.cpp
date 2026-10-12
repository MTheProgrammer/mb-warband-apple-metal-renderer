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

namespace {

struct Varyings {
    std::map<std::string, int> location;
    int next = 0;
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
        vec2 t = clamp(q.xy + o / size, vec2(0.5 / size), vec2(1.0 - 0.5 / size)); // stay in this window's half
        lit += textureLod(wbm_dyn_map, vec2((t.x + column) * 0.5, t.y), 0.0).x < q.z - bias + dot(slope, o) ? 0.0 : 1.0;
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
bool smooth_alpha_tests(std::string &fs)
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
    vec3 sky = pow(max(wbm_sky_sample(above, clamp(rough * 7.0, 0.0, 5.0)), vec3(0.0)), vec3(2.2));
    float blur = 0.04 + 0.3 * rough;
    vec3 env = mix(ground, sky, smoothstep(-blur, blur, d.z));
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
const char k_ambient_functions[] = R"(
uniform vec4 wbm_amb[6];

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
    return vec4(sky.xyz + (ambient.xyz + sky.xyz) * (mix(vec3(1.0), tint, k) - 1.0) + wbm_amb[5].xyz * (0.5 - 0.5 * up), sky.w);
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

// mapped and smooth_: the cosine of the sun's angle to the normal-mapped
// surface and to the smooth one.
vec3 wbm_skin_core(float mapped, float smooth_, float shadow)
{
    // Where the light only grazes, a sharp shadow map draws the head's own
    // polygons on it as steps. Light there is nearly gone anyway; the shadow is let go.
    shadow = mix(1.0, shadow, smoothstep(0.0, 0.3, smooth_));
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
vec4 wbm_sway(vec4 p, float height)
{
    if (wbm_wind.y <= 0.0) return p;
    float t = wbm_wind.x, metres, phase;
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

// The vertex colour as the shader should see it: with its alpha full where
// that alpha has been taken over for heights.
vec4 wbm_tint(vec4 colour)
{
    return wbm_wind.z > 0.5 ? vec4(colour.xyz, 1.0) : colour;
}
)";

bool add_wind(std::string &vs)
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
    vs = std::regex_replace(vs, project, std::string("(matWorldViewProj * wbm_sway($1, ") + (coloured ? "inColor0.w" : "-1.0") + "))");
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

vec3 wbm_eye(vec2 uv, vec3 view, vec3 to_sun, float shadow, vec3 sun, vec3 sky)
{
    vec2 d = wbm_eye_place(uv);
    float inside = 1.0 - smoothstep(0.55, 1.0, dot(d, d));
    if (inside <= 0.0) return vec3(0.0);
    vec3 n = normalize(vec3(d * vec2(0.8, 0.45), 1.0)); // the ball bulges out of the flat it is painted on
    vec3 v = normalize(view), h = normalize(v + normalize(to_sun));
    float glint = pow(max(dot(n, h), 0.0), 220.0) * 2.5 * shadow;
    float grazing = 0.03 + 0.97 * pow(1.0 - max(dot(n, v), 0.0), 5.0);
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
    fs = std::regex_replace(fs, encode, "$1.xyz += wbm_eye(Tex0, ViewDir, SunLightDir, " + lit + ", vSunColor.xyz, vAmbientColor.xyz + vSkyLightColor.xyz);\n  $&",
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
    vec3 over = max(x - knee, vec3(0.0));
    return min(x, vec3(knee)) + (1.0 - knee) * (1.0 - exp(-over / (1.0 - knee)));
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

// Rewrites the declarations and built-ins that changed between GLSL 1.20 and
// 4.50. Attributes and varyings get explicit locations so that the two stages
// and the vertex descriptor agree without relying on declaration order.
std::string modernise(const std::string &source, bool vertex, const std::map<std::string, int> &attribs, Varyings &varyings,
                      const XlOptions &options, std::vector<std::string> &srgb_samplers, std::string &log)
{
    std::string glsl = source;
    int shadow_filter = options.pcss_shadows ? 2 : options.soft_shadows ? 1 : 0;
    bool shadows = !vertex && (shadow_filter || options.dynamic_shadows) && rewrite_shadows(glsl, shadow_filter, options.dynamic_shadows);
    bool soft_shadows = shadows && shadow_filter != 0;
    // The vertex shader has the sun's matrix; the fragment shader needs it too now.
    bool needs_sun_matrix = soft_shadows && options.pcss_shadows && glsl.find("uniform mat4 matSunViewProj;") == std::string::npos;
    bool smooth_alpha = !vertex && options.smooth_alpha_test && smooth_alpha_tests(glsl);
    bool pbr_specular = !vertex && (options.pbr_specular || options.pbr_environment || options.pbr_sky) && use_pbr_specular(glsl);
    std::string pbr_extra;
    bool pbr_environment = pbr_specular && (options.pbr_environment || options.pbr_sky) && add_pbr_environment(glsl, options.pbr_sky, options.materials, options.debug_materials, pbr_extra);
    if (!vertex && options.srgb_textures) decode_srgb_in_hardware(glsl, srgb_samplers); // after the above: it removes a pow they look for
    // After the shadow rewrite, which looks for the sun term's shadow variable by its original shape.
    bool skin = (options.skin_scattering || options.foliage_backlight) && (vertex ? add_skin_scattering_vertex(glsl) : add_skin_scattering(glsl, options.skin_scattering));
    bool eyes = !vertex && options.eyes && add_eye_glint(glsl);
    bool leaf = vertex && options.foliage_backlight && add_leaf_glow(glsl); // before the wind wraps the position
    bool wind = vertex && options.wind && add_wind(glsl);
    bool sky_ambient = options.sky_ambient && add_sky_ambient(glsl, vertex);
    bool final_pass = !vertex && (options.filmic_tonemap || options.hdr_output) && rewrite_final_pass(glsl, options);

    static const std::regex decl(R"(^\s*(attribute|varying)\s+(\w+)\s+(\w+)\s*(\[\s*\d+\s*\])?\s*;\s*$)");
    std::set<int> used_attribs;
    for (auto &a : attribs) used_attribs.insert(a.second);

    std::ostringstream out;
    out << "#version 450\n";
    if (vertex) out << "uniform float " << k_xl_yflip_uniform << ";\n";
    else out << "layout(location = 0) out vec4 wbm_FragColor;\n";
    if (needs_sun_matrix) out << "uniform mat4 matSunViewProj;\n";
    if (soft_shadows) out << (options.pcss_shadows ? k_pcss_shadow_function : k_soft_shadow_function);
    if (shadows && options.dynamic_shadows)
        out << std::regex_replace(k_dynamic_shadow_function, std::regex("DYN_SIZE"), std::to_string((int)k_xl_dyn_size) + ".0");
    if (pbr_specular) out << k_pbr_globals << k_pbr_specular_function;
    if (pbr_environment) out << k_pbr_environment_function;
    if (!pbr_extra.empty()) out << std::regex_replace(pbr_extra, std::regex("FACE_SIZE"), std::to_string((int)k_xl_sky_face_size) + ".0");
    if (eyes) out << k_eye_function;
    if (leaf) out << k_leaf_function;
    if (wind) out << k_wind_function;
    if (skin && vertex) out << k_skin_vertex_function;
    if (skin && !vertex) { // with the materials debug view, skin that scatters is painted red
        std::string f = k_skin_function;
        if (options.debug_materials) f = std::regex_replace(f, std::regex("return lit \\* pow\\(vec3\\(shadow\\), vec3\\(0\\.7, 1\\.0, 1\\.25\\)\\);"), "return vec3(4.0, 0.0, 0.0);");
        out << f;
    }
    if (sky_ambient) out << k_ambient_functions;
    if (final_pass) out << k_tonemap_functions;
    if (final_pass && options.hdr_output) out << "uniform float " << k_xl_headroom_uniform << ";\n";
    if (!vertex && !final_pass && options.hdr_output) out << "uniform float " << k_xl_limit_uniform << ";\n";
    if (smooth_alpha) out << "uniform float " << k_xl_samples_uniform << ";\nint wbm_cover = -1;\n";

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
        // Appended to main, the last function the optimiser emits: the Y flip, and
        // the change from OpenGL's clip-space depth (-w..w) to Metal's (0..w),
        // reversed if asked. SPIRV-Cross's own depth fix-up is turned off.
        size_t end = s.rfind('}');
        if (end != std::string::npos)
            s.insert(end, std::string("  gl_Position.y *= ") + k_xl_yflip_uniform + ";\n  gl_Position.z = (" +
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
