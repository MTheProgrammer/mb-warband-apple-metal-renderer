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

// Swaps the engine's shadow filter for wbm_shadow. The block is generated code
// and has one shape in every shader that has it; a shader where the shape is
// not found is left as it is.
bool soften_shadows(std::string &fs)
{
    static const std::regex blend(
        R"((\w+) = clamp \(mix \(mix \(\w+, \w+, \w+\.x\), mix \(\w+, \w+, \w+\.x\), \w+\.y\), 0\.0, 1\.0\);)");
    for (const char *name : {"uniform sampler2D shadowmap_texture;", "uniform float fShadowMapSize;", "ShadowTexCoord"})
        if (fs.find(name) == std::string::npos) return false;
    std::smatch m;
    if (!std::regex_search(fs, m, blend)) return false;
    fs = std::string(m.prefix()) + std::string(m[1]) + " = wbm_shadow(shadowmap_texture, ShadowTexCoord.xyz, fShadowMapSize);" +
         std::string(m.suffix());
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
    float a2 = clamp(2.0 / (shininess + 2.0), 0.0004, 1.0);
    float d = nh * nh * (a2 - 1.0) + 1.0;
    float lobe = (a2 * a2) / (d * d);
    float gl = 2.0 * nl / (nl + sqrt(a2 + (1.0 - a2) * nl * nl));
    float gv = 2.0 * nv / (nv + sqrt(a2 + (1.0 - a2) * nv * nv));
    const float f0 = 0.25;
    float fresnel = (f0 + (1.0 - f0) * pow(1.0 - vh, 5.0)) / f0;
    wbm_n = n;
    wbm_a2 = a2;
    return lobe * gl * gv * fresnel;
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
    return env * fresnel * wbm_mask;
}
)";

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
bool add_pbr_environment(std::string &fs)
{
    static const std::regex decode(R"((\w+)\.xyz = pow \((\w+)\.xyz, vec3\(2\.2, 2\.2, 2\.2\)\);)");
    static const std::regex encode(R"((\w+)\.xyz = pow \(\1\.xyz, output_gamma_inv\.xyz\);)");
    static const std::regex specular_map(R"(texture2D \(specular_texture, ([\w.]+)\))");
    static const std::regex alpha_mask(R"(spec_coef\) \* \w+\) \* (\w+)\.w\))");
    auto has = [&](const char *text) { return fs.find(text) != std::string::npos; };

    if (std::distance(std::sregex_iterator(fs.begin(), fs.end(), decode), std::sregex_iterator()) != 1) return false;
    if (!std::regex_search(fs, encode) || !has("varying vec3 ViewDir;")) return false;
    // The opening brace of main. The engine's shader files have CRLF line ends,
    // so the brace is looked for separately from the signature.
    size_t body = fs.find("void main ()");
    if (body != std::string::npos) body = fs.find('{', body);
    if (body == std::string::npos) return false;

    std::string mask;
    std::smatch m;
    if (std::regex_search(fs, m, specular_map)) {
        mask = "dot(texture2D(specular_texture, " + std::string(m[1]) + ").xyz, vec3(0.33))";
    } else if (std::regex_search(fs, m, alpha_mask)) {
        std::smatch uv;
        std::regex sample("\\b" + std::string(m[1]) + " = texture2D \\(diffuse_texture, ([\\w.]+)\\);");
        if (!std::regex_search(fs, uv, sample)) return false;
        mask = "texture2D(diffuse_texture, " + std::string(uv[1]) + ").w";
    } else {
        return false;
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
    std::string sky = "(" + horizon + " * 1.3)";
    std::string ground = has("uniform vec4 vGroundAmbientColor;") ? "(mix(vGroundAmbientColor.xyz, " + horizon + ", 0.4) * 0.6)"
                                                                  : "(" + horizon + " * 0.35)";
    fs = std::regex_replace(fs, encode, "$1.xyz += wbm_environment(ViewDir, " + up + ", " + sky + ", " + ground + ");\n  $&",
                            std::regex_constants::format_first_only);
    fs = std::regex_replace(fs, decode,
                            "wbm_albedo = pow ($2.xyz, vec3(2.2, 2.2, 2.2));\n  $1.xyz = (wbm_albedo * (1.0 - (0.35 * wbm_metal)));");
    body = fs.find('{', fs.find("void main ()")) + 1; // again: the replacements above moved it
    // The mask alone does not say how reflective a material is: on many opaque
    // textures it is simply 1. The engine scales it by the material's specular
    // colour, whose brightness runs from under 1 (stone, cloth) through 3-15
    // (leather, wood, mail) to 50 and more (plate), so that is what separates
    // metal from non-metal here.
    fs.insert(body, "\n  float wbm_gloss = dot(vSpecularColor.xyz, vec3(0.3333));"
                    "\n  float wbm_raw_mask = clamp(" + mask + ", 0.0, 1.0);"
                    "\n  wbm_mask = wbm_raw_mask * clamp(wbm_gloss / 40.0, 0.0, 1.0);"
                    "\n  wbm_metal = smoothstep(0.2, 0.6, wbm_raw_mask) * smoothstep(6.0, 30.0, wbm_gloss);");
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
    bool soft_shadows = !vertex && options.soft_shadows && soften_shadows(glsl);
    bool smooth_alpha = !vertex && options.smooth_alpha_test && smooth_alpha_tests(glsl);
    bool pbr_specular = !vertex && (options.pbr_specular || options.pbr_environment) && use_pbr_specular(glsl);
    bool pbr_environment = pbr_specular && options.pbr_environment && add_pbr_environment(glsl);
    if (!vertex && options.srgb_textures) decode_srgb_in_hardware(glsl, srgb_samplers); // after the above: it removes a pow they look for
    bool final_pass = !vertex && (options.filmic_tonemap || options.hdr_output) && rewrite_final_pass(glsl, options);

    static const std::regex decl(R"(^\s*(attribute|varying)\s+(\w+)\s+(\w+)\s*(\[\s*\d+\s*\])?\s*;\s*$)");
    std::set<int> used_attribs;
    for (auto &a : attribs) used_attribs.insert(a.second);

    std::ostringstream out;
    out << "#version 450\n";
    if (vertex) out << "uniform float " << k_xl_yflip_uniform << ";\n";
    else out << "layout(location = 0) out vec4 wbm_FragColor;\n";
    if (soft_shadows) out << k_soft_shadow_function;
    if (pbr_specular) out << k_pbr_globals << k_pbr_specular_function;
    if (pbr_environment) out << k_pbr_environment_function;
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
