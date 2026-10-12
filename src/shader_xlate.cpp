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
    if (!vertex && options.srgb_textures) decode_srgb_in_hardware(glsl, srgb_samplers);

    static const std::regex decl(R"(^\s*(attribute|varying)\s+(\w+)\s+(\w+)\s*(\[\s*\d+\s*\])?\s*;\s*$)");
    std::set<int> used_attribs;
    for (auto &a : attribs) used_attribs.insert(a.second);

    std::ostringstream out;
    out << "#version 450\n";
    if (vertex) out << "uniform float " << k_xl_yflip_uniform << ";\n";
    else out << "layout(location = 0) out vec4 wbm_FragColor;\n";
    if (soft_shadows) out << k_soft_shadow_function;
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
