// GLSL 1.20 (as written by the engine's shader optimiser) -> GLSL 4.50 text ->
// SPIR-V (glslang) -> MSL (SPIRV-Cross).

#include "shader_xlate.h"

#include <SPIRV/GlslangToSpv.h>
#include <glslang/Public/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <spirv_msl.hpp>

#include <regex>
#include <set>
#include <sstream>

const char *const k_xl_yflip_uniform = "wbm_yflip";

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

// Rewrites the declarations and built-ins that changed between GLSL 1.20 and
// 4.50. Attributes and varyings get explicit locations so that the two stages
// and the vertex descriptor agree without relying on declaration order.
std::string modernise(const std::string &source, bool vertex, const std::map<std::string, int> &attribs, Varyings &varyings,
                      const XlOptions &options, std::string &log)
{
    std::string glsl = source;
    bool soft_shadows = !vertex && options.soft_shadows && soften_shadows(glsl);

    static const std::regex decl(R"(^\s*(attribute|varying)\s+(\w+)\s+(\w+)\s*(\[\s*\d+\s*\])?\s*;\s*$)");
    std::set<int> used_attribs;
    for (auto &a : attribs) used_attribs.insert(a.second);

    std::ostringstream out;
    out << "#version 450\n";
    if (vertex) out << "uniform float " << k_xl_yflip_uniform << ";\n";
    else out << "layout(location = 0) out vec4 wbm_FragColor;\n";
    if (soft_shadows) out << k_soft_shadow_function;

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
        size_t end = s.rfind('}'); // main is the last function the optimiser emits
        if (end != std::string::npos) s.insert(end, std::string("  gl_Position.y *= ") + k_xl_yflip_uniform + ";\n");
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
        auto common = msl.get_common_options();
        common.vertex.fixup_clipspace = true; // OpenGL clip-space Z is -w..w, Metal's is 0..w
        msl.set_common_options(common);

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
    std::string vs = modernise(vs_glsl, true, attrib_locations, varyings, options, p.log);
    std::string fs = modernise(fs_glsl, false, attrib_locations, varyings, options, p.log);
    std::vector<uint32_t> vs_spirv, fs_spirv;
    p.ok = to_spirv(EShLangVertex, vs, vs_spirv, p.log) && to_spirv(EShLangFragment, fs, fs_spirv, p.log) &&
           to_msl(vs_spirv, true, p.vs, p.log) && to_msl(fs_spirv, false, p.fs, p.log);
    return p;
}
