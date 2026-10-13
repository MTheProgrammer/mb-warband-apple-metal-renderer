// On-disk cache of translated programs. Translating a program takes glslang
// and SPIRV-Cross a few milliseconds each, several hundred times per launch;
// the result depends only on the two sources and the attribute bindings.

#include "shader_xlate.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/stat.h>
#include <unistd.h>

namespace {

// Bump when the translation changes (the rewriter, its options, or the
// versions of glslang and SPIRV-Cross), so that stale entries are ignored.
const char k_version[] = "wbx-98 glslang-15.1.0 spirv-cross-1.4.304.0";
const uint32_t k_magic = 0x31584257; // "WBX1"

uint64_t fnv(const std::string &s, uint64_t h)
{
    for (unsigned char c : s) h = (h ^ c) * 0x100000001b3ull;
    return h;
}

std::string cache_dir()
{
    static std::string dir = [] {
        const char *env = getenv("WBM_CACHE");
        if (env && !strcmp(env, "0")) return std::string();
        std::string d = env && *env ? env : std::string(getenv("HOME") ? getenv("HOME") : "/tmp") + "/Library/Caches/warband-metal";
        d += "/shaders";
        for (size_t i = 1; i <= d.size(); i++)
            if (i == d.size() || d[i] == '/') mkdir(d.substr(0, i).c_str(), 0755);
        return d;
    }();
    return dir;
}

struct Writer {
    std::string out;
    void u32(uint32_t v) { out.append((const char *)&v, 4); }
    void str(const std::string &s)
    {
        u32((uint32_t)s.size());
        out += s;
    }
};

struct Reader {
    const std::string &in;
    size_t pos = 0;
    bool ok = true;
    uint32_t u32()
    {
        uint32_t v = 0;
        if (pos + 4 > in.size()) return ok = false, 0;
        memcpy(&v, in.data() + pos, 4);
        pos += 4;
        return v;
    }
    std::string str()
    {
        uint32_t n = u32();
        if (!ok || pos + n > in.size()) return ok = false, std::string();
        std::string s = in.substr(pos, n);
        pos += n;
        return s;
    }
};

void write_stage(Writer &w, const XlStage &s)
{
    w.str(s.msl);
    w.str(s.entry);
    w.u32((uint32_t)s.uniform_buffer_index);
    w.u32((uint32_t)s.uniform_buffer_size);
    w.u32((uint32_t)s.uniforms.size());
    for (const XlUniform &u : s.uniforms) {
        w.str(u.name);
        for (int v : {u.offset, (int)u.is_int, u.components, u.columns, u.array_size, u.array_stride}) w.u32((uint32_t)v);
    }
    w.u32((uint32_t)s.samplers.size());
    for (const XlSampler &m : s.samplers) {
        w.str(m.name);
        w.u32((uint32_t)m.texture_index);
    }
    w.u32((uint32_t)s.input_locations.size());
    for (int l : s.input_locations) w.u32((uint32_t)l);
    w.u32((uint32_t)s.srgb_samplers.size());
    for (const std::string &name : s.srgb_samplers) w.str(name);
}

void read_stage(Reader &r, XlStage &s)
{
    s.msl = r.str();
    s.entry = r.str();
    s.uniform_buffer_index = (int)r.u32();
    s.uniform_buffer_size = (int)r.u32();
    for (uint32_t n = r.u32(); n && r.ok; n--) {
        XlUniform u;
        u.name = r.str();
        u.offset = (int)r.u32();
        u.is_int = r.u32() != 0;
        u.components = (int)r.u32();
        u.columns = (int)r.u32();
        u.array_size = (int)r.u32();
        u.array_stride = (int)r.u32();
        s.uniforms.push_back(u);
    }
    for (uint32_t n = r.u32(); n && r.ok; n--) {
        XlSampler m;
        m.name = r.str();
        m.texture_index = (int)r.u32();
        s.samplers.push_back(m);
    }
    for (uint32_t n = r.u32(); n && r.ok; n--) s.input_locations.push_back((int)r.u32());
    for (uint32_t n = r.u32(); n && r.ok; n--) s.srgb_samplers.push_back(r.str());
}

bool load(const std::string &path, XlProgram &p)
{
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    std::string data;
    char buf[65536];
    for (size_t n; (n = fread(buf, 1, sizeof buf, f)) > 0;) data.append(buf, n);
    fclose(f);
    Reader r{data};
    if (r.u32() != k_magic || r.str() != k_version) return false;
    read_stage(r, p.vs);
    read_stage(r, p.fs);
    return p.ok = r.ok && r.pos == data.size();
}

void store(const std::string &path, const XlProgram &p)
{
    Writer w;
    w.u32(k_magic);
    w.str(k_version);
    write_stage(w, p.vs);
    write_stage(w, p.fs);
    std::string tmp = path + "." + std::to_string(getpid()) + ".tmp"; // renamed into place, so readers never see half a file
    FILE *f = fopen(tmp.c_str(), "wb");
    if (!f) return;
    bool ok = fwrite(w.out.data(), 1, w.out.size(), f) == w.out.size();
    ok = fclose(f) == 0 && ok;
    if (!ok || rename(tmp.c_str(), path.c_str()) != 0) unlink(tmp.c_str());
}

} // namespace

XlProgram xl_translate_cached(const std::string &vs_glsl, const std::string &fs_glsl,
                              const std::map<std::string, int> &attrib_locations, const XlOptions &options,
                              bool *from_cache)
{
    *from_cache = false;
    std::string dir = cache_dir();
    if (dir.empty()) return xl_translate(vs_glsl, fs_glsl, attrib_locations, options);

    std::string key = std::string(k_version) + '\0' + vs_glsl + '\0' + fs_glsl + '\0';
    for (auto &a : attrib_locations) key += a.first + "=" + std::to_string(a.second) + ";";
    key += std::string("\0soft_shadows=", 14) + (options.soft_shadows ? "1" : "0");
    key += std::string("\0pcss_shadows=", 14) + (options.pcss_shadows ? "1" : "0");
    key += std::string("\0dynamic_shadows=", 17) + (options.dynamic_shadows ? "1" : "0");
    key += std::string("\0smooth_alpha_test=", 19) + (options.smooth_alpha_test ? "1" : "0");
    key += std::string("\0reverse_depth=", 15) + (options.reverse_depth ? "1" : "0");
    key += std::string("\0srgb_textures=", 15) + (options.srgb_textures ? "1" : "0");
    key += std::string("\0pbr_specular=", 14) + (options.pbr_specular ? "1" : "0");
    key += std::string("\0pbr_environment=", 17) + (options.pbr_environment ? "1" : "0");
    key += std::string("\0pbr_sky=", 9) + (options.pbr_sky ? "1" : "0");
    key += std::string("\0sky_ambient=", 13) + (options.sky_ambient ? "1" : "0");
    key += std::string("\0materials=", 11) + (options.materials ? "1" : "0");
    key += std::string("\0skin=", 6) + (options.skin_scattering ? "1" : "0");
    key += std::string("\0foliage=", 9) + (options.foliage_backlight ? "1" : "0");
    key += std::string("\0wind=", 6) + (options.wind ? "1" : "0");
    key += std::string("\0eyes=", 6) + (options.eyes ? "1" : "0");
    key += std::string("\0fur=", 5) + (options.fur ? "1" : "0");
    key += std::string("\0hairs=", 7) + (options.hair_strands ? "1" : "0");
    key += std::string("\0debug_materials=", 17) + (options.debug_materials ? "1" : "0");
    key += std::string("\0filmic_tonemap=", 16) + (options.filmic_tonemap ? "1" : "0");
    key += std::string("\0hdr_output=", 12) + (options.hdr_output ? "1" : "0");
    char name[40];
    snprintf(name, sizeof name, "/%016llx%016llx.wbx", (unsigned long long)fnv(key, 0xcbf29ce484222325ull),
             (unsigned long long)fnv(key, 0x84222325cbf29ce4ull));
    std::string path = dir + name;

    XlProgram p;
    if (load(path, p)) {
        *from_cache = true;
        return p;
    }
    p = xl_translate(vs_glsl, fs_glsl, attrib_locations, options);
    if (p.ok) store(path, p);
    return p;
}
