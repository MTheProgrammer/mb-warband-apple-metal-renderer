// Tracing wrappers for the 62 OpenGL functions the game uses. Each wrapper
// records the call and forwards it to real OpenGL through real_gl*.

#include "trace.h"

#include "backend.h"
#include "interpose.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <mach-o/getsect.h>

// Where the game's calls end up after tracing: real OpenGL, or the Metal
// backend once wb_gl_install_metal has run. For OpenGL, direct functions bind
// at load time (references from this image are not interposed) and GLEW ones
// are filled in by wb_glewInit below.
#define GL_DIRECT(n) static decltype(&gl##n) real_gl##n = &gl##n;
#define GL_GLEW(n) static decltype(__glew##n) real_gl##n;
#include "gl_api.def"

namespace {

// ---- shadow state, kept only to make the trace readable -----------------

GLuint s_unit, s_program, s_fbo, s_rbo, s_array_buf, s_elem_buf;
GLuint s_tex[32][2]; // [unit][0: 2D, 1: cube map]
int s_screen_w, s_screen_h; // extent of this frame's viewports on the default framebuffer

struct TexInfo {
    GLenum ifmt = 0;
    int w = 0, h = 0, levels = 0;
    bool compressed = false;
};
struct RbInfo {
    GLenum fmt = 0;
    int w = 0, h = 0;
};
struct Attachment {
    bool is_texture;
    GLuint id;
    GLenum textarget;
    GLint level;
};
struct UniformName {
    std::string name;
    uint64_t hash;
};
struct ShaderInfo {
    GLenum type = 0;
    std::string file;
};

std::mutex s_mu;
std::unordered_map<GLuint, TexInfo> s_texinfo;
std::unordered_map<GLuint, RbInfo> s_rbinfo;
std::map<GLuint, std::map<GLenum, Attachment>> s_fbos;
std::unordered_map<GLuint, ShaderInfo> s_shaders;
std::unordered_map<GLuint, std::vector<GLuint>> s_attached;
std::unordered_map<uint64_t, UniformName> s_uniforms; // program << 32 | location
const UniformName k_unknown_uniform{"?", 0};

GLuint bound_tex(GLenum target)
{
    return s_unit < 32 ? s_tex[s_unit][target == GL_TEXTURE_2D ? 0 : 1] : 0;
}

void remember_uniform(GLuint program, GLint loc, const char *name)
{
    if (loc < 0) return;
    std::lock_guard<std::mutex> lock(s_mu);
    UniformName &u = s_uniforms[(uint64_t)program << 32 | (uint32_t)loc];
    if (u.name != name) u = {name, wb_hash(name, strlen(name))};
}

// Entries are never erased, so the returned reference stays valid.
const UniformName &uniform(GLint loc)
{
    std::lock_guard<std::mutex> lock(s_mu);
    auto it = s_uniforms.find((uint64_t)s_program << 32 | (uint32_t)loc);
    return it == s_uniforms.end() ? k_unknown_uniform : it->second;
}

const char *ids(GLsizei n, const GLuint *p)
{
    static thread_local char buf[256];
    int o = 0;
    buf[0] = 0;
    for (GLsizei i = 0; i < n && p && o < (int)sizeof buf - 24; i++) o += snprintf(buf + o, sizeof buf - o, "%s%u", i ? " " : "", p[i]);
    if (n > 0 && p && o >= (int)sizeof buf - 24) snprintf(buf + o, sizeof buf - o, " ...");
    return buf;
}

const char *floats(const GLfloat *p, int n)
{
    static thread_local char buf[512];
    int o = 0;
    buf[0] = 0;
    for (int i = 0; i < n && p && o < (int)sizeof buf - 24; i++) o += snprintf(buf + o, sizeof buf - o, "%s%g", i ? " " : "", p[i]);
    return buf;
}

uint32_t fbits(GLfloat f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    return u;
}

// Size class for histograms: the exponent of the next power of two.
int size_class(uint64_t n)
{
    int b = 0;
    while (b < 63 && (1ull << b) < n) b++;
    return b;
}

const void *s_game_lo, *s_game_hi;

bool from_game(const void *return_address)
{
    if (g_native) return false;
    if (!s_game_lo) {
        unsigned long size = 0;
        const void *text = getsegmentdata((const struct mach_header_64 *)_dyld_get_image_header(0), "__TEXT", &size);
        s_game_hi = (const char *)text + size;
        s_game_lo = text;
    }
    return return_address >= s_game_lo && return_address < s_game_hi;
}

void foreign(int fn, const void *return_address)
{
    if (g_native || !wb_count_foreign(fn)) return;
    Dl_info info{};
    dladdr(return_address, &info);
    wb_note("note: %s is also called by %s; such calls are forwarded but not traced", g_fn_names[fn],
            info.dli_fname ? info.dli_fname : "an unknown image");
}

const char *rb_desc(const RbInfo &rb)
{
    static thread_local char buf[96];
    snprintf(buf, sizeof buf, "renderbuffer %s %dx%d", E(rb.fmt), rb.w, rb.h);
    return buf;
}

const char *tex_desc(const TexInfo &t)
{
    static thread_local char buf[96];
    snprintf(buf, sizeof buf, "texture %s %dx%d", E_ifmt(t.ifmt), t.w, t.h);
    return buf;
}

void post(int fn)
{
    if (!g_glerr) return;
    for (GLenum e; (e = glGetError()) != GL_NO_ERROR;) wb_note("  !! %s after %s", E(e), g_fn_names[fn]);
}

void log_info_log(const char *what, GLuint id, const char *text)
{
    if (text && *text) wb_note("  %s %u info log: %s", what, id, text);
}

// ---- the 22 direct functions --------------------------------------------
//
// Interposing redirects these for every loaded image, not just the game: SDL
// and system frameworks call some of them too. Those calls go straight to real
// OpenGL and are only counted, so the trace describes the game alone. In
// untraced Metal mode every call goes straight to the backend.

void wb_glBindTexture(GLenum target, GLuint texture)
{
    if (g_fast) return real_glBindTexture(target, texture);
    if (!from_game(__builtin_return_address(0))) return foreign(FN_glBindTexture, __builtin_return_address(0)), glBindTexture(target, texture);
    wb_count(FN_glBindTexture);
    if (s_unit < 32) s_tex[s_unit][target == GL_TEXTURE_2D ? 0 : 1] = texture;
    HIST(FN_glBindTexture, target, s_unit, texture != 0, 0, "%s on unit %u, %s", E(target), s_unit, texture ? "texture" : "0");
    VLOG(FN_glBindTexture, "%s, %u [unit %u]", E(target), texture, s_unit);
    real_glBindTexture(target, texture);
}

void wb_glBlendFunc(GLenum sfactor, GLenum dfactor)
{
    if (g_fast) return real_glBlendFunc(sfactor, dfactor);
    if (!from_game(__builtin_return_address(0))) return foreign(FN_glBlendFunc, __builtin_return_address(0)), glBlendFunc(sfactor, dfactor);
    wb_count(FN_glBlendFunc);
    HIST(FN_glBlendFunc, sfactor, dfactor, 0, 0, "%s, %s", E_blend(sfactor), E_blend(dfactor));
    VLOG(FN_glBlendFunc, "%s, %s", E_blend(sfactor), E_blend(dfactor));
    real_glBlendFunc(sfactor, dfactor);
}

void wb_glClear(GLbitfield mask)
{
    if (g_fast) return real_glClear(mask);
    if (!from_game(__builtin_return_address(0))) return foreign(FN_glClear, __builtin_return_address(0)), glClear(mask);
    wb_count(FN_glClear);
    HIST(FN_glClear, mask, s_fbo != 0, 0, 0, "%s on %s", E_clear(mask), s_fbo ? "an FBO" : "the screen");
    VLOG(FN_glClear, "%s [fbo %u]", E_clear(mask), s_fbo);
    real_glClear(mask);
}

void wb_glClearColor(GLclampf r, GLclampf g, GLclampf b, GLclampf a)
{
    if (g_fast) return real_glClearColor(r, g, b, a);
    if (!from_game(__builtin_return_address(0))) return foreign(FN_glClearColor, __builtin_return_address(0)), glClearColor(r, g, b, a);
    wb_count(FN_glClearColor);
    HIST(FN_glClearColor, fbits(r), fbits(g), fbits(b), fbits(a), "%g, %g, %g, %g", r, g, b, a);
    VLOG(FN_glClearColor, "%g, %g, %g, %g", r, g, b, a);
    real_glClearColor(r, g, b, a);
}

void wb_glCullFace(GLenum mode)
{
    if (g_fast) return real_glCullFace(mode);
    if (!from_game(__builtin_return_address(0))) return foreign(FN_glCullFace, __builtin_return_address(0)), glCullFace(mode);
    wb_count(FN_glCullFace);
    HIST(FN_glCullFace, mode, 0, 0, 0, "%s", E(mode));
    VLOG(FN_glCullFace, "%s", E(mode));
    real_glCullFace(mode);
}

void wb_glDeleteTextures(GLsizei n, const GLuint *textures)
{
    if (g_fast) return real_glDeleteTextures(n, textures);
    if (!from_game(__builtin_return_address(0))) return foreign(FN_glDeleteTextures, __builtin_return_address(0)), glDeleteTextures(n, textures);
    wb_count(FN_glDeleteTextures);
    wb_log(FN_glDeleteTextures, "%d: %s", n, ids(n, textures));
    real_glDeleteTextures(n, textures);
    post(FN_glDeleteTextures);
}

void wb_glDepthFunc(GLenum func)
{
    if (g_fast) return real_glDepthFunc(func);
    if (!from_game(__builtin_return_address(0))) return foreign(FN_glDepthFunc, __builtin_return_address(0)), glDepthFunc(func);
    wb_count(FN_glDepthFunc);
    HIST(FN_glDepthFunc, func, 0, 0, 0, "%s", E(func));
    VLOG(FN_glDepthFunc, "%s", E(func));
    real_glDepthFunc(func);
}

void wb_glDepthMask(GLboolean flag)
{
    if (g_fast) return real_glDepthMask(flag);
    if (!from_game(__builtin_return_address(0))) return foreign(FN_glDepthMask, __builtin_return_address(0)), glDepthMask(flag);
    wb_count(FN_glDepthMask);
    HIST(FN_glDepthMask, flag, 0, 0, 0, "%d", flag);
    VLOG(FN_glDepthMask, "%d", flag);
    real_glDepthMask(flag);
}

void wb_glDisable(GLenum cap)
{
    if (g_fast) return real_glDisable(cap);
    if (!from_game(__builtin_return_address(0))) return foreign(FN_glDisable, __builtin_return_address(0)), glDisable(cap);
    wb_count(FN_glDisable);
    HIST(FN_glDisable, cap, 0, 0, 0, "%s", E(cap));
    VLOG(FN_glDisable, "%s", E(cap));
    real_glDisable(cap);
}

void wb_glDrawArrays(GLenum mode, GLint first, GLsizei count)
{
    if (g_fast) return real_glDrawArrays(mode, first, count);
    if (!from_game(__builtin_return_address(0))) return foreign(FN_glDrawArrays, __builtin_return_address(0)), glDrawArrays(mode, first, count);
    wb_count(FN_glDrawArrays);
    g_frame_draws.fetch_add(1, std::memory_order_relaxed);
    HIST(FN_glDrawArrays, mode, s_array_buf != 0, 0, 0, "%s, last attribute source: %s", E_prim(mode),
         s_array_buf ? "VBO" : "CLIENT MEMORY");
    VLOG(FN_glDrawArrays, "%s, first=%d, count=%d [program %u, fbo %u]", E_prim(mode), first, count, s_program, s_fbo);
    real_glDrawArrays(mode, first, count);
}

void wb_glDrawElements(GLenum mode, GLsizei count, GLenum type, const GLvoid *indices)
{
    if (g_fast) return real_glDrawElements(mode, count, type, indices);
    if (!from_game(__builtin_return_address(0))) return foreign(FN_glDrawElements, __builtin_return_address(0)), glDrawElements(mode, count, type, indices);
    wb_count(FN_glDrawElements);
    g_frame_draws.fetch_add(1, std::memory_order_relaxed);
    HIST(FN_glDrawElements, mode, type, s_elem_buf != 0, 0, "%s, %s, indices from %s", E_prim(mode), E(type),
         s_elem_buf ? "element buffer" : "CLIENT MEMORY");
    VLOG(FN_glDrawElements, "%s, count=%d, %s, %s=%p [program %u, fbo %u, element buffer %u]", E_prim(mode), count,
         E(type), s_elem_buf ? "offset" : "pointer", indices, s_program, s_fbo, s_elem_buf);
    real_glDrawElements(mode, count, type, indices);
}

void wb_glEnable(GLenum cap)
{
    if (g_fast) return real_glEnable(cap);
    if (!from_game(__builtin_return_address(0))) return foreign(FN_glEnable, __builtin_return_address(0)), glEnable(cap);
    wb_count(FN_glEnable);
    HIST(FN_glEnable, cap, 0, 0, 0, "%s", E(cap));
    VLOG(FN_glEnable, "%s", E(cap));
    real_glEnable(cap);
}

void wb_glGenTextures(GLsizei n, GLuint *textures)
{
    if (g_fast) return real_glGenTextures(n, textures);
    if (!from_game(__builtin_return_address(0))) return foreign(FN_glGenTextures, __builtin_return_address(0)), glGenTextures(n, textures);
    wb_count(FN_glGenTextures);
    real_glGenTextures(n, textures);
    wb_log(FN_glGenTextures, "%d -> %s", n, ids(n, textures));
    post(FN_glGenTextures);
}

// glGet* writes a pname-dependent number of values. To report them without a
// table of sizes, the query is repeated into a sentinel-filled scratch array.
void wb_glGetFloatv(GLenum pname, GLfloat *params)
{
    if (g_fast) return real_glGetFloatv(pname, params);
    if (!from_game(__builtin_return_address(0))) return foreign(FN_glGetFloatv, __builtin_return_address(0)), glGetFloatv(pname, params);
    wb_count(FN_glGetFloatv);
    real_glGetFloatv(pname, params);
    bool first = wb_hist(FN_glGetFloatv, pname, 0, 0, 0);
    if (first || g_verbose.load(std::memory_order_relaxed)) {
        GLfloat v[16];
        for (GLfloat &x : v) x = -12345.678f;
        real_glGetFloatv(pname, v);
        int n = 16;
        while (n > 0 && v[n - 1] == -12345.678f) n--;
        if (first) wb_hist_text(FN_glGetFloatv, pname, 0, 0, 0, "%s -> %s (first answer)", E(pname), floats(v, n));
        VLOG(FN_glGetFloatv, "%s -> %s", E(pname), floats(v, n));
    }
    post(FN_glGetFloatv);
}

void wb_glGetIntegerv(GLenum pname, GLint *params)
{
    if (g_fast) return real_glGetIntegerv(pname, params);
    if (!from_game(__builtin_return_address(0))) return foreign(FN_glGetIntegerv, __builtin_return_address(0)), glGetIntegerv(pname, params);
    wb_count(FN_glGetIntegerv);
    real_glGetIntegerv(pname, params);
    bool first = wb_hist(FN_glGetIntegerv, pname, 0, 0, 0);
    if (first || g_verbose.load(std::memory_order_relaxed)) {
        GLint v[16];
        for (GLint &x : v) x = 0x7fabcdef;
        real_glGetIntegerv(pname, v);
        int n = 16, o = 0;
        while (n > 0 && v[n - 1] == 0x7fabcdef) n--;
        char buf[256] = "";
        for (int i = 0; i < n; i++) o += snprintf(buf + o, sizeof buf - o, "%s%d", i ? " " : "", v[i]);
        if (first) wb_hist_text(FN_glGetIntegerv, pname, 0, 0, 0, "%s -> %s (first answer)", E(pname), buf);
        VLOG(FN_glGetIntegerv, "%s -> %s", E(pname), buf);
    }
    post(FN_glGetIntegerv);
}

void wb_glPixelStorei(GLenum pname, GLint param)
{
    if (g_fast) return real_glPixelStorei(pname, param);
    if (!from_game(__builtin_return_address(0))) return foreign(FN_glPixelStorei, __builtin_return_address(0)), glPixelStorei(pname, param);
    wb_count(FN_glPixelStorei);
    HIST(FN_glPixelStorei, pname, param, 0, 0, "%s, %d", E(pname), param);
    VLOG(FN_glPixelStorei, "%s, %d", E(pname), param);
    real_glPixelStorei(pname, param);
}

void wb_glPolygonMode(GLenum face, GLenum mode)
{
    if (g_fast) return real_glPolygonMode(face, mode);
    if (!from_game(__builtin_return_address(0))) return foreign(FN_glPolygonMode, __builtin_return_address(0)), glPolygonMode(face, mode);
    wb_count(FN_glPolygonMode);
    HIST(FN_glPolygonMode, face, mode, 0, 0, "%s, %s", E(face), E(mode));
    VLOG(FN_glPolygonMode, "%s, %s", E(face), E(mode));
    real_glPolygonMode(face, mode);
}

void wb_glTexEnvf(GLenum target, GLenum pname, GLfloat param)
{
    if (g_fast) return real_glTexEnvf(target, pname, param);
    if (!from_game(__builtin_return_address(0))) return foreign(FN_glTexEnvf, __builtin_return_address(0)), glTexEnvf(target, pname, param);
    wb_count(FN_glTexEnvf);
    HIST(FN_glTexEnvf, target, pname, fbits(param), 0, "%s, %s, %g", E(target), E(pname), param);
    VLOG(FN_glTexEnvf, "%s, %s, %g [unit %u]", E(target), E(pname), param, s_unit);
    real_glTexEnvf(target, pname, param);
    post(FN_glTexEnvf);
}

void wb_glTexImage2D(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border,
                     GLenum format, GLenum type, const GLvoid *pixels)
{
    if (g_fast) return real_glTexImage2D(target, level, internalformat, width, height, border, format, type, pixels);
    if (!from_game(__builtin_return_address(0))) return foreign(FN_glTexImage2D, __builtin_return_address(0)), glTexImage2D(target, level, internalformat, width, height, border, format, type, pixels);
    wb_count(FN_glTexImage2D);
    GLuint tex = bound_tex(target);
    {
        std::lock_guard<std::mutex> lock(s_mu);
        TexInfo &t = s_texinfo[tex];
        if (level == 0) t = {(GLenum)internalformat, width, height, 1, false};
        else t.levels = std::max(t.levels, level + 1);
    }
    HIST(FN_glTexImage2D, target, internalformat, format, (uint64_t)type << 1 | (pixels != nullptr),
         "%s, internalformat %s, format %s, type %s, %s", E(target), E_ifmt(internalformat), E(format), E(type),
         pixels ? "with data" : "no data (render target or placeholder)");
    wb_log(FN_glTexImage2D, "%s, level %d, %s, %dx%d, border %d, %s, %s, data=%s [texture %u]", E(target), level,
           E_ifmt(internalformat), width, height, border, E(format), E(type), pixels ? "yes" : "NULL", tex);
    real_glTexImage2D(target, level, internalformat, width, height, border, format, type, pixels);
    post(FN_glTexImage2D);
}

void wb_glTexParameterf(GLenum target, GLenum pname, GLfloat param)
{
    if (g_fast) return real_glTexParameterf(target, pname, param);
    if (!from_game(__builtin_return_address(0))) return foreign(FN_glTexParameterf, __builtin_return_address(0)), glTexParameterf(target, pname, param);
    wb_count(FN_glTexParameterf);
    bool is_enum = param == std::floor(param) && param >= 0x2600;
    HIST(FN_glTexParameterf, target, pname, fbits(param), 0, "%s, %s, %s", E(target), E(pname),
         is_enum ? E((GLenum)param) : floats(&param, 1));
    VLOG(FN_glTexParameterf, "%s, %s, %s [texture %u]", E(target), E(pname),
         is_enum ? E((GLenum)param) : floats(&param, 1), bound_tex(target));
    real_glTexParameterf(target, pname, param);
    post(FN_glTexParameterf);
}

void wb_glTexParameteri(GLenum target, GLenum pname, GLint param)
{
    if (g_fast) return real_glTexParameteri(target, pname, param);
    if (!from_game(__builtin_return_address(0))) return foreign(FN_glTexParameteri, __builtin_return_address(0)), glTexParameteri(target, pname, param);
    wb_count(FN_glTexParameteri);
    bool is_enum = param >= 0x2600;
    char num[16];
    snprintf(num, sizeof num, "%d", param);
    HIST(FN_glTexParameteri, target, pname, param, 0, "%s, %s, %s", E(target), E(pname), is_enum ? E((GLenum)param) : num);
    VLOG(FN_glTexParameteri, "%s, %s, %s [texture %u]", E(target), E(pname), is_enum ? E((GLenum)param) : num,
         bound_tex(target));
    real_glTexParameteri(target, pname, param);
    post(FN_glTexParameteri);
}

void wb_glViewport(GLint x, GLint y, GLsizei width, GLsizei height)
{
    if (g_fast) return real_glViewport(x, y, width, height);
    if (!from_game(__builtin_return_address(0))) return foreign(FN_glViewport, __builtin_return_address(0)), glViewport(x, y, width, height);
    wb_count(FN_glViewport);
    HIST(FN_glViewport, x, y, width, (uint64_t)height << 1 | (s_fbo != 0), "%d, %d, %d, %d on %s", x, y, width, height,
         s_fbo ? "an FBO" : "the screen");
    VLOG(FN_glViewport, "%d, %d, %d, %d [fbo %u]", x, y, width, height, s_fbo);
    if (!s_fbo) {
        s_screen_w = std::max(s_screen_w, x + width);
        s_screen_h = std::max(s_screen_h, y + height);
    }
    real_glViewport(x, y, width, height);
}

// ---- the 40 GLEW functions ----------------------------------------------

void wb_glActiveTexture(GLenum texture)
{
    wb_count(FN_glActiveTexture);
    s_unit = texture - GL_TEXTURE0;
    HIST(FN_glActiveTexture, texture, 0, 0, 0, "%s", E(texture));
    VLOG(FN_glActiveTexture, "%s", E(texture));
    real_glActiveTexture(texture);
}

void wb_glAttachShader(GLuint program, GLuint shader)
{
    wb_count(FN_glAttachShader);
    {
        std::lock_guard<std::mutex> lock(s_mu);
        s_attached[program].push_back(shader);
    }
    wb_log(FN_glAttachShader, "program %u, shader %u", program, shader);
    real_glAttachShader(program, shader);
    post(FN_glAttachShader);
}

void wb_glBindAttribLocation(GLuint program, GLuint index, const GLchar *name)
{
    wb_count(FN_glBindAttribLocation);
    HIST(FN_glBindAttribLocation, index, wb_hash(name, strlen(name)), 0, 0, "index %u = \"%s\"", index, name);
    wb_log(FN_glBindAttribLocation, "program %u, index %u, \"%s\"", program, index, name);
    real_glBindAttribLocation(program, index, name);
    post(FN_glBindAttribLocation);
}

void wb_glBindBuffer(GLenum target, GLuint buffer)
{
    wb_count(FN_glBindBuffer);
    if (target == GL_ARRAY_BUFFER) s_array_buf = buffer;
    else if (target == GL_ELEMENT_ARRAY_BUFFER) s_elem_buf = buffer;
    HIST(FN_glBindBuffer, target, buffer != 0, 0, 0, "%s, %s", E(target), buffer ? "buffer" : "0");
    VLOG(FN_glBindBuffer, "%s, %u", E(target), buffer);
    real_glBindBuffer(target, buffer);
}

void wb_glBindFramebuffer(GLenum target, GLuint framebuffer)
{
    wb_count(FN_glBindFramebuffer);
    if (target != GL_READ_FRAMEBUFFER) s_fbo = framebuffer;
    HIST(FN_glBindFramebuffer, target, framebuffer, 0, 0, "%s, %u", E(target), framebuffer);
    VLOG(FN_glBindFramebuffer, "%s, %u", E(target), framebuffer);
    real_glBindFramebuffer(target, framebuffer);
}

void wb_glBindRenderbuffer(GLenum target, GLuint renderbuffer)
{
    wb_count(FN_glBindRenderbuffer);
    s_rbo = renderbuffer;
    HIST(FN_glBindRenderbuffer, target, renderbuffer != 0, 0, 0, "%s, %s", E(target), renderbuffer ? "renderbuffer" : "0");
    VLOG(FN_glBindRenderbuffer, "%s, %u", E(target), renderbuffer);
    real_glBindRenderbuffer(target, renderbuffer);
}

void wb_glBlendEquation(GLenum mode)
{
    wb_count(FN_glBlendEquation);
    HIST(FN_glBlendEquation, mode, 0, 0, 0, "%s", E(mode));
    VLOG(FN_glBlendEquation, "%s", E(mode));
    real_glBlendEquation(mode);
}

void wb_glBufferData(GLenum target, GLsizeiptr size, const void *data, GLenum usage)
{
    wb_count(FN_glBufferData);
    int cls = size_class((uint64_t)size);
    HIST(FN_glBufferData, target, usage, cls, data != nullptr, "%s, size <= 2^%d bytes, %s, %s", E(target), cls,
         data ? "with data" : "NULL data (orphan or reserve)", E(usage));
    VLOG(FN_glBufferData, "%s, %ld bytes, data=%s, %s [buffer %u]", E(target), (long)size, data ? "yes" : "NULL",
         E(usage), target == GL_ARRAY_BUFFER ? s_array_buf : s_elem_buf);
    real_glBufferData(target, size, data, usage);
}

void wb_glCompileShader(GLuint shader)
{
    wb_count(FN_glCompileShader);
    real_glCompileShader(shader);
    GLint ok = 0;
    char info[4096] = "";
    real_glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    real_glGetShaderInfoLog(shader, sizeof info, nullptr, info);
    wb_log(FN_glCompileShader, "shader %u -> %s", shader, ok ? "compiled" : "FAILED");
    log_info_log("shader", shader, info);
    post(FN_glCompileShader);
}

void wb_glCompressedTexImage2D(GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height,
                               GLint border, GLsizei imageSize, const void *data)
{
    wb_count(FN_glCompressedTexImage2D);
    GLuint tex = bound_tex(target);
    {
        std::lock_guard<std::mutex> lock(s_mu);
        TexInfo &t = s_texinfo[tex];
        if (level == 0) t = {internalformat, width, height, 1, true};
        else t.levels = std::max(t.levels, level + 1);
    }
    HIST(FN_glCompressedTexImage2D, target, internalformat, 0, 0, "%s, %s", E(target), E(internalformat));
    wb_log(FN_glCompressedTexImage2D, "%s, level %d, %s, %dx%d, border %d, %d bytes [texture %u]", E(target), level,
           E(internalformat), width, height, border, imageSize, tex);
    real_glCompressedTexImage2D(target, level, internalformat, width, height, border, imageSize, data);
    post(FN_glCompressedTexImage2D);
}

GLuint wb_glCreateProgram(void)
{
    wb_count(FN_glCreateProgram);
    GLuint r = real_glCreateProgram();
    wb_log(FN_glCreateProgram, "-> program %u", r);
    return r;
}

GLuint wb_glCreateShader(GLenum type)
{
    wb_count(FN_glCreateShader);
    GLuint r = real_glCreateShader(type);
    {
        std::lock_guard<std::mutex> lock(s_mu);
        s_shaders[r] = {type, ""};
    }
    wb_log(FN_glCreateShader, "%s -> shader %u", E(type), r);
    return r;
}

void wb_glDeleteBuffers(GLsizei n, const GLuint *buffers)
{
    wb_count(FN_glDeleteBuffers);
    wb_log(FN_glDeleteBuffers, "%d: %s", n, ids(n, buffers));
    real_glDeleteBuffers(n, buffers);
}

void wb_glDeleteFramebuffers(GLsizei n, const GLuint *framebuffers)
{
    wb_count(FN_glDeleteFramebuffers);
    wb_log(FN_glDeleteFramebuffers, "%d: %s", n, ids(n, framebuffers));
    real_glDeleteFramebuffers(n, framebuffers);
}

void wb_glDeleteProgram(GLuint program)
{
    wb_count(FN_glDeleteProgram);
    {
        std::lock_guard<std::mutex> lock(s_mu);
        s_attached.erase(program);
    }
    wb_log(FN_glDeleteProgram, "program %u", program);
    real_glDeleteProgram(program);
}

void wb_glDeleteRenderbuffers(GLsizei n, const GLuint *renderbuffers)
{
    wb_count(FN_glDeleteRenderbuffers);
    wb_log(FN_glDeleteRenderbuffers, "%d: %s", n, ids(n, renderbuffers));
    real_glDeleteRenderbuffers(n, renderbuffers);
}

void wb_glDeleteShader(GLuint shader)
{
    wb_count(FN_glDeleteShader);
    wb_log(FN_glDeleteShader, "shader %u", shader);
    real_glDeleteShader(shader);
}

void wb_glDisableVertexAttribArray(GLuint index)
{
    wb_count(FN_glDisableVertexAttribArray);
    HIST(FN_glDisableVertexAttribArray, index, 0, 0, 0, "%u", index);
    VLOG(FN_glDisableVertexAttribArray, "%u", index);
    real_glDisableVertexAttribArray(index);
}

void wb_glEnableVertexAttribArray(GLuint index)
{
    wb_count(FN_glEnableVertexAttribArray);
    HIST(FN_glEnableVertexAttribArray, index, 0, 0, 0, "%u", index);
    VLOG(FN_glEnableVertexAttribArray, "%u", index);
    real_glEnableVertexAttribArray(index);
}

void wb_glFramebufferRenderbuffer(GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer)
{
    wb_count(FN_glFramebufferRenderbuffer);
    RbInfo rb;
    {
        std::lock_guard<std::mutex> lock(s_mu);
        if (renderbuffer) s_fbos[s_fbo][attachment] = {false, renderbuffer, renderbuffertarget, 0};
        else s_fbos[s_fbo].erase(attachment);
        rb = s_rbinfo[renderbuffer];
    }
    // The engine re-attaches targets to one FBO many times per frame: hot calls.
    HIST(FN_glFramebufferRenderbuffer, attachment, rb.fmt, rb.w, (uint64_t)rb.h << 1 | (renderbuffer != 0),
         "%s = %s", E(attachment), renderbuffer ? rb_desc(rb) : "none");
    VLOG(FN_glFramebufferRenderbuffer, "%s, %s, %s, renderbuffer %u [fbo %u; %s %dx%d]", E(target), E(attachment),
           E(renderbuffertarget), renderbuffer, s_fbo, E(rb.fmt), rb.w, rb.h);
    real_glFramebufferRenderbuffer(target, attachment, renderbuffertarget, renderbuffer);
    post(FN_glFramebufferRenderbuffer);
}

void wb_glFramebufferTexture2D(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level)
{
    wb_count(FN_glFramebufferTexture2D);
    TexInfo t;
    {
        std::lock_guard<std::mutex> lock(s_mu);
        if (texture) s_fbos[s_fbo][attachment] = {true, texture, textarget, level};
        else s_fbos[s_fbo].erase(attachment);
        t = s_texinfo[texture];
    }
    HIST(FN_glFramebufferTexture2D, attachment, t.ifmt, t.w, (uint64_t)t.h << 1 | (texture != 0), "%s = %s", E(attachment),
         texture ? tex_desc(t) : "none");
    VLOG(FN_glFramebufferTexture2D, "%s, %s, %s, texture %u, level %d [fbo %u; %s %dx%d]", E(target), E(attachment),
           E(textarget), texture, level, s_fbo, E_ifmt(t.ifmt), t.w, t.h);
    real_glFramebufferTexture2D(target, attachment, textarget, texture, level);
    post(FN_glFramebufferTexture2D);
}

void wb_glGenBuffers(GLsizei n, GLuint *buffers)
{
    wb_count(FN_glGenBuffers);
    real_glGenBuffers(n, buffers);
    wb_log(FN_glGenBuffers, "%d -> %s", n, ids(n, buffers));
}

void wb_glGenFramebuffers(GLsizei n, GLuint *framebuffers)
{
    wb_count(FN_glGenFramebuffers);
    real_glGenFramebuffers(n, framebuffers);
    wb_log(FN_glGenFramebuffers, "%d -> %s", n, ids(n, framebuffers));
}

void wb_glGenRenderbuffers(GLsizei n, GLuint *renderbuffers)
{
    wb_count(FN_glGenRenderbuffers);
    real_glGenRenderbuffers(n, renderbuffers);
    wb_log(FN_glGenRenderbuffers, "%d -> %s", n, ids(n, renderbuffers));
}

void wb_glGenerateMipmap(GLenum target)
{
    wb_count(FN_glGenerateMipmap);
    GLuint tex = bound_tex(target);
    TexInfo t;
    {
        std::lock_guard<std::mutex> lock(s_mu);
        t = s_texinfo[tex];
    }
    HIST(FN_glGenerateMipmap, target, t.ifmt, t.compressed, 0, "%s, texture format %s", E(target), E_ifmt(t.ifmt));
    VLOG(FN_glGenerateMipmap, "%s [texture %u, %s %dx%d]", E(target), tex, E_ifmt(t.ifmt), t.w, t.h);
    real_glGenerateMipmap(target);
    post(FN_glGenerateMipmap);
}

void wb_glGetProgramInfoLog(GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog)
{
    wb_count(FN_glGetProgramInfoLog);
    real_glGetProgramInfoLog(program, bufSize, length, infoLog);
    wb_log(FN_glGetProgramInfoLog, "program %u, bufSize %d -> \"%s\"", program, bufSize, bufSize > 0 && infoLog ? infoLog : "");
}

void wb_glGetProgramiv(GLuint program, GLenum pname, GLint *params)
{
    wb_count(FN_glGetProgramiv);
    real_glGetProgramiv(program, pname, params);
    wb_log(FN_glGetProgramiv, "program %u, %s -> %d", program, E(pname), params ? *params : 0);
}

void wb_glGetShaderInfoLog(GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog)
{
    wb_count(FN_glGetShaderInfoLog);
    real_glGetShaderInfoLog(shader, bufSize, length, infoLog);
    wb_log(FN_glGetShaderInfoLog, "shader %u, bufSize %d -> \"%s\"", shader, bufSize, bufSize > 0 && infoLog ? infoLog : "");
}

void wb_glGetShaderiv(GLuint shader, GLenum pname, GLint *params)
{
    wb_count(FN_glGetShaderiv);
    real_glGetShaderiv(shader, pname, params);
    wb_log(FN_glGetShaderiv, "shader %u, %s -> %d", shader, E(pname), params ? *params : 0);
}

GLint wb_glGetUniformLocation(GLuint program, const GLchar *name)
{
    wb_count(FN_glGetUniformLocation);
    GLint r = real_glGetUniformLocation(program, name);
    remember_uniform(program, r, name);
    HIST(FN_glGetUniformLocation, wb_hash(name, strlen(name)), r >= 0, 0, 0, "\"%s\" %s", name, r >= 0 ? "found" : "-> -1");
    // The engine looks uniforms up again for every draw, so this is a hot call.
    VLOG(FN_glGetUniformLocation, "program %u, \"%s\" -> %d", program, name, r);
    post(FN_glGetUniformLocation);
    return r;
}

// After the real link, ask OpenGL what the program looks like. The game never
// makes these queries; the answers are the reference for shader reflection.
void wb_glLinkProgram(GLuint program)
{
    wb_count(FN_glLinkProgram);
    real_glLinkProgram(program);
    GLint ok = 0, n_uniforms = 0, n_attribs = 0;
    char info[4096] = "";
    real_glGetProgramiv(program, GL_LINK_STATUS, &ok);
    real_glGetProgramInfoLog(program, sizeof info, nullptr, info);
    wb_log(FN_glLinkProgram, "program %u -> %s", program, ok ? "linked" : "FAILED");
    log_info_log("program", program, info);

    std::string out = "program " + std::to_string(program) + (ok ? " linked" : " LINK FAILED") + " (frame " +
                      std::to_string(g_frame.load()) + ")\n";
    {
        std::lock_guard<std::mutex> lock(s_mu);
        for (GLuint sh : s_attached[program]) {
            const ShaderInfo &si = s_shaders[sh];
            out += "  shader  " + std::to_string(sh) + " " + E(si.type) + " " + si.file + "\n";
        }
    }
    if (ok && !g_metal) { // these queries need real OpenGL
        char name[256], line[512];
        GLint size;
        GLenum type;
        real_glGetProgramiv(program, GL_ACTIVE_ATTRIBUTES, &n_attribs);
        for (GLint i = 0; i < n_attribs; i++) {
            name[0] = 0;
            glGetActiveAttrib(program, i, sizeof name, nullptr, &size, &type, name);
            snprintf(line, sizeof line, "  attrib  location %2d  %-18s size %d  %s\n", glGetAttribLocation(program, name),
                     E(type), size, name);
            out += line;
        }
        real_glGetProgramiv(program, GL_ACTIVE_UNIFORMS, &n_uniforms);
        for (GLint i = 0; i < n_uniforms; i++) {
            name[0] = 0;
            glGetActiveUniform(program, i, sizeof name, nullptr, &size, &type, name);
            GLint loc = real_glGetUniformLocation(program, name);
            remember_uniform(program, loc, name);
            snprintf(line, sizeof line, "  uniform location %2d  %-18s size %d  %s\n", loc, E(type), size, name);
            out += line;
        }
    }
    static std::mutex file_mu;
    std::lock_guard<std::mutex> lock(file_mu);
    std::string path = std::string(g_out_dir) + "/programs.txt";
    if (FILE *f = fopen(path.c_str(), "a")) {
        fputs(out.c_str(), f);
        fputc('\n', f);
        fclose(f);
    }
    post(FN_glLinkProgram);
}

void wb_glRenderbufferStorage(GLenum target, GLenum internalformat, GLsizei width, GLsizei height)
{
    wb_count(FN_glRenderbufferStorage);
    {
        std::lock_guard<std::mutex> lock(s_mu);
        s_rbinfo[s_rbo] = {internalformat, width, height};
    }
    HIST(FN_glRenderbufferStorage, target, internalformat, 0, 0, "%s, %s", E(target), E(internalformat));
    wb_log(FN_glRenderbufferStorage, "%s, %s, %dx%d [renderbuffer %u]", E(target), E(internalformat), width, height, s_rbo);
    real_glRenderbufferStorage(target, internalformat, width, height);
    post(FN_glRenderbufferStorage);
}

// Writes the exact text the engine hands to the compiler, all strings joined.
void wb_glShaderSource(GLuint shader, GLsizei count, const GLchar *const *string, const GLint *length)
{
    wb_count(FN_glShaderSource);
    std::string src;
    for (GLsizei i = 0; i < count; i++) {
        if (!string[i]) continue;
        if (length && length[i] >= 0) src.append(string[i], (size_t)length[i]);
        else src.append(string[i]);
    }
    GLenum type;
    {
        std::lock_guard<std::mutex> lock(s_mu);
        type = s_shaders[shader].type;
    }
    char file[96];
    snprintf(file, sizeof file, "shaders/%05u_%s_%016llx.glsl", shader, type == GL_VERTEX_SHADER ? "vs" : "fs",
             (unsigned long long)wb_hash(src.data(), src.size()));
    std::string path = std::string(g_out_dir) + "/" + file;
    if (FILE *f = fopen(path.c_str(), "w")) {
        fwrite(src.data(), 1, src.size(), f);
        fclose(f);
    }
    {
        std::lock_guard<std::mutex> lock(s_mu);
        s_shaders[shader].file = file;
    }
    HIST(FN_glShaderSource, count, length != nullptr, 0, 0, "%d string(s), lengths %s", count, length ? "given" : "NULL");
    wb_log(FN_glShaderSource, "shader %u, %d string(s), %zu bytes -> %s", shader, count, src.size(), file);
    real_glShaderSource(shader, count, string, length);
    post(FN_glShaderSource);
}

void wb_glUniform1f(GLint location, GLfloat v0)
{
    wb_count(FN_glUniform1f);
    const UniformName &u = uniform(location);
    HIST(FN_glUniform1f, u.hash, 0, 0, 0, "%s", u.name.c_str());
    VLOG(FN_glUniform1f, "%d \"%s\" = %g", location, u.name.c_str(), v0);
    real_glUniform1f(location, v0);
}

void wb_glUniform1i(GLint location, GLint v0)
{
    wb_count(FN_glUniform1i);
    const UniformName &u = uniform(location);
    HIST(FN_glUniform1i, u.hash, v0, 0, 0, "%s = %d", u.name.c_str(), v0);
    VLOG(FN_glUniform1i, "%d \"%s\" = %d", location, u.name.c_str(), v0);
    real_glUniform1i(location, v0);
}

void wb_glUniform1iv(GLint location, GLsizei count, const GLint *value)
{
    wb_count(FN_glUniform1iv);
    const UniformName &u = uniform(location);
    HIST(FN_glUniform1iv, u.hash, count, 0, 0, "%s, count %d", u.name.c_str(), count);
    VLOG(FN_glUniform1iv, "%d \"%s\", count %d, first = %d", location, u.name.c_str(), count, count > 0 && value ? value[0] : 0);
    real_glUniform1iv(location, count, value);
}

void wb_glUniform4fv(GLint location, GLsizei count, const GLfloat *value)
{
    wb_count(FN_glUniform4fv);
    const UniformName &u = uniform(location);
    HIST(FN_glUniform4fv, u.hash, count, 0, 0, "%s, count %d", u.name.c_str(), count);
    VLOG(FN_glUniform4fv, "%d \"%s\", count %d, first = %s", location, u.name.c_str(), count, floats(value, count > 0 ? 4 : 0));
    real_glUniform4fv(location, count, value);
}

void wb_glUniformMatrix4fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
    wb_count(FN_glUniformMatrix4fv);
    const UniformName &u = uniform(location);
    HIST(FN_glUniformMatrix4fv, u.hash, count, transpose, 0, "%s, count %d, transpose %d", u.name.c_str(), count, transpose);
    VLOG(FN_glUniformMatrix4fv, "%d \"%s\", count %d, transpose %d, first = %s", location, u.name.c_str(), count,
         transpose, floats(value, count > 0 ? 16 : 0));
    real_glUniformMatrix4fv(location, count, transpose, value);
}

void wb_glUseProgram(GLuint program)
{
    wb_count(FN_glUseProgram);
    s_program = program;
    HIST(FN_glUseProgram, program, 0, 0, 0, "program %u", program);
    VLOG(FN_glUseProgram, "%u", program);
    real_glUseProgram(program);
}

void wb_glVertexAttribPointer(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride,
                              const void *pointer)
{
    wb_count(FN_glVertexAttribPointer);
    uint64_t offset = s_array_buf ? (uint64_t)(uintptr_t)pointer : ~0ull;
    HIST(FN_glVertexAttribPointer, index, (uint64_t)type << 16 | (uint64_t)size << 8 | normalized, stride, offset,
         "index %u, size %d, %s, normalized %d, stride %d, %s %llu", index, size, E(type), normalized, stride,
         s_array_buf ? "offset" : "CLIENT MEMORY", s_array_buf ? (unsigned long long)offset : 0ull);
    VLOG(FN_glVertexAttribPointer, "index %u, size %d, %s, normalized %d, stride %d, %s=%p [buffer %u]", index, size,
         E(type), normalized, stride, s_array_buf ? "offset" : "pointer", pointer, s_array_buf);
    real_glVertexAttribPointer(index, size, type, normalized, stride, pointer);
}

// ---- GLEW hook ------------------------------------------------------------

// The game calls the 40 extension functions through GLEW's __glew* variables.
// After the real glewInit has filled them in, keep the real pointers and put
// the wrappers in their place. A pointer GLEW left NULL stays NULL. In native
// mode the game keeps calling the real functions directly.
GLenum wb_glewInit(void)
{
    wb_count(FN_glewInit);
    if (g_metal) { // there is no OpenGL context for the real glewInit to inspect
        wb_gl_install_metal();
        wb_log(FN_glewInit, "-> GLEW_OK, all pointers routed to the Metal backend");
        wb_flush();
        return GLEW_OK;
    }
    GLenum r = glewInit();
    int swapped = 0;
#define GL_DIRECT(n)
#define GL_GLEW(n)                                                                  \
    if (__glew##n && __glew##n != wb_gl##n) {                                      \
        real_gl##n = __glew##n;                                                    \
        if (!g_native) __glew##n = wb_gl##n;                                       \
        swapped++;                                                                 \
    } else if (!__glew##n) {                                                       \
        wb_note("  !! GLEW has no pointer for gl" #n);                             \
    }
#include "gl_api.def"
    wb_log(FN_glewInit, "-> %u (glewExperimental=%d), %d of 40 pointers redirected", r, glewExperimental, swapped);
    wb_flush();
    return r;
}

#define GL_DIRECT(n) WB_INTERPOSE(wb_gl##n, gl##n);
#define GL_GLEW(n)
#include "gl_api.def"
WB_INTERPOSE(wb_glewInit, glewInit);

} // namespace

// ---- helpers used by the SDL hooks and the summary -------------------------

void wb_gl_install_metal()
{
#define GL_DIRECT(n) real_gl##n = mtl_gl##n;
#define GL_GLEW(n)                                                                  \
    real_gl##n = mtl_gl##n;                                                        \
    __glew##n = g_fast ? mtl_gl##n : wb_gl##n;
#include "gl_api.def"
}

void wb_gl_context_created()
{
    static const GLenum strings[] = {GL_VENDOR, GL_RENDERER, GL_VERSION, GL_SHADING_LANGUAGE_VERSION};
    static const char *const string_names[] = {"vendor", "renderer", "version", "GLSL"};
    static const GLenum limits[] = {GL_MAX_TEXTURE_SIZE, GL_MAX_TEXTURE_IMAGE_UNITS, GL_MAX_VERTEX_ATTRIBS,
                                    GL_MAX_VERTEX_UNIFORM_COMPONENTS, GL_MAX_FRAGMENT_UNIFORM_COMPONENTS,
                                    GL_MAX_VARYING_FLOATS, GL_MAX_RENDERBUFFER_SIZE, GL_MAX_COLOR_ATTACHMENTS,
                                    GL_MAX_DRAW_BUFFERS, GL_MAX_SAMPLES, GL_DEPTH_BITS, GL_STENCIL_BITS, GL_ALPHA_BITS};
    std::string info;
    for (int i = 0; i < 4; i++) {
        const char *s = (const char *)glGetString(strings[i]);
        info += std::string(string_names[i]) + ": " + (s ? s : "(null)") + "\n";
    }
    for (GLenum pname : limits) {
        GLint v = -1;
        real_glGetIntegerv(pname, &v);
        info += std::string(E(pname)) + " = " + std::to_string(v) + "\n";
    }
    while (glGetError() != GL_NO_ERROR) {}
    wb_set_context_info(info.c_str());
    wb_note("GL context:\n%s", info.c_str());
    const char *ext = (const char *)glGetString(GL_EXTENSIONS);
    std::string path = std::string(g_out_dir) + "/extensions.txt";
    if (FILE *f = fopen(path.c_str(), "w")) {
        for (const char *p = ext ? ext : ""; *p; p++) fputc(*p == ' ' ? '\n' : *p, f);
        fclose(f);
    }
}

void wb_gl_dump_state(FILE *f)
{
    std::lock_guard<std::mutex> lock(s_mu);
    fprintf(f, "\n== framebuffer objects (last known attachments) ==\n");
    if (s_fbos.empty()) fprintf(f, "  (none)\n");
    for (auto &fbo : s_fbos) {
        fprintf(f, "fbo %u\n", fbo.first);
        for (auto &a : fbo.second) {
            if (a.second.is_texture) {
                const TexInfo &t = s_texinfo[a.second.id];
                fprintf(f, "  %-28s texture %u (%s level %d): %s %dx%d\n", E(a.first), a.second.id, E(a.second.textarget),
                        a.second.level, E_ifmt(t.ifmt), t.w, t.h);
            } else {
                const RbInfo &r = s_rbinfo[a.second.id];
                fprintf(f, "  %-28s renderbuffer %u: %s %dx%d\n", E(a.first), a.second.id, E(r.fmt), r.w, r.h);
            }
        }
    }
    fprintf(f, "\ntextures created: %zu, shaders: %zu\n", s_texinfo.size(), s_shaders.size());
    if (g_metal) mtl_dump_stats(f);
}

// The extent of the viewports the game set on the default framebuffer since
// the last call. Logged next to SDL's drawable size, which can be smaller.
void wb_gl_take_screen_extent(int *w, int *h)
{
    *w = s_screen_w;
    *h = s_screen_h;
    s_screen_w = s_screen_h = 0;
}

// Reads the back buffer of the default framebuffer, leaving GL state as found.
bool wb_gl_read_backbuffer(int w, int h, unsigned char *rgba)
{
    if (!real_glBindFramebuffer) return false;
    GLint fbo = 0, read_buffer = GL_BACK, pack = 4;
    while (glGetError() != GL_NO_ERROR) {} // the game never reads errors, so stale ones pile up
    real_glGetIntegerv(GL_FRAMEBUFFER_BINDING, &fbo);
    real_glGetIntegerv(GL_PACK_ALIGNMENT, &pack);
    real_glBindFramebuffer(GL_FRAMEBUFFER, 0);
    real_glGetIntegerv(GL_READ_BUFFER, &read_buffer);
    glReadBuffer(GL_BACK);
    real_glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
    real_glPixelStorei(GL_PACK_ALIGNMENT, pack);
    glReadBuffer((GLenum)read_buffer);
    real_glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)fbo);
    return glGetError() == GL_NO_ERROR;
}
