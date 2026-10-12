// Capturing a frame, and playing it back. Included by metal_backend.mm, inside
// its anonymous namespace.
//
// A capture file holds everything the renderer has been given up to the start
// of one frame (buffers, textures, programs, framebuffers, OpenGL state),
// followed by every OpenGL call of that frame. tools/replay.mm loads it into a
// fresh renderer and issues the calls again, as often as it likes, so that a
// scene can be looked at with other settings without starting the game.
//
// The file is a sequence of records: a 32-bit kind, a 32-bit payload length,
// the payload. Integers are stored as 64 bits, floats as doubles, byte strings
// with a 64-bit length in front (all ones for a null pointer).

enum : uint32_t {
    // Kinds below 1000 are OpenGL calls, numbered as in the Fn enumeration.
    CAP_HEADER = 1000,
    CAP_COUNTERS,
    CAP_BUFFER,
    CAP_TEXTURE,
    CAP_RENDERBUFFER,
    CAP_FRAMEBUFFER,
    CAP_SHADER,
    CAP_PROGRAM,
    CAP_GL_STATE,
    CAP_EXTRA,
    CAP_FRAME, // the frame's calls follow
    CAP_END,
};
enum { CAP_VERSION = 1 };

FILE *s_rec; // open while a frame is being recorded
volatile sig_atomic_t s_capture_requested;
const uint64_t k_null_blob = ~0ull;

struct Rec {
    uint32_t kind;
    std::vector<uint8_t> b;
    explicit Rec(uint32_t k) : kind(k) {}
    Rec &raw(const void *p, size_t n)
    {
        const uint8_t *q = (const uint8_t *)p;
        b.insert(b.end(), q, q + n);
        return *this;
    }
    Rec &i(int64_t v) { return raw(&v, 8); }
    Rec &f(double v) { return raw(&v, 8); }
    Rec &blob(const void *p, size_t n)
    {
        uint64_t len = p ? n : k_null_blob;
        raw(&len, 8);
        return p ? raw(p, n) : *this;
    }
    Rec &s(const std::string &v) { return blob(v.data(), v.size()); }
    ~Rec()
    {
        if (!s_rec) return;
        uint32_t head[2] = {kind, (uint32_t)b.size()};
        fwrite(head, 4, 2, s_rec);
        fwrite(b.data(), 1, b.size(), s_rec);
    }
};
#define REC(fn) \
    if (s_rec) Rec{(uint32_t)(fn)}

struct Rd {
    const uint8_t *p, *end;
    bool ok = true;
    template <class T> T get()
    {
        T v{};
        if (p + sizeof(T) > end) ok = false;
        else memcpy(&v, p, sizeof(T)), p += sizeof(T);
        return v;
    }
    int64_t i() { return get<int64_t>(); }
    double f() { return get<double>(); }
    // Null pointers come back as {nullptr, 0}.
    std::pair<const uint8_t *, size_t> blob()
    {
        uint64_t len = get<uint64_t>();
        if (len == k_null_blob || !ok) return {nullptr, 0};
        if (len > (uint64_t)(end - p)) {
            ok = false;
            return {nullptr, 0};
        }
        const uint8_t *at = p;
        p += len;
        return {at, (size_t)len};
    }
    std::string s()
    {
        auto v = blob();
        return v.first ? std::string((const char *)v.first, v.second) : std::string();
    }
};

// The name a uniform call's location stands for. Calls are recorded by name:
// locations depend on which uniforms the translated program ended up with, and
// that depends on the options in force.
std::string uniform_name(GLint location)
{
    Program *p = find(s_programs, s_program);
    if (!p || location < 0 || (size_t)location >= p->locations.size()) return std::string();
    return p->locations[(size_t)location].name;
}

size_t uniform_extent(const XlUniform &u) { return u.array_size > 0 ? (size_t)u.array_stride * (size_t)u.array_size : (size_t)u.components * (size_t)u.columns * 4; }

// ---- writing the state ---------------------------------------------------------------------

// Bytes per row and in all of one level of a texture, or false for a format
// whose contents are not saved.
bool level_layout(MTLPixelFormat format, NSUInteger w, NSUInteger h, NSUInteger *row, NSUInteger *size)
{
    NSUInteger block = 0, pixel = 0;
    switch (format) {
    case MTLPixelFormatBC1_RGBA:
    case MTLPixelFormatBC1_RGBA_sRGB: block = 8; break;
    case MTLPixelFormatBC2_RGBA:
    case MTLPixelFormatBC2_RGBA_sRGB:
    case MTLPixelFormatBC3_RGBA:
    case MTLPixelFormatBC3_RGBA_sRGB: block = 16; break;
    case MTLPixelFormatR8Unorm: pixel = 1; break;
    case MTLPixelFormatRG8Unorm:
    case MTLPixelFormatR16Float: pixel = 2; break;
    case MTLPixelFormatRGBA8Unorm:
    case MTLPixelFormatRGBA8Unorm_sRGB:
    case MTLPixelFormatBGRA8Unorm:
    case MTLPixelFormatBGRA8Unorm_sRGB:
    case MTLPixelFormatR32Float:
    case MTLPixelFormatRG16Float: pixel = 4; break;
    case MTLPixelFormatRGBA16Float: pixel = 8; break;
    case MTLPixelFormatRGBA32Float: pixel = 16; break;
    default: return false;
    }
    if (block) {
        *row = ((w + 3) / 4) * block;
        *size = *row * ((h + 3) / 4);
    } else {
        *row = w * pixel;
        *size = *row * h;
    }
    return true;
}

// Reads levels 0..last of a texture back from the GPU, whatever its storage.
std::vector<std::vector<uint8_t>> read_texture(id<MTLTexture> t, int last)
{
    std::vector<std::vector<uint8_t>> levels;
    std::vector<id<MTLBuffer>> buffers;
    id<MTLCommandBuffer> cmd = [s_queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    for (int level = 0; level <= last && level < (int)t.mipmapLevelCount; level++) {
        NSUInteger w = std::max<NSUInteger>(1, t.width >> level), h = std::max<NSUInteger>(1, t.height >> level), row, size;
        if (!level_layout(t.pixelFormat, w, h, &row, &size)) break;
        id<MTLBuffer> buffer = [s_device newBufferWithLength:size options:MTLResourceStorageModeShared];
        [blit copyFromTexture:t sourceSlice:0 sourceLevel:(NSUInteger)level sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(w, h, 1)
                     toBuffer:buffer destinationOffset:0 destinationBytesPerRow:row destinationBytesPerImage:size];
        buffers.push_back(buffer);
    }
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    for (id<MTLBuffer> buffer : buffers) {
        const uint8_t *p = (const uint8_t *)buffer.contents;
        levels.emplace_back(p, p + buffer.length);
    }
    return levels;
}

void write_gl_state()
{
    Rec r(CAP_GL_STATE);
    r.i(s_fbo).i(s_rbo).i(s_array_buffer).i(s_element_buffer).i(s_program).i(s_unit);
    r.i(MAX_UNITS);
    for (int u = 0; u < MAX_UNITS; u++) r.i(s_bound_texture[u]);
    r.i(MAX_ATTRIBS);
    for (int a = 0; a < MAX_ATTRIBS; a++) {
        const Attrib &at = s_attribs[a];
        r.i(at.enabled).i(at.buffer).i(at.size).i(at.type).i(at.normalized).i(at.stride).i((int64_t)at.offset);
    }
    r.f(s_clear_color.red).f(s_clear_color.green).f(s_clear_color.blue).f(s_clear_color.alpha);
    r.i(s_viewport.x).i(s_viewport.y).i(s_viewport.w).i(s_viewport.h);
    r.i(s_depth_test).i(s_depth_mask).i(s_cull).i(s_blend);
    r.i(s_depth_func).i(s_cull_face).i(s_polygon_mode).i(s_blend_src).i(s_blend_dst).i(s_blend_eq);
}

// Writes everything a frame can depend on. Called between frames.
void write_state()
{
    @autoreleasepool {
        Rec(CAP_HEADER).i(CAP_VERSION).i(s_screen_w).i(s_screen_h).i((int64_t)s_frame_serial);
        Rec(CAP_COUNTERS).i(s_next_object).i(s_next_texture).i(s_next_buffer).i(s_next_framebuffer).i(s_next_renderbuffer);
        for (auto &e : s_buffers)
            if (e.second.mtl) Rec(CAP_BUFFER).i(e.first).blob(e.second.mtl.contents, e.second.mtl.length);
        for (auto &e : s_textures) {
            const Texture &t = e.second;
            Rec r(CAP_TEXTURE);
            // Screen-sized render targets are made afresh on playback, at whatever
            // scale and sample count are in force then; their contents last a frame.
            int kind = !t.mtl ? 0 : (t.screen_sized && t.gl_w > 0) ? 2 : 1;
            r.i(e.first).i(kind).i(t.min_filter).i(t.mag_filter).i(t.wrap_s).i(t.wrap_t).f(t.anisotropy).i(t.max_level).i(t.material);
            if (kind == 2) {
                r.i(t.gl_internal).i(t.gl_format).i(t.gl_type).i(t.gl_w).i(t.gl_h);
            } else if (kind == 1) {
                MTLTextureSwizzleChannels sw = t.mtl.swizzle;
                r.i((int64_t)t.mtl.pixelFormat).i((int64_t)t.mtl.width).i((int64_t)t.mtl.height).i((int64_t)t.mtl.mipmapLevelCount).i((int64_t)t.mtl.usage);
                r.i(sw.red).i(sw.green).i(sw.blue).i(sw.alpha).i(t.top_level).i(t.scale);
                std::vector<std::vector<uint8_t>> levels = read_texture(t.mtl, std::max(t.top_level, 0));
                r.i((int64_t)levels.size());
                for (auto &level : levels) r.blob(level.data(), level.size());
            }
        }
        for (auto &e : s_renderbuffers) Rec(CAP_RENDERBUFFER).i(e.first).i(e.second.gl_internal).i(e.second.gl_w).i(e.second.gl_h);
        for (auto &e : s_framebuffers) Rec(CAP_FRAMEBUFFER).i(e.first).i(e.second.color_texture).i(e.second.depth_renderbuffer).i(e.second.depth_texture);
        for (auto &e : s_shaders) Rec(CAP_SHADER).i(e.first).i(e.second->type).s(e.second->source);
        for (auto &e : s_programs) {
            const Program &p = e.second;
            Rec r(CAP_PROGRAM);
            r.i(e.first).i(p.linked).i((int64_t)p.shaders.size());
            for (auto &sh : p.shaders) r.i(sh->type).s(sh->source);
            r.i((int64_t)p.attribs.size());
            for (auto &a : p.attribs) r.s(a.first).i(a.second);
            r.i((int64_t)p.locations.size());
            for (const UniformLocation &l : p.locations) {
                r.s(l.name).i(l.unit);
                for (int stage = 0; stage < 2; stage++) {
                    const XlUniform *u = l.uniform[stage];
                    size_t have = u && (size_t)u->offset < p.uniform_data[stage].size() ? p.uniform_data[stage].size() - (size_t)u->offset : 0;
                    if (u) r.blob(p.uniform_data[stage].data() + u->offset, std::min(have, uniform_extent(*u)));
                    else r.blob(nullptr, 0);
                }
            }
        }
        write_gl_state();
        {
            Rec r(CAP_EXTRA);
            r.i(s_sun_matrix_valid);
            for (float v : s_sun_matrix) r.f(v);
            r.i((int64_t)s_scenery.size());
            for (uint64_t m : s_scenery) r.i((int64_t)m);
            r.i((int64_t)s_scenery_frame);
            r.i((int64_t)s_overlays.size());
            for (uint64_t m : s_overlays) r.i((int64_t)m);
        }
        Rec{CAP_FRAME};
    }
}

// Called at the end of mtl_present: finishes a recording, or starts one.
void capture_at_frame_end()
{
    if (s_rec) {
        { Rec end(CAP_END); }
        long size = ftell(s_rec);
        fclose(s_rec);
        s_rec = nullptr;
        wb_note("frame capture written, %.0f MB", (double)size / (1024.0 * 1024.0));
        return;
    }
    if (!s_capture_requested) return;
    s_capture_requested = 0;
    const char *dir = getenv("WBM_OUT");
    char path[1200];
    snprintf(path, sizeof path, "%s/capture_%06llu.wbcap", dir ? dir : ".", (unsigned long long)s_frame_serial);
    s_rec = fopen(path, "wb");
    if (!s_rec) {
        complain("cannot write a frame capture to %s", path);
        return;
    }
    wb_note("capturing the next frame to %s", path);
    write_state();
}

// ---- playing a capture back ----------------------------------------------------------------

struct Replay {
    std::vector<uint8_t> file;
    size_t state_at = 0, frame_at = 0; // offsets of the first state record and of the first call
    std::vector<uint8_t> gl_state, counters; // payloads, applied again before each pass over the frame
    int width = 0, height = 0;
    uint64_t calls = 0, skipped_uniforms = 0;
    // The free camera. The captured frame's camera is found once (find_camera);
    // `delta` then takes a position in that camera's view space to the same
    // position seen from the moved camera, and `turn` is its rotation alone.
    bool camera_known = false, camera_moved = false, hide_ui = false;
    double p00 = 1, p11 = 1, A = 0, B = 0, C = -1; // the projection: x and y scales, z = A * z_view + B, w = C * z_view
    double view[16] = {};                          // world to view space
    double delta[16] = {}, turn[16] = {};
    float eye[4] = {};                             // the moved camera's position in the world
    double orbit[6] = {};                          // yaw, pitch (radians), pan x, pan y, forward (metres), pivot distance
};
Replay s_replay;

bool next_record(const std::vector<uint8_t> &file, size_t *at, uint32_t *kind, Rd *rd)
{
    if (*at + 8 > file.size()) return false;
    uint32_t head[2];
    memcpy(head, file.data() + *at, 8);
    if (*at + 8 + head[1] > file.size()) return false;
    *kind = head[0];
    rd->p = file.data() + *at + 8;
    rd->end = rd->p + head[1];
    rd->ok = true;
    *at += 8 + head[1];
    return true;
}

void apply_counters(Rd r)
{
    s_next_object = (GLuint)r.i();
    s_next_texture = (GLuint)r.i();
    s_next_buffer = (GLuint)r.i();
    s_next_framebuffer = (GLuint)r.i();
    s_next_renderbuffer = (GLuint)r.i();
}

void apply_gl_state(Rd r)
{
    s_fbo = (GLuint)r.i(), s_rbo = (GLuint)r.i(), s_array_buffer = (GLuint)r.i(), s_element_buffer = (GLuint)r.i();
    s_program = (GLuint)r.i(), s_unit = (GLuint)r.i();
    int64_t units = r.i();
    for (int64_t u = 0; u < units; u++) {
        GLuint tex = (GLuint)r.i();
        if (u < MAX_UNITS) s_bound_texture[u] = tex;
    }
    int64_t attribs = r.i();
    for (int64_t a = 0; a < attribs; a++) {
        Attrib at;
        at.enabled = r.i() != 0, at.buffer = (GLuint)r.i(), at.size = (GLint)r.i(), at.type = (GLenum)r.i();
        at.normalized = (GLboolean)r.i(), at.stride = (GLsizei)r.i(), at.offset = (uintptr_t)r.i();
        if (a < MAX_ATTRIBS) s_attribs[a] = at;
    }
    double red = r.f(), green = r.f(), blue = r.f(), alpha = r.f();
    s_clear_color = MTLClearColorMake(red, green, blue, alpha);
    s_viewport.x = (GLint)r.i(), s_viewport.y = (GLint)r.i(), s_viewport.w = (GLsizei)r.i(), s_viewport.h = (GLsizei)r.i();
    s_depth_test = r.i() != 0, s_depth_mask = r.i() != 0, s_cull = r.i() != 0, s_blend = r.i() != 0;
    s_depth_func = (GLenum)r.i(), s_cull_face = (GLenum)r.i(), s_polygon_mode = (GLenum)r.i();
    s_blend_src = (GLenum)r.i(), s_blend_dst = (GLenum)r.i(), s_blend_eq = (GLenum)r.i();
    s_last_key_program = 0;
}

struct SavedUniform {
    std::string name;
    int unit = 0;
    std::vector<uint8_t> raw[2];
    bool has[2] = {false, false};
};

void restore_uniform(Program &p, const SavedUniform &u)
{
    auto it = p.location_by_name.find(u.name);
    if (it == p.location_by_name.end()) return;
    UniformLocation &l = p.locations[(size_t)it->second];
    l.unit = u.unit;
    for (int stage = 0; stage < 2; stage++) {
        const XlUniform *x = l.uniform[stage];
        if (!x) continue;
        const std::vector<uint8_t> &raw = u.has[stage] ? u.raw[stage] : u.raw[1 - stage]; // the same value in either stage
        if ((size_t)x->offset >= p.uniform_data[stage].size()) continue;
        size_t n = std::min({raw.size(), uniform_extent(*x), p.uniform_data[stage].size() - (size_t)x->offset});
        if (n) memcpy(p.uniform_data[stage].data() + x->offset, raw.data(), n);
        p.uniform_dirty[stage] = true;
    }
}

bool restore_state()
{
    Replay &rp = s_replay;
    size_t at = rp.state_at;
    uint32_t kind;
    Rd r{};
    // Engine uniforms are global in effect (the sun, the fog, the camera): a
    // program that never received one in the captured run, because its shader
    // had no use for it under the options then, takes the value another got.
    std::unordered_map<std::string, SavedUniform> by_name;
    std::vector<std::pair<GLuint, std::set<std::string>>> given;
    int textures = 0, programs = 0, buffers = 0, replaced = 0, resized = 0;
    while (next_record(rp.file, &at, &kind, &r)) {
        if (kind == CAP_FRAME) break;
        @autoreleasepool {
            switch (kind) {
            case CAP_COUNTERS:
                rp.counters.assign(r.p, r.end);
                apply_counters(r);
                break;
            case CAP_BUFFER: {
                GLuint id = (GLuint)r.i();
                auto data = r.blob();
                if (data.first && data.second) s_buffers[id].mtl = [s_device newBufferWithBytes:data.first length:data.second options:MTLResourceStorageModeShared];
                buffers++;
                break;
            }
            case CAP_TEXTURE: {
                GLuint id = (GLuint)r.i();
                int tex_kind = (int)r.i();
                Texture &t = s_textures[id];
                t.min_filter = (GLenum)r.i(), t.mag_filter = (GLenum)r.i(), t.wrap_s = (GLenum)r.i(), t.wrap_t = (GLenum)r.i();
                t.anisotropy = (float)r.f();
                t.max_level = (int)r.i();
                t.material = (uint8_t)r.i();
                if (tex_kind == 2) {
                    GLint internal = (GLint)r.i();
                    GLenum format = (GLenum)r.i(), type = (GLenum)r.i();
                    GLsizei w = (GLsizei)r.i(), h = (GLsizei)r.i();
                    GLuint unit = s_unit, bound = s_bound_texture[0];
                    s_unit = 0, s_bound_texture[0] = id;
                    mtl_glTexImage2D(GL_TEXTURE_2D, 0, internal, w, h, 0, format, type, nullptr);
                    s_unit = unit, s_bound_texture[0] = bound;
                } else if (tex_kind == 1) {
                    MTLPixelFormat format = (MTLPixelFormat)r.i();
                    NSUInteger w = (NSUInteger)r.i(), h = (NSUInteger)r.i(), mips = (NSUInteger)r.i();
                    MTLTextureUsage usage = (MTLTextureUsage)r.i();
                    MTLTextureSwizzleChannels sw;
                    sw.red = (MTLTextureSwizzle)r.i(), sw.green = (MTLTextureSwizzle)r.i(), sw.blue = (MTLTextureSwizzle)r.i(), sw.alpha = (MTLTextureSwizzle)r.i();
                    t.top_level = (int)r.i();
                    t.scale = (int)r.i();
                    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h mipmapped:NO];
                    d.mipmapLevelCount = mips;
                    d.usage = usage;
                    const MTLTextureSwizzleChannels none = MTLTextureSwizzleChannelsDefault;
                    bool swizzled = memcmp(&sw, &none, sizeof none) != 0;
                    if (s_shader_options.srgb_textures && !swizzled) d.usage |= MTLTextureUsagePixelFormatView;
                    d.swizzle = sw;
                    t.mtl = [s_device newTextureWithDescriptor:d];
                    t.format = format;
                    int64_t levels = r.i();
                    for (int64_t level = 0; level < levels; level++) {
                        auto data = r.blob();
                        // A texture pack applies to a played-back frame as it does to the
                        // game: the largest level says which of the game's files this is.
                        if (level == 0 && data.first && !s_material_by_content.empty()) {
                            int alpha = format == MTLPixelFormatBC2_RGBA ? 3 : format == MTLPixelFormatBC3_RGBA ? 5 : format == MTLPixelFormatBC1_RGBA ? 0 : -1;
                            auto it = alpha < 0 ? s_material_by_content.end() : s_material_by_content.find(content_key((uint32_t)w, (uint32_t)h, data.first, data.second, alpha));
                            if (it != s_material_by_content.end()) attach_pbr_map(t, it->second);
                            if (it != s_material_by_content.end() && replace_from_pack(t, it->second)) {
                                replaced++;
                                if (t.mtl.width != w || t.mtl.height != h) resized++;
                            }
                        }
                        if (t.replaced) continue;
                        NSUInteger lw = std::max<NSUInteger>(1, w >> level), lh = std::max<NSUInteger>(1, h >> level), row, size;
                        if (!data.first || !t.mtl || !level_layout(format, lw, lh, &row, &size) || data.second < size) continue;
                        [t.mtl replaceRegion:MTLRegionMake2D(0, 0, lw, lh) mipmapLevel:(NSUInteger)level withBytes:data.first bytesPerRow:row];
                    }
                }
                textures++;
                break;
            }
            case CAP_RENDERBUFFER: {
                GLuint id = (GLuint)r.i();
                GLenum internal = (GLenum)r.i();
                GLsizei w = (GLsizei)r.i(), h = (GLsizei)r.i();
                GLuint was = s_rbo;
                s_rbo = id;
                if (w > 0 && h > 0) mtl_glRenderbufferStorage(GL_RENDERBUFFER, internal, w, h);
                else s_renderbuffers[id];
                s_rbo = was;
                break;
            }
            case CAP_FRAMEBUFFER: {
                GLuint id = (GLuint)r.i();
                Framebuffer &fb = s_framebuffers[id];
                fb.color_texture = (GLuint)r.i(), fb.depth_renderbuffer = (GLuint)r.i(), fb.depth_texture = (GLuint)r.i();
                break;
            }
            case CAP_SHADER: {
                GLuint id = (GLuint)r.i();
                auto sh = std::make_shared<Shader>();
                sh->type = (GLenum)r.i();
                sh->source = r.s();
                s_shaders[id] = sh;
                break;
            }
            case CAP_PROGRAM: {
                GLuint id = (GLuint)r.i();
                bool linked = r.i() != 0;
                Program &p = s_programs[id];
                int64_t shaders = r.i();
                for (int64_t n = 0; n < shaders; n++) {
                    auto sh = std::make_shared<Shader>();
                    sh->type = (GLenum)r.i();
                    sh->source = r.s();
                    p.shaders.push_back(sh);
                }
                int64_t attribs = r.i();
                for (int64_t n = 0; n < attribs; n++) {
                    std::string name = r.s();
                    p.attribs[name] = (int)r.i();
                }
                if (linked) link_program(id, p);
                int64_t uniforms = r.i();
                given.emplace_back(id, std::set<std::string>());
                for (int64_t n = 0; n < uniforms; n++) {
                    SavedUniform u;
                    u.name = r.s();
                    u.unit = (int)r.i();
                    for (int stage = 0; stage < 2; stage++) {
                        auto raw = r.blob();
                        u.has[stage] = raw.first != nullptr;
                        if (raw.first) u.raw[stage].assign(raw.first, raw.first + raw.second);
                    }
                    if (p.linked) restore_uniform(p, u);
                    given.back().second.insert(u.name);
                    if (u.has[0] || u.has[1]) by_name[u.name] = u;
                }
                programs++;
                break;
            }
            case CAP_GL_STATE:
                rp.gl_state.assign(r.p, r.end);
                apply_gl_state(r);
                break;
            case CAP_EXTRA: {
                s_sun_matrix_valid = r.i() != 0;
                for (float &v : s_sun_matrix) v = (float)r.f();
                int64_t n = r.i();
                for (int64_t k = 0; k < n; k++) s_scenery.insert((uint64_t)r.i());
                s_scenery_frame = ~0ull >> 1; // never taken for stale: the engine does not draw its map again
                (void)r.i();
                n = r.i();
                for (int64_t k = 0; k < n; k++) s_overlays.insert((uint64_t)r.i());
                break;
            }
            default: break;
            }
        }
        if (!r.ok) {
            wb_note("replay: a record of kind %u is shorter than its contents", kind);
            return false;
        }
    }
    rp.frame_at = at;
    for (auto &g : given) {
        Program *p = find(s_programs, g.first);
        if (!p || !p->linked) continue;
        for (const UniformLocation &l : p->locations) {
            if (g.second.count(l.name)) continue;
            auto it = by_name.find(l.name);
            if (it != by_name.end()) {
                SavedUniform u = it->second;
                u.unit = l.unit; // a sampler's unit is the program's own business
                restore_uniform(*p, u);
            }
        }
    }
    wb_note("replay: %d buffers, %d textures, %d programs restored", buffers, textures, programs);
    if (replaced) wb_note("replay: %d textures taken from texture packs, %d of them at another size than the game's", replaced, resized);
    return true;
}

// Looks through the frame for a draw of the scene proper that sets its
// world-view-projection, world-view and world matrices, and takes the camera
// apart from them: projection = WVP x inverse(WV), view = WV x inverse(W).
void find_camera()
{
    Replay &rp = s_replay;
    size_t at = rp.frame_at;
    uint32_t kind;
    Rd r{};
    double wvp[16], wv[16], w[16];
    bool have_wvp = false, have_wv = false, have_w = false, depth_test = true;
    while (!rp.camera_known && next_record(rp.file, &at, &kind, &r) && kind != CAP_END) {
        if (kind == FN_glUseProgram) {
            have_wvp = have_wv = have_w = false;
        } else if (kind == FN_glEnable || kind == FN_glDisable) {
            if ((GLenum)r.i() == GL_DEPTH_TEST) depth_test = kind == FN_glEnable;
        } else if (kind == FN_glUniformMatrix4fv) {
            std::string name = r.s();
            (void)r.i();
            auto data = r.blob();
            double *to = name == "matWorldViewProj" ? wvp : name == "matWorldView" ? wv : name == "matWorld" ? w : nullptr;
            if (to && data.first && data.second >= 64) {
                float m[16];
                memcpy(m, data.first, 64);
                for (int k = 0; k < 16; k++) to[k] = m[k];
                (to == wvp ? have_wvp : to == wv ? have_wv : have_w) = true;
            }
        } else if ((kind == FN_glDrawElements || kind == FN_glDrawArrays) && have_wvp && have_wv && have_w && depth_test) {
            double inv[16], proj[16];
            if (!mat_inverse(wv, inv)) continue;
            mat_mul(wvp, inv, proj);
            bool perspective = fabs(fabs(proj[11]) - 1.0) < 0.01 && fabs(proj[15]) < 0.01 && fabs(proj[0]) > 1e-3 && fabs(proj[5]) > 1e-3 && fabs(proj[14]) > 1e-6;
            if (!perspective || !mat_inverse(w, inv)) continue;
            rp.p00 = proj[0], rp.p11 = proj[5], rp.A = proj[10], rp.B = proj[14], rp.C = proj[11];
            mat_mul(wv, inv, rp.view);
            rp.camera_known = true;
        }
    }
    if (!rp.camera_known) wb_note("replay: no camera was found in the frame; it cannot be moved");
}

void rotation_about(const double *axis, double angle, double *out)
{
    double x = axis[0], y = axis[1], z = axis[2], len = sqrt(x * x + y * y + z * z);
    if (len < 1e-12) len = 1, x = 0, y = 0, z = 1;
    x /= len, y /= len, z /= len;
    double c = cos(angle), s = sin(angle), t = 1 - c;
    const double m[16] = {t * x * x + c, t * x * y + s * z, t * x * z - s * y, 0, t * x * y - s * z, t * y * y + c, t * y * z + s * x, 0,
                          t * x * z + s * y, t * y * z - s * x, t * z * z + c, 0, 0, 0, 0, 1};
    memcpy(out, m, sizeof m);
}

void translation(double x, double y, double z, double *out)
{
    const double m[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1};
    memcpy(out, m, sizeof m);
}

// Turns the orbit settings into `delta` and `turn`. The scene is turned about
// a point straight ahead of the captured camera, by yaw about the world's up
// axis and by pitch about the camera's own horizontal, then shifted.
void update_camera()
{
    Replay &rp = s_replay;
    const double *o = rp.orbit;
    rp.camera_moved = rp.camera_known && (o[0] != 0 || o[1] != 0 || o[2] != 0 || o[3] != 0 || o[4] != 0);
    if (!rp.camera_moved) return;
    const double up[3] = {rp.view[8], rp.view[9], rp.view[10]}, across[3] = {1, 0, 0}; // the world's Z axis, in view space
    double yaw[16], pitch[16], to_pivot[16], from_pivot[16], shift[16], a[16], b[16];
    rotation_about(up, o[0], yaw);
    rotation_about(across, o[1], pitch);
    mat_mul(pitch, yaw, rp.turn);
    double pivot_z = o[5] / rp.C; // view-space z of a point that far ahead
    translation(0, 0, -pivot_z, from_pivot);
    translation(0, 0, pivot_z, to_pivot);
    translation(o[2], o[3], -o[4] / rp.C, shift);
    mat_mul(rp.turn, from_pivot, a);
    mat_mul(to_pivot, a, b);
    mat_mul(shift, b, rp.delta);
    double moved[16], inv[16];
    mat_mul(rp.delta, rp.view, moved);
    if (mat_inverse(moved, inv))
        for (int k = 0; k < 3; k++) rp.eye[k] = (float)inv[12 + k];
}

// Whether the framebuffer bound now is one the scene's camera draws into.
bool replay_scene_target()
{
    Target t;
    return s_fbo != 0 && resolve_target(t) && (t.scene || t.prepass);
}

// Rewrites one matrix of a uniform call for the moved camera. clip: the
// matrix ends in the projection (rows x, y, z, w of clip space); otherwise it
// ends in view space. The sky and whatever else is drawn without a depth test
// sits at no particular distance, and only turns.
void move_matrix(float *m, bool clip)
{
    Replay &rp = s_replay;
    const double *d = s_depth_test ? rp.delta : rp.turn;
    for (int c = 0; c < 4; c++) {
        double h = c == 3 ? 1.0 : 0.0; // these matrices are affine up to the projection
        double v[3] = {clip ? m[c * 4 + 0] / rp.p00 : m[c * 4 + 0], clip ? m[c * 4 + 1] / rp.p11 : m[c * 4 + 1], clip ? m[c * 4 + 3] / rp.C : m[c * 4 + 2]};
        double n[3];
        for (int row = 0; row < 3; row++) n[row] = d[row] * v[0] + d[4 + row] * v[1] + d[8 + row] * v[2] + d[12 + row] * h;
        if (clip) {
            m[c * 4 + 0] = (float)(rp.p00 * n[0]);
            m[c * 4 + 1] = (float)(rp.p11 * n[1]);
            m[c * 4 + 2] = (float)(rp.A * n[2] + rp.B * h);
            m[c * 4 + 3] = (float)(rp.C * n[2]);
        } else {
            m[c * 4 + 0] = (float)n[0], m[c * 4 + 1] = (float)n[1], m[c * 4 + 2] = (float)n[2];
        }
    }
}

GLint replay_location(const std::string &name)
{
    GLint loc = name.empty() ? -1 : mtl_glGetUniformLocation(s_program, name.c_str());
    if (loc < 0) s_replay.skipped_uniforms++;
    return loc;
}

// Issues the captured frame's calls once.
void replay_calls()
{
    Replay &rp = s_replay;
    if (!rp.counters.empty()) apply_counters(Rd{rp.counters.data(), rp.counters.data() + rp.counters.size()});
    if (!rp.gl_state.empty()) apply_gl_state(Rd{rp.gl_state.data(), rp.gl_state.data() + rp.gl_state.size()});
    update_camera();
    int screen_draws = 0;
    size_t at = rp.frame_at;
    uint32_t kind;
    Rd r{};
    while (next_record(rp.file, &at, &kind, &r) && kind != CAP_END) {
        rp.calls++;
        // Without the interface: on the screen itself, only the first draw, which
        // is the finished scene, is let through.
        if (rp.hide_ui && s_fbo == 0 && (kind == FN_glDrawElements || kind == FN_glDrawArrays) && screen_draws++ > 0) continue;
        switch (kind) {
        case FN_glClear: mtl_glClear((GLbitfield)r.i()); break;
        case FN_glClearColor: {
            float c[4];
            for (float &v : c) v = (float)r.f();
            mtl_glClearColor(c[0], c[1], c[2], c[3]);
            break;
        }
        case FN_glViewport: {
            int64_t v[4];
            for (int64_t &x : v) x = r.i();
            mtl_glViewport((GLint)v[0], (GLint)v[1], (GLsizei)v[2], (GLsizei)v[3]);
            break;
        }
        case FN_glDepthMask: mtl_glDepthMask((GLboolean)r.i()); break;
        case FN_glDepthFunc: mtl_glDepthFunc((GLenum)r.i()); break;
        case FN_glCullFace: mtl_glCullFace((GLenum)r.i()); break;
        case FN_glBlendEquation: mtl_glBlendEquation((GLenum)r.i()); break;
        case FN_glPolygonMode: mtl_glPolygonMode(GL_FRONT_AND_BACK, (GLenum)r.i()); break;
        case FN_glBlendFunc: {
            GLenum src = (GLenum)r.i(), dst = (GLenum)r.i();
            mtl_glBlendFunc(src, dst);
            break;
        }
        case FN_glEnable: mtl_glEnable((GLenum)r.i()); break;
        case FN_glDisable: mtl_glDisable((GLenum)r.i()); break;
        case FN_glBindBuffer: {
            GLenum target = (GLenum)r.i();
            mtl_glBindBuffer(target, (GLuint)r.i());
            break;
        }
        case FN_glBufferData: {
            GLenum target = (GLenum)r.i();
            int64_t size = r.i();
            auto data = r.blob();
            mtl_glBufferData(target, (GLsizeiptr)size, data.first, GL_STATIC_DRAW);
            break;
        }
        case FN_glEnableVertexAttribArray: mtl_glEnableVertexAttribArray((GLuint)r.i()); break;
        case FN_glDisableVertexAttribArray: mtl_glDisableVertexAttribArray((GLuint)r.i()); break;
        case FN_glVertexAttribPointer: {
            int64_t v[6];
            for (int64_t &x : v) x = r.i();
            mtl_glVertexAttribPointer((GLuint)v[0], (GLint)v[1], (GLenum)v[2], (GLboolean)v[3], (GLsizei)v[4], (const void *)(uintptr_t)v[5]);
            break;
        }
        case FN_glGenTextures:
        case FN_glGenBuffers:
        case FN_glGenFramebuffers:
        case FN_glGenRenderbuffers: {
            GLsizei n = (GLsizei)r.i();
            GLuint first = (GLuint)r.i();
            std::vector<GLuint> ids((size_t)std::max(n, 1));
            if (kind == FN_glGenTextures) mtl_glGenTextures(n, ids.data());
            else if (kind == FN_glGenBuffers) mtl_glGenBuffers(n, ids.data());
            else if (kind == FN_glGenFramebuffers) mtl_glGenFramebuffers(n, ids.data());
            else mtl_glGenRenderbuffers(n, ids.data());
            if (n > 0 && ids[0] != first) complain("replay: an object was numbered %u in the capture and %u now", first, ids[0]);
            break;
        }
        case FN_glActiveTexture: mtl_glActiveTexture((GLenum)r.i()); break;
        case FN_glBindTexture: {
            GLenum target = (GLenum)r.i();
            mtl_glBindTexture(target, (GLuint)r.i());
            break;
        }
        case FN_glTexImage2D: {
            int64_t v[7];
            for (int64_t &x : v) x = r.i();
            auto data = r.blob();
            mtl_glTexImage2D((GLenum)v[0], (GLint)v[1], (GLint)v[2], (GLsizei)v[3], (GLsizei)v[4], 0, (GLenum)v[5], (GLenum)v[6], data.first);
            break;
        }
        case FN_glCompressedTexImage2D: {
            int64_t v[5];
            for (int64_t &x : v) x = r.i();
            auto data = r.blob();
            mtl_glCompressedTexImage2D((GLenum)v[0], (GLint)v[1], (GLenum)v[2], (GLsizei)v[3], (GLsizei)v[4], 0, (GLsizei)data.second, data.first);
            break;
        }
        case FN_glTexParameteri: {
            GLenum pname = (GLenum)r.i();
            mtl_glTexParameteri(GL_TEXTURE_2D, pname, (GLint)r.i());
            break;
        }
        case FN_glTexParameterf: {
            GLenum pname = (GLenum)r.i();
            mtl_glTexParameterf(GL_TEXTURE_2D, pname, (GLfloat)r.f());
            break;
        }
        case FN_glGenerateMipmap: mtl_glGenerateMipmap(GL_TEXTURE_2D); break;
        case FN_glTexEnvf: {
            GLenum pname = (GLenum)r.i();
            mtl_glTexEnvf(GL_TEXTURE_FILTER_CONTROL, pname, (GLfloat)r.f());
            break;
        }
        case FN_glPixelStorei: {
            GLenum pname = (GLenum)r.i();
            mtl_glPixelStorei(pname, (GLint)r.i());
            break;
        }
        case FN_glBindFramebuffer: {
            GLenum target = (GLenum)r.i();
            mtl_glBindFramebuffer(target, (GLuint)r.i());
            break;
        }
        case FN_glBindRenderbuffer: mtl_glBindRenderbuffer(GL_RENDERBUFFER, (GLuint)r.i()); break;
        case FN_glRenderbufferStorage: {
            GLenum internal = (GLenum)r.i();
            GLsizei w = (GLsizei)r.i(), h = (GLsizei)r.i();
            mtl_glRenderbufferStorage(GL_RENDERBUFFER, internal, w, h);
            break;
        }
        case FN_glFramebufferTexture2D: {
            GLenum attachment = (GLenum)r.i();
            GLuint texture = (GLuint)r.i();
            mtl_glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, texture, (GLint)r.i());
            break;
        }
        case FN_glFramebufferRenderbuffer: {
            GLenum attachment = (GLenum)r.i();
            mtl_glFramebufferRenderbuffer(GL_FRAMEBUFFER, attachment, GL_RENDERBUFFER, (GLuint)r.i());
            break;
        }
        case FN_glCreateShader: {
            GLenum type = (GLenum)r.i();
            GLuint was = (GLuint)r.i(), now = mtl_glCreateShader(type);
            if (was != now) complain("replay: a shader was numbered %u in the capture and %u now", was, now);
            break;
        }
        case FN_glCreateProgram: {
            GLuint was = (GLuint)r.i(), now = mtl_glCreateProgram();
            if (was != now) complain("replay: a program was numbered %u in the capture and %u now", was, now);
            break;
        }
        case FN_glShaderSource: {
            GLuint shader = (GLuint)r.i();
            std::string source = r.s();
            const GLchar *text = source.c_str();
            GLint length = (GLint)source.size();
            mtl_glShaderSource(shader, 1, &text, &length);
            break;
        }
        case FN_glAttachShader: {
            GLuint program = (GLuint)r.i();
            mtl_glAttachShader(program, (GLuint)r.i());
            break;
        }
        case FN_glBindAttribLocation: {
            GLuint program = (GLuint)r.i(), index = (GLuint)r.i();
            std::string name = r.s();
            mtl_glBindAttribLocation(program, index, name.c_str());
            break;
        }
        case FN_glLinkProgram: mtl_glLinkProgram((GLuint)r.i()); break;
        case FN_glUseProgram: mtl_glUseProgram((GLuint)r.i()); break;
        case FN_glUniform1f: {
            GLint loc = replay_location(r.s());
            float v = (float)r.f();
            if (loc >= 0) mtl_glUniform1f(loc, v);
            break;
        }
        case FN_glUniform1i: {
            GLint loc = replay_location(r.s());
            GLint v = (GLint)r.i();
            if (loc >= 0) mtl_glUniform1i(loc, v);
            break;
        }
        case FN_glUniform1iv:
        case FN_glUniform4fv:
        case FN_glUniformMatrix4fv: {
            std::string name = r.s();
            GLint loc = replay_location(name);
            GLsizei count = (GLsizei)r.i();
            auto data = r.blob();
            size_t each = kind == FN_glUniform1iv ? 4 : kind == FN_glUniform4fv ? 16 : 64;
            if (loc < 0 || !data.first || data.second < each * (size_t)count) break;
            if (rp.camera_moved && count == 1 && kind != FN_glUniform1iv && replay_scene_target()) {
                bool clip = name == "matWorldViewProj" || name == "matViewProj", view = name == "matWorldView";
                if (kind == FN_glUniformMatrix4fv && (clip || view)) {
                    float m[16];
                    memcpy(m, data.first, sizeof m);
                    move_matrix(m, clip);
                    mtl_glUniformMatrix4fv(loc, 1, GL_FALSE, m);
                    break;
                }
                if (kind == FN_glUniform4fv && name == "vCameraPos") {
                    float v[4];
                    memcpy(v, data.first, sizeof v);
                    memcpy(v, rp.eye, 12);
                    mtl_glUniform4fv(loc, 1, v);
                    break;
                }
            }
            if (kind == FN_glUniform1iv) mtl_glUniform1iv(loc, count, (const GLint *)data.first);
            else if (kind == FN_glUniform4fv) mtl_glUniform4fv(loc, count, (const GLfloat *)data.first);
            else mtl_glUniformMatrix4fv(loc, count, GL_FALSE, (const GLfloat *)data.first);
            break;
        }
        case FN_glDrawArrays: {
            GLenum mode = (GLenum)r.i();
            GLint first = (GLint)r.i();
            mtl_glDrawArrays(mode, first, (GLsizei)r.i());
            break;
        }
        case FN_glDrawElements: {
            GLenum mode = (GLenum)r.i();
            GLsizei count = (GLsizei)r.i();
            GLenum type = (GLenum)r.i();
            mtl_glDrawElements(mode, count, type, (const GLvoid *)(uintptr_t)r.i());
            break;
        }
        default: break;
        }
    }
}
