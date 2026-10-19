// Strands of hair: horses' manes and tails, and people's loose hair.
// Included by metal_backend.mm, inside its anonymous namespace.

// Strands. Where a coat's hair map marks long hair, roots are scattered over
// the mesh's triangles, as many as their area asks for, and for each root a
// ribbon of k_strand_segments segments is made whose vertices are all copies of
// the mesh's vertex nearest the root, moved to the root. So the ribbon is
// drawn with the horse's own program, bones and lighting; what tells its
// vertices apart is their colour, from which the vertex shader lays the strand
// out (see add_shell_push in shader_xlate.cpp). The model itself is untouched.
const int k_strand_segments = 8; // the vertex shader divides by this

struct StrandMesh {
    GLuint buffer = 0; // in s_buffers, laid out like the mesh's own vertices
    id<MTLBuffer> indices;
    NSUInteger count = 0;
};
std::unordered_map<uint64_t, StrandMesh> s_strand_meshes;
GLuint s_strand_names = 0x7f000000; // buffer names of the shim's own, out of the game's way

// vertices, elements: the mesh as the game gave it. Attributes 0 (position),
// 2 (colour), 4 (texture coordinate), 7 and 8 (bone weights and numbers) must share the buffer.
// Without a map it is a person's hair (see add_hair_strands): a rigid mesh,
// all of it hair, with roots everywhere; the texture's clear parts are left to the shader.
const StrandMesh &build_strands(uint64_t key, const uint8_t *vertices, size_t vertex_bytes, const uint8_t *elements, size_t count, size_t width, id<MTLTexture> map)
{
    const bool person = map == nil;
    StrandMesh &mesh = s_strand_meshes[key];
    if (mesh.buffer) return mesh;
    mesh.buffer = s_strand_names++;
    const Attrib &pa = s_attribs[0], &ca = s_attribs[2], &ta = s_attribs[4], &wa = s_attribs[7], &ba = s_attribs[8];
    size_t stride = (size_t)pa.stride;
    bool fits = stride > 0 && pa.type == GL_FLOAT && ta.type == GL_FLOAT && ca.type == GL_UNSIGNED_BYTE && ca.size == 4 && ta.buffer == pa.buffer && ca.buffer == pa.buffer &&
                (person || (wa.buffer == pa.buffer && ba.buffer == pa.buffer && wa.enabled && ba.enabled)) && (size_t)ta.stride == stride && (size_t)ca.stride == stride &&
                (person || map.pixelFormat == MTLPixelFormatRGBA8Unorm);
    if (!fits) return mesh;
    size_t n = vertex_bytes / stride, mw = person ? 1 : map.width, mh = person ? 1 : map.height;
    std::vector<uint8_t> px(mw * mh * 4, 255); // a person's: long hair everywhere
    if (!person) [map getBytes:px.data() bytesPerRow:mw * 4 fromRegion:MTLRegionMake2D(0, 0, mw, mh) mipmapLevel:0];
    auto at = [&](float u, float v) {
        size_t x = std::min(mw - 1, (size_t)((u - floorf(u)) * (float)mw)), y = std::min(mh - 1, (size_t)((v - floorf(v)) * (float)mh));
        return &px[(y * mw + x) * 4];
    };
    auto component = [](const uint8_t *p, GLenum type, int k) {
        float f;
        if (type == GL_FLOAT) return memcpy(&f, p + k * 4, 4), f;
        return (float)p[k];
    };
    uint32_t seed = (uint32_t)key | 1;
    auto chance = [&] { // xorshift
        seed ^= seed << 13, seed ^= seed >> 17, seed ^= seed << 5;
        return (float)(seed & 0xffffff) / 16777216.0f;
    };
    std::vector<uint8_t> out;
    std::vector<uint32_t> indices;
    const size_t most = person ? 6000 : 12000; // strands to a mesh
    size_t strands = 0;
    float density = person ? s_people_hair_density : s_strand_density;
    // The bone a vertex hangs on most. The tail is what hangs on the last two bones of a horse's skeleton.
    auto bone_of = [&](const uint8_t *from) {
        int bone = 0;
        if (person) return bone;
        float heaviest = -1;
        for (int k = 0; k < std::min<int>(wa.size, 4); k++) {
            float weight = component(from + wa.offset, wa.type, k);
            if (weight > heaviest) heaviest = weight, bone = (int)component(from + ba.offset, ba.type, k);
        }
        return bone;
    };
    // A mane grows from the crest of the neck only, which lies on the model's
    // plane of symmetry (x = 0), and the hair there is closer set than a tail's.
    const float crest = 0.035f, closer = 6.0f;
    float tail_top = -1e9f, tail_bottom = 1e9f; // the model's z over the tail, found in the first pass
    for (int pass = 0; pass < 2; pass++) { // first to see how much skin there is to cover, so that the allowance is spread evenly
    double covered = 0;
    for (size_t i = 0; i + 2 < count && strands < most; i += 3) {
        uint32_t v[3];
        for (int k = 0; k < 3; k++) v[k] = width == 2 ? *(const uint16_t *)(elements + (i + (size_t)k) * 2) : *(const uint32_t *)(elements + (i + (size_t)k) * 4);
        if (v[0] >= n || v[1] >= n || v[2] >= n) continue;
        float p[3][3], t[3][2];
        for (int k = 0; k < 3; k++) memcpy(p[k], vertices + v[k] * stride + pa.offset, 12), memcpy(t[k], vertices + v[k] * stride + ta.offset, 8);
        // Any long hair on it? Its corners and middle are asked.
        bool any = at((t[0][0] + t[1][0] + t[2][0]) / 3, (t[0][1] + t[1][1] + t[2][1]) / 3)[2] > 204;
        for (int k = 0; k < 3; k++) any = any || at(t[k][0], t[k][1])[2] > 204;
        if (!any) continue;
        float e1[3] = {p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]}, e2[3] = {p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]};
        float cx = e1[1] * e2[2] - e1[2] * e2[1], cy = e1[2] * e2[0] - e1[0] * e2[2], cz = e1[0] * e2[1] - e1[1] * e2[0];
        // A person's hair runs the way its texture is painted: down the picture, which is towards v = 0.
        // (The band along the picture's top is a braid and a ribbon, lying sideways: nothing is rooted there.)
        float flow[3] = {0, 0, 0}, reach = 1; // reach: how long strands are here, as a share of the longest
        if (person) {
            float du1 = t[1][0] - t[0][0], dv1 = t[1][1] - t[0][1], du2 = t[2][0] - t[0][0], dv2 = t[2][1] - t[0][1], det = du1 * dv2 - du2 * dv1;
            if (fabsf(det) < 1e-9f) continue;
            if (std::max({t[0][1], t[1][1], t[2][1]}) > 0.93f) continue;
            for (int k = 0; k < 3; k++) flow[k] = -(e2[k] * du1 - e1[k] * du2) / det;
            float len = sqrtf(flow[0] * flow[0] + flow[1] * flow[1] + flow[2] * flow[2]);
            if (len < 1e-9f) continue;
            for (float &f : flow) f /= len;
            // len is how many metres the whole texture's length of hair is drawn over here: little for a
            // beard or a cropped head, whose painted hairs are short, and loose strands are short with them.
            reach = std::min(1.0f, std::max(0.12f, len * 0.12f / s_people_hair_length));
        }
        bool tail_here = bone_of(vertices + v[0] * stride) >= 26;
        if (!person && !tail_here && std::min({fabsf(p[0][0]), fabsf(p[1][0]), fabsf(p[2][0])}) > crest) continue;
        float area = 0.5f * sqrtf(cx * cx + cy * cy + cz * cz) * (tail_here || person ? 1.0f : closer);
        float wanted = area * density;
        covered += area;
        if (pass == 0) {
            if (tail_here)
                for (int k = 0; k < 3; k++) tail_top = std::max(tail_top, p[k][2]), tail_bottom = std::min(tail_bottom, p[k][2]);
            continue;
        }
        int roots = (int)wanted + (chance() < wanted - floorf(wanted) ? 1 : 0);
        for (int r = 0; r < roots && strands < most; r++) {
            float a = chance(), b = chance();
            if (a + b > 1) a = 1 - a, b = 1 - b;
            float w[3] = {1 - a - b, a, b};
            float root[3], uv[2];
            for (int k = 0; k < 3; k++) root[k] = w[0] * p[0][k] + w[1] * p[1][k] + w[2] * p[2][k];
            for (int k = 0; k < 2; k++) uv[k] = w[0] * t[0][k] + w[1] * t[1][k] + w[2] * t[2][k];
            const uint8_t *here = at(uv[0], uv[1]);
            float luck = chance(), luck2 = chance(); // drawn whether or not the root is kept, so that a map's edit moves no other strand
            if (here[2] <= 204) continue;
            int nearest = w[1] > w[0] ? (w[2] > w[1] ? 2 : 1) : (w[2] > w[0] ? 2 : 0);
            const uint8_t *from = vertices + v[nearest] * stride;
            int bone = bone_of(from);
            bool tail = bone >= 26;
            if (!person && !tail && fabsf(root[0]) > crest) continue;
            bool to_plus = fabsf(root[0]) > 0.004f ? root[0] > 0 : luck2 > 0.5f; // the side the mane falls to
            int far = tail && tail_top > tail_bottom ? std::min(7, std::max(0, (int)lroundf(7.0f * (tail_top - root[2]) / (tail_top - tail_bottom)))) : !tail && bone == 7 ? 7 : 0; // bone 7 is the head: the forelock
            uint32_t first = (uint32_t)(out.size() / stride);
            for (int segment = 0; segment <= k_strand_segments; segment++) {
                for (int side = 0; side < 2; side++) {
                    size_t o = out.size();
                    out.insert(out.end(), from, from + stride);
                    memcpy(&out[o + pa.offset], root, 12), memcpy(&out[o + ta.offset], uv, 8);
                    // Normal, tangent and binormal are blended between the corners as well, or the lighting would go in patches.
                    for (int attrib : {1, 5, 6}) {
                        const Attrib &na = s_attribs[attrib];
                        if (!na.enabled || na.buffer != pa.buffer || na.type != GL_FLOAT || na.size < 3) continue;
                        float blended[3] = {0, 0, 0}, corner[3];
                        for (int c = 0; c < 3; c++) {
                            memcpy(corner, vertices + v[c] * stride + na.offset, 12);
                            for (int k = 0; k < 3; k++) blended[k] += w[c] * corner[k];
                        }
                        float len = sqrtf(blended[0] * blended[0] + blended[1] * blended[1] + blended[2] * blended[2]);
                        if (len > 1e-6f) {
                            for (float &f : blended) f /= len;
                            memcpy(&out[o + na.offset], blended, 12);
                        }
                    }
                    uint8_t *colour = &out[o + ca.offset];
                    colour[0] = (uint8_t)(segment * 2 + side + far * 32), colour[1] = (uint8_t)((int)(luck * 63.0f) + (!tail && to_plus ? 64 : 0) + (tail ? 128 : 0)), colour[2] = here[0], colour[3] = here[1];
                    if (person) {
                        colour[0] = (uint8_t)(segment * 2 + side), colour[1] = (uint8_t)(luck * 255.0f), colour[2] = (uint8_t)(reach * 255.0f);
                        const Attrib &fa = s_attribs[6]; // the binormal, which this shader takes for the hair's run
                        if (fa.enabled && fa.buffer == pa.buffer && fa.type == GL_FLOAT && fa.size >= 3) memcpy(&out[o + fa.offset], flow, 12);
                    }
                }
                if (segment < k_strand_segments) {
                    uint32_t q = first + (uint32_t)segment * 2;
                    for (uint32_t k : {q, q + 1, q + 2, q + 1, q + 3, q + 2}) indices.push_back(k);
                }
            }
            strands++;
        }
    }
    if (pass == 0 && covered * density > (double)most * 0.9) density = (float)((double)most * 0.9 / covered);
    }
    if (indices.empty()) return mesh;
    s_buffers[mesh.buffer].mtl = [s_device newBufferWithBytes:out.data() length:out.size() options:MTLResourceStorageModeShared];
    mesh.indices = [s_device newBufferWithBytes:indices.data() length:indices.size() * 4 options:MTLResourceStorageModeShared];
    mesh.count = indices.size();
    if (getenv("WBM_DEBUG_STRANDS")) fprintf(stderr, "strands: %zu on a mesh of %zu triangles\n", strands, count / 3);
    return mesh;
}

// Long hair casts shadows too. Its layers exist only as draws of the main
// pass, so that is where they are taken from: a few of them per coat, to be
// drawn into the next frame's map like the grass (see s_late_casters).
void record_hair_casters(Program &p, Texture &coat, NSUInteger count, MTLIndexType index_type, id<MTLBuffer> indices, NSUInteger index_offset)
{
    if (!p.u_world || !p.u_bones || !coat.hair || s_late_casters.size() >= 4000) return;
    CasterDraw c{};
    memcpy(c.key.format, s_last_key.format, sizeof c.key.format);
    memcpy(c.key.slot, s_last_key.slot, sizeof c.key.slot);
    memcpy(c.key.offset, s_last_key.offset, sizeof c.key.offset);
    memcpy(c.key.stride, s_last_key.stride, sizeof c.key.stride);
    if (!c.key.format[0] || !c.key.format[1] || !c.key.format[7] || !c.key.format[8]) return;
    c.slots = s_draw_binding.slots;
    for (int i = 0; i < c.slots; i++) {
        Buffer *b = find(s_buffers, s_draw_binding.buffer[i]);
        if (!b || !b->mtl) return;
        c.vertex[i] = b->mtl;
        c.vertex_offset[i] = s_draw_binding.base[i];
    }
    c.indices = indices, c.index_offset = index_offset, c.index_count = count, c.index_type = index_type;
    memcpy(c.wvp, p.uniform_data[0].data() + p.u_world->offset, sizeof c.wvp);
    c.world = true;
    c.bones = -1;
    c.own_bones = std::make_shared<std::vector<float>>(32 * 16, 0.0f);
    memcpy(c.own_bones->data(), p.uniform_data[0].data() + p.u_bones->offset, std::min<size_t>(32 * 64, p.uniform_data[0].size() - (size_t)p.u_bones->offset));
    c.alpha = coat.hair;
    c.alpha_sampler = sampler_for(coat);
    c.hair = true;
    // Down, in the space the world matrix starts from: the world's, turned back by the matrix's rotation.
    float len = sqrtf(c.wvp[2] * c.wvp[2] + c.wvp[6] * c.wvp[6] + c.wvp[10] * c.wvp[10]);
    if (len > 1e-6f) c.down[0] = -c.wvp[2] / len, c.down[1] = -c.wvp[6] / len, c.down[2] = -c.wvp[10] / len;
    for (int layer = 1; layer <= 2; layer++) {
        c.share = (float)layer / 2.0f;
        c.push = s_hair_length * c.share;
        s_late_casters.push_back(c);
    }
}

// Strands cast shadows as strands: the ribbons just drawn are drawn again
// into the next frame's map (see s_late_casters), laid out by the map's own
// shaders. Called while the strands' buffer is still bound and prepared.
// kind: 1 a horse's, 2 a person's. drift: as the strands were given it, in the world's space.
void record_strand_caster(Program &p, const StrandMesh &strands, int kind, float first, float second, const float drift[3])
{
    if (!p.u_world || s_late_casters.size() >= 4000 || (kind == 1 && !p.u_bones)) return;
    CasterDraw c{};
    memcpy(c.key.format, s_last_key.format, sizeof c.key.format);
    memcpy(c.key.slot, s_last_key.slot, sizeof c.key.slot);
    memcpy(c.key.offset, s_last_key.offset, sizeof c.key.offset);
    memcpy(c.key.stride, s_last_key.stride, sizeof c.key.stride);
    const int horse[6] = {0, 1, 2, 4, 7, 8}, person[5] = {0, 1, 2, 4, 6};
    for (int k = 0; k < (kind == 1 ? 6 : 5); k++)
        if (!c.key.format[(kind == 1 ? horse : person)[k]]) return;
    c.slots = s_draw_binding.slots;
    for (int i = 0; i < c.slots; i++) {
        Buffer *b = find(s_buffers, s_draw_binding.buffer[i]);
        if (!b || !b->mtl) return;
        c.vertex[i] = b->mtl;
        c.vertex_offset[i] = s_draw_binding.base[i];
    }
    c.indices = strands.indices, c.index_offset = 0, c.index_count = strands.count, c.index_type = MTLIndexTypeUInt32;
    memcpy(c.wvp, p.uniform_data[0].data() + p.u_world->offset, sizeof c.wvp);
    c.world = true;
    c.bones = -1;
    if (kind == 1) {
        c.own_bones = std::make_shared<std::vector<float>>(32 * 16, 0.0f);
        memcpy(c.own_bones->data(), p.uniform_data[0].data() + p.u_bones->offset, std::min<size_t>(32 * 64, p.uniform_data[0].size() - (size_t)p.u_bones->offset));
    }
    c.strand = kind, c.push = first, c.share = second;
    // Down and the drift, in the space the world matrix starts from: the world's, turned back by the matrix's rotation.
    const float *w = c.wvp;
    float len = sqrtf(w[2] * w[2] + w[6] * w[6] + w[10] * w[10]);
    if (len > 1e-6f) c.down[0] = -w[2] / len, c.down[1] = -w[6] / len, c.down[2] = -w[10] / len;
    for (int k = 0; k < 3; k++) c.drift[k] = drift[0] * w[k * 4] + drift[1] * w[k * 4 + 1] + drift[2] * w[k * 4 + 2];
    s_late_casters.push_back(c);
}

// Hair has weight: when its owner moves off, the strands' tips are left
// behind a moment, and settle again when it stops. Each thing that has
// strands is followed from frame to frame by where its world matrix puts it
// (several may share a mesh: the nearest of last frame's is taken for the
// same one), and how far its strands trail is eased after how fast it goes.
// The result: which way and how far the tips trail, in the world's space, as a
// share of a strand's length.
struct Mover {
    float place[3], drift[3];
    double time;
    uint64_t frame;
};
std::unordered_map<uint64_t, std::vector<Mover>> s_movers;

void drift_of(uint64_t mesh, const float *world, float out[3])
{
    static const char *fixed = getenv("WBM_TEST_DRIFT"); // "x y z", to see it in a still
    if (fixed && sscanf(fixed, "%f %f %f", &out[0], &out[1], &out[2]) == 3) return;
    std::vector<Mover> &movers = s_movers[mesh];
    const float place[3] = {world[12], world[13], world[14]};
    Mover *same = nullptr;
    float nearest = 2.0f * 2.0f; // metres, squared: farther off than that it is another
    for (Mover &m : movers) {
        if (m.frame == s_frame_serial) continue; // taken by another this frame
        float d = 0;
        for (int k = 0; k < 3; k++) d += (place[k] - m.place[k]) * (place[k] - m.place[k]);
        if (d < nearest) nearest = d, same = &m;
    }
    if (!same) {
        for (size_t i = 0; i < movers.size();) // those not seen for a while make room
            if (movers[i].frame + 120 < s_frame_serial) movers[i] = movers.back(), movers.pop_back();
            else i++;
        movers.push_back(Mover{{place[0], place[1], place[2]}, {0, 0, 0}, s_wind_time, s_frame_serial});
        out[0] = out[1] = out[2] = 0;
        return;
    }
    double dt = s_wind_time - same->time;
    if (dt > 1e-4 && dt < 0.5) {
        float target[3], length = 0;
        float ahead = sqrtf(world[4] * world[4] + world[5] * world[5] + world[6] * world[6]); // the model's +y is the way it faces
        for (int k = 0; k < 3; k++) {
            float speed = (place[k] - same->place[k]) / (float)dt + (ahead > 1e-6f ? world[4 + k] / ahead * s_pretend_speed : 0.0f);
            target[k] = -speed * 0.09f, length += target[k] * target[k]; // at a gallop, 10 m/s, the tips trail by nine tenths of the length
        }
        length = sqrtf(length);
        float ease = 1.0f - expf(-(float)dt * 5.0f);
        for (int k = 0; k < 3; k++) same->drift[k] += ((length > 0.95f ? target[k] * 0.95f / length : target[k]) - same->drift[k]) * ease;
    } else if (dt >= 0.5) {
        same->drift[0] = same->drift[1] = same->drift[2] = 0;
    }
    if (dt > 1e-4 || dt < 0) memcpy(same->place, place, sizeof place), same->time = s_wind_time;
    same->frame = s_frame_serial;
    memcpy(out, same->drift, 12);
}
float s_drift_now[4]; // the shaders' wbm_drift for the draw being prepared
