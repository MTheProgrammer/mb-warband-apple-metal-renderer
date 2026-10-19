// The per-frame shadow map: what casts, and the casters' and torches' shaders.
// Included by metal_backend.mm, inside namespace wbm.

// ---- the per-frame shadow map -------------------------------------------------------------
//
// Before its main pass the engine draws every visible object once into a depth
// buffer (when "depth effects" are on). Those draws are recorded here, and then
// drawn again from the sun's direction into a shadow map of the shim's own, so
// that characters, horses and their gear cast shadows on everything, at any
// distance, where the engine gives them only a decal on the ground.
//
// The map lives in the engine's own shadow space, the one its matSunViewProj
// maps the world into, restricted to a window around the camera. A shader's
// existing shadow coordinate therefore indexes it directly (see k_xl_dyn_sampler).
//
// To move a recorded draw there, its world-view-projection has to be turned
// back into a world transform, which needs the camera's view-projection. The
// pre-pass shaders are never given that, so the map is drawn at the first draw
// of the main pass whose program has both a world-view-projection and a world
// matrix (the sky dome, normally the very first): view-projection is then
// WVP x inverse(W), exact for this frame. It is drawn in a command buffer of
// its own, committed at once, since a render pass is open on the main one.

struct CasterDraw {
    PipelineKey key;          // vertex layout; the other fields are zero
    id<MTLBuffer> vertex[MAX_VERTEX_SLOTS];
    NSUInteger vertex_offset[MAX_VERTEX_SLOTS];
    int slots;
    id<MTLBuffer> indices;
    NSUInteger index_offset, index_count;
    MTLIndexType index_type;
    float wvp[16];            // object to clip space; object to world space when `world` is set
    bool world;
    float cutoff;             // alpha below which a cut-out texel casts nothing
    int bones;                // index into s_caster_bones of 32 matrices, or -1
    id<MTLTexture> alpha;     // set for alpha-tested geometry
    id<MTLSamplerState> alpha_sampler;
    // A layer of long hair (see record_hair_casters): the mesh pushed out by
    // `push` metres, hanging as the drawn hair does at `share` of its length,
    // casting only where `alpha`, here the hair map, says long hair.
    uint64_t mesh = 0; // see mesh_id
    bool scenery = false; // in the engine's own shadow map already: not for the sun's
    bool hair = false;
    int strand = 0; // 1: a horse's mane and tail as strands (push: the mane's length, share: the tail's); 2: a person's (push: the longest strand)
    float drift[3] = {0, 0, 0};
    float push = 0, share = 0, down[3] = {0, 0, 0};
    std::shared_ptr<std::vector<float>> own_bones; // 32 matrices, kept with the record: it outlives the frame's own
};
std::vector<CasterDraw> s_casters;
std::vector<float> s_caster_bones;
// For motion blur: this frame's objects with their bones, and where last frame's were.
struct VelocityDraw {
    CasterDraw c;
    int bones; // index into s_velocity_bones of 32 matrices, or -1
    float origin[4]; // where it is, in clip space: the engine places things with bones by their bones, under one matrix for all
};
struct PastObject {
    uint64_t mesh;
    float wvp[16];
    int bones; // index into s_past_bones, or -1
    bool taken;
    float origin[4];
};
std::vector<VelocityDraw> s_velocity_draws;
std::vector<PastObject> s_past_objects;
std::vector<float> s_velocity_bones, s_past_bones;
uint64_t s_past_frame, s_velocity_objects;
id<MTLTexture> s_velocity_texture, s_velocity_depth;
id<MTLDepthStencilState> s_velocity_depth_state;

// Grass never reaches the pre-pass: the engine draws it in the main pass only.
// Cut-out geometry that turns up there without having been in the pre-pass is
// recorded too, with its world matrix, which the main pass does supply, and
// goes into the next frame's map. Grass stands still, so a frame's delay does
// not show.
std::vector<CasterDraw> s_late_casters, s_late_ready; // from this frame, and from the last for use in this one
std::unordered_set<uint64_t> s_prepass_meshes;        // everything this frame's pre-pass drew
std::unordered_set<uint64_t> s_prepass_solid;         // and what of it had no texture to be cut out by

// Scenery is in the engine's own shadow map already, drawn there once per scene.
// Drawing it into this one as well would cost a draw and change the look of its
// shadow wherever the window happens to reach, so the meshes the engine drew
// into its map are remembered and passed over.
std::unordered_set<uint64_t> s_scenery;
uint64_t s_scenery_frame;

uint64_t mesh_id(NSUInteger count, NSUInteger index_offset)
{
    return ((uint64_t)s_element_buffer * 0x9E3779B97F4A7C15ull) ^ ((uint64_t)s_draw_binding.buffer[0] * 0xC2B2AE3D27D4EB4Full) ^
           ((uint64_t)index_offset * 0x165667B19E3779F9ull) ^ (uint64_t)count;
}

// Some of what the pre-pass draws is not part of the world at all: the aiming
// mark, for one. The main pass gives such things away by drawing them on top of
// everything, without a depth test, and a mesh seen drawn that way is kept out
// of the map from the next frame on. (Not writing depth is no such sign: grass
// is drawn that way, and grass should cast.)
std::unordered_set<uint64_t> s_overlays;

void record_overlay(NSUInteger count, NSUInteger index_offset)
{
    if (s_overlays.size() > 4096) s_overlays.clear();
    s_overlays.insert(mesh_id(count, index_offset));
}

void record_scenery(NSUInteger count, NSUInteger index_offset)
{
    if (s_scenery_frame + 1 < s_frame_serial) s_scenery.clear(); // the engine is drawing its map afresh
    s_scenery_frame = s_frame_serial;
    s_scenery.insert(mesh_id(count, index_offset));
}

const char k_caster_source[] = R"(
#include <metal_stdlib>
using namespace metal;

// m takes object space to the engine's shadow space: map coordinates in x and
// y, the depth its shaders compare against in z.
struct Caster {
    float4x4 m;
    float4 window[2]; // the wide window and the fine one. xy: centre, zw: 2 / (width, height), all in map coordinates
    float4 params;    // x: alpha below which a cut-out texel casts nothing; for hair, y: metres out, z: share of the hair's length
    float4 hang;      // for hair, xyz: which way is down, in the space the bones leave the vertex in
    float4 drift;     // for strands, xyz: how far their tips trail behind, as a share of their length, in the same space
};
struct SkinnedCaster {
    float4x4 m;
    float4 window[2];
    float4 params;
    float4 hang;
    float4 drift;
    float4x4 bones[32];
};
struct HairIn {
    float3 position [[attribute(0)]];
    float3 normal [[attribute(1)]];
    float2 uv [[attribute(4)]];
    float4 weights [[attribute(7)]];
    float4 indices [[attribute(8)]];
};
// Strands (see build_strands): every vertex sits at its strand's root, and its colour says which point of the strand it is.
struct StrandIn {
    float3 position [[attribute(0)]];
    float3 normal [[attribute(1)]];
    float4 colour [[attribute(2)]];
    float2 uv [[attribute(4)]];
    float4 weights [[attribute(7)]];
    float4 indices [[attribute(8)]];
};
struct PersonStrandIn {
    float3 position [[attribute(0)]];
    float3 normal [[attribute(1)]];
    float4 colour [[attribute(2)]];
    float2 uv [[attribute(4)]];
    float3 run [[attribute(6)]];
};
struct StaticIn {
    float3 position [[attribute(0)]];
    float2 uv [[attribute(4)]];
};
struct SkinnedIn {
    float3 position [[attribute(0)]];
    float2 uv [[attribute(4)]];
    float4 weights [[attribute(7)]];
    float4 indices [[attribute(8)]];
};
struct Out {
    float4 position [[position]];
    float depth;
    float2 uv;
    float cutoff [[flat]];
};

// Rows of the map are stored in OpenGL's order like every other texture the
// renderer draws, hence the flipped y. The depth buffer only has to order the
// casters; the comparison value is written as colour, unscaled.
static Out finish(float4 s, float2 uv, float4 window, float4 params)
{
    float3 q = s.xyz / s.w;
    Out out;
    out.position = float4((q.x - window.x) * window.z, -(q.y - window.y) * window.w, clamp(q.z * 0.25 + 0.5, 0.0, 1.0), 1.0);
    out.depth = q.z;
    out.uv = uv;
    out.cutoff = params.x;
    return out;
}

// Each draw is rasterised twice, once into either half of the map (vertex
// amplification): `side` says which, and only the position depends on it.
vertex Out caster_static(StaticIn in [[stage_in]], constant Caster &c [[buffer(0)]], ushort side [[amplification_id]])
{
    return finish(c.m * float4(in.position, 1.0), in.uv, c.window[side], c.params);
}

vertex Out caster_skinned(SkinnedIn in [[stage_in]], constant SkinnedCaster &c [[buffer(0)]], ushort side [[amplification_id]])
{
    float4 p = float4(in.position, 1.0);
    float4 skinned = c.bones[int(in.indices.x)] * p * in.weights.x + c.bones[int(in.indices.y)] * p * in.weights.y +
                     c.bones[int(in.indices.z)] * p * in.weights.z + c.bones[int(in.indices.w)] * p * in.weights.w;
    return finish(c.m * skinned, in.uv, c.window[side], c.params);
}

// A layer of long hair, moved as the renderer's own layers are (add_shell_push in shader_xlate.cpp).
vertex Out caster_hair(HairIn in [[stage_in]], constant SkinnedCaster &c [[buffer(0)]], ushort side [[amplification_id]])
{
    float4 p = float4(in.position, 1.0), q = float4(in.normal, 0.0);
    float4 skinned = c.bones[int(in.indices.x)] * p * in.weights.x + c.bones[int(in.indices.y)] * p * in.weights.y +
                     c.bones[int(in.indices.z)] * p * in.weights.z + c.bones[int(in.indices.w)] * p * in.weights.w;
    float3 n = normalize((c.bones[int(in.indices.x)] * q * in.weights.x + c.bones[int(in.indices.y)] * q * in.weights.y +
                          c.bones[int(in.indices.z)] * q * in.weights.z + c.bones[int(in.indices.w)] * q * in.weights.w).xyz);
    skinned.xyz += c.params.y * (n * (1.0 - 0.55 * c.params.z) + c.hang.xyz * (2.2 * c.params.z));
    return finish(c.m * skinned, in.uv, c.window[side], c.params);
}

// A strand is a ribbon turned to face the sun, wide enough for the map to hold.
static float3 strand_point(float3 out_, float3 down, float len, float k, float t, float side, float half_width, float3 drift, float4x4 m)
{
    float3 dir = out_ * (1.0 - k * t) + down * (k * t);
    float3 p = len * (out_ * (t - 0.5 * k * t * t) + down * (0.5 * k * t * t)) + drift * (len * t * t);
    p *= min(1.0, len * t / max(length(p), 1e-6)); // however it is blown about, a strand is no longer than it is
    float3 sunward = float3(m[0][2], m[1][2], m[2][2]);
    float3 across = cross(dir, sunward);
    across = dot(across, across) > 1e-12 ? normalize(across) : float3(1.0, 0.0, 0.0);
    return p + across * (side * half_width);
}
// Hair that streams behind does not hang as well: the faster, the less it droops.
static float streaming(float k, float3 drift) { return k * (1.0 - 0.9 * min(length(drift), 1.0)); }

// The mane and tail of a horse, laid out as add_shell_push in shader_xlate.cpp lays them out. params.y: the mane's length, z: the tail's.
vertex Out caster_strand(StrandIn in [[stage_in]], constant SkinnedCaster &c [[buffer(0)]], ushort side_ [[amplification_id]])
{
    float4 p = float4(in.position, 1.0);
    float4 skinned = c.bones[int(in.indices.x)] * p * in.weights.x + c.bones[int(in.indices.y)] * p * in.weights.y +
                     c.bones[int(in.indices.z)] * p * in.weights.z + c.bones[int(in.indices.w)] * p * in.weights.w;
    float4x4 b = c.bones[int(in.indices.x)] * in.weights.x + c.bones[int(in.indices.y)] * in.weights.y + c.bones[int(in.indices.z)] * in.weights.z + c.bones[int(in.indices.w)] * in.weights.w;
    float3x3 turn = float3x3(b[0].xyz, b[1].xyz, b[2].xyz);
    float v = floor(in.colour.x * 255.0 + 0.5), r = floor(in.colour.y * 255.0 + 0.5);
    float far = floor(v / 32.0) / 7.0;
    v -= floor(v / 32.0) * 32.0;
    float side = fmod(v, 2.0) * 2.0 - 1.0, t = floor(v / 2.0) / 8.0;
    float tail = step(128.0, r);
    r -= 128.0 * tail;
    float fall = step(64.0, r) * 2.0 - 1.0, chance = (r - 64.0 * step(64.0, r)) / 63.0;
    float3 n = normalize(turn * in.normal), down = c.hang.xyz;
    float3 out_ = tail > 0.5 ? normalize(normalize(turn * float3(0.0, -0.12, -1.0)) + n * 0.12)
                             : normalize(normalize(turn * (far > 0.5 ? float3(fall * 0.25, 1.0, 0.0) : float3(fall, (fract(chance * 5.7) - 0.5) * 0.9, 0.0))) * (0.3 + 0.35 * fract(chance * 3.1)) + n * 0.2 + down * 0.35);
    float len = (tail > 0.5 ? c.params.z * (1.0 - 0.6 * far) : c.params.y * (1.0 - 0.45 * far)) * (tail > 0.5 ? 0.7 + 0.3 * chance : 0.45 + 0.55 * chance);
    skinned.xyz += strand_point(out_, down, len, streaming(tail > 0.5 ? 0.85 : 1.0, c.drift.xyz), t, side, 0.006, c.drift.xyz, c.m);
    return finish(c.m * skinned, in.uv + float2(chance, far), c.window[side_], c.params);
}

// A person's, as add_hair_strands lays them out. params.y: the longest strand's length.
vertex Out caster_strand_person(PersonStrandIn in [[stage_in]], constant Caster &c [[buffer(0)]], ushort side_ [[amplification_id]])
{
    float v = floor(in.colour.x * 255.0 + 0.5), chance = in.colour.y;
    float side = fmod(v, 2.0) * 2.0 - 1.0, t = floor(v / 2.0) / 8.0;
    float3 n = normalize(in.normal), down = c.hang.xyz;
    float3 run = dot(in.run, in.run) > 0.0004 ? normalize(in.run) : down;
    float3 out_ = normalize(run + n * (fract(chance * 3.1) > 0.95 ? 0.45 : 0.02 + 0.14 * fract(chance * 3.1)));
    float len = c.params.y * in.colour.z * (0.4 + 0.6 * chance);
    float4 p = float4(in.position + strand_point(out_, down, len, streaming(0.4, c.drift.xyz), t, side, 0.003, c.drift.xyz, c.m), 1.0);
    return finish(c.m * p, in.uv + float2(chance, 0.0), c.window[side_], c.params);
}

// Hair lets about half the light past it: every other strand casts.
fragment float4 caster_strand_only(Out in [[stage_in]])
{
    if (fract(sin(dot(floor(in.uv * 4096.0), float2(12.9898, 78.233))) * 43758.5453) > 0.5) discard_fragment();
    return float4(in.depth);
}

fragment float4 caster_hair_only(Out in [[stage_in]], texture2d<float> hair [[texture(0)]], sampler s [[sampler(0)]])
{
    if (hair.sample(s, in.uv).b < 0.8) discard_fragment(); // long hair is 1 in a hair map's blue
    // Hair lets about half the light past it: every other strand casts, which
    // the receiving filter turns into half a shadow. Solid, a mane would put
    // itself and the neck under it in the dark.
    float strand = fract(sin(floor(in.uv.x * 700.0) * 12.9898 + floor(in.uv.y * 700.0) * 78.233) * 43758.5453);
    if (strand > 0.45) discard_fragment();
    return float4(in.depth);
}

fragment float4 caster_opaque(Out in [[stage_in]])
{
    return float4(in.depth);
}

fragment float4 caster_cutout(Out in [[stage_in]], texture2d<float> diffuse [[texture(0)]], sampler s [[sampler(0)]])
{
    if (diffuse.sample(s, in.uv).a < in.cutoff) discard_fragment();
    return float4(in.depth);
}

// The same casters, seen from the point lights (see render_torch_shadows). Here m takes object space to the
// world. Every draw is made six times per light, once for each face of the light's cube: the instance says
// which, and the layer of the map to draw into follows from it.
struct Torches {
    float4 light[4]; // xyz: where the light of each cube is
    float4 depth;    // a face holds x + y / (distance along its axis); z: scenery this near the light is the light's own holder
};
struct TorchOut {
    float4 position [[position]];
    float2 uv;
    float cutoff [[flat]];
    float clip [[clip_distance]];
    uint layer [[render_target_array_index]];
};

static TorchOut torch_finish(float4 world, float2 uv, float4 params, constant Torches &t, uint instance, bool scenery)
{
    float3 r = world.xyz / world.w - t.light[instance / 6].xyz;
    float3 c; // across the face, down it, and along its axis, as a cube map's faces are laid out
    switch (instance % 6) {
    case 0: c = float3(-r.z, -r.y, r.x); break;
    case 1: c = float3(r.z, -r.y, -r.x); break;
    case 2: c = float3(r.x, r.z, r.y); break;
    case 3: c = float3(r.x, -r.z, -r.y); break;
    case 4: c = float3(r.x, -r.y, r.z); break;
    default: c = float3(-r.x, -r.y, -r.z); break;
    }
    TorchOut out;
    out.position = float4(c.x, -c.y, t.depth.x * c.z + t.depth.y, c.z);
    // A torch's light sits in its own bracket, and a point there throws the bracket's shadow across the whole
    // wall, where a flame, which is no point and burns above it, throws next to none. Scenery is cut away
    // where its corners are this near the light: that takes the fine things at the light, and leaves the wall
    // behind it, whose corners are far off, whole.
    out.clip = scenery ? length(r) - t.depth.z : 1.0;
    out.uv = uv;
    out.cutoff = params.x;
    out.layer = instance;
    return out;
}

vertex TorchOut torch_static(StaticIn in [[stage_in]], constant Caster &c [[buffer(0)]], constant Torches &t [[buffer(1)]], uint instance [[instance_id]])
{
    return torch_finish(c.m * float4(in.position, 1.0), in.uv, c.params, t, instance, true);
}

vertex TorchOut torch_skinned(SkinnedIn in [[stage_in]], constant SkinnedCaster &c [[buffer(0)]], constant Torches &t [[buffer(1)]], uint instance [[instance_id]])
{
    float4 p = float4(in.position, 1.0);
    float4 skinned = c.bones[int(in.indices.x)] * p * in.weights.x + c.bones[int(in.indices.y)] * p * in.weights.y +
                     c.bones[int(in.indices.z)] * p * in.weights.z + c.bones[int(in.indices.w)] * p * in.weights.w;
    return torch_finish(c.m * skinned, in.uv, c.params, t, instance, false);
}

fragment void torch_cutout(TorchOut in [[stage_in]], texture2d<float> diffuse [[texture(0)]], sampler s [[sampler(0)]])
{
    if (diffuse.sample(s, in.uv).a < in.cutoff) discard_fragment();
}
)";

id<MTLLibrary> s_caster_library;
id<MTLDepthStencilState> s_caster_depth_state;
NSUInteger s_caster_halves = 1; // 2 when the fine window is drawn too
std::unordered_map<PipelineKey, id<MTLRenderPipelineState>, PipelineKeyHash> s_velocity_pipelines[2]; // [skinned]
std::unordered_map<PipelineKey, id<MTLRenderPipelineState>, PipelineKeyHash> s_caster_pipelines[7]; // [skinned * 2 + cutout], 4 for hair in layers, 5 and 6 for strands

// Called for each draw of the engine's depth pre-pass, after it has been issued.
void record_caster(NSUInteger count, MTLIndexType index_type, id<MTLBuffer> indices, NSUInteger index_offset, bool late)
{
    Program *p = find(s_programs, s_program);
    if (!p || !p->u_wvp || s_casters.size() >= 20000) return;
    uint64_t mesh = mesh_id(count, index_offset);
    if (late) {
        if (!p->u_world || p->u_bones || p->alpha_texture_location < 0 || p->is_blob_shadow || s_late_casters.size() >= 4000) return;
        if (s_prepass_meshes.count(mesh)) return;
    } else {
        s_prepass_meshes.insert(mesh);
    }
    // Cut-out scenery (bushes, grass, leaves) is drawn here all the same: the
    // engine's map is too coarse for twigs, whose shadows all but vanish in it.
    bool scenery = !late && p->alpha_texture_location < 0 && s_scenery.count(mesh);
    if (scenery) {
        s_dyn_stats.scenery++;
        if (!(s_torch_shadows > 0)) return; // a torch's map has no other to lean on: it takes the scenery too
    }
    if (!late && s_overlays.count(mesh)) {
        s_dyn_stats.overlays++;
        return;
    }
    CasterDraw c{};
    c.scenery = scenery;
    memcpy(c.key.format, s_last_key.format, sizeof c.key.format);
    memcpy(c.key.slot, s_last_key.slot, sizeof c.key.slot);
    memcpy(c.key.offset, s_last_key.offset, sizeof c.key.offset);
    memcpy(c.key.stride, s_last_key.stride, sizeof c.key.stride);
    if (!c.key.format[0]) return; // no position
    c.slots = s_draw_binding.slots;
    for (int i = 0; i < c.slots; i++) {
        Buffer *b = find(s_buffers, s_draw_binding.buffer[i]);
        if (!b || !b->mtl) return;
        c.vertex[i] = b->mtl;
        c.vertex_offset[i] = s_draw_binding.base[i];
    }
    c.indices = indices;
    c.index_offset = index_offset;
    c.index_count = count;
    c.index_type = index_type;
    memcpy(c.wvp, p->uniform_data[0].data() + (late ? p->u_world : p->u_wvp)->offset, sizeof c.wvp);
    c.world = late;
    c.cutoff = 0.03137255f; // the engine's own threshold in its pre-pass
    if (late) {
        c.cutoff = 0.5f;
        auto it = p->location_by_name.find("alpha_test_val");
        const XlUniform *u = it != p->location_by_name.end() ? p->locations[(size_t)it->second].uniform[1] : nullptr;
        if (u && u->components == 1 && !u->is_int) memcpy(&c.cutoff, p->uniform_data[1].data() + u->offset, 4);
        // Blended cut-outs test alpha against next to nothing and leave the soft
        // edge to the blend. A shadow has no such edge; half is where the shape ends.
        if (s_blend) c.cutoff = std::max(c.cutoff, 0.5f);
    }
    c.bones = -1;
    if (!late && p->u_bones && c.key.format[7] && c.key.format[8]) {
        c.bones = (int)(s_caster_bones.size() / (32 * 16));
        size_t at = s_caster_bones.size();
        s_caster_bones.resize(at + 32 * 16);
        size_t bytes = std::min<size_t>(32 * 64, p->uniform_data[0].size() - (size_t)p->u_bones->offset);
        memcpy(&s_caster_bones[at], p->uniform_data[0].data() + p->u_bones->offset, bytes);
    }
    if (p->alpha_texture_location >= 0) {
        UniformLocation &l = p->locations[(size_t)p->alpha_texture_location];
        Texture *t = (l.unit >= 0 && l.unit < MAX_UNITS) ? find(s_textures, s_bound_texture[l.unit]) : nullptr;
        if (t && t->mtl) {
            c.alpha = t->mtl;
            c.alpha_sampler = sampler_for(*t);
        }
    }
    if (!late && !c.alpha) s_prepass_solid.insert(mesh);
    c.mesh = mesh;
    if (!late && c.alpha && c.bones < 0 && s_dyn_foliage_range > 0 && c.wvp[15] > s_dyn_foliage_range) { // w of its origin: how far ahead it is
        c.scenery = true; // not for the sun's map; a torch's still takes it
        s_dyn_stats.far_foliage++;
    }
    if (!late && (s_motion_blur || s_temporal) && s_velocity_draws.size() < 1500) { // kept apart: the shadow map takes its own away when it is drawn
        VelocityDraw v{c, -1, {c.wvp[12], c.wvp[13], c.wvp[14], c.wvp[15]}};
        if (c.bones >= 0) {
            v.bones = (int)(s_velocity_bones.size() / (32 * 16));
            const float *from = &s_caster_bones[(size_t)c.bones * 32 * 16];
            s_velocity_bones.insert(s_velocity_bones.end(), from, from + 32 * 16);
            for (int r = 0; r < 4; r++) v.origin[r] = c.wvp[r] * from[12] + c.wvp[4 + r] * from[13] + c.wvp[8 + r] * from[14] + c.wvp[12 + r];
        }
        s_velocity_draws.push_back(v);
    }
    if (late || casting()) (late ? s_late_casters : s_casters).push_back(c);
    else if (c.bones >= 0) s_caster_bones.resize((size_t)c.bones * 32 * 16); // not wanted for shadows: give the matrices back
}

// Motion blur of things that move by themselves. Each object of the depth
// pre-pass is found again among last frame's (the same mesh, nearest to where
// the camera's own movement would have put it), and those that have moved
// are drawn into a half-size texture with both poses: see the shaders in
// k_blur_source. `reproject` takes this frame's clip space to the last one's.
bool objects_recorded() { return !s_velocity_draws.empty() && s_past_frame == s_frame_serial && !s_past_objects.empty(); }
