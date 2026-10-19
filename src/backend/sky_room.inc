// The sky atlas, the room probe and the sky's ambient light.
// Included by metal_backend.mm, inside namespace wbm.

// ---- the sky atlas ------------------------------------------------------------------------

// Redraws the draw that was just issued, the engine's sky dome, six more times:
// once per face of the atlas, each with a 90-degree view along one axis. The
// dome's own shader, textures and vertex data are reused; only its
// world-view-projection matrix is replaced. The dome is centred on the camera,
// so its world matrix with the translation removed puts it around the origin.
//
// This runs in a command buffer of its own, committed at once: a render pass is
// open on the main one, and the atlas has to be finished before the draws that
// follow in that pass sample it.
void capture_sky(MTLPrimitiveType primitive, NSUInteger count, MTLIndexType index_type, id<MTLBuffer> indices, NSUInteger index_offset)
{
    Program *p = find(s_programs, s_program);
    double world[16];
    if (!p || !vertex_matrix(*p, "matWorld", world)) return;
    auto it = p->location_by_name.find("matWorldViewProj");
    const XlUniform *wvp = it == p->location_by_name.end() ? nullptr : p->locations[(size_t)it->second].uniform[0];
    if (!wvp) return;
    s_sky_frame = s_frame_serial;

    const int N = k_xl_sky_face_size;
    if (!s_sky_atlas) {
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:6 * N height:N mipmapped:YES];
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModePrivate;
        s_sky_atlas = [s_device newTextureWithDescriptor:td];
        MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
        sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterLinear;
        sd.mipFilter = MTLSamplerMipFilterLinear;
        s_sky_sampler = [s_device newSamplerStateWithDescriptor:sd];
    }
    PipelineKey key = s_last_key; // the dome's vertex layout, onto the atlas instead of the scene
    key.color_format = (uint16_t)MTLPixelFormatRGBA16Float;
    key.depth_format = (uint16_t)MTLPixelFormatInvalid;
    key.samples = 1;
    key.blend = 0;
    key.blend_src = key.blend_dst = key.blend_eq = 0;
    id<MTLRenderPipelineState> pipeline = pipeline_for(*p, key);
    if (!pipeline) return;

    world[12] = world[13] = world[14] = 0;
    const double near = 0.1, far = 10000;
    double projection[16] = {};
    projection[0] = projection[5] = 1; // 90 degrees, square
    projection[10] = -(far + near) / (far - near);
    projection[11] = -1;
    projection[14] = -2 * far * near / (far - near);
    static const double faces[6][2][3] = { // forward, up; see k_xl_sky_sampler
        {{1, 0, 0}, {0, 0, 1}}, {{-1, 0, 0}, {0, 0, 1}}, {{0, 1, 0}, {0, 0, 1}},
        {{0, -1, 0}, {0, 0, 1}}, {{0, 0, 1}, {0, 1, 0}}, {{0, 0, -1}, {0, 1, 0}}};

    id<MTLCommandBuffer> cmd = timed([s_queue commandBuffer]);
    for (int f = 0; f < 6; f++) {
        const double *fw = faces[f][0], *up = faces[f][1];
        double right[3] = {fw[1] * up[2] - fw[2] * up[1], fw[2] * up[0] - fw[0] * up[2], fw[0] * up[1] - fw[1] * up[0]};
        double view[16] = {}, vw[16], m[16]; // rows: right, up, -forward
        for (int c = 0; c < 3; c++) view[c * 4 + 0] = right[c], view[c * 4 + 1] = up[c], view[c * 4 + 2] = -fw[c];
        view[15] = 1;
        mat_mul(view, world, vw);
        mat_mul(projection, vw, m);

        std::vector<uint8_t> vs = p->uniform_data[0], fs = p->uniform_data[1];
        float m32[16], yflip = -1.0f, limit = 65000.0f; // drawn like any other texture target: rows in OpenGL's order
        for (int i = 0; i < 16; i++) m32[i] = (float)m[i];
        memcpy(vs.data() + wvp->offset, m32, sizeof m32);
        if (p->yflip_offset >= 0) memcpy(vs.data() + p->yflip_offset, &yflip, 4);
        if (p->jitter_offset >= 0) memset(vs.data() + p->jitter_offset, 0, 8);
        if (p->limit_offset >= 0) memcpy(fs.data() + p->limit_offset, &limit, 4);

        MTLRenderPassDescriptor *d = [MTLRenderPassDescriptor renderPassDescriptor];
        d.colorAttachments[0].texture = s_sky_atlas;
        d.colorAttachments[0].loadAction = f == 0 ? MTLLoadActionClear : MTLLoadActionLoad;
        d.colorAttachments[0].storeAction = MTLStoreActionStore;
        d.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
        id<MTLRenderCommandEncoder> pass = [cmd renderCommandEncoderWithDescriptor:d];
        [pass setViewport:(MTLViewport){(double)(f * N), 0.0, (double)N, (double)N, 0.0, 1.0}];
        [pass setScissorRect:(MTLScissorRect){(NSUInteger)(f * N), 0, (NSUInteger)N, (NSUInteger)N}];
        [pass setRenderPipelineState:pipeline];
        [pass setDepthStencilState:depth_state(false)];
        [pass setCullMode:MTLCullModeNone];
        if (p->xl.vs.uniform_buffer_index >= 0 && !vs.empty())
            [pass setVertexBuffer:[s_device newBufferWithBytes:vs.data() length:vs.size() options:MTLResourceStorageModeShared]
                           offset:0
                          atIndex:(NSUInteger)p->xl.vs.uniform_buffer_index];
        if (p->xl.fs.uniform_buffer_index >= 0 && !fs.empty())
            [pass setFragmentBuffer:[s_device newBufferWithBytes:fs.data() length:fs.size() options:MTLResourceStorageModeShared]
                             offset:0
                            atIndex:(NSUInteger)p->xl.fs.uniform_buffer_index];
        for (UniformLocation &l : p->locations) {
            for (int stage = 0; stage < 2; stage++) {
                int index = l.texture_index[stage];
                if (index < 0) continue;
                Texture *t = (l.unit >= 0 && l.unit < MAX_UNITS) ? find(s_textures, s_bound_texture[l.unit]) : nullptr;
                id<MTLTexture> texture = t && t->mtl ? t->mtl : s_white;
                id<MTLSamplerState> sampler = t && t->mtl ? sampler_for(*t) : s_samplers.begin()->second;
                if (l.srgb && t && t->mtl) {
                    if (id<MTLTexture> srgb = srgb_view_for(*t)) texture = srgb;
                }
                if (stage == 0) {
                    [pass setVertexTexture:texture atIndex:(NSUInteger)index];
                    [pass setVertexSamplerState:sampler atIndex:(NSUInteger)index];
                } else {
                    [pass setFragmentTexture:texture atIndex:(NSUInteger)index];
                    [pass setFragmentSamplerState:sampler atIndex:(NSUInteger)index];
                }
            }
        }
        bool complete = true;
        for (int slot = 0; slot < s_draw_binding.slots; slot++) {
            Buffer *b = find(s_buffers, s_draw_binding.buffer[slot]);
            if (!b || !b->mtl) {
                complete = false;
                break;
            }
            [pass setVertexBuffer:b->mtl offset:s_draw_binding.base[slot] atIndex:(NSUInteger)(VERTEX_SLOT_BASE + slot)];
        }
        [pass setVertexBuffer:s_constant_attrib offset:0 atIndex:CONSTANT_SLOT];
        if (complete)
            [pass drawIndexedPrimitives:primitive indexCount:count indexType:index_type indexBuffer:indices indexBufferOffset:index_offset];
        [pass endEncoding];
    }
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit generateMipmapsForTexture:s_sky_atlas];
    if (s_sky_ambient) { // the smallest level with a texel per face: each is that face's mean colour
        NSUInteger level = 0;
        while ((N >> level) > 1) level++;
        id<MTLBuffer> means = [s_device newBufferWithLength:6 * 8 options:MTLResourceStorageModeShared];
        [blit copyFromTexture:s_sky_atlas sourceSlice:0 sourceLevel:level sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(6, 1, 1)
                     toBuffer:means destinationOffset:0 destinationBytesPerRow:6 * 8 destinationBytesPerImage:6 * 8];
        [cmd addCompletedHandler:^(id<MTLCommandBuffer>) {
            const uint16_t *half = (const uint16_t *)means.contents;
            auto to_float = [](uint16_t h) {
                int e = (h >> 10) & 31, m = h & 1023;
                float v = e == 0 ? ldexpf((float)m, -24) : e == 31 ? 65504.0f : ldexpf((float)(m + 1024), e - 25);
                return (h & 0x8000) ? -v : v;
            };
            if (s_interior && s_room_atlas) return; // the room's colours are in use, not the sky's
            std::lock_guard<std::mutex> lock(s_amb_mu);
            for (int face = 0; face < 6; face++)
                for (int c = 0; c < 3; c++) s_amb_faces[face][c] = to_float(half[face * 4 + c]);
            s_amb_faces_valid = true;
        }];
    }
    [blit endEncoding];
    [cmd commit];
}

// The room probe. Laid out as the sky atlas is, six faces side by side, and
// read by the same shader code; but filled from the finished picture of the
// scene itself. Each texel is a direction from the camera: where that
// direction is on screen, the picture's colour there is blended in, a fifth at
// a time; where it is not, the texel keeps what it had, drifting very slowly
// towards the picture's nearest edge so that what was never looked at is not
// left black. Looking round a room fills it in. It has no depth: everything is
// taken to be far away, as with any such probe.
const char k_room_source[] = R"(
#include <metal_stdlib>
using namespace metal;
struct Params {
    float4 view[3]; // rows: the world's directions into view space, third axis forwards
    float4 lens;    // x, y: the projection's scale; z: face size in texels; w: 1 for the first filling
};
struct VertexOut {
    float4 position [[position]];
};
vertex VertexOut room_vertex(uint id [[vertex_id]])
{
    VertexOut out;
    out.position = float4(float2((id << 1) & 2, id & 2) * 2.0 - 1.0, 0.0, 1.0);
    return out;
}
fragment float4 room_fragment(VertexOut in [[stage_in]], texture2d<float> scene [[texture(0)]], constant Params &p [[buffer(0)]])
{
    constexpr sampler smooth(filter::linear, address::clamp_to_edge);
    float n = p.lens.z;
    int face = clamp(int(in.position.x / n), 0, 5);
    float2 uv = float2(in.position.x - float(face) * n, in.position.y) / n * 2.0 - 1.0;
    const float3 forwards[6] = {float3(1, 0, 0), float3(-1, 0, 0), float3(0, 1, 0), float3(0, -1, 0), float3(0, 0, 1), float3(0, 0, -1)};
    float3 forward = forwards[face], up = face < 4 ? float3(0, 0, 1) : float3(0, 1, 0);
    float3 d = normalize(forward + cross(forward, up) * uv.x + up * uv.y);
    float3 v = float3(dot(p.view[0].xyz, d), dot(p.view[1].xyz, d), dot(p.view[2].xyz, d));
    float2 ndc = v.z > 1.0e-3 ? float2(v.x * p.lens.x, v.y * p.lens.y) / v.z : normalize(v.xy + 1.0e-5) * 4.0;
    float seen = v.z > 1.0e-3 && all(abs(ndc) < float2(0.98)) ? 1.0 : 0.0;
    float3 colour = scene.sample(smooth, clamp(ndc, -0.98, 0.98) * 0.5 + 0.5).rgb;
    return float4(min(max(colour, 0.0), float3(8.0)), p.lens.w > 0.5 ? 1.0 : seen > 0.5 ? 0.2 : 0.006); // lens.w: the first filling, all at once
}
)";
struct RoomParams {
    float view[3][4];
    float lens[4];
};
id<MTLLibrary> s_room_library;
id<MTLRenderPipelineState> s_room_pipeline;

void update_room_probe()
{
    if (!s_last.color || s_camera.view_frame != s_frame_serial || s_camera.projection_frame != s_frame_serial) return;
    if (s_room_frame + (uint64_t)s_room_every > s_frame_serial && s_room_atlas) return;
    const int N = k_xl_sky_face_size;
    if (!s_room_library) {
        NSError *error = nil;
        s_room_library = [s_device newLibraryWithSource:@(k_room_source) options:nil error:&error];
        MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
        d.vertexFunction = [s_room_library newFunctionWithName:@"room_vertex"];
        d.fragmentFunction = [s_room_library newFunctionWithName:@"room_fragment"];
        MTLRenderPipelineColorAttachmentDescriptor *c = d.colorAttachments[0];
        c.pixelFormat = MTLPixelFormatRGBA16Float;
        c.blendingEnabled = YES;
        c.sourceRGBBlendFactor = MTLBlendFactorSourceAlpha, c.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        c.sourceAlphaBlendFactor = MTLBlendFactorZero, c.destinationAlphaBlendFactor = MTLBlendFactorOne;
        if (s_room_library) s_room_pipeline = [s_device newRenderPipelineStateWithDescriptor:d error:&error];
        if (!s_room_pipeline) {
            complain("room probe shaders failed: %s", error.localizedDescription.UTF8String);
            s_room_probe = false;
            return;
        }
    }
    bool fresh = !s_room_atlas;
    if (fresh) {
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:6 * N height:N mipmapped:YES];
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModePrivate;
        s_room_atlas = [s_device newTextureWithDescriptor:td];
        if (!s_sky_sampler) {
            MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
            sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterLinear;
            sd.mipFilter = MTLSamplerMipFilterLinear;
            s_sky_sampler = [s_device newSamplerStateWithDescriptor:sd];
        }
    }
    RoomParams params = {};
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) params.view[r][c] = (float)s_camera.view[r][c];
    params.lens[0] = (float)s_camera.p00, params.lens[1] = (float)s_camera.p11, params.lens[2] = (float)N, params.lens[3] = fresh ? 1.0f : 0.0f;
    MTLRenderPassDescriptor *d = [MTLRenderPassDescriptor renderPassDescriptor];
    d.colorAttachments[0].texture = s_room_atlas;
    d.colorAttachments[0].loadAction = fresh ? MTLLoadActionClear : MTLLoadActionLoad;
    d.colorAttachments[0].storeAction = MTLStoreActionStore;
    d.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0.25); // the fourth channel marks it as a room's: see wbm_environment_sky
    id<MTLRenderCommandEncoder> pass = [command_buffer() renderCommandEncoderWithDescriptor:d];
    [pass setRenderPipelineState:s_room_pipeline];
    [pass setFragmentTexture:s_last.color atIndex:0];
    [pass setFragmentBytes:&params length:sizeof params atIndex:0];
    // (A room just entered is filled at once with what can be seen of it, and its nearest edges elsewhere.)
    [pass drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [pass endEncoding];
    id<MTLBlitCommandEncoder> blit = [command_buffer() blitCommandEncoder];
    [blit generateMipmapsForTexture:s_room_atlas];
    if (s_sky_ambient) {
        NSUInteger level = 0;
        while ((N >> level) > 1) level++;
        id<MTLBuffer> means = [s_device newBufferWithLength:6 * 8 options:MTLResourceStorageModeShared];
        [blit copyFromTexture:s_room_atlas sourceSlice:0 sourceLevel:level sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(6, 1, 1)
                     toBuffer:means destinationOffset:0 destinationBytesPerRow:6 * 8 destinationBytesPerImage:6 * 8];
        [command_buffer() addCompletedHandler:^(id<MTLCommandBuffer>) {
            const uint16_t *half = (const uint16_t *)means.contents;
            auto to_float = [](uint16_t h) {
                int e = (h >> 10) & 31, m = h & 1023;
                float v = e == 0 ? ldexpf((float)m, -24) : e == 31 ? 65504.0f : ldexpf((float)(m + 1024), e - 25);
                return (h & 0x8000) ? -v : v;
            };
            std::lock_guard<std::mutex> lock(s_amb_mu);
            for (int face = 0; face < 6; face++)
                for (int c = 0; c < 3; c++) s_amb_faces[face][c] = to_float(half[face * 4 + c]);
            s_amb_faces_valid = true;
        }];
    }
    [blit endEncoding];
    s_room_frame = s_frame_serial;
    s_room_updates++;
}

// Works out the shaders' wbm_amb for this frame. Called at present.
void update_sky_ambient()
{
    float amb[24] = {};
    float faces[6][3];
    bool valid;
    {
        std::lock_guard<std::mutex> lock(s_amb_mu);
        memcpy(faces, s_amb_faces, sizeof faces);
        valid = s_amb_faces_valid;
    }
    // Only while a sky is being drawn: indoors, and in menus, there is none.
    if (valid && s_sky_frame + 2 >= s_frame_serial) {
        float linear[5][3], mean[3] = {};
        for (int face = 0; face < 5; face++) // the sixth looks down, at the dome's mirrored underside
            for (int c = 0; c < 3; c++) mean[c] += (linear[face][c] = powf(std::max(faces[face][c], 0.0f), 2.2f)) / 5;
        float brightness = 0.2126f * mean[0] + 0.7152f * mean[1] + 0.0722f * mean[2];
        if (brightness > 1e-5f) {
            for (int face = 0; face < 5; face++)
                for (int c = 0; c < 3; c++) amb[face * 4 + c] = std::min(2.5f, std::max(0.4f, linear[face][c] / brightness));
            // A room's own colours are stronger than a sky's (torchlit stone has next to no blue): taken whole they turn the shade red.
            if (s_interior)
                for (int k = 0; k < 20; k++)
                    if (k % 4 != 3) amb[k] = 1.0f + (amb[k] - 1.0f) * 0.4f;
            amb[23] = s_sky_ambient_strength;
            // vSunDir is the way the light travels; what reaches the ground is its
            // downward part. A fifth of that comes back up, a little warm.
            float down = s_sun_valid ? std::max(0.0f, -s_sun_dir[2]) : 0.0f;
            const float ground[3] = {0.22f, 0.19f, 0.14f};
            for (int c = 0; c < 3; c++) amb[20 + c] = std::max(0.0f, s_sun_color[c]) * down * ground[c] * s_bounce_strength;
            s_amb_stats_frames++;
        }
    }
    memcpy(s_amb, amb, sizeof s_amb);
}
