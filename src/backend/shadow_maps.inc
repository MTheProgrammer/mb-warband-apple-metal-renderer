// Drawing the shadow maps: the per-frame map and the torches'.
// Included by metal_backend.mm, inside its anonymous namespace.

id<MTLRenderPipelineState> caster_pipeline(const PipelineKey &key, bool skinned, bool cutout, bool hair = false, int strand = 0)
{
    auto &cache = s_caster_pipelines[strand ? 4 + strand : hair ? 4 : skinned * 2 + cutout];
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = [s_caster_library newFunctionWithName:strand == 1 ? @"caster_strand" : strand == 2 ? @"caster_strand_person" : hair ? @"caster_hair" : skinned ? @"caster_skinned" : @"caster_static"];
    d.fragmentFunction = [s_caster_library newFunctionWithName:strand ? @"caster_strand_only" : hair ? @"caster_hair_only" : cutout ? @"caster_cutout" : @"caster_opaque"];
    MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
    const int plain[5] = {0, 4, 7, 8, 1}, horse[6] = {0, 4, 7, 8, 1, 2}, person[5] = {0, 4, 1, 2, 6};
    const int *wanted = strand == 1 ? horse : strand == 2 ? person : plain;
    bool uses_constant = false;
    for (int n = 0; n < (strand == 1 ? 6 : strand == 2 ? 5 : hair ? 5 : skinned ? 4 : 2); n++) {
        int i = wanted[n];
        if (key.format[i] && key.slot[i] != CONSTANT_SLOT) {
            vd.attributes[i].format = (MTLVertexFormat)key.format[i];
            vd.attributes[i].offset = key.offset[i];
            vd.attributes[i].bufferIndex = VERTEX_SLOT_BASE + key.slot[i];
        } else { // the mesh has no such attribute: it reads the constant buffer
            vd.attributes[i].format = MTLVertexFormatFloat4;
            vd.attributes[i].offset = 0;
            vd.attributes[i].bufferIndex = CONSTANT_SLOT;
            uses_constant = true;
        }
    }
    for (int s = 0; s < MAX_VERTEX_SLOTS; s++) {
        if (!key.stride[s]) continue;
        vd.layouts[VERTEX_SLOT_BASE + s].stride = key.stride[s];
        vd.layouts[VERTEX_SLOT_BASE + s].stepFunction = MTLVertexStepFunctionPerVertex;
    }
    if (uses_constant) {
        vd.layouts[CONSTANT_SLOT].stride = 16;
        vd.layouts[CONSTANT_SLOT].stepFunction = MTLVertexStepFunctionConstant;
        vd.layouts[CONSTANT_SLOT].stepRate = 0;
    }
    d.vertexDescriptor = vd;
    d.colorAttachments[0].pixelFormat = MTLPixelFormatR32Float;
    d.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    d.maxVertexAmplificationCount = s_caster_halves;
    NSError *error = nil;
    id<MTLRenderPipelineState> state = [s_device newRenderPipelineStateWithDescriptor:d error:&error];
    if (!state) complain("shadow caster pipeline failed: %s", error.localizedDescription.UTF8String);
    cache[key] = state;
    return state;
}

// Compiles the casters' shaders, once. False when they cannot be had; shadows are then off for good.
bool caster_setup()
{
    if (s_caster_library) return true;
    NSError *error = nil;
    // The default language version follows the SDK the game was built with,
    // which is too old for vertex amplification.
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion2_2;
    s_caster_library = [s_device newLibraryWithSource:@(k_caster_source) options:options error:&error];
    if (!s_caster_library) {
        complain("shadow caster shaders did not compile: %s", error.localizedDescription.UTF8String);
        s_dyn_shadows = false;
        s_torch_shadows = 0;
        return false;
    }
    MTLDepthStencilDescriptor *dd = [MTLDepthStencilDescriptor new];
    dd.depthCompareFunction = MTLCompareFunctionLess;
    dd.depthWriteEnabled = YES;
    s_caster_depth_state = [s_device newDepthStencilStateWithDescriptor:dd];
    return true;
}

// ---- shadows from point lights -------------------------------------------------------------
//
// The engine's point lights (torches, candles, fires; four at most) cast no shadows. Here the casters
// recorded for the sun's map are drawn once more from each light, into the six faces of a cube of depths
// per light, in one pass: every draw is instanced, six times per light, and each instance goes to its own
// layer of the map. Shaders lit by these lights then compare their surface's distance from a light with
// what the light's cube holds that way (add_torch_fragment in shader_xlate.cpp).
//
// Which lights there are is learnt from the draws themselves (see note_torches), so the maps are drawn for
// the lights of the frame before: they stand still, or nearly.

// The cubes: one of the right size once there is something to draw into it, a token one before that, since
// a shader that samples the map has to be given a texture of its kind whether or not there are shadows.
id<MTLTexture> torch_map(bool real)
{
    NSUInteger size = real ? (NSUInteger)s_torch_size : 1;
    if (!s_torch_map || (real && s_torch_map.width != size)) {
        MTLTextureDescriptor *td = [MTLTextureDescriptor new];
        td.textureType = MTLTextureTypeCubeArray;
        td.pixelFormat = MTLPixelFormatDepth32Float;
        td.width = td.height = size;
        td.arrayLength = 4;
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModePrivate;
        s_torch_map = [s_device newTextureWithDescriptor:td];
    }
    if (!s_torch_sampler) {
        MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new]; // compares, and blends the four answers around the point asked for
        sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterLinear;
        sd.compareFunction = MTLCompareFunctionLessEqual;
        s_torch_sampler = [s_device newSamplerStateWithDescriptor:sd];
    }
    return s_torch_map;
}

std::unordered_map<PipelineKey, id<MTLRenderPipelineState>, PipelineKeyHash> s_torch_pipelines[4]; // [skinned * 2 + cutout]

id<MTLRenderPipelineState> torch_pipeline(const PipelineKey &key, bool skinned, bool cutout)
{
    auto &cache = s_torch_pipelines[skinned * 2 + cutout];
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = [s_caster_library newFunctionWithName:skinned ? @"torch_skinned" : @"torch_static"];
    d.fragmentFunction = cutout ? [s_caster_library newFunctionWithName:@"torch_cutout"] : nil; // depth alone otherwise
    MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
    const int wanted[4] = {0, 4, 7, 8};
    bool uses_constant = false;
    for (int n = 0; n < (skinned ? 4 : 2); n++) {
        int i = wanted[n];
        if (key.format[i] && key.slot[i] != CONSTANT_SLOT) {
            vd.attributes[i].format = (MTLVertexFormat)key.format[i];
            vd.attributes[i].offset = key.offset[i];
            vd.attributes[i].bufferIndex = VERTEX_SLOT_BASE + key.slot[i];
        } else { // the mesh has no such attribute: it reads the constant buffer
            vd.attributes[i].format = MTLVertexFormatFloat4;
            vd.attributes[i].offset = 0;
            vd.attributes[i].bufferIndex = CONSTANT_SLOT;
            uses_constant = true;
        }
    }
    for (int sl = 0; sl < MAX_VERTEX_SLOTS; sl++) {
        if (!key.stride[sl]) continue;
        vd.layouts[VERTEX_SLOT_BASE + sl].stride = key.stride[sl];
        vd.layouts[VERTEX_SLOT_BASE + sl].stepFunction = MTLVertexStepFunctionPerVertex;
    }
    if (uses_constant) {
        vd.layouts[CONSTANT_SLOT].stride = 16;
        vd.layouts[CONSTANT_SLOT].stepFunction = MTLVertexStepFunctionConstant;
        vd.layouts[CONSTANT_SLOT].stepRate = 0;
    }
    d.vertexDescriptor = vd;
    d.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    d.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle; // needed to choose a layer per instance
    NSError *error = nil;
    id<MTLRenderPipelineState> state = [s_device newRenderPipelineStateWithDescriptor:d error:&error];
    if (!state) complain("point light shadow pipeline failed: %s", error.localizedDescription.UTF8String);
    cache[key] = state;
    return state;
}

// inv_frame, p00 and p11 are render_dynamic_shadows' own: they turn a recorded draw's clip space back into the world.
void render_torch_shadows(const std::vector<CasterDraw> &casters, const std::vector<float> &bones, const double *inv_frame, double p00, double p11)
{
    s_torch_cubes = 0;
    if (s_torch_wanted.empty()) return;
    const size_t static_size = 256, skinned_size = 2304; // as for the sun's map
    size_t total = 0;
    for (const CasterDraw &c : casters) total += c.bones >= 0 || c.own_bones ? skinned_size : static_size;
    if (!total) return;
    id<MTLBuffer> uniforms = [s_device newBufferWithLength:total options:MTLResourceStorageModeShared];
    uint8_t *base = (uint8_t *)uniforms.contents;

    // Depth as a perspective projection gives it, from 5 cm to as far as a torch's light is worth following.
    const double near = 0.05, far = 80.0;
    struct {
        float light[4][4], depth[4];
    } torches = {};
    int cubes = (int)std::min<size_t>(4, s_torch_wanted.size());
    for (int k = 0; k < cubes; k++) memcpy(torches.light[k], s_torch_wanted[(size_t)k].at, 12);
    torches.depth[0] = (float)(far / (far - near));
    torches.depth[1] = (float)(-far * near / (far - near));
    torches.depth[2] = s_torch_self;

    id<MTLTexture> map = torch_map(true);
    id<MTLCommandBuffer> cmd = timed([s_queue commandBuffer]);
    MTLRenderPassDescriptor *d = [MTLRenderPassDescriptor renderPassDescriptor];
    d.depthAttachment.texture = map;
    d.depthAttachment.loadAction = MTLLoadActionClear;
    d.depthAttachment.storeAction = MTLStoreActionStore;
    d.depthAttachment.clearDepth = 1.0;
    d.renderTargetArrayLength = 24;
    id<MTLRenderCommandEncoder> pass = [cmd renderCommandEncoderWithDescriptor:d];
    [pass setDepthStencilState:s_caster_depth_state];
    [pass setCullMode:MTLCullModeNone];
    [pass setVertexBuffer:s_constant_attrib offset:0 atIndex:CONSTANT_SLOT];
    [pass setVertexBytes:&torches length:sizeof torches atIndex:1];
    size_t at = 0, drawn = 0;
    void *last_pipeline = nullptr;
    for (const CasterDraw &c : casters) {
        bool skinned = c.bones >= 0 || c.own_bones;
        size_t size = skinned ? skinned_size : static_size;
        // Hair's extra layers and strands are the sun's affair; the head they grow on casts here.
        id<MTLRenderPipelineState> pipeline = c.hair || c.strand ? nil : torch_pipeline(c.key, skinned, c.alpha != nil);
        if (!pipeline) {
            at += size;
            continue;
        }
        double object[16], placed[16], m[16];
        for (int i = 0; i < 16; i++) object[i] = c.wvp[i];
        if (!c.world) {
            for (int col = 0; col < 4; col++) {
                placed[col * 4 + 0] = object[col * 4 + 0] / p00;
                placed[col * 4 + 1] = object[col * 4 + 1] / p11;
                placed[col * 4 + 2] = object[col * 4 + 3];
                placed[col * 4 + 3] = col == 3 ? 1.0 : 0.0;
            }
            mat_mul(inv_frame, placed, m);
        }
        float m32[16];
        for (int i = 0; i < 16; i++) m32[i] = (float)(c.world ? object[i] : m[i]);
        memcpy(base + at, m32, 64);
        const float params[4] = {c.cutoff, 0, 0, 0};
        memcpy(base + at + 96, params, 16);
        if (skinned) memcpy(base + at + 144, c.own_bones ? c.own_bones->data() : &bones[(size_t)c.bones * 32 * 16], 32 * 64);
        if (last_pipeline != (__bridge void *)pipeline) {
            [pass setRenderPipelineState:pipeline];
            last_pipeline = (__bridge void *)pipeline;
        }
        [pass setVertexBuffer:uniforms offset:at atIndex:0];
        for (int i = 0; i < c.slots; i++) [pass setVertexBuffer:c.vertex[i] offset:c.vertex_offset[i] atIndex:(NSUInteger)(VERTEX_SLOT_BASE + i)];
        if (c.alpha) {
            [pass setFragmentTexture:c.alpha atIndex:0];
            [pass setFragmentSamplerState:c.alpha_sampler atIndex:0];
        }
        [pass drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:c.index_count indexType:c.index_type indexBuffer:c.indices indexBufferOffset:c.index_offset instanceCount:(NSUInteger)cubes * 6];
        at += size;
        drawn++;
    }
    [pass endEncoding];
    [cmd commit];

    s_torch_cubes = cubes;
    for (int k = 0; k < cubes; k++) memcpy(s_torch_cube[k], s_torch_wanted[(size_t)k].at, 12);
    s_torch_params[0] = s_torch_shadows, s_torch_params[1] = torches.depth[0], s_torch_params[2] = torches.depth[1], s_torch_params[3] = 2.0f / (float)s_torch_size;
    s_torch_frame = s_frame_serial;
    s_torch_frames++;
    s_torch_casters += drawn;
}

// One value of a uniform the game sets, from whichever stage's block holds it.
bool uniform_value(Program &p, int location, int element, void *out, size_t bytes)
{
    if (location < 0) return false;
    const UniformLocation &l = p.locations[(size_t)location];
    for (int s = 0; s < 2; s++) {
        const XlUniform *u = l.uniform[s];
        if (!u || element >= std::max(1, u->array_size)) continue;
        size_t at = (size_t)u->offset + (size_t)element * (size_t)u->array_stride;
        if (at + bytes > p.uniform_data[s].size()) continue;
        memcpy(out, p.uniform_data[s].data() + at, bytes);
        return true;
    }
    return false;
}

// Called for a draw into the scene by a program that takes shadows from point lights: notes the lights the
// engine has given it, for the next frame's maps, and fills in the shaders' wbm_torch_light, which says for
// each of them which cube, if any, was drawn from where it stands.
void note_torches(Program &p, float lights[4][4])
{
    int count = 4; // a shader that is not told how many uses what it is given
    uniform_value(p, p.light_count_location, 0, &count, 4);
    bool drawn = s_torch_frame == s_frame_serial;
    for (int i = 0; i < std::min(count, 4); i++) {
        float at[4], colour[4];
        if (!uniform_value(p, p.light_pos_location, i, at, 16) || !uniform_value(p, p.light_colour_location, i, colour, 16)) continue;
        // The slots past those in use hold whatever was there before.
        if (!(0.2126f * colour[0] + 0.7152f * colour[1] + 0.0722f * colour[2] > 0.02f) || !(fabsf(at[0]) + fabsf(at[1]) + fabsf(at[2]) < 1.0e6f)) continue;
        auto near_to = [&](const float *other) {
            float d = 0;
            for (int k = 0; k < 3; k++) d += (at[k] - other[k]) * (at[k] - other[k]);
            return d < 0.3f * 0.3f; // a flame flickers about; a light that has moved farther is another light
        };
        bool known = false;
        for (const TorchPlace &seen : s_torch_seen) known = known || near_to(seen.at);
        if (!known && s_torch_seen.size() < 4) s_torch_seen.push_back(TorchPlace{{at[0], at[1], at[2]}});
        memcpy(lights[i], at, 12);
        for (int k = 0; drawn && k < s_torch_cubes; k++)
            if (near_to(s_torch_cube[k])) {
                memcpy(lights[i], s_torch_cube[k], 12); // where the map was drawn from, to the letter
                lights[i][3] = (float)(k + 1);
                break;
            }
    }
}

// p is the program of a main-pass draw that has both a world-view-projection
// and a world matrix set, from which the camera is recovered.
void render_dynamic_shadows(Program &p)
{
    s_dyn_frame = s_frame_serial; // one attempt per frame, whatever comes of it
    std::vector<CasterDraw> casters;
    casters.swap(s_casters);
    std::vector<float> bones;
    bones.swap(s_caster_bones);
    size_t from_prepass = casters.size();
    casters.insert(casters.end(), s_late_ready.begin(), s_late_ready.end());
    s_dyn_window[2] = s_dyn_window[3] = 0; // no map, until this succeeds
    s_dyn_near[2] = s_dyn_near[3] = 0;

    double wvp[16], world[16], inv[16], view_proj[16], sun[16];
    if (!vertex_matrix(p, "matWorldViewProj", wvp) || !vertex_matrix(p, "matWorld", world) || !mat_inverse(world, inv)) return;
    mat_mul(wvp, inv, view_proj);
    for (int i = 0; i < 16; i++) sun[i] = s_sun_matrix[i];

    // Inverting the view-projection itself to get from a draw's clip space back
    // to the world loses the position along the view axis to rounding: a
    // centimetre or two at a hundred metres, different every frame the camera
    // moves, which made distant shadows shimmer. The parts of a projected
    // position that do not go through the depth mapping are enough: x and y
    // divided by the projection's scales, and w, are a rigid transform of the
    // world position (`frame`), and that inverts without loss.
    double p00 = sqrt(view_proj[0] * view_proj[0] + view_proj[4] * view_proj[4] + view_proj[8] * view_proj[8]);
    double p11 = sqrt(view_proj[1] * view_proj[1] + view_proj[5] * view_proj[5] + view_proj[9] * view_proj[9]);
    if (p00 < 1e-9 || p11 < 1e-9) return;
    auto rigid = [&](const double *clip, double *out) { // rows (x / p00, y / p11, w, 0 0 0 1) of a clip-space matrix
        for (int c = 0; c < 4; c++) {
            out[c * 4 + 0] = clip[c * 4 + 0] / p00;
            out[c * 4 + 1] = clip[c * 4 + 1] / p11;
            out[c * 4 + 2] = clip[c * 4 + 3];
            out[c * 4 + 3] = c == 3 ? 1.0 : 0.0;
        }
    };
    double frame[16], inv_frame[16];
    rigid(view_proj, frame);
    if (!mat_inverse(frame, inv_frame)) return;

    if (!caster_setup()) return;
    if (s_torch_shadows > 0) render_torch_shadows(casters, bones, inv_frame, p00, p11);
    if (!s_dyn_shadows || !s_sun_matrix_valid) return;
    if (!s_dyn_map) {
        // The fine window needs every draw rasterised twice in one go.
        s_caster_halves = s_dyn_near_range > 0 && [s_device supportsVertexAmplificationCount:2] ? 2 : 1;
        const NSUInteger size = k_xl_dyn_size;
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float width:size * 2 height:size mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModePrivate;
        s_dyn_map = [s_device newTextureWithDescriptor:td];
        td.pixelFormat = MTLPixelFormatDepth32Float;
        td.usage = MTLTextureUsageRenderTarget;
        s_dyn_depth = [s_device newTextureWithDescriptor:td];
        MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new]; // nearest: the shader compares depths itself
        sd.sAddressMode = sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
        s_dyn_sampler = [s_device newSamplerStateWithDescriptor:sd];
    }

    // Where the camera is, and which way it faces: the direction in which w grows.
    double eye[3] = {inv_frame[12], inv_frame[13], inv_frame[14]};
    for (int k = 0; k < 3; k++) s_eye[k] = (float)eye[k];
    s_eye_frame = s_frame_serial;
    double fx = view_proj[3], fy = view_proj[7], flat = sqrt(fx * fx + fy * fy); // Z is up
    double reach = s_dyn_range * 0.6;
    double focus[3] = {eye[0] + (flat > 1e-6 ? fx / flat * reach : 0), eye[1] + (flat > 1e-6 ? fy / flat * reach : 0), eye[2]};

    // The window: centred on that point in map coordinates, s_dyn_range metres to
    // each side, snapped to its own texel grid so that shadows do not crawl as
    // the camera moves.
    auto row_length = [&](int r) { return sqrt(sun[r] * sun[r] + sun[4 + r] * sun[4 + r] + sun[8 + r] * sun[8 + r]); };
    double half_u = s_dyn_range * row_length(0), half_v = s_dyn_range * row_length(1);
    if (half_u < 1e-9 || half_v < 1e-9) return;
    double cu = sun[0] * focus[0] + sun[4] * focus[1] + sun[8] * focus[2] + sun[12];
    double cv = sun[1] * focus[0] + sun[5] * focus[1] + sun[9] * focus[2] + sun[13];
    double texel_u = 2 * half_u / k_xl_dyn_size, texel_v = 2 * half_v / k_xl_dyn_size;
    cu = round(cu / texel_u) * texel_u;
    cv = round(cv / texel_v) * texel_v;
    // The fine window: the same, s_dyn_near_range metres to each side of a point
    // just ahead of the camera.
    double near_u = s_dyn_near_range * row_length(0), near_v = s_dyn_near_range * row_length(1), nu = 0, nv = 0;
    bool fine = s_caster_halves == 2 && near_u > 1e-9 && near_v > 1e-9;
    if (fine) {
        double ahead_by = s_dyn_near_range * 0.5;
        double point[3] = {eye[0] + (flat > 1e-6 ? fx / flat * ahead_by : 0), eye[1] + (flat > 1e-6 ? fy / flat * ahead_by : 0), eye[2]};
        nu = sun[0] * point[0] + sun[4] * point[1] + sun[8] * point[2] + sun[12];
        nv = sun[1] * point[0] + sun[5] * point[1] + sun[9] * point[2] + sun[13];
        double step_u = 2 * near_u / k_xl_dyn_size, step_v = 2 * near_v / k_xl_dyn_size;
        nu = round(nu / step_u) * step_u;
        nv = round(nv / step_v) * step_v;
    }
    float window[8] = {(float)cu, (float)cv, (float)(1 / half_u), (float)(1 / half_v), 0, 0, 0, 0};
    if (fine) window[4] = (float)nu, window[5] = (float)nv, window[6] = (float)(1 / near_u), window[7] = (float)(1 / near_v);

    double to_shadow[16]; // from `frame` to shadow space
    mat_mul(sun, inv_frame, to_shadow);

    // All the casters' uniforms go into one buffer.
    const size_t static_size = 256, skinned_size = 2304; // 144 and 2192 bytes, rounded up to the offset alignment
    size_t total = 0;
    for (const CasterDraw &c : casters) total += c.bones >= 0 || c.own_bones ? skinned_size : static_size;
    if (!total) return;
    id<MTLBuffer> uniforms = [s_device newBufferWithLength:total options:MTLResourceStorageModeShared];
    uint8_t *base = (uint8_t *)uniforms.contents;

    id<MTLCommandBuffer> cmd = timed([s_queue commandBuffer]);
    MTLRenderPassDescriptor *d = [MTLRenderPassDescriptor renderPassDescriptor];
    d.colorAttachments[0].texture = s_dyn_map;
    d.colorAttachments[0].loadAction = MTLLoadActionClear;
    d.colorAttachments[0].storeAction = MTLStoreActionStore;
    d.colorAttachments[0].clearColor = MTLClearColorMake(1e9, 0, 0, 0); // nothing nearer the sun
    d.depthAttachment.texture = s_dyn_depth;
    d.depthAttachment.loadAction = MTLLoadActionClear;
    d.depthAttachment.storeAction = MTLStoreActionDontCare;
    d.depthAttachment.clearDepth = 1.0;
    id<MTLRenderCommandEncoder> pass = [cmd renderCommandEncoderWithDescriptor:d];
    [pass setDepthStencilState:s_caster_depth_state];
    [pass setCullMode:MTLCullModeNone];
    [pass setVertexBuffer:s_constant_attrib offset:0 atIndex:CONSTANT_SLOT];
    const double N = k_xl_dyn_size;
    const MTLViewport viewports[2] = {{0, 0, N, N, 0, 1}, {N, 0, N, N, 0, 1}};
    const MTLScissorRect scissors[2] = {{0, 0, k_xl_dyn_size, k_xl_dyn_size}, {k_xl_dyn_size, 0, k_xl_dyn_size, k_xl_dyn_size}};
    NSUInteger halves = fine ? 2 : 1;
    [pass setViewports:viewports count:halves];
    [pass setScissorRects:scissors count:halves];
    if (s_caster_halves == 2) {
        MTLVertexAmplificationViewMapping mappings[2] = {{0, 0}, {1, 0}};
        [pass setVertexAmplificationCount:halves viewMappings:mappings];
    }
    size_t at = 0;
    void *last_pipeline = nullptr;
    for (const CasterDraw &c : casters) {
        bool skinned = c.bones >= 0 || c.own_bones;
        size_t size = skinned ? skinned_size : static_size;
        id<MTLRenderPipelineState> pipeline = c.scenery ? nil : caster_pipeline(c.key, skinned, c.alpha != nil, c.hair, c.strand);
        if (!pipeline) {
            at += size;
            continue;
        }
        double m[16], object[16], placed[16];
        for (int i = 0; i < 16; i++) object[i] = c.wvp[i];
        if (!c.world) rigid(object, placed);
        mat_mul(c.world ? sun : to_shadow, c.world ? object : placed, m);
        float m32[16];
        for (int i = 0; i < 16; i++) m32[i] = (float)m[i];
        memcpy(base + at, m32, 64);
        memcpy(base + at + 64, window, 32);
        const float params[4] = {c.cutoff, c.push, c.share, 0}, hang[4] = {c.down[0], c.down[1], c.down[2], 0};
        memcpy(base + at + 96, params, 16);
        memcpy(base + at + 112, hang, 16);
        const float drift[4] = {c.drift[0], c.drift[1], c.drift[2], 0};
        memcpy(base + at + 128, drift, 16);
        if (skinned) memcpy(base + at + 144, c.own_bones ? c.own_bones->data() : &bones[(size_t)c.bones * 32 * 16], 32 * 64);
        if (last_pipeline != (__bridge void *)pipeline) {
            [pass setRenderPipelineState:pipeline];
            last_pipeline = (__bridge void *)pipeline;
        }
        [pass setVertexBuffer:uniforms offset:at atIndex:0];
        for (int i = 0; i < c.slots; i++) [pass setVertexBuffer:c.vertex[i] offset:c.vertex_offset[i] atIndex:(NSUInteger)(VERTEX_SLOT_BASE + i)];
        if (c.alpha) {
            [pass setFragmentTexture:c.alpha atIndex:0];
            [pass setFragmentSamplerState:c.alpha_sampler atIndex:0];
        }
        [pass drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:c.index_count indexType:c.index_type indexBuffer:c.indices indexBufferOffset:c.index_offset];
        at += size;
    }
    [pass endEncoding];
    [cmd commit];

    s_dyn_window[0] = (float)cu;
    s_dyn_window[1] = (float)cv;
    s_dyn_window[2] = (float)(1 / (2 * half_u));
    s_dyn_window[3] = (float)(1 / (2 * half_v));
    if (fine) {
        s_dyn_near[0] = (float)nu;
        s_dyn_near[1] = (float)nv;
        s_dyn_near[2] = (float)(1 / (2 * near_u));
        s_dyn_near[3] = (float)(1 / (2 * near_v));
    }
    s_dyn_bias = (float)(s_dyn_bias_m * row_length(2));
    s_dyn_stats.frames++;
    s_dyn_stats.casters += from_prepass;
    s_dyn_stats.late += casters.size() - from_prepass;
}
