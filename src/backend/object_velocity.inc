// The movement of objects that move by themselves.
// Included by metal_backend.mm, inside namespace wbm.

id<MTLRenderPipelineState> velocity_pipeline(const PipelineKey &key, bool skinned)
{
    auto &cache = s_velocity_pipelines[skinned];
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = [s_blur_library newFunctionWithName:skinned ? @"velocity_skinned" : @"velocity_static"];
    d.fragmentFunction = [s_blur_library newFunctionWithName:@"velocity_fragment"];
    MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
    const int wanted[3] = {0, 7, 8};
    bool uses_constant = false;
    for (int n = 0; n < (skinned ? 3 : 1); n++) {
        int i = wanted[n];
        if (key.format[i] && key.slot[i] != CONSTANT_SLOT) {
            vd.attributes[i].format = (MTLVertexFormat)key.format[i];
            vd.attributes[i].offset = key.offset[i];
            vd.attributes[i].bufferIndex = VERTEX_SLOT_BASE + key.slot[i];
        } else {
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
    d.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA16Float;
    d.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    NSError *error = nil;
    id<MTLRenderPipelineState> state = [s_device newRenderPipelineStateWithDescriptor:d error:&error];
    if (!state) complain("motion blur object pipeline failed: %s", error.localizedDescription.UTF8String);
    cache[key] = state;
    return state;
}

id<MTLTexture> render_object_velocities(const double *reproject, NSUInteger w, NSUInteger h)
{
    if (!objects_recorded() || !s_blur_library) return nil;
    struct Pair {
        const VelocityDraw *now;
        const PastObject *before;
    };
    std::vector<Pair> pairs;
    size_t bytes = 0;
    // Last frame's objects by mesh: in a forest there are a thousand and more of each, and comparing every one
    // with every other took longer than drawing them.
    std::unordered_multimap<uint64_t, PastObject *> past_by_mesh;
    past_by_mesh.reserve(s_past_objects.size());
    for (PastObject &o : s_past_objects) past_by_mesh.emplace(o.mesh, &o);
    const size_t static_size = 256, skinned_size = 4352; // 128 and 4224 bytes, rounded up to the offset alignment
    for (const VelocityDraw &v : s_velocity_draws) {
        const float *m = v.c.wvp, *at = v.origin;
        if (at[3] < 0.05f || at[3] > 80.0f) continue; // behind the camera, or too far off for it to show
        double expected[4]; // where it was, had only the camera moved
        for (int r = 0; r < 4; r++) expected[r] = reproject[12 + r] * at[3] + reproject[r] * at[0] + reproject[4 + r] * at[1] + reproject[8 + r] * at[2];
        if (expected[3] < 1e-3) continue;
        // Not one past object to each: a mesh drawn twice in one place (in two parts, or once more for some effect)
        // would send its second draw to the nearest other of its kind, across the room, and under temporal
        // upscaling a wrong movement drags the picture with it. Nor from far off, for the same reason.
        PastObject *best = nullptr;
        double nearest = 0.25;
        auto same = past_by_mesh.equal_range(v.c.mesh);
        for (auto it = same.first; it != same.second; ++it) {
            PastObject &o = *it->second;
            if (o.origin[3] < 1e-3f || (o.bones >= 0) != (v.bones >= 0)) continue;
            double d = hypot(o.origin[0] / o.origin[3] - expected[0] / expected[3], o.origin[1] / o.origin[3] - expected[1] / expected[3]) + fabs(log(o.origin[3] / expected[3])) * 3.0;
            if (d < nearest) nearest = d, best = &o;
        }
        if (!best) continue;
        if (v.bones < 0) { // a thing without bones that has only gone where the camera's movement took it has no blur of its own
            double most = 0;
            for (int c = 0; c < 4; c++)
                for (int r = 0; r < 4; r++) {
                    double e = 0;
                    for (int k = 0; k < 4; k++) e += reproject[k * 4 + r] * m[c * 4 + k];
                    most = std::max(most, fabs(e - best->wvp[c * 4 + r]));
                }
            if (most < 2e-3 * std::max(1.0f, m[15])) continue;
        }
        if (v.bones >= 0 && !memcmp(&s_velocity_bones[(size_t)v.bones * 512], &s_past_bones[(size_t)best->bones * 512], 512 * sizeof(float))) {
            double most = 0;
            for (int c = 0; c < 4; c++)
                for (int r = 0; r < 4; r++) {
                    double e = 0;
                    for (int k = 0; k < 4; k++) e += reproject[k * 4 + r] * m[c * 4 + k];
                    most = std::max(most, fabs(e - best->wvp[c * 4 + r]));
                }
            if (most < 2e-3 * std::max(1.0f, m[15])) continue; // nor one with bones that has held its pose
        }
        pairs.push_back({&v, best});
        bytes += v.bones >= 0 ? skinned_size : static_size;
    }
    if (pairs.empty()) return nil;
    NSUInteger vw = std::max<NSUInteger>(1, w / 2), vh = std::max<NSUInteger>(1, h / 2);
    if (!s_velocity_texture || s_velocity_texture.width != vw || s_velocity_texture.height != vh) {
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:vw height:vh mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModePrivate;
        s_velocity_texture = [s_device newTextureWithDescriptor:td];
        s_velocity_depth = new_depth_texture(vw, vh, 1);
    }
    if (!s_velocity_depth_state) {
        MTLDepthStencilDescriptor *dd = [MTLDepthStencilDescriptor new];
        dd.depthCompareFunction = MTLCompareFunctionLess;
        dd.depthWriteEnabled = YES;
        s_velocity_depth_state = [s_device newDepthStencilStateWithDescriptor:dd];
    }
    id<MTLBuffer> uniforms = [s_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    uint8_t *base = (uint8_t *)uniforms.contents;
    MTLRenderPassDescriptor *d = [MTLRenderPassDescriptor renderPassDescriptor];
    d.colorAttachments[0].texture = s_velocity_texture;
    d.colorAttachments[0].loadAction = MTLLoadActionClear;
    d.colorAttachments[0].storeAction = MTLStoreActionStore;
    d.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, -1, 0); // nothing moving here
    d.depthAttachment.texture = s_velocity_depth;
    d.depthAttachment.loadAction = MTLLoadActionClear;
    d.depthAttachment.storeAction = MTLStoreActionDontCare;
    d.depthAttachment.clearDepth = 1.0;
    id<MTLRenderCommandEncoder> pass = [command_buffer() renderCommandEncoderWithDescriptor:d];
    [pass setDepthStencilState:s_velocity_depth_state];
    [pass setCullMode:MTLCullModeNone];
    [pass setVertexBuffer:s_constant_attrib offset:0 atIndex:CONSTANT_SLOT];
    size_t at = 0;
    for (const Pair &pair : pairs) {
        const CasterDraw &c = pair.now->c;
        bool skinned = pair.now->bones >= 0;
        size_t size = skinned ? skinned_size : static_size;
        id<MTLRenderPipelineState> pipeline = velocity_pipeline(c.key, skinned);
        if (pipeline) {
            memcpy(base + at, c.wvp, 64);
            memcpy(base + at + 64, pair.before->wvp, 64);
            if (skinned) {
                memcpy(base + at + 128, &s_velocity_bones[(size_t)pair.now->bones * 32 * 16], 32 * 64);
                memcpy(base + at + 128 + 32 * 64, &s_past_bones[(size_t)pair.before->bones * 32 * 16], 32 * 64);
            }
            [pass setRenderPipelineState:pipeline];
            [pass setVertexBuffer:uniforms offset:at atIndex:0];
            for (int i = 0; i < c.slots; i++) [pass setVertexBuffer:c.vertex[i] offset:c.vertex_offset[i] atIndex:(NSUInteger)(VERTEX_SLOT_BASE + i)];
            [pass drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:c.index_count indexType:c.index_type indexBuffer:c.indices indexBufferOffset:c.index_offset];
        }
        at += size;
    }
    [pass endEncoding];
    s_velocity_objects += pairs.size();
    return s_velocity_texture;
}
