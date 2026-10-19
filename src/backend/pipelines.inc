// Programs (linking, uniforms) and pipeline and depth states.
// Included by metal_backend.mm, inside namespace wbm.

// ---- programs -----------------------------------------------------------------------------

void link_program(GLuint program_id, Program &p)
{
    p.linked = false;
    p.pipelines.clear();
    p.locations.clear();
    p.location_by_name.clear();
    p.yflip_offset = p.jitter_offset = p.hash_frame_offset = p.reproject_offset = p.wind_before_offset = p.samples_offset = p.water_offset = p.parallax_offset = p.grade_offset = p.headroom_offset = p.limit_offset = p.dyn_window_offset = p.dyn_near_offset = p.dyn_bias_offset = p.torch_offset = p.torch_light_offset = -1;
    p.amb_offset[0] = p.amb_offset[1] = p.material_offset = p.pbr_offset = p.skin_offset[0] = p.skin_offset[1] = p.wind_offset = p.leaf_sun_offset = p.gaze_offset = p.fur_offset = p.hair_colour_offset = p.horse_eye_offset = p.sheen_offset = p.shell_offset[0] = p.shell_offset[1] = p.drift_offset = -1;
    p.u_wvp = p.u_world = p.u_bones = nullptr;
    p.alpha_texture_location = -1;
    s_last_key_program = 0; // the pipeline remembered from the previous draw may be this program's
    std::string vs, fs;
    for (auto &sh : p.shaders) (sh->type == GL_VERTEX_SHADER ? vs : fs) = sh->source;
    bool from_cache = false;
    CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
    p.xl = xl_translate_cached(vs, fs, p.attribs, s_shader_options, &from_cache);
    s_shader_stats.translate_ms += (CFAbsoluteTimeGetCurrent() - started) * 1000;
    s_shader_stats.links++;
    s_shader_stats.cache_hits += from_cache;
    if (!p.xl.ok) {
        wb_note("  !! metal: program %u did not translate:\n%s", program_id, p.xl.log.c_str());
        return;
    }
    const XlStage *stages[2] = {&p.xl.vs, &p.xl.fs};
    p.functions_ready = false;
    for (int s = 0; s < 2; s++) {
        // Metal compiles the source in the background; the program waits for it
        // only when it first draws (see finish_compile). The engine links every
        // program at start-up and uses a few of them right away.
        std::shared_ptr<StageCompile> &compile = s_stage_compiles[stages[s]->msl];
        if (!compile) {
            compile = std::make_shared<StageCompile>();
            compile->group = dispatch_group_create();
            dispatch_group_enter(compile->group);
            std::shared_ptr<StageCompile> result = compile;
            NSString *entry = @(stages[s]->entry.c_str());
            [s_device newLibraryWithSource:@(stages[s]->msl.c_str())
                                   options:nil
                         completionHandler:^(id<MTLLibrary> library, NSError *error) {
                           if (library) result->function = [library newFunctionWithName:entry];
                           if (!result->function) result->error = error ? error.localizedDescription.UTF8String : "entry point not found";
                           dispatch_group_leave(result->group);
                         }];
            s_shader_stats.stage_compiles++;
        }
        p.compile[s] = compile;
        p.uniform_data[s].assign((size_t)stages[s]->uniform_buffer_size, 0);

        auto location = [&](const std::string &name) -> UniformLocation & {
            auto it = p.location_by_name.find(name);
            if (it == p.location_by_name.end()) {
                it = p.location_by_name.emplace(name, (GLint)p.locations.size()).first;
                p.locations.emplace_back();
                p.locations.back().name = name;
                p.locations.back().is_sun_dir = name == "vSunDir";
                p.locations.back().is_sun_color = name == "vSunColor";
                p.locations.back().is_sky = name == k_xl_sky_sampler;
                p.locations.back().is_dyn_map = name == k_xl_dyn_sampler;
                p.locations.back().is_torch_map = name == k_xl_torch_sampler;
                p.locations.back().is_light_indices = name == "iLightIndices";
                p.locations.back().is_scene_copy = name == "wbm_scene";
                p.locations.back().is_pbr_map = name == k_xl_pbr_sampler;
                p.locations.back().is_depth_map = name == "wbm_depth_map";
                p.locations.back().is_sheen_map = name == "wbm_sheen_map";
                p.locations.back().is_fur_map = name == "wbm_fur_map";
                p.locations.back().is_hair_map = name == "wbm_hair_map";
                p.locations.back().is_sun_matrix = name == "matSunViewProj";
            }
            return p.locations[(size_t)it->second];
        };
        for (const XlUniform &u : stages[s]->uniforms) {
            if (u.name == k_xl_yflip_uniform) p.yflip_offset = u.offset;
            if (u.name == k_xl_jitter_uniform) p.jitter_offset = u.offset;
            else if (u.name == k_xl_samples_uniform) p.samples_offset = u.offset;
            else if (u.name == k_xl_headroom_uniform) p.headroom_offset = u.offset;
            else if (u.name == "wbm_grade" && s == 1) p.grade_offset = u.offset;
            else if (u.name == "wbm_parallax" && s == 1) p.parallax_offset = u.offset;
            else if (u.name == "wbm_water" && s == 1) p.water_offset = u.offset;
            else if (u.name == k_xl_limit_uniform) p.limit_offset = u.offset;
            else if (u.name == k_xl_dyn_window_uniform) p.dyn_window_offset = u.offset;
            else if (u.name == k_xl_dyn_near_uniform) p.dyn_near_offset = u.offset;
            else if (u.name == k_xl_dyn_bias_uniform) p.dyn_bias_offset = u.offset;
            else if (u.name == k_xl_torch_uniform && s == 1) p.torch_offset = u.offset;
            else if (u.name == k_xl_torch_light_uniform && s == 1 && u.array_stride == 16) p.torch_light_offset = u.offset;
            else if (u.name == k_xl_ambient_uniform) p.amb_offset[s] = u.offset;
            else if (u.name == k_xl_material_uniform) p.material_offset = u.offset;
            else if (u.name == k_xl_pbr_uniform) p.pbr_offset = u.offset;
            else if (u.name == k_xl_skin_uniform) p.skin_offset[s] = u.offset;
            else if (u.name == k_xl_wind_uniform) p.wind_offset = u.offset;
            else if (u.name == "wbm_reproject") p.reproject_offset = u.offset;
            else if (u.name == "wbm_hash_frame") p.hash_frame_offset = u.offset;
            else if (u.name == "wbm_wind_before") p.wind_before_offset = u.offset;
            else if (u.name == "wbm_leaf_sun") p.leaf_sun_offset = u.offset;
            else if (u.name == "wbm_gaze") p.gaze_offset = u.offset;
            else if (u.name == "wbm_fur") p.fur_offset = u.offset;
            else if (u.name == "wbm_horse_eye") p.horse_eye_offset = u.offset;
            else if (u.name == "wbm_sheen") p.sheen_offset = u.offset;
            else if (u.name == "wbm_hair_colour") p.hair_colour_offset = u.offset;
            else if (u.name == "wbm_shell") p.shell_offset[s] = u.offset;
            else if (u.name == "wbm_drift" && s == 0) p.drift_offset = u.offset;
            else location(u.name).uniform[s] = &u;
        }
        for (const XlSampler &smp : stages[s]->samplers) location(smp.name).texture_index[s] = smp.texture_index;
        for (const std::string &name : stages[s]->srgb_samplers) location(name).srgb = true;
        p.uniform_dirty[s] = true;
        p.pushed_frame[s] = 0;
    }
    auto vertex_uniform = [&](const char *name) -> const XlUniform * {
        auto it = p.location_by_name.find(name);
        return it == p.location_by_name.end() ? nullptr : p.locations[(size_t)it->second].uniform[0];
    };
    auto location_of = [&](const char *name) {
        auto it = p.location_by_name.find(name);
        return it == p.location_by_name.end() ? -1 : (int)it->second;
    };
    p.light_pos_location = location_of("vLightPosDir");
    p.light_colour_location = location_of("vLightDiffuse");
    p.light_count_location = location_of("iLightPointCount");
    p.u_wvp = vertex_uniform("matWorldViewProj");
    p.u_world = vertex_uniform("matWorld");
    p.u_bones = vertex_uniform("matWorldArray");
    auto diffuse = p.location_by_name.find("diffuse_texture");
    if (diffuse != p.location_by_name.end() && p.locations[(size_t)diffuse->second].texture_index[1] >= 0) p.alpha_texture_location = diffuse->second;
    p.is_blob_shadow = fs.find("0.0234375") != std::string::npos;
    p.input_mask = 0;
    for (int loc : p.xl.vs.input_locations)
        if (loc >= 0 && loc < MAX_ATTRIBS) p.input_mask |= 1u << loc;
    p.linked = true;
}

// Waits for the program's two shaders to finish compiling. Returns false, and
// retires the program, if either failed.
bool finish_compile(Program &p)
{
    CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
    for (int s = 0; s < 2; s++) {
        StageCompile &c = *p.compile[s];
        dispatch_group_wait(c.group, DISPATCH_TIME_FOREVER);
        if (!c.function) {
            complain("program %u %s shader did not compile: %s", s_program, s ? "fragment" : "vertex", c.error.c_str());
            p.linked = false;
            return false;
        }
        p.function[s] = c.function;
    }
    s_shader_stats.wait_ms += (CFAbsoluteTimeGetCurrent() - started) * 1000;
    return p.functions_ready = true;
}

// Writes `count` values of `components` 32-bit words each into the uniform's
// place in both stages' buffers, converting between int and float as needed.
void set_uniform(GLint loc, const void *data, int count, int components, bool data_is_int)
{
    Program *p = find(s_programs, s_program);
    if (!p || loc < 0 || (size_t)loc >= p->locations.size()) return;
    UniformLocation &l = p->locations[(size_t)loc];
    if ((l.texture_index[0] >= 0 || l.texture_index[1] >= 0) && data_is_int) l.unit = *(const GLint *)data;
    // The shaders take an object's point lights in the order of iLightIndices, four numbers, the nearest light
    // first. The game sets it as though it were one number, so the other three stay 0, and every light after
    // the first is the scene's light 0 over again: of two torches in a room, the walls saw one. The rest of the
    // order is filled in here.
    GLint light_order[4];
    if (s_torch_shadows > 0 && l.is_light_indices && data_is_int && count == 1 && components == 1) {
        light_order[0] = *(const GLint *)data;
        for (int i = 0, next = 1; i < 4 && next < 4; i++)
            if (i != light_order[0]) light_order[next++] = i;
        data = light_order;
        count = 4;
    }
    if (s_dyn_shadows && l.is_sun_matrix && components == 16 && count >= 1) {
        memcpy(s_sun_matrix, data, sizeof s_sun_matrix); // the same for every program in a scene
        s_sun_matrix_valid = true;
    }
    if ((s_contact_shadows || s_light_shafts || s_sky_ambient || s_foliage_backlight > 0) && (l.is_sun_dir || l.is_sun_color) && components == 4 && count >= 1) {
        // The UI and other passes set these too; only the values in use while the
        // scene buffer is bound describe the scene's sun.
        Target t;
        if (s_fbo != 0 && resolve_target(t) && t.scene) {
            if (l.is_sun_dir) {
                memcpy(s_sun_dir, data, sizeof s_sun_dir);
                s_sun_valid = true;
            } else {
                memcpy(s_sun_color, data, sizeof s_sun_color);
            }
        }
    }
    for (int s = 0; s < 2; s++) {
        const XlUniform *u = l.uniform[s];
        if (!u) continue;
        int words = std::min(components, u->components * u->columns);
        int n = std::min(count, std::max(1, u->array_size));
        int stride = u->array_size ? u->array_stride : words * 4;
        uint8_t *dst = p->uniform_data[s].data() + u->offset;
        const uint32_t *src = (const uint32_t *)data;
        for (int i = 0; i < n; i++, dst += stride, src += components) {
            uint32_t converted;
            const void *value = src;
            size_t bytes = (size_t)words * 4;
            if (u->is_int != data_is_int) {
                if (data_is_int) {
                    float v = (float)*(const GLint *)src;
                    memcpy(&converted, &v, 4);
                } else {
                    GLint v = (GLint) * (const GLfloat *)src;
                    memcpy(&converted, &v, 4);
                }
                value = &converted;
                bytes = 4;
            }
            if (memcmp(dst, value, bytes) != 0) { // the engine re-sends unchanged values for every draw
                memcpy(dst, value, bytes);
                p->uniform_dirty[s] = true;
            }
        }
    }
}

// ---- pipeline and fixed-function state ------------------------------------------------------

MTLVertexFormat vertex_format(const Attrib &a)
{
    if (a.type == GL_FLOAT && a.size >= 1 && a.size <= 4) return (MTLVertexFormat)(MTLVertexFormatFloat + a.size - 1);
    if (a.type == GL_UNSIGNED_BYTE && a.size == 4) return a.normalized ? MTLVertexFormatUChar4Normalized : MTLVertexFormatUChar4;
    if (a.type == GL_SHORT && a.size == 2) return a.normalized ? MTLVertexFormatShort2Normalized : MTLVertexFormatShort2;
    if (a.type == GL_SHORT && a.size == 4) return a.normalized ? MTLVertexFormatShort4Normalized : MTLVertexFormatShort4;
    return MTLVertexFormatInvalid;
}

MTLBlendFactor blend_factor(GLenum f)
{
    switch (f) {
    case GL_ZERO: return MTLBlendFactorZero;
    case GL_ONE: return MTLBlendFactorOne;
    case GL_SRC_COLOR: return MTLBlendFactorSourceColor;
    case GL_ONE_MINUS_SRC_COLOR: return MTLBlendFactorOneMinusSourceColor;
    case GL_SRC_ALPHA: return MTLBlendFactorSourceAlpha;
    case GL_ONE_MINUS_SRC_ALPHA: return MTLBlendFactorOneMinusSourceAlpha;
    case GL_DST_ALPHA: return MTLBlendFactorDestinationAlpha;
    case GL_ONE_MINUS_DST_ALPHA: return MTLBlendFactorOneMinusDestinationAlpha;
    case GL_DST_COLOR: return MTLBlendFactorDestinationColor;
    case GL_ONE_MINUS_DST_COLOR: return MTLBlendFactorOneMinusDestinationColor;
    case GL_SRC_ALPHA_SATURATE: return MTLBlendFactorSourceAlphaSaturated;
    default: complain("unsupported blend factor %s", E_blend(f)); return MTLBlendFactorOne;
    }
}

MTLBlendOperation blend_operation(GLenum eq)
{
    switch (eq) {
    case GL_FUNC_SUBTRACT: return MTLBlendOperationSubtract;
    case GL_FUNC_REVERSE_SUBTRACT: return MTLBlendOperationReverseSubtract;
    case GL_MIN: return MTLBlendOperationMin;
    case GL_MAX: return MTLBlendOperationMax;
    default: return MTLBlendOperationAdd;
    }
}

struct VertexBinding {
    GLuint buffer[MAX_VERTEX_SLOTS];
    uintptr_t base[MAX_VERTEX_SLOTS];
    int slots = 0;
};
VertexBinding s_draw_binding; // of the draw most recently prepared

// Describes the current vertex arrays for the attributes the program reads.
// Attributes that share a buffer and stride share a Metal buffer slot.
bool build_vertex_layout(const Program &p, PipelineKey &key, VertexBinding &binding)
{
    for (int i = 0; i < MAX_ATTRIBS; i++) {
        if (!(p.input_mask & (1u << i))) continue;
        const Attrib &a = s_attribs[i];
        if (!a.enabled) { // OpenGL supplies a constant; so does the constant buffer
            key.format[i] = MTLVertexFormatFloat4;
            key.slot[i] = CONSTANT_SLOT;
            continue;
        }
        MTLVertexFormat format = vertex_format(a);
        if (format == MTLVertexFormatInvalid) {
            complain("unsupported vertex attribute: size %d, type %s", a.size, E(a.type));
            return false;
        }
        if (!a.buffer) {
            complain("vertex attribute %d comes from client memory, which is not supported", i);
            return false;
        }
        int slot = 0;
        while (slot < binding.slots && !(binding.buffer[slot] == a.buffer && key.stride[slot] == a.stride)) slot++;
        if (slot == binding.slots) {
            if (slot == MAX_VERTEX_SLOTS) {
                complain("more than %d vertex buffers in one draw", MAX_VERTEX_SLOTS);
                return false;
            }
            binding.buffer[slot] = a.buffer;
            binding.base[slot] = a.offset;
            key.stride[slot] = (uint16_t)a.stride;
            binding.slots++;
        }
        binding.base[slot] = std::min(binding.base[slot], a.offset);
        key.format[i] = (uint8_t)format;
        key.slot[i] = (uint8_t)slot;
    }
    for (int i = 0; i < MAX_ATTRIBS; i++)
        if (key.format[i] && key.slot[i] != CONSTANT_SLOT) key.offset[i] = (uint16_t)(s_attribs[i].offset - binding.base[key.slot[i]]);
    return true;
}

id<MTLRenderPipelineState> pipeline_for(Program &p, const PipelineKey &key)
{
    auto it = p.pipelines.find(key);
    if (it != p.pipelines.end()) return it->second;

    MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = p.function[0];
    d.fragmentFunction = p.function[1];
    MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
    bool uses_constant = false;
    for (int i = 0; i < MAX_ATTRIBS; i++) {
        if (!key.format[i]) continue;
        vd.attributes[i].format = (MTLVertexFormat)key.format[i];
        vd.attributes[i].offset = key.offset[i];
        vd.attributes[i].bufferIndex = key.slot[i] == CONSTANT_SLOT ? CONSTANT_SLOT : VERTEX_SLOT_BASE + key.slot[i];
        uses_constant |= key.slot[i] == CONSTANT_SLOT;
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
    MTLRenderPipelineColorAttachmentDescriptor *c = d.colorAttachments[0];
    c.pixelFormat = (MTLPixelFormat)key.color_format;
    if (key.blend) {
        c.blendingEnabled = YES;
        c.sourceRGBBlendFactor = c.sourceAlphaBlendFactor = blend_factor(key.blend_src);
        c.destinationRGBBlendFactor = c.destinationAlphaBlendFactor = blend_factor(key.blend_dst);
        c.rgbBlendOperation = c.alphaBlendOperation = blend_operation(key.blend_eq);
    }
    if (key.motion) {
        d.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA16Float;
        d.colorAttachments[1].writeMask = key.motion == 2 ? MTLColorWriteMaskAll : MTLColorWriteMaskNone;
    }
    d.depthAttachmentPixelFormat = (MTLPixelFormat)key.depth_format;
    d.rasterSampleCount = key.samples;

    NSError *error = nil;
    id<MTLRenderPipelineState> state = [s_device newRenderPipelineStateWithDescriptor:d error:&error];
    if (!state) complain("pipeline for program %u failed: %s", s_program, error.localizedDescription.UTF8String);
    p.pipelines[key] = state; // nil is cached too, so a failure is reported and attempted once
    return state;
}

id<MTLDepthStencilState> depth_state(bool has_depth)
{
    bool test = s_depth_test && has_depth;
    if (!test) { // OpenGL neither tests nor writes depth while the test is disabled
        if (!s_depth_off) s_depth_off = [s_device newDepthStencilStateWithDescriptor:[MTLDepthStencilDescriptor new]];
        return s_depth_off;
    }
    unsigned func = (s_depth_func - GL_NEVER) & 7;
    id<MTLDepthStencilState> __strong &state = s_depth_states[1][func][s_depth_mask];
    if (!state) {
        static const MTLCompareFunction compare[8] = {MTLCompareFunctionNever,        MTLCompareFunctionLess,
                                                      MTLCompareFunctionEqual,        MTLCompareFunctionLessEqual,
                                                      MTLCompareFunctionGreater,      MTLCompareFunctionNotEqual,
                                                      MTLCompareFunctionGreaterEqual, MTLCompareFunctionAlways};
        MTLDepthStencilDescriptor *d = [MTLDepthStencilDescriptor new];
        // With reversed depth nearer is larger, so the ordering comparisons swap.
        static const unsigned reversed[8] = {0, 4, 2, 6, 1, 5, 3, 7};
        d.depthCompareFunction = compare[s_reverse_depth ? reversed[func] : func];
        d.depthWriteEnabled = s_depth_mask;
        state = [s_device newDepthStencilStateWithDescriptor:d];
    }
    return state;
}
