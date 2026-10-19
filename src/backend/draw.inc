// This frame's clip space to the last one's (OpenGL's, column by column), once this
// frame's camera has been seen; until then, and after a gap, as if nothing had moved.
const float *frame_reproject()
{
    static float m[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    static uint64_t frame = ~0ull;
    if (frame == s_frame_serial) return m;
    static const float still[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    double inverse[16], r[16];
    if (s_camera.view_proj_frame != s_frame_serial) return still; // not seen yet: ask again at the next draw
    frame = s_frame_serial;
    memcpy(m, still, sizeof m);
    if (s_camera.last_view_proj_frame + 1 == s_frame_serial && mat_inverse(s_camera.view_proj, inverse)) {
        mat_mul(s_camera.last_view_proj, inverse, r);
        for (int i = 0; i < 16; i++) m[i] = (float)r[i];
    }
    return m;
}

// Binds everything a draw needs. Returns false when the draw has to be skipped.
bool prepare_draw()
{
    Program *p = find(s_programs, s_program);
    if (!p || !p->linked) return false;
    if (!p->functions_ready && !finish_compile(*p)) return false;
    if (!ensure_pass()) return false;

    PipelineKey key;
    memset(&key, 0, sizeof key);
    VertexBinding binding;
    if (!build_vertex_layout(*p, key, binding)) return false;
    key.color_format = (uint16_t)(s_pass_color ? s_pass_color.pixelFormat : MTLPixelFormatInvalid);
    key.depth_format = (uint16_t)(s_pass_depth ? s_pass_depth.pixelFormat : MTLPixelFormatInvalid);
    key.samples = (uint8_t)s_pass_samples;
    key.motion = (uint8_t)(!s_pass_motion ? 0 : p->reproject_offset >= 0 ? 2 : 1);
    if (s_blend) {
        key.blend = 1;
        key.blend_src = (uint16_t)s_blend_src;
        key.blend_dst = (uint16_t)s_blend_dst;
        key.blend_eq = (uint16_t)s_blend_eq;
    }
    id<MTLRenderPipelineState> pipeline;
    if (s_state_cache && s_last_key_program == s_program && s_last_key == key) {
        pipeline = s_last_pipeline;
    } else {
        pipeline = pipeline_for(*p, key);
        s_last_key_program = s_program;
        s_last_key = key;
        s_last_pipeline = pipeline;
    }
    if (!pipeline) return false;
    if (!already(s_set.pipeline == (__bridge void *)pipeline)) {
        [s_pass setRenderPipelineState:pipeline];
        s_set.pipeline = (__bridge void *)pipeline;
    }
    id<MTLDepthStencilState> depth = depth_state(s_pass_depth != nil);
    if (!already(s_set.depth_state == (__bridge void *)depth)) {
        [s_pass setDepthStencilState:depth];
        s_set.depth_state = (__bridge void *)depth;
    }

    // Textures are rendered upside down so that their rows are stored in OpenGL's
    // order (see wbm_yflip in shader_xlate.h). That mirrors the winding, and the
    // viewport then counts from the top like Metal's own.
    bool flip = !s_pass_screen;
    double target_h = s_pass_color ? (double)s_pass_color.height : (double)s_pass_depth.height;
    double k = s_pass_scale, vw = s_viewport.w * k, vh = s_viewport.h * k;
    double viewport[4] = {s_viewport.x * k, flip ? s_viewport.y * k : target_h - s_viewport.y * k - vh, vw, vh};
    if (!already(!memcmp(viewport, s_set.viewport, sizeof viewport))) {
        [s_pass setViewport:(MTLViewport){viewport[0], viewport[1], viewport[2], viewport[3], 0.0, 1.0}];
        memcpy(s_set.viewport, viewport, sizeof viewport);
    }
    int winding = flip ? MTLWindingClockwise : MTLWindingCounterClockwise;
    if (!already(s_set.winding == winding)) [s_pass setFrontFacingWinding:(MTLWinding)(s_set.winding = winding)];
    int cull = !s_cull ? MTLCullModeNone : s_cull_face == GL_FRONT ? MTLCullModeFront : MTLCullModeBack;
    if (!already(s_set.cull == cull)) [s_pass setCullMode:(MTLCullMode)(s_set.cull = cull)];
    int fill = s_polygon_mode == GL_LINE ? MTLTriangleFillModeLines : MTLTriangleFillModeFill;
    if (!already(s_set.fill == fill)) [s_pass setTriangleFillMode:(MTLTriangleFillMode)(s_set.fill = fill)];

    float samples = (float)s_pass_samples, yflip = flip ? -1.0f : 1.0f;
    if (p->samples_offset >= 0 && memcmp(p->uniform_data[1].data() + p->samples_offset, &samples, 4)) {
        memcpy(p->uniform_data[1].data() + p->samples_offset, &samples, 4);
        p->uniform_dirty[1] = true;
    }
    if (p->dyn_window_offset >= 0 && memcmp(p->uniform_data[1].data() + p->dyn_window_offset, s_dyn_window, 16)) {
        memcpy(p->uniform_data[1].data() + p->dyn_window_offset, s_dyn_window, 16);
        p->uniform_dirty[1] = true;
    }
    for (int stage = 0; stage < 2; stage++) {
        if (p->amb_offset[stage] >= 0 && memcmp(p->uniform_data[stage].data() + p->amb_offset[stage], s_amb, sizeof s_amb)) {
            memcpy(p->uniform_data[stage].data() + p->amb_offset[stage], s_amb, sizeof s_amb);
            p->uniform_dirty[stage] = true;
        }
    }
    Texture *diffuse = nullptr; // the diffuse texture says what the surface is
    if ((p->material_offset >= 0 || p->parallax_offset >= 0 || p->pbr_offset >= 0 || p->skin_offset[0] >= 0 || p->skin_offset[1] >= 0 || p->wind_offset >= 0 || p->leaf_sun_offset >= 0 || p->gaze_offset >= 0 || p->fur_offset >= 0 || s_debug_skin) && p->alpha_texture_location >= 0) {
        UniformLocation &l = p->locations[(size_t)p->alpha_texture_location];
        diffuse = (l.unit >= 0 && l.unit < MAX_UNITS) ? find(s_textures, s_bound_texture[l.unit]) : nullptr;
    }
    if (s_debug_skin && diffuse && diffuse->material == MAT_FLORA && (p->wind_offset < 0 || (p->skin_offset[0] < 0 && p->skin_offset[1] < 0 && p->leaf_sun_offset < 0)))
        complain("foliage drawn by program %u: wind %s, light through leaves %s", s_program, p->wind_offset >= 0 ? "yes" : "NO",
                 p->skin_offset[0] >= 0 || p->skin_offset[1] >= 0 || p->leaf_sun_offset >= 0 ? "yes" : "NO");
    bool unlit_leaf = diffuse && diffuse->material == MAT_FLORA && p->skin_offset[0] < 0 && p->skin_offset[1] < 0 && p->leaf_sun_offset < 0;
    if (s_debug_skin && diffuse && (unlit_leaf || (diffuse->material == MAT_SKIN && p->skin_offset[0] < 0 && p->skin_offset[1] < 0))) {
        // Skin drawn by a shader the scattering rewrite did not recognise: show how it lights.
        for (auto &sh : p->shaders) {
            std::string lines;
            size_t at = 0;
            while (at < sh->source.size() && lines.size() < 700) {
                size_t end = sh->source.find(';', at);
                if (end == std::string::npos) break;
                std::string st = sh->source.substr(at, end - at);
                if (st.find("Sun") != std::string::npos && st.find("uniform") == std::string::npos && st.find("varying") == std::string::npos) {
                    for (char &c : st)
                        if (c == '\n' || c == '\r') c = ' ';
                    lines += st + ";";
                }
                at = end + 1;
            }
            complain("%s without scattering, program %u %s: %s", unlit_leaf ? "foliage" : "skin", s_program, sh->type == GL_VERTEX_SHADER ? "vs" : "fs", lines.c_str());
        }
    }
    for (int stage = 0; stage < 2; stage++) {
        if (p->shell_offset[stage] < 0 || !memcmp(p->uniform_data[stage].data() + p->shell_offset[stage], s_shell_now, 16)) continue;
        memcpy(p->uniform_data[stage].data() + p->shell_offset[stage], s_shell_now, 16);
        p->uniform_dirty[stage] = true;
    }
    if (p->drift_offset >= 0 && memcmp(p->uniform_data[0].data() + p->drift_offset, s_drift_now, 16)) {
        memcpy(p->uniform_data[0].data() + p->drift_offset, s_drift_now, 16);
        p->uniform_dirty[0] = true;
    }
    if (p->fur_offset >= 0) {
        const float fur[4] = {diffuse && (diffuse->pbr || diffuse->hair) ? s_fur_strength : 0.0f, s_dyn_near[2] > 0 && s_dyn_bias > 0 ? s_dyn_bias_m / s_dyn_bias : 0.0f,
                              s_skin_backlight, diffuse && diffuse->hair ? s_texture_per_metre : -s_texture_per_metre};
        if (p->hair_colour_offset >= 0) {
            float colour[4] = {0, 0, 0, 0};
            if (diffuse && diffuse->hair_colour[3] > 0) {
                // In whatever form the shader gets the texture's own colour: decoded by the GPU, or as painted.
                for (int k = 0; k < 3; k++) colour[k] = s_shader_options.srgb_textures ? powf(diffuse->hair_colour[k], 2.2f) : diffuse->hair_colour[k];
                colour[3] = 1;
            }
            if (memcmp(p->uniform_data[1].data() + p->hair_colour_offset, colour, 16)) {
                memcpy(p->uniform_data[1].data() + p->hair_colour_offset, colour, 16);
                p->uniform_dirty[1] = true;
            }
        }
        if (p->horse_eye_offset >= 0) {
            const float eye[4] = {diffuse ? diffuse->horse_eye[0] : 0, diffuse ? diffuse->horse_eye[1] : 0, diffuse && diffuse->horse_eye[2] > 0 ? diffuse->horse_eye[2] : 1.0f,
                                  diffuse && diffuse->horse_eye[2] > 0 && s_shader_options.eyes ? 1.0f : 0.0f};
            if (memcmp(p->uniform_data[1].data() + p->horse_eye_offset, eye, 16)) {
                memcpy(p->uniform_data[1].data() + p->horse_eye_offset, eye, 16);
                p->uniform_dirty[1] = true;
            }
        }
        if (p->sheen_offset >= 0) {
            // The glTF materials' way: a highlight stretched along the hair and a sheen, where the pack has both maps.
            const float sheen[4] = {diffuse && diffuse->sheen && diffuse->hair && diffuse->pbr ? 1.0f : 0.0f, s_fur_anisotropy, s_fur_sheen, 0};
            if (memcmp(p->uniform_data[1].data() + p->sheen_offset, sheen, 16)) {
                memcpy(p->uniform_data[1].data() + p->sheen_offset, sheen, 16);
                p->uniform_dirty[1] = true;
            }
        }
        if (memcmp(p->uniform_data[1].data() + p->fur_offset, fur, 16)) {
            memcpy(p->uniform_data[1].data() + p->fur_offset, fur, 16);
            p->uniform_dirty[1] = true;
        }
    }
    if (p->gaze_offset >= 0) {
        float gaze[4] = {0, 0, 0, 0};
        float phase = 0; // a face's own, from where its head is, so that a crowd does not glance about as one
        if (p->u_world) {
            const float *w = (const float *)(p->uniform_data[0].data() + p->u_world->offset);
            phase = floorf(w[12] * 3.0f) * 12.9898f + floorf(w[13] * 3.0f) * 78.233f;
        }
        eye_gaze(s_wind_time, phase, gaze);
        gaze[2] = diffuse && diffuse->woman ? 1.0f : 0.0f;
        if (memcmp(p->uniform_data[1].data() + p->gaze_offset, gaze, 16)) {
            memcpy(p->uniform_data[1].data() + p->gaze_offset, gaze, 16);
            p->uniform_dirty[1] = true;
        }
    }
    if (p->leaf_sun_offset >= 0) {
        float sun[4] = {0, 0, 0, 0};
        // The direction to the sun in view space, from the camera worked out for this frame or the last.
        if (diffuse && diffuse->material == MAT_FLORA && s_sun_valid && s_camera.view_frame + 2 >= s_frame_serial) {
            double v[3], len = 0;
            for (int r = 0; r < 3; r++) {
                v[r] = -(s_camera.view[r][0] * s_sun_dir[0] + s_camera.view[r][1] * s_sun_dir[1] + s_camera.view[r][2] * s_sun_dir[2]);
                len += v[r] * v[r];
            }
            v[2] *= s_camera.forward_sign; // the third axis is kept pointing forwards; the shaders' own may point back
            if (len > 1e-12)
                for (int r = 0; r < 3; r++) sun[r] = (float)(v[r] / sqrt(len));
            sun[3] = s_foliage_backlight;
        }
        if (memcmp(p->uniform_data[0].data() + p->leaf_sun_offset, sun, 16)) {
            memcpy(p->uniform_data[0].data() + p->leaf_sun_offset, sun, 16);
            p->uniform_dirty[0] = true;
        }
    }
    if (p->wind_offset >= 0) {
        float wind[4] = {0, 0, 0, 0};
        if (diffuse && diffuse->material == MAT_FLORA && s_wind_strength > 0) {
            // A mesh in its own space has its foot at height 0. A batch placed in the
            // world (its vertices far from the origin) needs heights worked out for it.
            Buffer *vb = s_attribs[0].enabled ? find(s_buffers, s_attribs[0].buffer) : nullptr;
            bool batch = false;
            if (vb && vb->mtl && s_attribs[0].stride > 0 && s_attribs[0].type == GL_FLOAT && vb->mtl.length >= (NSUInteger)s_attribs[0].offset + 12) {
                float first[3];
                memcpy(first, (const uint8_t *)vb->mtl.contents + s_attribs[0].offset, 12);
                batch = fabsf(first[0]) + fabsf(first[1]) > 40.0f;
            }
            float phase = 0;
            if (p->u_world) {
                const float *w = (const float *)(p->uniform_data[0].data() + p->u_world->offset);
                phase = w[12] * 0.37f + w[13] * 0.71f;
            }
            // A palm's leaves are weighed whether batched or not: height above the
            // foot says nothing about them, their crown being at the top of a trunk.
            if (diffuse->radial_wind && vb && vb->mtl) batch = true;
            bool ready = !batch || weigh_foliage(*vb, diffuse->radial_wind);
            wind[0] = (float)s_wind_time, wind[1] = ready ? s_wind_strength : 0.0f, wind[2] = batch ? 1.0f : 0.0f, wind[3] = fmodf(phase, 6.2831853f);
            // Once a buffer's alpha holds heights, every shader has to be told, sway or not.
            if (vb && vb->wind_weights == (__bridge void *)vb->mtl) wind[2] = 1.0f;
        }
        if (memcmp(p->uniform_data[0].data() + p->wind_offset, wind, 16)) {
            memcpy(p->uniform_data[0].data() + p->wind_offset, wind, 16);
            p->uniform_dirty[0] = true;
        }
    }
    for (int stage = 0; stage < 2; stage++) {
        if (p->skin_offset[stage] < 0) continue;
        const float skin[4] = {s_shader_options.skin_scattering && diffuse && diffuse->material == MAT_SKIN ? 1.0f : 0.0f,
                               s_dyn_near[2] > 0 && s_dyn_bias > 0 ? s_dyn_bias_m / s_dyn_bias : 0.0f, s_skin_backlight,
                               diffuse && diffuse->material == MAT_FLORA ? s_foliage_backlight : 0.0f};
        if (memcmp(p->uniform_data[stage].data() + p->skin_offset[stage], skin, 16)) {
            memcpy(p->uniform_data[stage].data() + p->skin_offset[stage], skin, 16);
            p->uniform_dirty[stage] = true;
        }
    }
    if (p->pbr_offset >= 0) {
        const float has[4] = {diffuse && diffuse->pbr ? 1.0f : 0.0f, 0, 0, 0};
        if (memcmp(p->uniform_data[1].data() + p->pbr_offset, has, 16)) {
            memcpy(p->uniform_data[1].data() + p->pbr_offset, has, 16);
            p->uniform_dirty[1] = true;
        }
    }
    if (p->water_offset >= 0) {
        // [0]: the projection (p00, p11, a, b: clip z = a * w + b); [1]: the world's up in view space, and the far plane's
        // distance; [2] x: 1 when the scene's copy is this frame's and the camera known, y: steps of the search for what is reflected.
        float water[12] = {};
        bool known = s_water_reflections && s_water_scene && s_water_frame + (uint64_t)s_water_every > s_frame_serial && s_camera.projection_frame == s_frame_serial &&
                     s_camera.view_frame == s_frame_serial && fabs(s_camera.a) > 1e-9;
        if (known) {
            water[0] = (float)s_camera.p00, water[1] = (float)s_camera.p11, water[2] = (float)s_camera.a, water[3] = (float)s_camera.b;
            for (int r = 0; r < 3; r++) water[4 + r] = (float)s_camera.view[r][2];
            water[7] = fabs(1.0 - s_camera.a) > 1e-9 ? (float)fabs(s_camera.b / (1.0 - s_camera.a)) : 1250.0f;
            water[8] = 1.0f, water[9] = (float)s_water_steps;
        }
        if (memcmp(p->uniform_data[1].data() + p->water_offset, water, sizeof water)) {
            memcpy(p->uniform_data[1].data() + p->water_offset, water, sizeof water);
            p->uniform_dirty[1] = true;
        }
    }
    if (p->parallax_offset >= 0) { // stone, brick and the ground have depth; a painted shield has not
        // (And whatever a pack has given a map of its heights: w says that there is one to read.)
        bool mapped = diffuse && diffuse->depth;
        const float parallax[4] = {mapped || (diffuse && diffuse->material == MAT_STONE) ? s_parallax : 0.0f, s_parallax_sign[0], s_parallax_sign[1], mapped ? 1.0f : 0.0f};
        if (memcmp(p->uniform_data[1].data() + p->parallax_offset, parallax, 16)) {
            memcpy(p->uniform_data[1].data() + p->parallax_offset, parallax, 16);
            p->uniform_dirty[1] = true;
        }
    }
    if (p->material_offset >= 0) {
        int cls = diffuse ? diffuse->material : MAT_UNKNOWN;
        if (memcmp(p->uniform_data[1].data() + p->material_offset, k_material_params[cls], 16)) {
            memcpy(p->uniform_data[1].data() + p->material_offset, k_material_params[cls], 16);
            p->uniform_dirty[1] = true;
        }
    }
    if (p->dyn_near_offset >= 0 && memcmp(p->uniform_data[1].data() + p->dyn_near_offset, s_dyn_near, 16)) {
        memcpy(p->uniform_data[1].data() + p->dyn_near_offset, s_dyn_near, 16);
        p->uniform_dirty[1] = true;
    }
    if (p->dyn_bias_offset >= 0 && memcmp(p->uniform_data[1].data() + p->dyn_bias_offset, &s_dyn_bias, 4)) {
        memcpy(p->uniform_data[1].data() + p->dyn_bias_offset, &s_dyn_bias, 4);
        p->uniform_dirty[1] = true;
    }
    if (p->torch_offset >= 0 && p->torch_light_offset >= 0) {
        float lights[4][4] = {}, torch[4] = {};
        if (s_pass_scene) {
            note_torches(*p, lights);
            if (s_torch_frame == s_frame_serial) memcpy(torch, s_torch_params, sizeof torch);
        }
        if (memcmp(p->uniform_data[1].data() + p->torch_light_offset, lights, sizeof lights)) {
            memcpy(p->uniform_data[1].data() + p->torch_light_offset, lights, sizeof lights);
            p->uniform_dirty[1] = true;
        }
        if (memcmp(p->uniform_data[1].data() + p->torch_offset, torch, sizeof torch)) {
            memcpy(p->uniform_data[1].data() + p->torch_offset, torch, sizeof torch);
            p->uniform_dirty[1] = true;
        }
    }
    float limit = s_pass_screen ? 1.0f : 65000.0f;
    if (p->limit_offset >= 0 && memcmp(p->uniform_data[1].data() + p->limit_offset, &limit, 4)) {
        memcpy(p->uniform_data[1].data() + p->limit_offset, &limit, 4);
        p->uniform_dirty[1] = true;
    }
    if (p->grade_offset >= 0 && memcmp(p->uniform_data[1].data() + p->grade_offset, s_grade, 16)) {
        memcpy(p->uniform_data[1].data() + p->grade_offset, s_grade, 16);
        p->uniform_dirty[1] = true;
    }
    float headroom = s_headroom.load(std::memory_order_relaxed);
    if (p->headroom_offset >= 0 && memcmp(p->uniform_data[1].data() + p->headroom_offset, &headroom, 4)) {
        memcpy(p->uniform_data[1].data() + p->headroom_offset, &headroom, 4);
        p->uniform_dirty[1] = true;
    }
    if (p->yflip_offset >= 0 && memcmp(p->uniform_data[0].data() + p->yflip_offset, &yflip, 4)) {
        memcpy(p->uniform_data[0].data() + p->yflip_offset, &yflip, 4);
        p->uniform_dirty[0] = true;
    }
    // The scene and its depth pre-pass, shifted alike by this frame's fraction of a pixel; nothing else.
    float jitter[2] = {0, 0};
    if (s_temporal && (s_pass_scene || s_pass_prepass) && s_pass_color) {
        jitter[0] = 2.0f * s_jitter[0] / (float)s_pass_color.width;
        jitter[1] = -2.0f * s_jitter[1] / (float)s_pass_color.height; // clip space counts upwards, the picture's rows down
    }
    if (p->jitter_offset >= 0 && memcmp(p->uniform_data[0].data() + p->jitter_offset, jitter, 8)) {
        memcpy(p->uniform_data[0].data() + p->jitter_offset, jitter, 8);
        p->uniform_dirty[0] = true;
    }
    if (p->reproject_offset >= 0) {
        const float *reproject = frame_reproject();
        if (memcmp(p->uniform_data[0].data() + p->reproject_offset, reproject, 64)) {
            memcpy(p->uniform_data[0].data() + p->reproject_offset, reproject, 64);
            p->uniform_dirty[0] = true;
        }
    }
    // The noise of the alpha test moves on each frame only where frames are averaged; elsewhere it would crawl.
    float hash_frame = s_temporal ? (float)(s_frame_serial % 64) : 0.0f;
    if (p->hash_frame_offset >= 0 && memcmp(p->uniform_data[1].data() + p->hash_frame_offset, &hash_frame, 4)) {
        memcpy(p->uniform_data[1].data() + p->hash_frame_offset, &hash_frame, 4);
        p->uniform_dirty[1] = true;
    }
    float wind_before = (float)s_wind_time_before;
    if (p->wind_before_offset >= 0 && memcmp(p->uniform_data[0].data() + p->wind_before_offset, &wind_before, 4)) {
        memcpy(p->uniform_data[0].data() + p->wind_before_offset, &wind_before, 4);
        p->uniform_dirty[0] = true;
    }
    const XlStage *stages[2] = {&p->xl.vs, &p->xl.fs};
    for (int s = 0; s < 2; s++) {
        if (stages[s]->uniform_buffer_index < 0 || p->uniform_data[s].empty()) continue;
        if (!s_state_cache || p->uniform_dirty[s] || p->pushed_frame[s] != s_frame_serial) {
            p->pushed_buffer[s] = push_uniforms(p->uniform_data[s], &p->pushed_offset[s]);
            p->pushed_frame[s] = s_frame_serial;
            p->uniform_dirty[s] = false;
            s_call_stats.uniform_copies++;
        } else {
            s_call_stats.uniform_reuses++;
        }
        bind_buffer(s, p->pushed_buffer[s], p->pushed_offset[s], (NSUInteger)stages[s]->uniform_buffer_index);
    }
    for (UniformLocation &l : p->locations) {
        for (int s = 0; s < 2; s++) {
            int index = l.texture_index[s];
            if (index < 0) continue;
            Texture *t = (l.unit >= 0 && l.unit < MAX_UNITS) ? find(s_textures, s_bound_texture[l.unit]) : nullptr;
            id<MTLTexture> texture = t && t->mtl ? t->mtl : s_white;
            id<MTLSamplerState> sampler = t && t->mtl ? sampler_for(*t) : s_samplers.begin()->second;
            if (l.is_sky) { // not one of the game's textures
                texture = s_interior && s_room_atlas ? s_room_atlas : s_sky_atlas ? s_sky_atlas : s_white;
                if (s_sky_sampler) sampler = s_sky_sampler;
            } else if (l.is_scene_copy) {
                texture = s_water_scene ? s_water_scene : s_white;
                if (s_sky_sampler) sampler = s_sky_sampler;
            } else if (l.is_torch_map) {
                texture = torch_map(false);
                sampler = s_torch_sampler;
            } else if (l.is_dyn_map) {
                texture = s_dyn_map ? s_dyn_map : s_white;
                if (s_dyn_sampler) sampler = s_dyn_sampler;
            } else if (l.is_hair_map) {
                texture = diffuse && diffuse->hair ? diffuse->hair : s_white;
                sampler = diffuse && diffuse->mtl ? sampler_for(*diffuse) : s_samplers.begin()->second;
            } else if (l.is_sheen_map) {
                texture = diffuse && diffuse->sheen ? diffuse->sheen : s_white;
                sampler = diffuse && diffuse->mtl ? sampler_for(*diffuse) : s_samplers.begin()->second;
            } else if (l.is_depth_map) {
                texture = diffuse && diffuse->depth ? diffuse->depth : s_white;
                sampler = diffuse && diffuse->mtl ? sampler_for(*diffuse) : s_samplers.begin()->second;
            } else if (l.is_pbr_map || l.is_fur_map) {
                texture = diffuse && diffuse->pbr ? diffuse->pbr : s_white;
                sampler = diffuse && diffuse->mtl ? sampler_for(*diffuse) : s_samplers.begin()->second;
            } else if (l.srgb && t && t->mtl) {
                // The shader no longer decodes this sampler's colour itself.
                if (id<MTLTexture> view = srgb_view_for(*t)) texture = view;
                else complain("a texture without an sRGB format is bound to %s; it will look too bright", l.name.c_str());
            }
            // The pass onto the screen reads the scene enlarged, when it has been (see apply_upscale).
            if (s_pass_screen && t && texture == t->mtl && t->upscaled && t->upscaled_frame == s_frame_serial) texture = t->upscaled;
            bool tracked = index < 16;
            if (!already(tracked && s_set.texture[s][index] == (__bridge void *)texture)) {
                if (s == 0) [s_pass setVertexTexture:texture atIndex:(NSUInteger)index];
                else [s_pass setFragmentTexture:texture atIndex:(NSUInteger)index];
                if (tracked) s_set.texture[s][index] = (__bridge void *)texture;
            }
            if (!already(tracked && s_set.sampler[s][index] == (__bridge void *)sampler)) {
                if (s == 0) [s_pass setVertexSamplerState:sampler atIndex:(NSUInteger)index];
                else [s_pass setFragmentSamplerState:sampler atIndex:(NSUInteger)index];
                if (tracked) s_set.sampler[s][index] = (__bridge void *)sampler;
            }
        }
    }
    for (int slot = 0; slot < binding.slots; slot++) {
        Buffer *b = find(s_buffers, binding.buffer[slot]);
        if (!b || !b->mtl) return false;
        bind_buffer(0, b->mtl, binding.base[slot], (NSUInteger)(VERTEX_SLOT_BASE + slot));
    }
    bind_buffer(0, s_constant_attrib, 0, CONSTANT_SLOT);
    s_last.draws++;
    s_draw_binding = binding;
    // (Played back with the camera moved, the sky only turns while the scene also
    // shifts, so the sky's matrices do not give the scene's camera; a draw that
    // tests depth does. In the game the two agree and the sky comes first.)
    if (casting() && s_pass_scene && s_dyn_frame != s_frame_serial && !s_casters.empty() && p->u_wvp && p->u_world && (!s_replaying || s_depth_test))
        render_dynamic_shadows(*p);
    if (s_pass_scene && (((s_ao || s_contact_shadows || s_light_shafts || s_foliage_backlight > 0) && s_camera.view_frame != s_frame_serial) ||
                         ((s_motion_blur || s_temporal) && s_camera.view_proj_frame != s_frame_serial)))
        observe_camera(*p);
    if (s_callsites) {
        char kind[96];
        snprintf(kind, sizeof kind, "%s%s%s", pass_kind(), p->u_bones ? ", with bones" : "", s_blend ? ", blended" : "");
        note_callsite("draw into", kind);
    }
    return true;
}
