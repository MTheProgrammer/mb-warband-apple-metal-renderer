// ---- draws -----------------------------------------------------------------------------------------------------------

void mtl_glDrawArrays(GLenum mode, GLint first, GLsizei count)
{
    REC(FN_glDrawArrays).i(mode).i(first).i(count);
    if (count <= 0) return;
    @autoreleasepool {
        if (mode == GL_QUADS) {
            int quads = count / 4;
            if (quads > 65535 / 4) {
                complain("glDrawArrays(GL_QUADS) with %d vertices is not supported", count);
                return;
            }
            if (!quads || !prepare_draw()) return;
            [s_pass drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                               indexCount:(NSUInteger)quads * 6
                                indexType:MTLIndexTypeUInt16
                              indexBuffer:quad_indices(quads)
                        indexBufferOffset:0
                            instanceCount:1
                               baseVertex:first
                             baseInstance:0];
            return;
        }
        MTLPrimitiveType type;
        if (!primitive_type(mode, &type) || !prepare_draw()) return;
        [s_pass drawPrimitives:type vertexStart:(NSUInteger)first vertexCount:(NSUInteger)count];
    }
}

void mtl_glDrawElements(GLenum mode, GLsizei count, GLenum type, const GLvoid *indices)
{
    REC(FN_glDrawElements).i(mode).i(count).i(type).i((int64_t)(uintptr_t)indices);
    if (count <= 0) return;
    @autoreleasepool {
        MTLPrimitiveType primitive;
        if (!primitive_type(mode, &primitive)) return;
        if (type != GL_UNSIGNED_SHORT && type != GL_UNSIGNED_INT) {
            complain("index type %s is not supported", E(type));
            return;
        }
        Buffer *b = find(s_buffers, s_element_buffer);
        if (!b || !b->mtl) {
            complain("glDrawElements without an element buffer is not supported");
            return;
        }
        // A character's ground decal would double up with its real shadow.
        if (s_dyn_shadows && s_dyn_window[2] > 0) {
            Program *p = find(s_programs, s_program);
            if (p && p->is_blob_shadow && s_pass_scene) return;
        }
        NSUInteger offset = (NSUInteger)(uintptr_t)indices, index_size = type == GL_UNSIGNED_SHORT ? 2 : 4;
        if (offset + (NSUInteger)count * index_size > b->mtl.length) {
            complain("glDrawElements reads past the end of its element buffer");
            return;
        }
        // A coat near the camera is drawn again in layers, as fur (see k_fur_function).
        // How much of the texture a metre of this mesh covers, from its triangles; once per vertex buffer.
        if (s_fur_strength > 0 && s_pass_scene) {
            Program *fp = find(s_programs, s_program);
            Buffer *vb = fp && fp->fur_offset >= 0 && s_attribs[0].enabled ? find(s_buffers, s_attribs[0].buffer) : nullptr;
            const Attrib &pa = s_attribs[0], &ta = s_attribs[4];
            if (vb && vb->mtl && vb->scale_of != (__bridge void *)vb->mtl) {
                vb->scale_of = (__bridge void *)vb->mtl;
                vb->texture_per_metre = 0.3f;
                if (ta.enabled && ta.buffer == pa.buffer && pa.type == GL_FLOAT && ta.type == GL_FLOAT && pa.stride > 0) {
                    const uint8_t *vertices = (const uint8_t *)vb->mtl.contents, *elements = (const uint8_t *)b->mtl.contents + (uintptr_t)indices;
                    size_t stride = (size_t)pa.stride, n = vb->mtl.length / stride, width = type == GL_UNSIGNED_SHORT ? 2 : 4;
                    double in_texture = 0, in_space = 0;
                    for (GLsizei i = 0; i + 2 < std::min<GLsizei>(count, 3000); i += 3) {
                        uint32_t v[3];
                        for (int k = 0; k < 3; k++) v[k] = width == 2 ? *(const uint16_t *)(elements + (size_t)(i + k) * 2) : *(const uint32_t *)(elements + (size_t)(i + k) * 4);
                        if (v[0] >= n || v[1] >= n || v[2] >= n) continue;
                        for (int k = 0; k < 3; k++) {
                            float p0[3], p1[3], t0[2], t1[2];
                            memcpy(p0, vertices + v[k] * stride + pa.offset, 12), memcpy(p1, vertices + v[(k + 1) % 3] * stride + pa.offset, 12);
                            memcpy(t0, vertices + v[k] * stride + ta.offset, 8), memcpy(t1, vertices + v[(k + 1) % 3] * stride + ta.offset, 8);
                            in_space += sqrt((p1[0] - p0[0]) * (p1[0] - p0[0]) + (p1[1] - p0[1]) * (p1[1] - p0[1]) + (p1[2] - p0[2]) * (p1[2] - p0[2]));
                            in_texture += hypot(t1[0] - t0[0], t1[1] - t0[1]);
                        }
                    }
                    if (in_space > 1e-6 && in_texture > 1e-6) vb->texture_per_metre = (float)(in_texture / in_space);
                }
            }
            if (vb) s_texture_per_metre = vb->texture_per_metre;
        }
        int shells = 0;
        bool long_hair = false; // the coat has a hair map, which says where mane and tail are
        if (s_fur_shells > 0 && s_fur_strength > 0 && s_pass_scene && !s_blend) {
            Program *fp = find(s_programs, s_program);
            if (fp && fp->shell_offset[0] >= 0 && fp->shell_offset[1] >= 0 && fp->u_wvp && fp->alpha_texture_location >= 0) {
                UniformLocation &l = fp->locations[(size_t)fp->alpha_texture_location];
                Texture *coat = (l.unit >= 0 && l.unit < MAX_UNITS) ? find(s_textures, s_bound_texture[l.unit]) : nullptr;
                float distance; // of the object's origin ahead of the camera
                memcpy(&distance, fp->uniform_data[0].data() + fp->u_wvp->offset + 60, 4);
                if (coat && (coat->pbr || coat->hair) && distance > 0 && distance < s_fur_range) {
                    shells = std::max(2, (int)lroundf((float)s_fur_shells * std::min(1.0f, 1.6f * (1.0f - distance / s_fur_range))));
                    long_hair = coat->hair != nil && s_hair_length > 0;
                }
            }
        }
        float clock = 1.0f + (float)fmod(s_wind_time, 600.0); // above 0.5, which is all the coat's layers ask of it
        const float surface[4] = {0, 0, s_fur_density, shells ? clock : 0.0f};
        memcpy(s_shell_now, surface, sizeof surface);
        if (s_water_reflections && s_pass_scene && s_pass_color) {
            Program *wp = find(s_programs, s_program);
            if (wp && wp->water_offset >= 0 && s_water_frame + (uint64_t)s_water_every <= s_frame_serial) {
                id<MTLTexture> scene = s_pass_color; // what later passes read: with multisampling, filled when the pass ends
                end_pass();
                if (!s_water_scene || s_water_scene.width != scene.width || s_water_scene.height != scene.height || s_water_scene.pixelFormat != scene.pixelFormat) {
                    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:scene.pixelFormat width:scene.width height:scene.height mipmapped:NO];
                    td.usage = MTLTextureUsageShaderRead;
                    td.storageMode = MTLStorageModePrivate;
                    s_water_scene = [s_device newTextureWithDescriptor:td];
                }
                id<MTLBlitCommandEncoder> blit = [command_buffer() blitCommandEncoder];
                [blit copyFromTexture:scene sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(scene.width, scene.height, 1)
                            toTexture:s_water_scene destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
                [blit endEncoding];
                s_water_frame = s_frame_serial;
                s_water_copies++;
            }
        }
        if (!prepare_draw()) return;
        MTLIndexType index_type = type == GL_UNSIGNED_SHORT ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
        [s_pass drawIndexedPrimitives:primitive indexCount:(NSUInteger)count indexType:index_type indexBuffer:b->mtl indexBufferOffset:offset];
        if (long_hair && s_dyn_shadows && !s_hair_strands && primitive == MTLPrimitiveTypeTriangle) { // strands cast for themselves
            Program *fp = find(s_programs, s_program);
            UniformLocation &l = fp->locations[(size_t)fp->alpha_texture_location];
            if (Texture *coat = (l.unit >= 0 && l.unit < MAX_UNITS) ? find(s_textures, s_bound_texture[l.unit]) : nullptr)
                record_hair_casters(*fp, *coat, (NSUInteger)count, index_type, b->mtl, offset);
        }
        // The coat's layers are tested against the depth of the body but leave
        // none of their own: ambient occlusion and contact shadows, which work
        // from depth, would take the gaps between hairs for pits and speckle the horse.
        // Each layer is blended over what lies beneath it, by how much of the
        // pixel its hairs cover (wbm_shell_cover): the layers go from the body outwards.
        bool depth_mask = s_depth_mask, blend = s_blend;
        GLenum blend_src = s_blend_src, blend_dst = s_blend_dst, blend_eq = s_blend_eq;
        if (shells) s_depth_mask = false, s_blend = true, s_blend_src = GL_SRC_ALPHA, s_blend_dst = GL_ONE_MINUS_SRC_ALPHA, s_blend_eq = GL_FUNC_ADD;
        for (int layer = 1; layer <= shells; layer++) {
            float share = (float)layer / (float)shells;
            const float now[4] = {s_fur_length * share, share, s_fur_density, clock};
            memcpy(s_shell_now, now, sizeof now);
            if (!prepare_draw()) break;
            [s_pass drawIndexedPrimitives:primitive indexCount:(NSUInteger)count indexType:index_type indexBuffer:b->mtl indexBufferOffset:offset];
            s_shell_draws++;
        }
        s_depth_mask = depth_mask; // mane and tail do leave depth: their layers hang far from the body and must hide one another
        // People's hair as strands over the hair mesh, near the camera (see add_hair_strands).
        if (s_people_hair && s_pass_scene && primitive == MTLPrimitiveTypeTriangle) {
            Program *hp = find(s_programs, s_program);
            Buffer *vb = s_attribs[0].enabled ? find(s_buffers, s_attribs[0].buffer) : nullptr;
            const uint32_t needed = 1u << 0 | 1u << 1 | 1u << 2 | 1u << 4 | 1u << 6;
            // The hair shader is the one that takes wbm_shell in both stages without being a coat's or having bones.
            bool hair = hp && hp->shell_offset[0] >= 0 && hp->shell_offset[1] >= 0 && hp->fur_offset < 0 && !hp->u_bones && hp->u_wvp && (hp->input_mask & needed) == needed;
            float distance = 0;
            if (hair) memcpy(&distance, hp->uniform_data[0].data() + hp->u_wvp->offset + 60, 4);
            if (s_people_hair_frame != s_frame_serial) s_people_hair_frame = s_frame_serial, s_people_hair_drawn = 0;
            if (hair && vb && vb->mtl && distance > 0 && distance < s_people_hair_range && s_people_hair_drawn < s_people_hair_most) {
                const StrandMesh &strands = build_strands(mesh_id((NSUInteger)count, offset) ^ 0x5bd1e995a4c7f3d1ull, (const uint8_t *)vb->mtl.contents, vb->mtl.length,
                                                          (const uint8_t *)b->mtl.contents + offset, (size_t)count, type == GL_UNSIGNED_SHORT ? 2 : 4, nil);
                if (strands.count) {
                    Attrib attribs[MAX_ATTRIBS];
                    memcpy(attribs, s_attribs, sizeof attribs);
                    bool cull = s_cull, was_blend = s_blend, depth_mask = s_depth_mask;
                    GLenum src = s_blend_src, dst = s_blend_dst, eq = s_blend_eq;
                    for (Attrib &a : s_attribs)
                        if (a.enabled && a.buffer == attribs[0].buffer) a.buffer = strands.buffer;
                    s_cull = false;
                    if (hp->u_world) drift_of(mesh_id((NSUInteger)count, offset), (const float *)(hp->uniform_data[0].data() + hp->u_world->offset), s_drift_now);
                    for (int soft = 0; soft < 2; soft++) { // what is solid first, leaving depth; then the soft edges, blended, leaving none
                        s_blend = soft != 0, s_depth_mask = soft == 0;
                        if (soft) s_blend_src = GL_SRC_ALPHA, s_blend_dst = GL_ONE_MINUS_SRC_ALPHA, s_blend_eq = GL_FUNC_ADD;
                        const float now[4] = {-s_people_hair_length, s_wind_strength, soft ? 1.0f : -1.0f, clock}; // y: how hard the wind blows
                        memcpy(s_shell_now, now, sizeof now);
                        if (prepare_draw())
                            [s_pass drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:strands.count indexType:MTLIndexTypeUInt32 indexBuffer:strands.indices indexBufferOffset:0];
                        s_shell_draws++;
                    }
                    if (s_dyn_shadows) record_strand_caster(*hp, strands, 2, s_people_hair_length, 0, s_drift_now);
                    memset(s_drift_now, 0, sizeof s_drift_now);
                    memcpy(s_attribs, attribs, sizeof attribs);
                    s_cull = cull, s_blend = was_blend, s_depth_mask = depth_mask, s_blend_src = src, s_blend_dst = dst, s_blend_eq = eq;
                    s_people_hair_drawn++;
                }
            }
        }
        // Mane and tail as strands (see build_strands): one more draw of the same program, over ribbons.
        if (long_hair && s_hair_strands && primitive == MTLPrimitiveTypeTriangle) {
            Program *fp = find(s_programs, s_program);
            UniformLocation &l = fp->locations[(size_t)fp->alpha_texture_location];
            Texture *coat = (l.unit >= 0 && l.unit < MAX_UNITS) ? find(s_textures, s_bound_texture[l.unit]) : nullptr;
            Buffer *vb = s_attribs[0].enabled ? find(s_buffers, s_attribs[0].buffer) : nullptr;
            const uint32_t needed = 1u << 0 | 1u << 1 | 1u << 2 | 1u << 4 | 1u << 5 | 1u << 7 | 1u << 8;
            if (coat && coat->hair && vb && vb->mtl && (fp->input_mask & needed) == needed) {
                const StrandMesh &strands = build_strands(mesh_id((NSUInteger)count, offset) ^ (uint64_t)(uintptr_t)(__bridge void *)coat->hair, (const uint8_t *)vb->mtl.contents,
                                                          vb->mtl.length, (const uint8_t *)b->mtl.contents + offset, (size_t)count, type == GL_UNSIGNED_SHORT ? 2 : 4, coat->hair);
                if (strands.count) {
                    Attrib attribs[MAX_ATTRIBS];
                    memcpy(attribs, s_attribs, sizeof attribs);
                    bool cull = s_cull, was_blend = s_blend;
                    GLenum src = s_blend_src, dst = s_blend_dst, eq = s_blend_eq;
                    for (Attrib &a : s_attribs)
                        if (a.enabled && a.buffer == attribs[0].buffer) a.buffer = strands.buffer;
                    // What is solid first, leaving depth; then the soft edges, blended, leaving none (the sign of z tells the shader which).
                    bool depth_mask = s_depth_mask;
                    s_cull = false;
                    if (fp->u_world) drift_of(mesh_id((NSUInteger)count, offset), (const float *)(fp->uniform_data[0].data() + fp->u_world->offset), s_drift_now);
                    for (int soft = 0; soft < 2; soft++) {
                        if (soft) s_blend = true, s_blend_src = GL_SRC_ALPHA, s_blend_dst = GL_ONE_MINUS_SRC_ALPHA, s_blend_eq = GL_FUNC_ADD, s_depth_mask = false;
                        const float now[4] = {-std::max(s_mane_length, 0.001f), s_tail_length, soft ? s_fur_density : -s_fur_density, clock};
                        memcpy(s_shell_now, now, sizeof now);
                        if (prepare_draw())
                            [s_pass drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:strands.count indexType:MTLIndexTypeUInt32 indexBuffer:strands.indices indexBufferOffset:0];
                        s_shell_draws++;
                    }
                    if (s_dyn_shadows) record_strand_caster(*fp, strands, 1, s_mane_length, s_tail_length, s_drift_now);
                    memset(s_drift_now, 0, sizeof s_drift_now);
                    memcpy(s_attribs, attribs, sizeof attribs);
                    s_depth_mask = depth_mask;
                    s_cull = cull, s_blend = was_blend, s_blend_src = src, s_blend_dst = dst, s_blend_eq = eq;
                    long_hair = false; // no layers as well
                }
            }
        }
        // Mane and tail: layers of their own, reaching much farther out (see wbm_shell_bare).
        for (int layer = 1; long_hair && layer <= shells * 2; layer++) {
            float share = (float)layer / (float)(shells * 2);
            const float now[4] = {s_hair_length * share, share, -s_fur_density, clock};
            memcpy(s_shell_now, now, sizeof now);
            if (!prepare_draw()) break;
            [s_pass drawIndexedPrimitives:primitive indexCount:(NSUInteger)count indexType:index_type indexBuffer:b->mtl indexBufferOffset:offset];
            s_shell_draws++;
        }
        s_blend = blend, s_blend_src = blend_src, s_blend_dst = blend_dst, s_blend_eq = blend_eq;
        if (shells) memset(s_shell_now, 0, sizeof s_shell_now);
        if ((casting() || s_motion_blur) && s_pass_prepass && primitive == MTLPrimitiveTypeTriangle) record_caster((NSUInteger)count, index_type, b->mtl, offset, false);
        if (casting() && s_pass_scene && primitive == MTLPrimitiveTypeTriangle && s_depth_test && s_depth_mask)
            record_caster((NSUInteger)count, index_type, b->mtl, offset, true);
        if (s_dyn_shadows && s_pass_shadow_map) record_scenery((NSUInteger)count, offset);
        if (casting() && s_pass_scene && !s_depth_test) record_overlay((NSUInteger)count, offset);
        // So is a marker that floats over a soldier's head: the pre-pass draws it as a solid thing, without a
        // texture to cut it out by, and the main pass then blends it in, unlit. (Soldiers fading in at a
        // distance are blended too, but lit.)
        if (casting() && s_pass_scene && s_depth_test && s_blend && s_prepass_solid.count(mesh_id((NSUInteger)count, offset))) {
            Program *mp = find(s_programs, s_program);
            if (mp && !(mp->input_mask & 2u)) record_overlay((NSUInteger)count, offset); // and it is unlit: its shader takes no normals
        }
        // The engine draws the sky first, with the depth test off. That draw is the one to capture.
        if ((s_shader_options.pbr_sky || s_sky_ambient) && s_pass_scene && !s_depth_test && s_sky_frame != s_frame_serial)
            capture_sky(primitive, (NSUInteger)count, index_type, b->mtl, offset);
    }
}
