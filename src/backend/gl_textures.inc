// ---- textures ------------------------------------------------------------------------------------------------

void mtl_glGenTextures(GLsizei n, GLuint *textures) { gen_ids(s_next_texture, n, textures); REC(FN_glGenTextures).i(n).i(n > 0 ? textures[0] : 0); }

void mtl_glDeleteTextures(GLsizei n, const GLuint *textures)
{
    for (GLsizei i = 0; i < n; i++) s_textures.erase(textures[i]);
}

void mtl_glActiveTexture(GLenum texture) { REC(FN_glActiveTexture).i(texture); s_unit = texture - GL_TEXTURE0; }

void mtl_glBindTexture(GLenum target, GLuint texture)
{
    REC(FN_glBindTexture).i(target).i(texture);
    if (target != GL_TEXTURE_2D) {
        complain("texture target %s is not supported", E(target));
        return;
    }
    if (s_unit < MAX_UNITS) s_bound_texture[s_unit] = texture;
    if (texture) s_textures[texture]; // the object exists from its first bind
}

void mtl_glTexImage2D(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint, GLenum format,
                      GLenum type, const GLvoid *pixels)
{
    Texture *t = bound_texture();
    if (target != GL_TEXTURE_2D || !t || width <= 0 || height <= 0) return;
    FormatInfo f = pick_format(internalformat, format, type);
    if (f.format == MTLPixelFormatInvalid) {
        complain("unsupported texture format: internalformat %s, format %s, type %s", E_ifmt(internalformat), E(format), E(type));
        return;
    }
    REC(FN_glTexImage2D).i(target).i(level).i(internalformat).i(width).i(height).i(format).i(type).blob(
        pixels, (((size_t)width * (size_t)f.bytes_per_pixel + 3) & ~(size_t)3) * (size_t)height);
    if (level == 0) t->gl_internal = internalformat, t->gl_format = format, t->gl_type = type, t->gl_w = width, t->gl_h = height;
    @autoreleasepool {
        // Without data this is a render target: screen-sized ones get the screen's scale.
        double k = (!pixels && level == 0) ? scale_for(width, height) : 1;
        if (!ensure_storage(*t, f, level, (int)lround(width * k), (int)lround(height * k))) return;
        t->scale = k;
        if (level == 0) t->screen_sized = !pixels && is_screen_sized(width, height);
        if (!pixels) return;
        bool type_matches = type == GL_UNSIGNED_BYTE || type == GL_UNSIGNED_SHORT ||
                            (type == GL_FLOAT && (f.format == MTLPixelFormatR32Float || f.format == MTLPixelFormatRGBA32Float));
        if (!type_matches) {
            complain("texture data of type %s for internalformat %s is not converted", E(type), E_ifmt(internalformat));
            return;
        }
        size_t src_row = ((size_t)width * f.bytes_per_pixel + 3) & ~(size_t)3; // GL_UNPACK_ALIGNMENT is always 4
        MTLRegion region = MTLRegionMake2D(0, 0, (NSUInteger)width, (NSUInteger)height);
        if (f.expand_rgb) {
            std::vector<uint8_t> rgba((size_t)width * height * 4);
            for (int y = 0; y < height; y++) {
                const uint8_t *src = (const uint8_t *)pixels + src_row * y;
                uint8_t *dst = &rgba[(size_t)y * width * 4];
                for (int x = 0; x < width; x++, src += 3, dst += 4) dst[0] = src[0], dst[1] = src[1], dst[2] = src[2], dst[3] = 255;
            }
            [t->mtl replaceRegion:region mipmapLevel:(NSUInteger)level withBytes:rgba.data() bytesPerRow:(NSUInteger)width * 4];
        } else {
            [t->mtl replaceRegion:region mipmapLevel:(NSUInteger)level withBytes:pixels bytesPerRow:src_row];
        }
        if (level > t->top_level) {
            t->top_level = level;
            t->sampler = nil;
        }
    }
}

void mtl_glCompressedTexImage2D(GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLint,
                                GLsizei imageSize, const void *data)
{
    Texture *t = bound_texture();
    if (target != GL_TEXTURE_2D || !t || width <= 0 || height <= 0) return;
    FormatInfo f = pick_compressed_format(internalformat);
    if (f.format == MTLPixelFormatInvalid) {
        complain("unsupported compressed texture format %s", E(internalformat));
        return;
    }
    REC(FN_glCompressedTexImage2D).i(target).i(level).i(internalformat).i(width).i(height).blob(data, (size_t)std::max(imageSize, 0));
    @autoreleasepool {
        if (t->replaced && level > 0) return; // the pack's image came with its own smaller levels
        if (level == 0) t->replaced = false;
        if (!s_material_by_content.empty() && level == 0 && data) {
            int alpha = internalformat == 0x83F2 ? 3 : internalformat == 0x83F3 ? 5 : 0; // GL_COMPRESSED_RGBA_S3TC_DXT3_EXT, ..._DXT5_EXT
            auto it = s_material_by_content.find(content_key((uint32_t)width, (uint32_t)height, data, (size_t)imageSize, alpha));
            // The game loads some textures small first and in full later; whatever
            // was recognised stays until something else is.
            if (level == 0) t->pbr = nil;
            if (it != s_material_by_content.end()) {
                t->material = it->second.cls;
                t->radial_wind = s_texture_names[it->second.name].find("palm") != std::string::npos;
                t->woman = is_womans_face(s_texture_names[it->second.name]);
                horse_eye_for(s_texture_names[it->second.name], t->horse_eye);
                attach_pbr_map(*t, it->second);
                if (replace_from_pack(*t, it->second)) return;
            }
        }
        if (!ensure_storage(*t, f, level, width, height) || !data) return;
        if (s_materials && level == 0) {
            int alpha = internalformat == 0x83F2 ? 3 : internalformat == 0x83F3 ? 5 : 0;
            auto it = s_material_by_content.find(content_key((uint32_t)width, (uint32_t)height, data, (size_t)imageSize, alpha));
            s_material_stats.uploads++;
            if (it != s_material_by_content.end()) s_material_stats.matched++;
            else if (s_material_stats.strangers++ < 60 && imageSize >= 32) {
                const uint8_t *b = (const uint8_t *)data, *e = b + imageSize - 16;
                char first[40], last[40];
                for (int i = 0; i < 16; i++) snprintf(first + i * 2, 3, "%02x", b[i]), snprintf(last + i * 2, 3, "%02x", e[i]);
                wb_note("material classes: no file matches a %dx%d %s texture of %d bytes, first %s last %s", width, height, E(internalformat), imageSize, first, last);
                if (imageSize >= 16384 && s_material_stats.dumped < 12) { // a few of them whole, to compare with the files by hand
                    if (const char *dir = getenv("WBM_OUT")) {
                        char path[1200];
                        snprintf(path, sizeof path, "%s/unmatched_%02d_%dx%d_%s.bin", dir, s_material_stats.dumped++, width, height, E(internalformat));
                        if (FILE *out = fopen(path, "wb")) {
                            fwrite(data, 1, (size_t)imageSize, out);
                            fclose(out);
                        }
                    }
                }
            }
            if (t->material) s_material_stats.recognised++;
        }
        size_t blocks_w = ((size_t)width + 3) / 4, blocks_h = ((size_t)height + 3) / 4;
        if ((size_t)imageSize < blocks_w * blocks_h * f.block_bytes) {
            complain("compressed texture level is shorter than its size requires");
            return;
        }
        [t->mtl replaceRegion:MTLRegionMake2D(0, 0, (NSUInteger)width, (NSUInteger)height)
                  mipmapLevel:(NSUInteger)level
                        slice:0
                    withBytes:data
                  bytesPerRow:blocks_w * f.block_bytes
                bytesPerImage:0];
        if (level > t->top_level) {
            t->top_level = level;
            t->sampler = nil;
        }
    }
}

static void set_texture_parameter(GLenum pname, GLint value, GLfloat fvalue)
{
    Texture *t = bound_texture();
    if (!t) return;
    switch (pname) {
    case GL_TEXTURE_MIN_FILTER: t->min_filter = (GLenum)value; break;
    case GL_TEXTURE_MAG_FILTER: t->mag_filter = (GLenum)value; break;
    case GL_TEXTURE_WRAP_S: t->wrap_s = (GLenum)value; break;
    case GL_TEXTURE_WRAP_T: t->wrap_t = (GLenum)value; break;
    case GL_TEXTURE_MAX_LEVEL: t->max_level = value; break;
    case GL_TEXTURE_MAX_ANISOTROPY_EXT: t->anisotropy = fvalue; break;
    default: complain("texture parameter %s is ignored", E(pname)); return;
    }
    t->sampler = nil;
}

void mtl_glTexParameteri(GLenum, GLenum pname, GLint param) { REC(FN_glTexParameteri).i(pname).i(param); set_texture_parameter(pname, param, (GLfloat)param); }
void mtl_glTexParameterf(GLenum, GLenum pname, GLfloat param) { REC(FN_glTexParameterf).i(pname).f(param); set_texture_parameter(pname, (GLint)param, param); }

void mtl_glGenerateMipmap(GLenum)
{
    REC(FN_glGenerateMipmap);
    Texture *t = bound_texture();
    if (!t || !t->mtl || t->mtl.mipmapLevelCount < 2) return;
    @autoreleasepool {
        end_pass(); // the texture may have just been rendered to
        id<MTLBlitCommandEncoder> blit = [command_buffer() blitCommandEncoder];
        [blit generateMipmapsForTexture:t->mtl];
        [blit endEncoding];
        t->top_level = (int)t->mtl.mipmapLevelCount - 1;
        t->sampler = nil;
    }
}

// The only glTexEnv call the engine makes sets a LOD bias of 0; glPixelStorei
// only ever sets the unpack alignment to 4, which the uploads assume.
void mtl_glTexEnvf(GLenum, GLenum pname, GLfloat param)
{
    REC(FN_glTexEnvf).i(pname).f(param);
    if (pname != GL_TEXTURE_LOD_BIAS || param != 0) complain("glTexEnvf(%s, %g) is ignored", E(pname), param);
}

void mtl_glPixelStorei(GLenum pname, GLint param)
{
    REC(FN_glPixelStorei).i(pname).i(param);
    if (pname != GL_UNPACK_ALIGNMENT || param != 4) complain("glPixelStorei(%s, %d) is ignored", E(pname), param);
}
