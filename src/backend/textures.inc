// Texture formats, storage and samplers.
// Included by metal_backend.mm, inside namespace wbm.

// ---- textures and samplers ---------------------------------------------------------------

struct FormatInfo {
    MTLPixelFormat format = MTLPixelFormatInvalid;
    int bytes_per_pixel = 0;      // of the data the game supplies; 0 for compressed
    int block_bytes = 0;          // compressed formats
    bool expand_rgb = false;      // 24-bit RGB data has to be widened to RGBA
    MTLTextureSwizzleChannels swizzle = MTLTextureSwizzleChannelsDefault;
};

FormatInfo pick_format(GLint internalformat, GLenum format, GLenum type)
{
    FormatInfo f;
    const MTLTextureSwizzleChannels luminance = {MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleOne};
    switch (internalformat) { // sized formats, used for render targets
    case GL_RG16F: f.format = MTLPixelFormatRG16Float; f.bytes_per_pixel = 4; return f;
    case GL_RGBA16F: f.format = MTLPixelFormatRGBA16Float; f.bytes_per_pixel = 8; return f;
    case GL_RGBA32F: f.format = MTLPixelFormatRGBA32Float; f.bytes_per_pixel = 16; return f;
    case GL_R8: f.format = MTLPixelFormatR8Unorm; f.bytes_per_pixel = 1; return f;
    case GL_R16F: f.format = MTLPixelFormatR16Float; f.bytes_per_pixel = 2; return f;
    case GL_R32F: f.format = MTLPixelFormatR32Float; f.bytes_per_pixel = 4; return f;
    case GL_DEPTH_COMPONENT:
    case GL_DEPTH_COMPONENT16:
    case GL_DEPTH_COMPONENT24:
    case GL_DEPTH_COMPONENT32: f.format = MTLPixelFormatDepth32Float; f.bytes_per_pixel = 4; return f;
    }
    if (type == GL_UNSIGNED_BYTE) {
        switch (format) {
        case GL_BGRA: f.format = MTLPixelFormatBGRA8Unorm; f.bytes_per_pixel = 4; return f;
        case GL_RGBA: f.format = MTLPixelFormatRGBA8Unorm; f.bytes_per_pixel = 4; return f;
        case GL_RGB: f.format = MTLPixelFormatRGBA8Unorm; f.bytes_per_pixel = 3; f.expand_rgb = true; return f;
        case GL_RED: f.format = MTLPixelFormatR8Unorm; f.bytes_per_pixel = 1; return f;
        case GL_ALPHA: f.format = MTLPixelFormatA8Unorm; f.bytes_per_pixel = 1; return f;
        case GL_LUMINANCE: f.format = MTLPixelFormatR8Unorm; f.bytes_per_pixel = 1; f.swizzle = luminance; return f;
        case GL_LUMINANCE_ALPHA:
            f.format = MTLPixelFormatRG8Unorm;
            f.bytes_per_pixel = 2;
            f.swizzle = {MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleGreen};
            return f;
        }
    }
    if (type == GL_UNSIGNED_SHORT && format == GL_LUMINANCE) {
        f.format = MTLPixelFormatR16Unorm;
        f.bytes_per_pixel = 2;
        f.swizzle = luminance;
    }
    return f;
}

FormatInfo pick_compressed_format(GLenum internalformat)
{
    FormatInfo f;
    switch (internalformat) {
    case GL_COMPRESSED_RGB_S3TC_DXT1_EXT:
    case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT: f.format = MTLPixelFormatBC1_RGBA; f.block_bytes = 8; break;
    case GL_COMPRESSED_RGBA_S3TC_DXT3_EXT: f.format = MTLPixelFormatBC2_RGBA; f.block_bytes = 16; break;
    case GL_COMPRESSED_RGBA_S3TC_DXT5_EXT: f.format = MTLPixelFormatBC3_RGBA; f.block_bytes = 16; break;
    }
    return f;
}

int mip_count(int w, int h)
{
    int n = 1;
    while (w > 1 || h > 1) {
        w = std::max(1, w / 2);
        h = std::max(1, h / 2);
        n++;
    }
    return n;
}

// Makes sure the texture has storage whose level `level` is w x h in the given
// format, recreating it when the game respecifies it differently.
bool ensure_storage(Texture &t, const FormatInfo &f, int level, int w, int h)
{
    if (level < 0 || level > 15) { // GL_INVALID_VALUE; shifting by it would be undefined
        complain("mip level %d is out of range", level);
        return false;
    }
    if (t.mtl && t.format == f.format && level < (int)t.mtl.mipmapLevelCount &&
        std::max(1, (int)t.mtl.width >> level) == w && std::max(1, (int)t.mtl.height >> level) == h)
        return true;
    if (t.mtl && level != 0) {
        complain("mip level %d (%dx%d) does not fit its %dx%d texture", level, w, h, (int)t.mtl.width, (int)t.mtl.height);
        return false;
    }
    int base_w = w << level, base_h = h << level;
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f.format
                                                                                 width:base_w
                                                                                height:base_h
                                                                             mipmapped:NO];
    d.mipmapLevelCount = mip_count(base_w, base_h);
    // Any uncompressed texture may later be attached to a framebuffer, except
    // the luminance ones: Metal does not allow rendering to a swizzled texture.
    const MTLTextureSwizzleChannels none = MTLTextureSwizzleChannelsDefault;
    bool swizzled = memcmp(&f.swizzle, &none, sizeof none) != 0;
    d.usage = (f.block_bytes || swizzled) ? MTLTextureUsageShaderRead : MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    if (s_shader_options.srgb_textures && !swizzled) d.usage |= MTLTextureUsagePixelFormatView;
    d.swizzle = f.swizzle;
    t.mtl = [s_device newTextureWithDescriptor:d];
    t.format = f.format;
    t.top_level = -1;
    t.sampler = nil;
    t.ms = nil;
    t.srgb_view = nil;
    return t.mtl != nil;
}

Texture *bound_texture()
{
    return s_unit < MAX_UNITS ? find(s_textures, s_bound_texture[s_unit]) : nullptr;
}

MTLSamplerAddressMode address_mode(GLenum wrap)
{
    switch (wrap) {
    case GL_REPEAT: return MTLSamplerAddressModeRepeat;
    case GL_MIRRORED_REPEAT: return MTLSamplerAddressModeMirrorRepeat;
    case GL_CLAMP_TO_BORDER: return MTLSamplerAddressModeClampToBorderColor;
    default: return MTLSamplerAddressModeClampToEdge;
    }
}

// The texture seen through the sRGB twin of its format, or nil if it has none.
id<MTLTexture> srgb_view_for(Texture &t)
{
    if (t.srgb_view) return t.srgb_view;
    MTLPixelFormat format;
    switch (t.format) {
    case MTLPixelFormatBC1_RGBA: format = MTLPixelFormatBC1_RGBA_sRGB; break;
    case MTLPixelFormatBC2_RGBA: format = MTLPixelFormatBC2_RGBA_sRGB; break;
    case MTLPixelFormatBC3_RGBA: format = MTLPixelFormatBC3_RGBA_sRGB; break;
    case MTLPixelFormatRGBA8Unorm: format = MTLPixelFormatRGBA8Unorm_sRGB; break;
    case MTLPixelFormatBGRA8Unorm: format = MTLPixelFormatBGRA8Unorm_sRGB; break;
    default: return nil;
    }
    if (!(t.mtl.usage & MTLTextureUsagePixelFormatView)) return nil;
    return t.srgb_view = [t.mtl newTextureViewWithPixelFormat:format];
}

id<MTLSamplerState> sampler_for(Texture &t)
{
    if (t.sampler) return t.sampler;
    bool mipmapped = t.min_filter >= GL_NEAREST_MIPMAP_NEAREST && t.min_filter <= GL_LINEAR_MIPMAP_LINEAR;
    int top = mipmapped ? std::max(0, std::min(t.max_level, t.top_level)) : 0;
    int aniso = (int)std::max(1.0f, std::min(16.0f, t.anisotropy));
    uint64_t key = (uint64_t)t.min_filter | (uint64_t)t.mag_filter << 16 | (uint64_t)(t.wrap_s & 0xff) << 32 |
                   (uint64_t)(t.wrap_s >> 8 & 0xf) << 40 | (uint64_t)(t.wrap_t & 0xff) << 44 | (uint64_t)(t.wrap_t >> 8 & 0xf) << 52 |
                   (uint64_t)top << 56 | (uint64_t)(aniso > 1) << 63;
    id<MTLSamplerState> __strong &cached = s_samplers[key];
    if (!cached) {
        MTLSamplerDescriptor *d = [MTLSamplerDescriptor new];
        bool min_linear = t.min_filter == GL_LINEAR || t.min_filter == GL_LINEAR_MIPMAP_NEAREST || t.min_filter == GL_LINEAR_MIPMAP_LINEAR;
        d.minFilter = min_linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
        d.magFilter = t.mag_filter == GL_LINEAR ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
        d.mipFilter = !mipmapped ? MTLSamplerMipFilterNotMipmapped
                                 : (s_trilinear || t.min_filter == GL_NEAREST_MIPMAP_LINEAR || t.min_filter == GL_LINEAR_MIPMAP_LINEAR)
                                       ? MTLSamplerMipFilterLinear
                                       : MTLSamplerMipFilterNearest;
        d.sAddressMode = address_mode(t.wrap_s);
        d.tAddressMode = address_mode(t.wrap_t);
        d.borderColor = MTLSamplerBorderColorTransparentBlack;
        d.lodMaxClamp = top;
        d.maxAnisotropy = min_linear ? aniso : 1;
        if (s_lod_bias != 0) {
            if (@available(macOS 26.0, *)) {
#if defined(__MAC_26_0) && __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
                d.lodBias = s_lod_bias;
#else
                [d setValue:@(s_lod_bias) forKey:@"lodBias"]; // built with an older SDK (as CI's): the property is there at run time
#endif
            }
        }
        cached = [s_device newSamplerStateWithDescriptor:d];
    }
    return t.sampler = cached;
}
