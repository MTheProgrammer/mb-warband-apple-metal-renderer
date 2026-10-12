// Replacement textures. Included by metal_backend.mm, inside its anonymous
// namespace.
//
// A texture pack is a folder of image files named after the game's own texture
// files: helmets_new_a.png, or .dds, stands in for the game's
// helmets_new_a.dds. When the game uploads a texture that is recognised as one
// of its files (see scan_game_textures) and a pack has a file of that name,
// the pack's image is loaded in its place, at whatever size it has. The game's
// files are not touched.
//
// WBM_TEXTURES lists the folders, separated by colons, the first one winning;
// without it ~/Library/Application Support/warband-metal/textures is used if
// it exists. Subfolders are searched too.

std::unordered_map<std::string, std::string> s_pack_files; // lower-case name without extension -> path
// WBM_TEXTURES_SELF=1, for testing the loader: every recognised texture is
// loaded again from the game's own file, which should change nothing.
bool s_pack_self_test = getenv("WBM_TEXTURES_SELF") != nullptr;
struct {
    std::atomic<int> replaced{0}, failed{0}, pbr_maps{0};
    std::mutex mu;
    std::set<std::string> names; // of the textures replaced, for the summary
} s_pack_stats;

void index_pack_folder(const std::string &dir, int depth)
{
    DIR *d = opendir(dir.c_str());
    if (!d) return;
    while (struct dirent *e = readdir(d)) {
        std::string name = e->d_name;
        if (name.empty() || name[0] == '.') continue;
        std::string path = dir + "/" + name;
        if (e->d_type == DT_DIR) {
            if (depth < 4) index_pack_folder(path, depth + 1);
            continue;
        }
        size_t dot = name.rfind('.');
        if (dot == std::string::npos) continue;
        std::string ext = name.substr(dot + 1), base = name.substr(0, dot);
        for (char &c : ext) c = (char)tolower((unsigned char)c);
        for (char &c : base) c = (char)tolower((unsigned char)c);
        if (ext != "dds" && ext != "png" && ext != "jpg" && ext != "jpeg" && ext != "tga" && ext != "tif" && ext != "tiff" && ext != "bmp") continue;
        s_pack_files.emplace(base, path); // the first folder to have a name keeps it
    }
    closedir(d);
}

void index_texture_packs()
{
    std::string list;
    if (const char *v = getenv("WBM_TEXTURES")) {
        list = v;
    } else if (const char *home = getenv("HOME")) {
        list = std::string(home) + "/Library/Application Support/warband-metal/textures";
    }
    size_t at = 0;
    while (at <= list.size() && !list.empty()) {
        size_t colon = list.find(':', at);
        std::string dir = list.substr(at, colon == std::string::npos ? std::string::npos : colon - at);
        if (!dir.empty()) index_pack_folder(dir, 0);
        if (colon == std::string::npos) break;
        at = colon + 1;
    }
}

// ---- reading image files -------------------------------------------------------------------

// Turns one 4x4 block upside down. alpha: 0 for DXT1, 3 for DXT3, 5 for DXT5.
void flip_block(uint8_t *q, int alpha)
{
    if (alpha == 3) { // four rows of explicit alpha, two bytes each
        std::swap(q[0], q[6]), std::swap(q[1], q[7]), std::swap(q[2], q[4]), std::swap(q[3], q[5]);
    } else if (alpha == 5) { // sixteen 3-bit alpha indices after the two alpha values
        uint64_t bits = 0;
        for (int i = 0; i < 6; i++) bits |= (uint64_t)q[2 + i] << (8 * i);
        uint64_t turned = ((bits & 0xFFF) << 36) | ((bits >> 12 & 0xFFF) << 24) | ((bits >> 24 & 0xFFF) << 12) | (bits >> 36 & 0xFFF);
        for (int i = 0; i < 6; i++) q[2 + i] = (uint8_t)(turned >> (8 * i));
    }
    uint8_t *colour = q + (alpha ? 8 : 0); // two colours, then four rows of indices, a byte each
    std::swap(colour[4], colour[7]), std::swap(colour[5], colour[6]);
}

// Turns a level of block-compressed data upside down, as the game does before
// it uploads a texture. Levels less than a block high are left as they are.
void flip_compressed(std::vector<uint8_t> &level, int w, int h, int alpha)
{
    size_t block = alpha ? 16 : 8, row = (size_t)((w + 3) / 4) * block, rows = (size_t)((h + 3) / 4);
    if (h < 4 || level.size() < row * rows) return;
    std::vector<uint8_t> out(level.size());
    for (size_t r = 0; r < rows; r++) {
        memcpy(&out[r * row], &level[(rows - 1 - r) * row], row);
        for (size_t b = 0; b + block <= row; b += block) flip_block(&out[r * row + b], alpha);
    }
    level.swap(out);
}

// Decodes block-compressed data to 8-bit RGBA.
std::vector<uint8_t> decode_compressed(const uint8_t *data, int w, int h, int alpha)
{
    std::vector<uint8_t> px((size_t)w * h * 4, 255);
    size_t block = alpha ? 16 : 8;
    int bw = (w + 3) / 4, bh = (h + 3) / 4;
    auto c565 = [](uint16_t c, int *o) {
        o[0] = ((c >> 11) & 31) * 255 / 31, o[1] = ((c >> 5) & 63) * 255 / 63, o[2] = (c & 31) * 255 / 31;
    };
    for (int by = 0; by < bh; by++) {
        for (int bx = 0; bx < bw; bx++) {
            const uint8_t *b = data + ((size_t)by * bw + bx) * block, *cb = b + (alpha ? 8 : 0);
            uint16_t c0 = (uint16_t)(cb[0] | cb[1] << 8), c1 = (uint16_t)(cb[2] | cb[3] << 8);
            int p[4][4];
            c565(c0, p[0]), c565(c1, p[1]);
            bool four = alpha || c0 > c1;
            for (int k = 0; k < 3; k++) {
                p[2][k] = four ? (2 * p[0][k] + p[1][k]) / 3 : (p[0][k] + p[1][k]) / 2;
                p[3][k] = four ? (p[0][k] + 2 * p[1][k]) / 3 : 0;
            }
            p[0][3] = p[1][3] = p[2][3] = 255, p[3][3] = four ? 255 : 0;
            int a[8] = {};
            uint64_t abits = 0;
            if (alpha == 5) {
                a[0] = b[0], a[1] = b[1];
                for (int i = 1; i < 7; i++) a[i + 1] = a[0] > a[1] ? ((7 - i) * a[0] + i * a[1]) / 7 : i < 5 ? ((5 - i) * a[0] + i * a[1]) / 5 : (i == 5 ? 0 : 255);
                for (int i = 0; i < 6; i++) abits |= (uint64_t)b[2 + i] << (8 * i);
            }
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    int X = bx * 4 + x, Y = by * 4 + y;
                    if (X >= w || Y >= h) continue;
                    uint8_t *o = &px[((size_t)Y * w + X) * 4];
                    const int *c = p[(cb[4 + y] >> (2 * x)) & 3];
                    o[0] = (uint8_t)c[0], o[1] = (uint8_t)c[1], o[2] = (uint8_t)c[2], o[3] = (uint8_t)c[3];
                    if (alpha == 3) o[3] = (uint8_t)(((b[y * 2 + x / 2] >> (4 * (x & 1))) & 15) * 17);
                    else if (alpha == 5) o[3] = (uint8_t)a[(abits >> (3 * (y * 4 + x))) & 7];
                }
            }
        }
    }
    return px;
}

struct LoadedImage {
    int w = 0, h = 0;
    int alpha = -1;                           // -1: `levels[0]` is 8-bit RGBA; otherwise block-compressed, 0, 3 or 5
    std::vector<std::vector<uint8_t>> levels; // top-down rows, largest first
};

bool read_dds(const std::string &path, LoadedImage *img)
{
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    std::vector<uint8_t> file;
    uint8_t buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) file.insert(file.end(), buf, buf + n);
    fclose(f);
    if (file.size() < 128 || memcmp(file.data(), "DDS ", 4)) return false;
    uint32_t h, w, mips, pf_flags, fourcc, bits, rmask;
    memcpy(&h, &file[12], 4), memcpy(&w, &file[16], 4), memcpy(&mips, &file[28], 4), memcpy(&pf_flags, &file[80], 4);
    memcpy(&fourcc, &file[84], 4), memcpy(&bits, &file[88], 4), memcpy(&rmask, &file[92], 4);
    if (!w || !h || w > 16384 || h > 16384) return false;
    img->w = (int)w, img->h = (int)h;
    mips = std::max<uint32_t>(mips, 1);
    size_t at = 128;
    bool compressed = (pf_flags & 4) != 0;
    int alpha = !memcmp(&fourcc, "DXT1", 4) ? 0 : !memcmp(&fourcc, "DXT3", 4) ? 3 : !memcmp(&fourcc, "DXT5", 4) ? 5 : -1;
    if (compressed && alpha < 0) return false;
    if (compressed) {
        size_t block = alpha ? 16 : 8;
        bool whole_chain = (int)mips >= mip_count((int)w, (int)h);
        if (!whole_chain) { // decoded, so that the missing levels can be made on the GPU
            size_t size = (size_t)((w + 3) / 4) * ((h + 3) / 4) * block;
            if (file.size() < at + size) return false;
            img->levels.push_back(decode_compressed(&file[at], (int)w, (int)h, alpha));
            return true;
        }
        img->alpha = alpha;
        for (uint32_t level = 0, lw = w, lh = h; level < mips; level++, lw = std::max<uint32_t>(lw / 2, 1), lh = std::max<uint32_t>(lh / 2, 1)) {
            size_t size = (size_t)((lw + 3) / 4) * ((lh + 3) / 4) * block;
            if (file.size() < at + size) break;
            img->levels.emplace_back(file.begin() + (long)at, file.begin() + (long)(at + size));
            at += size;
        }
        return !img->levels.empty();
    }
    if (bits != 32 && bits != 24) return false;
    size_t bpp = bits / 8, size = (size_t)w * h * bpp;
    if (file.size() < at + size) return false;
    std::vector<uint8_t> px((size_t)w * h * 4);
    bool bgr = rmask == 0x00ff0000;
    for (size_t i = 0; i < (size_t)w * h; i++) {
        const uint8_t *s = &file[at + i * bpp];
        px[i * 4 + 0] = bgr ? s[2] : s[0], px[i * 4 + 1] = s[1], px[i * 4 + 2] = bgr ? s[0] : s[2], px[i * 4 + 3] = bpp == 4 ? s[3] : 255;
    }
    img->levels.push_back(std::move(px));
    return true;
}

// PNG and the other formats the system reads, as 8-bit RGBA with alpha kept apart from colour.
bool read_image(const std::string &path, LoadedImage *img)
{
    NSURL *url = [NSURL fileURLWithPath:@(path.c_str())];
    CGImageSourceRef source = CGImageSourceCreateWithURL((__bridge CFURLRef)url, nullptr);
    if (!source) return false;
    CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, nullptr);
    CFRelease(source);
    if (!image) return false;
    size_t w = CGImageGetWidth(image), h = CGImageGetHeight(image);
    bool ok = w && h && w <= 16384 && h <= 16384;
    if (ok) {
        std::vector<uint8_t> px(w * h * 4);
        CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB(); // the file's numbers, not a conversion of them
        CGContextRef ctx = CGBitmapContextCreate(px.data(), w, h, 8, w * 4, space, (uint32_t)kCGImageAlphaPremultipliedLast | (uint32_t)kCGBitmapByteOrder32Big);
        CGColorSpaceRelease(space);
        ok = ctx != nullptr;
        if (ok) {
            CGContextSetBlendMode(ctx, kCGBlendModeCopy);
            CGContextDrawImage(ctx, CGRectMake(0, 0, w, h), image);
            CGContextRelease(ctx);
            for (size_t i = 0; i < w * h; i++) { // undo the premultiplication the context applies
                uint8_t *p = &px[i * 4];
                if (p[3] && p[3] != 255)
                    for (int k = 0; k < 3; k++) p[k] = (uint8_t)std::min(255, (p[k] * 255 + p[3] / 2) / p[3]);
            }
            img->w = (int)w, img->h = (int)h;
            img->levels.push_back(std::move(px));
        }
    }
    CGImageRelease(image);
    return ok;
}

// ---- putting a replacement in place ----------------------------------------------------

// Loads the file into the texture. flip: the game uploads this texture upside
// down, and the replacement has to be turned the same way.
bool load_replacement(Texture &t, const std::string &path, bool flip)
{
    @autoreleasepool {
        LoadedImage img;
        bool dds = path.size() > 4 && !strcasecmp(path.c_str() + path.size() - 4, ".dds");
        if (!(dds ? read_dds(path, &img) : read_image(path, &img)) || img.levels.empty()) {
            complain("texture pack: %s could not be read", path.c_str());
            return false;
        }
        t.mtl = nil; // whatever the game put there has another size
        if (img.alpha >= 0) {
            FormatInfo f;
            f.format = img.alpha == 0 ? MTLPixelFormatBC1_RGBA : img.alpha == 3 ? MTLPixelFormatBC2_RGBA : MTLPixelFormatBC3_RGBA;
            f.block_bytes = img.alpha ? 16 : 8;
            if (!ensure_storage(t, f, 0, img.w, img.h)) return false;
            int level = 0;
            for (int lw = img.w, lh = img.h; level < (int)img.levels.size() && level < (int)t.mtl.mipmapLevelCount; level++, lw = std::max(lw / 2, 1), lh = std::max(lh / 2, 1)) {
                if (flip) flip_compressed(img.levels[(size_t)level], lw, lh, img.alpha);
                [t.mtl replaceRegion:MTLRegionMake2D(0, 0, (NSUInteger)lw, (NSUInteger)lh)
                         mipmapLevel:(NSUInteger)level
                           withBytes:img.levels[(size_t)level].data()
                         bytesPerRow:(NSUInteger)((lw + 3) / 4) * (NSUInteger)f.block_bytes];
            }
            t.top_level = level - 1;
        } else {
            std::vector<uint8_t> &px = img.levels[0];
            if (flip) {
                size_t row = (size_t)img.w * 4;
                std::vector<uint8_t> line(row);
                for (int y = 0; y < img.h / 2; y++) {
                    uint8_t *a = &px[(size_t)y * row], *b = &px[(size_t)(img.h - 1 - y) * row];
                    memcpy(line.data(), a, row), memcpy(a, b, row), memcpy(b, line.data(), row);
                }
            }
            FormatInfo f;
            f.format = MTLPixelFormatRGBA8Unorm;
            f.bytes_per_pixel = 4;
            if (!ensure_storage(t, f, 0, img.w, img.h)) return false;
            [t.mtl replaceRegion:MTLRegionMake2D(0, 0, (NSUInteger)img.w, (NSUInteger)img.h) mipmapLevel:0 withBytes:px.data() bytesPerRow:(NSUInteger)img.w * 4];
            if (t.mtl.mipmapLevelCount > 1) {
                id<MTLCommandBuffer> cmd = [s_queue commandBuffer];
                id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
                [blit generateMipmapsForTexture:t.mtl];
                [blit endEncoding];
                [cmd commit];
            }
            t.top_level = (int)t.mtl.mipmapLevelCount - 1;
        }
        t.scale = 1;
        t.screen_sized = false;
        t.sampler = nil;
        t.replaced = true;
        return true;
    }
}

// Called when a texture has been recognised as one of the game's files.
// Returns true when a pack's file took its place.
bool replace_from_pack(Texture &t, const KnownTexture &known)
{
    std::string name = s_texture_names[known.name], path;
    const char suffix[] = "_degraded_"; // the small stand-in the game loads first for some textures
    size_t n = sizeof suffix - 1;
    if (name.size() > n && !name.compare(name.size() - n, n, suffix)) name.resize(name.size() - n);
    auto it = s_pack_files.find(name);
    if (it != s_pack_files.end()) path = it->second;
    else if (s_pack_self_test) path = s_texture_paths[known.name];
    else return false;
    if (!load_replacement(t, path, known.flipped)) {
        s_pack_stats.failed++;
        return false;
    }
    s_pack_stats.replaced++;
    std::lock_guard<std::mutex> lock(s_pack_stats.mu);
    s_pack_stats.names.insert(name);
    return true;
}

// A pack may also hold, for the game's texture `name`, a file name_pbr: red is
// metalness and green roughness, laid out like the texture itself. It is kept
// beside the texture and given to shaders that ask for one (k_xl_pbr_sampler).
void attach_pbr_map(Texture &t, const KnownTexture &known)
{
    t.pbr = nil;
    t.hair = nil;
    t.hair_colour[3] = 0;
    if (s_pack_files.empty()) return;
    std::string name = s_texture_names[known.name];
    const char suffix[] = "_degraded_";
    size_t n = sizeof suffix - 1;
    if (name.size() > n && !name.compare(name.size() - n, n, suffix)) name.resize(name.size() - n);
    auto hair = s_pack_files.find(name + "_hair");
    if (hair != s_pack_files.end()) {
        Texture map;
        if (load_replacement(map, hair->second, known.flipped)) {
            t.hair = map.mtl;
            read_hair_colour(hair->second, t.hair_colour);
        }
        else s_pack_stats.failed++;
    }
    auto it = s_pack_files.find(name + "_pbr");
    if (it == s_pack_files.end()) return;
    Texture map;
    if (!load_replacement(map, it->second, known.flipped)) {
        s_pack_stats.failed++;
        return;
    }
    t.pbr = map.mtl;
    s_pack_stats.pbr_maps++;
}
