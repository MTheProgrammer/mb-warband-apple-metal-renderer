// Works on the game's texture files, to make texture packs (see WBM_TEXTURES):
//
//   texture_tool export <out-folder> [name...]
//       writes the named textures (all of them without names) as PNG, to be
//       edited or enlarged and used as replacements;
//   texture_tool pbr <out-folder> [name...]
//       writes <name>_pbr.png for textures that have metal on them: metalness
//       in red and roughness in green, worked out from the texture's specular
//       map (bright there means metal) and, for textures the name says are
//       metal, from which parts are grey; and, for horses, in blue how much of
//       the surface is coat (not armour, cloth or saddle).
//
// The game's folder is taken from WBM_GAME, or from where Steam puts it.

#include "../src/material_names.h"

#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>

struct Image {
    int w = 0, h = 0;
    std::vector<uint8_t> px; // RGBA, top row first
};

static bool read_dds(const std::string &path, Image *img)
{
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    std::vector<uint8_t> d;
    uint8_t buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) d.insert(d.end(), buf, buf + n);
    fclose(f);
    if (d.size() < 128 || memcmp(d.data(), "DDS ", 4)) return false;
    uint32_t h, w, pf, fourcc, bits, rmask;
    memcpy(&h, &d[12], 4), memcpy(&w, &d[16], 4), memcpy(&pf, &d[80], 4), memcpy(&fourcc, &d[84], 4), memcpy(&bits, &d[88], 4), memcpy(&rmask, &d[92], 4);
    if (!w || !h || w > 16384 || h > 16384) return false;
    img->w = (int)w, img->h = (int)h;
    img->px.assign((size_t)w * h * 4, 255);
    const uint8_t *data = &d[128];
    size_t have = d.size() - 128;
    if (!(pf & 4)) {
        size_t bpp = bits / 8;
        if ((bpp != 3 && bpp != 4) || have < (size_t)w * h * bpp) return false;
        bool bgr = rmask == 0x00ff0000;
        for (size_t i = 0; i < (size_t)w * h; i++) {
            const uint8_t *s = data + i * bpp;
            uint8_t *o = &img->px[i * 4];
            o[0] = bgr ? s[2] : s[0], o[1] = s[1], o[2] = bgr ? s[0] : s[2], o[3] = bpp == 4 ? s[3] : 255;
        }
        return true;
    }
    int alpha = !memcmp(&fourcc, "DXT1", 4) ? 0 : !memcmp(&fourcc, "DXT3", 4) ? 3 : !memcmp(&fourcc, "DXT5", 4) ? 5 : -1;
    if (alpha < 0) return false;
    size_t block = alpha ? 16 : 8;
    int bw = ((int)w + 3) / 4, bh = ((int)h + 3) / 4;
    if (have < (size_t)bw * bh * block) return false;
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
                    if (X >= (int)w || Y >= (int)h) continue;
                    uint8_t *o = &img->px[((size_t)Y * w + X) * 4];
                    const int *c = p[(cb[4 + y] >> (2 * x)) & 3];
                    o[0] = (uint8_t)c[0], o[1] = (uint8_t)c[1], o[2] = (uint8_t)c[2], o[3] = (uint8_t)c[3];
                    if (alpha == 3) o[3] = (uint8_t)(((b[y * 2 + x / 2] >> (4 * (x & 1))) & 15) * 17);
                    else if (alpha == 5) o[3] = (uint8_t)a[(abits >> (3 * (y * 4 + x))) & 7];
                }
            }
        }
    }
    return true;
}

static bool write_png(const std::string &path, const Image &img)
{
    @autoreleasepool {
        CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
        CGDataProviderRef provider = CGDataProviderCreateWithData(nullptr, img.px.data(), img.px.size(), nullptr);
        CGImageRef image = CGImageCreate((size_t)img.w, (size_t)img.h, 8, 32, (size_t)img.w * 4, space, (uint32_t)kCGImageAlphaLast | (uint32_t)kCGBitmapByteOrder32Big,
                                         provider, nullptr, false, kCGRenderingIntentDefault);
        NSURL *url = [NSURL fileURLWithPath:@(path.c_str())];
        CGImageDestinationRef dest = CGImageDestinationCreateWithURL((__bridge CFURLRef)url, CFSTR("public.png"), 1, nullptr);
        bool ok = image && dest;
        if (ok) {
            CGImageDestinationAddImage(dest, image, nullptr);
            ok = CGImageDestinationFinalize(dest);
        }
        if (dest) CFRelease(dest);
        if (image) CGImageRelease(image);
        CGDataProviderRelease(provider);
        CGColorSpaceRelease(space);
        return ok;
    }
}

// The game's texture files by lower-case name; the HD folder's version wins, as the larger.
static std::map<std::string, std::string> game_textures()
{
    std::string root;
    if (const char *g = getenv("WBM_GAME")) root = g;
    else if (const char *home = getenv("HOME")) root = std::string(home) + "/Library/Application Support/Steam/steamapps/common/MountBlade Warband";
    std::map<std::string, std::string> files;
    for (const char *sub : {"/TexturesHD", "/Textures"}) {
        DIR *d = opendir((root + sub).c_str());
        if (!d) continue;
        while (struct dirent *e = readdir(d)) {
            std::string name = e->d_name;
            if (name.size() < 5 || strcasecmp(name.c_str() + name.size() - 4, ".dds")) continue;
            std::string base = name.substr(0, name.size() - 4);
            for (char &c : base) c = (char)tolower((unsigned char)c);
            files.emplace(base, root + sub + "/" + name);
        }
        closedir(d);
    }
    return files;
}

static bool ends_with(const std::string &s, const char *suffix)
{
    size_t n = strlen(suffix);
    return s.size() > n && !s.compare(s.size() - n, n, suffix);
}

// Not a base-colour texture: a normal map, a specular map, or a small stand-in.
static bool is_auxiliary(const std::string &name)
{
    return name.find("normalmap") != std::string::npos || name.find("specular") != std::string::npos || name.find("_degraded_") != std::string::npos ||
           name.find("bump") != std::string::npos || ends_with(name, "_n") || ends_with(name, "_s") || ends_with(name, "_spec");
}

static float smoothstep(float a, float b, float x)
{
    float t = std::min(1.0f, std::max(0.0f, (x - a) / (b - a)));
    return t * t * (3 - 2 * t);
}

// Brightness of the specular map at a place given in the base texture's pixels.
static float sample_luminance(const Image &m, float u, float v)
{
    float x = u * m.w - 0.5f, y = v * m.h - 0.5f;
    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    float fx = x - x0, fy = y - y0, sum = 0;
    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < 2; i++) {
            int X = std::min(m.w - 1, std::max(0, x0 + i)), Y = std::min(m.h - 1, std::max(0, y0 + j));
            const uint8_t *p = &m.px[((size_t)Y * m.w + X) * 4];
            sum += (p[0] + p[1] + p[2]) / (3 * 255.0f) * (i ? fx : 1 - fx) * (j ? fy : 1 - fy);
        }
    }
    return sum;
}

int main(int argc, char **argv)
{
    if (argc < 3 || (strcmp(argv[1], "export") && strcmp(argv[1], "pbr"))) {
        fprintf(stderr, "usage: texture_tool export <out-folder> [name...]\n       texture_tool pbr <out-folder> [name...]\n");
        return 2;
    }
    bool pbr = !strcmp(argv[1], "pbr");
    std::string out = argv[2];
    mkdir(out.c_str(), 0755);
    std::map<std::string, std::string> files = game_textures();
    if (files.empty()) {
        fprintf(stderr, "no texture files found; set WBM_GAME to the game's folder\n");
        return 1;
    }
    std::vector<std::string> names;
    for (int i = 3; i < argc; i++) {
        std::string n = argv[i];
        for (char &c : n) c = (char)tolower((unsigned char)c);
        names.push_back(n);
    }
    if (names.empty())
        for (auto &f : files)
            if (!pbr || !is_auxiliary(f.first)) names.push_back(f.first);

    int written = 0, skipped = 0;
    for (const std::string &name : names) {
        auto it = files.find(name);
        Image base;
        if (it == files.end() || !read_dds(it->second, &base)) {
            fprintf(stderr, "%s: not one of the game's textures, or in a format that is not read\n", name.c_str());
            skipped++;
            continue;
        }
        if (!pbr) {
            written += write_png(out + "/" + name + ".png", base);
            continue;
        }
        // The specular map that goes with this texture, by the game's naming.
        Image spec;
        bool has_spec = false;
        for (const char *suffix : {"_specular", "_spec", "_s"}) {
            auto s = files.find(name + suffix);
            if (s != files.end() && read_dds(s->second, &spec)) {
                has_spec = true;
                break;
            }
        }
        bool metal_class = classify_texture_name(name) == MAT_METAL;
        bool horse = false;
        for (const char *word : {"horse", "charger", "courser", "hunter", "steed"}) horse = horse || name.find(word) != std::string::npos;
        if (name.compare(0, 4, "pic_") == 0 || name.find("boot") != std::string::npos || name.find("horseman") != std::string::npos) horse = false; // pictures and riders' gear
        if (!has_spec && !metal_class && !horse) {
            skipped++;
            continue;
        }
        Image map;
        map.w = base.w, map.h = base.h;
        map.px.assign((size_t)base.w * base.h * 4, 255);
        float most = 0;
        for (int y = 0; y < base.h; y++) {
            for (int x = 0; x < base.w; x++) {
                const uint8_t *c = &base.px[((size_t)y * base.w + x) * 4];
                float lin[3];
                for (int k = 0; k < 3; k++) lin[k] = powf(c[k] / 255.0f, 2.2f);
                float hi = std::max({lin[0], lin[1], lin[2]}), lo = std::min({lin[0], lin[1], lin[2]});
                float grey = 1 - smoothstep(0.1f, 0.28f, (hi - lo) / std::max(hi, 0.08f));
                float s = has_spec ? sample_luminance(spec, (x + 0.5f) / base.w, (y + 0.5f) / base.h) : 0;
                // Measured on the game's helmet textures: fur and leather are under
                // 0.08 in the specular map, steel from 0.3 up.
                float metal = has_spec ? smoothstep(0.12f, 0.3f, s) : 0;
                // Grey means steel on a texture of metal things, except where the
                // texture is simply empty: unused corners are black.
                if (metal_class) metal = std::max(metal, 0.95f * grey * smoothstep(0.004f, 0.02f, hi));
                // Brighter in the specular map is smoother; bare metal with no map is fairly smooth.
                float rough = has_spec ? 0.85f - 0.5f * smoothstep(0.05f, 0.6f, s) : 0.75f - 0.3f * metal;
                uint8_t *o = &map.px[((size_t)y * base.w + x) * 4];
                // Blue: how much of the surface is a horse's coat. With a specular map,
                // what is neither metal (bright) nor cloth (black). Without one, all that
                // is painted, less the saddle, which every plain horse has at the same place.
                float fur = 0;
                if (horse) {
                    float u = (x + 0.5f) / base.w, v = (y + 0.5f) / base.h;
                    if (has_spec) {
                        fur = smoothstep(0.03f, 0.10f, s) * (1 - smoothstep(0.24f, 0.36f, s));
                    } else {
                        bool saddle = (u > 0.485f && u < 0.755f && v > 0.02f && v < 0.335f) || (u > 0.195f && u < 0.315f && v > 0.455f && v < 0.715f) ||
                                      (u > 0.03f && u < 0.10f && v > 0.44f && v < 0.60f);
                        fur = saddle ? 0.0f : smoothstep(0.0002f, 0.0012f, hi); // a black coat is dark, but not as black as the unused parts
                    }
                }
                o[0] = (uint8_t)lroundf(metal * 255), o[1] = (uint8_t)lroundf(rough * 255), o[2] = (uint8_t)lroundf(fur * 255), o[3] = 255;
                most = std::max(most, std::max(metal, fur));
            }
        }
        if (most < 0.5f) { // nothing on it is metal: the renderer's own estimate will do
            skipped++;
            continue;
        }
        written += write_png(out + "/" + name + "_pbr.png", map);
    }
    printf("%d files written to %s, %d textures passed over\n", written, out.c_str(), skipped);
    return 0;
}
