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
//   texture_tool hair <picture> <name> <out-folder> [tack-picture]
//       writes <name>_hair.png for the game's texture <name> from a picture of
//       that texture with everything but the long hair (mane, tail, forelock)
//       blacked out, or simply white where that hair is and black elsewhere
//       (such a mask is taken exactly; in a picture, parts of the marked area
//       that have the coat's colour are left out):
//       red and green give the way the hair is combed, worked out from the run
//       of the strands in the game's texture and its normal map; blue says what grows there (255 long hair, about 128 coat, 0 none).
//       In place of the picture, "boxes:x0,y0,x1,y1/..." names rectangles of
//       the texture inside which whatever is painted is long hair. What was
//       taken for long hair is also written, white on black, to masks/.
//       A second picture, of the saddle, straps and other gear alone, keeps
//       the coat off those.
//
//   texture_tool import <folder of pictures> <pack-folder> [--gain g] [--normals directx] [--dds]
//       makes a texture pack of pictures that come back from being enlarged or
//       repainted elsewhere (Blender, say), named after the game's textures:
//         <name>, <name>_basecolor     the texture itself (also _albedo, _diffuse);
//         <name>_normal                its normal map, whatever the game calls
//                                      it (a picture under the game's own name
//                                      for the normal map will do as well);
//         <name>_metallic, <name>_roughness
//                                      grey pictures of each, or
//         <name>_orm                   the two as glTF packs them (roughness in
//                                      green, metalness in blue);
//         <name>_opacity               what goes into the texture's alpha;
//         <name>_ao                    how much of the light from all round reaches
//                                      each spot (ambient occlusion);
//         <name>_height                the surface's heights, white the highest;
//       and writes <name>.png, <name>_pbr.png as the renderer wants it (metalness
//       in red, roughness in green, occlusion in alpha; a horse's coat, in blue,
//       is kept from the map already in the pack or worked out), and
//       <name>_depth.png, the heights, for WBM_PARALLAX. Normal maps are taken to have green
//       pointing up the picture, as glTF and Blender have it, and are turned
//       about for the game, which has it pointing down (--normals directx:
//       they are the game's way already); where the game keeps a normal's x in
//       alpha (the textures of rooms), it is put there. A texture that has lost
//       its alpha on the way gets the game's own back. --gain g: the pictures
//       were made g times brighter than the game's, in linear light, to look
//       right elsewhere; they are brought back. --dds: the pack is written as
//       .dds files in the game's own compression, with their smaller sizes,
//       not as PNG: a quarter to an eighth of the video memory, and no
//       unpacking when a scene loads, for a little blockiness. Alpha that is
//       all or nothing gets soft edges (see soften_cutout); --hard-alpha: not.
//
// The game's folder is taken from WBM_GAME, or from where Steam puts it.

#include "../src/brf.h"
#include "../src/material_names.h"
#include "dds.h"

#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>

#include <algorithm>
#include <dispatch/dispatch.h>
#include <unistd.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>

// PNG, JPEG and whatever else the system reads.
static bool read_picture(const std::string &path, Image *img)
{
    @autoreleasepool {
        NSURL *url = [NSURL fileURLWithPath:@(path.c_str())];
        CGImageSourceRef source = CGImageSourceCreateWithURL((__bridge CFURLRef)url, nullptr);
        if (!source) return false;
        CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, nullptr);
        CFRelease(source);
        if (!image) return false;
        size_t w = CGImageGetWidth(image), h = CGImageGetHeight(image);
        img->w = (int)w, img->h = (int)h;
        img->px.assign(w * h * 4, 255);
        // Eight bits each of red, green, blue and an alpha that the colours are not multiplied by: taken as
        // they are, alpha and all. (Drawn, as below, the alpha is lost, and with it whatever it stood for.)
        CGImageAlphaInfo alpha = CGImageGetAlphaInfo(image);
        bool plain = CGImageGetBitsPerComponent(image) == 8 && CGImageGetBitsPerPixel(image) == 32 && (alpha == kCGImageAlphaLast || alpha == kCGImageAlphaNoneSkipLast) &&
                     (CGImageGetBitmapInfo(image) & kCGBitmapByteOrderMask) <= kCGBitmapByteOrder32Big && CGColorSpaceGetModel(CGImageGetColorSpace(image)) == kCGColorSpaceModelRGB;
        if (plain) {
            CFDataRef data = CGDataProviderCopyData(CGImageGetDataProvider(image));
            size_t stride = CGImageGetBytesPerRow(image);
            if (data && (size_t)CFDataGetLength(data) >= stride * (h - 1) + w * 4) {
                const uint8_t *from = CFDataGetBytePtr(data);
                for (size_t y = 0; y < h; y++) memcpy(&img->px[y * w * 4], from + y * stride, w * 4);
                if (alpha == kCGImageAlphaNoneSkipLast)
                    for (size_t i = 3; i < img->px.size(); i += 4) img->px[i] = 255;
                CFRelease(data);
                CGImageRelease(image);
                return true;
            }
            if (data) CFRelease(data);
        }
        // Anything else (16 bits a channel, another order) is drawn into eight bits. A picture with an alpha of its
        // own is drawn keeping it, which Core Graphics does only with the colours multiplied by it, and they are
        // divided back after: drawn without, the alpha was lost and the colours were left multiplied by it (a
        // glTF sheen map's colour came out wrong; a cut-out's colour, where alpha is 0, came out black).
        bool has_alpha = alpha != kCGImageAlphaNone && alpha != kCGImageAlphaNoneSkipLast && alpha != kCGImageAlphaNoneSkipFirst;
        CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
        CGContextRef ctx = CGBitmapContextCreate(img->px.data(), w, h, 8, w * 4, space,
                                                 (uint32_t)(has_alpha ? kCGImageAlphaPremultipliedLast : kCGImageAlphaNoneSkipLast) | (uint32_t)kCGBitmapByteOrder32Big);
        CGColorSpaceRelease(space);
        if (ctx) {
            CGContextSetBlendMode(ctx, kCGBlendModeCopy);
            CGContextDrawImage(ctx, CGRectMake(0, 0, w, h), image);
            CGContextRelease(ctx);
            if (has_alpha)
                for (size_t i = 0; i < w * h; i++) {
                    uint8_t *px = &img->px[i * 4];
                    if (px[3] > 0 && px[3] < 255)
                        for (int k = 0; k < 3; k++) px[k] = (uint8_t)std::min(255, (int)lroundf((float)px[k] * 255.0f / (float)px[3]));
                }
            else
                for (size_t i = 3; i < img->px.size(); i += 4) img->px[i] = 255;
        }
        CGImageRelease(image);
        return ctx != nullptr;
    }
}

// A separable box blur of a plane of floats, run three times: near enough a Gaussian.
static void blur(std::vector<float> &plane, int w, int h, int radius)
{
    if (radius < 1) return;
    std::vector<float> tmp(plane.size());
    for (int pass = 0; pass < 3; pass++) {
        for (int y = 0; y < h; y++) {
            double sum = 0;
            for (int x = -radius; x <= radius; x++) sum += plane[(size_t)y * w + std::min(w - 1, std::max(0, x))];
            for (int x = 0; x < w; x++) {
                tmp[(size_t)y * w + x] = (float)(sum / (2 * radius + 1));
                sum += plane[(size_t)y * w + std::min(w - 1, x + radius + 1)] - plane[(size_t)y * w + std::max(0, x - radius)];
            }
        }
        for (int x = 0; x < w; x++) {
            double sum = 0;
            for (int y = -radius; y <= radius; y++) sum += tmp[(size_t)std::min(h - 1, std::max(0, y)) * w + x];
            for (int y = 0; y < h; y++) {
                plane[(size_t)y * w + x] = (float)(sum / (2 * radius + 1));
                sum += tmp[(size_t)std::min(h - 1, y + radius + 1) * w + x] - tmp[(size_t)std::max(0, y - radius) * w + x];
            }
        }
    }
}

// Adds to a structure tensor (xx, xy, yy) the gradients of one channel of an
// image, sampled at the tensor's own size.
static void add_gradients(const Image &img, int channel, int w, int h, std::vector<float> &xx, std::vector<float> &xy, std::vector<float> &yy)
{
    auto at = [&](int x, int y) {
        int X = std::min(img.w - 1, std::max(0, (int)((x + 0.5f) * img.w / w))), Y = std::min(img.h - 1, std::max(0, (int)((y + 0.5f) * img.h / h)));
        const uint8_t *p = &img.px[((size_t)Y * img.w + X) * 4];
        return channel < 0 ? (p[0] + p[1] + p[2]) / 765.0f : p[channel] / 255.0f;
    };
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float gx = (at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2 * at(x - 1, y) + at(x - 1, y + 1));
            float gy = (at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2 * at(x, y - 1) + at(x + 1, y - 1));
            size_t i = (size_t)y * w + x;
            xx[i] += gx * gx, xy[i] += gx * gy, yy[i] += gy * gy;
        }
    }
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

// Makes a folder and whatever folders above it are missing.
static void make_folders(const std::string &path)
{
    for (size_t at = 1; at <= path.size(); at++)
        if (at == path.size() || path[at] == '/') mkdir(path.substr(0, at).c_str(), 0755);
}

static bool is_horse(const std::string &name)
{
    bool horse = false;
    for (const char *word : {"horse", "charger", "courser", "hunter", "steed"}) horse = horse || name.find(word) != std::string::npos;
    if (name.compare(0, 4, "pic_") == 0 || name.find("boot") != std::string::npos || name.find("horseman") != std::string::npos) horse = false; // pictures and riders' gear
    return horse;
}

// How much of a horse's texture at (u, v) is coat. On an armoured horse,
// whose specular map tells its parts apart: what is neither metal (bright
// there) nor cloth (black). On a plain one, all that is painted; less the
// saddle where the layout is known, which it is for horse_a and horse_b.
static float coat_at(const std::string &name, bool has_spec, float spec, float u, float v, float brightest)
{
    bool armoured = name.find("war") != std::string::npos || name.find("charger") != std::string::npos;
    if (armoured && has_spec) return smoothstep(0.03f, 0.10f, spec) * (1 - smoothstep(0.24f, 0.36f, spec));
    bool saddle = (name == "horse_a" || name == "horse_b") &&
                  ((u > 0.485f && u < 0.755f && v > 0.02f && v < 0.335f) || (u > 0.195f && u < 0.315f && v > 0.455f && v < 0.715f) || (u > 0.03f && u < 0.10f && v > 0.44f && v < 0.60f));
    return saddle ? 0.0f : smoothstep(0.0002f, 0.0012f, brightest); // a black coat is dark, but not as black as the unused parts
}

static bool load_specular(const std::map<std::string, std::string> &files, const std::string &name, Image *spec)
{
    for (const char *suffix : {"_specular", "_spec", "_s"}) {
        auto s = files.find(name + suffix);
        if (s != files.end() && read_dds(s->second, spec)) return true;
    }
    return false;
}

// tack: optionally, a picture of the same texture with all but the saddle, straps and other gear blacked out.
static int hair_map(const std::string &picture, std::string name, const std::string &out, const char *tack_picture)
{
    Image tack;
    bool has_tack = tack_picture && read_picture(tack_picture, &tack);
    if (tack_picture && !has_tack) fprintf(stderr, "%s could not be read; going on without it\n", tack_picture);
    for (char &c : name) c = (char)tolower((unsigned char)c);
    std::map<std::string, std::string> files = game_textures();
    Image hair, base, spec, normal;
    // In place of a picture: "boxes:x0,y0,x1,y1/x0,y0,x1,y1...", rectangles of
    // the texture (0 to 1, from its top left) inside which whatever is painted is long hair.
    std::vector<std::array<float, 4>> boxes;
    if (picture.compare(0, 6, "boxes:") == 0) {
        const char *at = picture.c_str() + 6;
        std::array<float, 4> box;
        while (sscanf(at, "%f,%f,%f,%f", &box[0], &box[1], &box[2], &box[3]) == 4) {
            boxes.push_back(box);
            at = strchr(at, '/');
            if (!at) break;
            at++;
        }
        if (boxes.empty()) {
            fprintf(stderr, "no rectangles in %s\n", picture.c_str());
            return 1;
        }
    } else if (!read_picture(picture, &hair)) {
        fprintf(stderr, "%s could not be read\n", picture.c_str());
        return 1;
    }
    auto it = files.find(name);
    if (it == files.end() || !read_dds(it->second, &base)) {
        fprintf(stderr, "%s is not one of the game's textures\n", name.c_str());
        return 1;
    }
    bool has_spec = load_specular(files, name, &spec), has_normal = false;
    for (const char *suffix : {"_normalmap", "_n", "_bump"}) {
        auto n = files.find(name + suffix);
        if (!has_normal && n != files.end() && read_dds(n->second, &normal)) has_normal = true;
    }
    // Arabian horses and some others share another's normal map; the caller may know better.
    int w = base.w, h = base.h;
    size_t count = (size_t)w * h;
    auto picture_at = [&](int x, int y) {
        if (!boxes.empty()) {
            float u = (x + 0.5f) / w, v = (y + 0.5f) / h;
            const uint8_t *c = &base.px[((size_t)y * w + x) * 4];
            bool painted = std::max({c[0], c[1], c[2]}) > 3;
            for (auto &b : boxes)
                if (painted && u >= b[0] && u <= b[2] && v >= b[1] && v <= b[3]) return 1.0f;
            return 0.0f;
        }
        int X = std::min(hair.w - 1, (int)((x + 0.5f) * hair.w / w)), Y = std::min(hair.h - 1, (int)((y + 0.5f) * hair.h / h));
        const uint8_t *p = &hair.px[((size_t)Y * hair.w + X) * 4];
        return (float)std::max({p[0], p[1], p[2]}) / 255.0f;
    };
    auto tack_at = [&](int x, int y) {
        int X = std::min(tack.w - 1, (int)((x + 0.5f) * tack.w / w)), Y = std::min(tack.h - 1, (int)((y + 0.5f) * tack.h / h));
        const uint8_t *p = &tack.px[((size_t)Y * tack.w + X) * 4];
        return (float)std::max({p[0], p[1], p[2]}) / 255.0f;
    };
    // Where the long hair is: what the picture did not black out. Smoothed, so
    // that the dark gaps between strands and the picture's compression do not show.
    // "Blacked out" is rarely quite black: the commonest brightness is taken for
    // the background, and hair is what is brighter than that. Dark hair is
    // strands with gaps of background between them, so the mask is spread
    // widely enough to close those before it is cut off again.
    int histogram[256] = {};
    for (int y = 0; y < h; y += 2)
        for (int x = 0; x < w; x += 2) histogram[std::min(255, (int)(picture_at(x, y) * 255))]++;
    int background = 0;
    for (int v = 1; v < 256; v++)
        if (histogram[v] > histogram[background]) background = v;
    float floor_ = (float)background / 255.0f + 0.03f;
    std::vector<float> mask(count);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) mask[(size_t)y * w + x] = picture_at(x, y) > floor_ ? 1.0f : 0.0f;
    blur(mask, w, h, std::max(2, w / 130));
    for (float &m : mask) m = smoothstep(0.42f, 0.58f, m); // neither wider nor narrower than the picture has it
    // A picture made for one horse is used for others of the same layout, whose
    // manes do not end in quite the same places; and a hand may slip. So within
    // the marked area, what has the colour of the coat next to it rather than
    // of the hair is left out. Not so for a plain mask, white on black with
    // nothing in between: that was made or corrected on purpose and is taken as it is.
    int extremes = 0, all = 0;
    for (int v = 0; v < 256; v++) all += histogram[v], extremes += v < 13 || v > 242 ? histogram[v] : 0;
    bool plain_mask = boxes.empty() && extremes > all * 0.97;
    if (plain_mask) printf("  the picture is a plain mask and is taken as it is\n");
    else {
        std::vector<float> around = mask;
        blur(around, w, h, std::max(3, w / 40));
        std::vector<uint8_t> hair_c[3], coat_c[3];
        for (size_t i = 0; i < count; i += 3) {
            const uint8_t *c = &base.px[i * 4];
            if (std::max({c[0], c[1], c[2]}) <= 10) continue; // not painted
            if (mask[i] > 0.5f)
                for (int k = 0; k < 3; k++) hair_c[k].push_back(c[k]);
            else if (around[i] > 0.05f)
                for (int k = 0; k < 3; k++) coat_c[k].push_back(c[k]);
        }
        if (hair_c[0].size() > 50 && coat_c[0].size() > 50) {
            float hair_m[3], coat_m[3], apart = 0;
            for (int k = 0; k < 3; k++) {
                std::nth_element(hair_c[k].begin(), hair_c[k].begin() + (long)hair_c[k].size() / 2, hair_c[k].end());
                std::nth_element(coat_c[k].begin(), coat_c[k].begin() + (long)coat_c[k].size() / 2, coat_c[k].end());
                hair_m[k] = hair_c[k][hair_c[k].size() / 2], coat_m[k] = coat_c[k][coat_c[k].size() / 2];
                apart += (hair_m[k] - coat_m[k]) * (hair_m[k] - coat_m[k]);
            }
            if (apart > 30.0f * 30.0f) { // the two can be told apart
                // Judged on the texture smoothed a little, or single strands and the gaps between them would decide.
                std::vector<float> smooth[3];
                for (int k = 0; k < 3; k++) {
                    smooth[k].resize(count);
                    for (size_t i = 0; i < count; i++) smooth[k][i] = base.px[i * 4 + k];
                    blur(smooth[k], w, h, std::max(1, w / 250));
                }
                size_t dropped = 0, kept = 0;
                for (size_t i = 0; i < count; i++) {
                    if (mask[i] <= 0.5f) continue;
                    float to_hair = 0, to_coat = 0;
                    for (int k = 0; k < 3; k++) {
                        to_hair += (smooth[k][i] - hair_m[k]) * (smooth[k][i] - hair_m[k]);
                        to_coat += (smooth[k][i] - coat_m[k]) * (smooth[k][i] - coat_m[k]);
                    }
                    if (to_coat < to_hair) mask[i] = 0, dropped++;
                    else kept++;
                }
                blur(mask, w, h, std::max(1, w / 300));
                for (float &m : mask) m = smoothstep(0.42f, 0.58f, m);
                printf("  %.0f%% of the marked area has the coat's colour and is left out\n", 100.0 * (double)dropped / (double)std::max<size_t>(dropped + kept, 1));
            }
        }
    }
    // Which way the strands run: across a strand brightness changes fast, along
    // it hardly at all. Summed over a neighbourhood, the squared gradients say
    // which direction that is (the one they vary least in).
    std::vector<float> xx(count, 0.0f), xy(count, 0.0f), yy(count, 0.0f);
    add_gradients(base, -1, w, h, xx, xy, yy); // the game's own strands, not the picture's: a picture may have been redrawn
    if (has_normal) {
        add_gradients(normal, 0, w, h, xx, xy, yy);
        add_gradients(normal, 1, w, h, xx, xy, yy);
    }
    // The coat has a lie too: the painted strokes of the texture and the relief
    // of the normal map run along it. So the direction is worked out wherever
    // anything grows, not only in the long hair. How far it is to be trusted,
    // which is how much the strokes agree with one another, goes into its length.
    std::vector<float> grows(count);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t i = (size_t)y * w + x;
            const uint8_t *c = &base.px[i * 4];
            float brightest = powf(std::max({c[0], c[1], c[2]}) / 255.0f, 2.2f);
            float s = has_spec ? sample_luminance(spec, (x + 0.5f) / w, (y + 0.5f) / h) : 0;
            float coat = coat_at(name, has_spec, s, (x + 0.5f) / w, (y + 0.5f) / h, brightest);
            if (has_tack && tack_at(x, y) > 0.07f) coat = 0;
            grows[i] = std::max(mask[i], coat);
        }
    }
    // For the long hair, only what lies well inside it is read: at its edge the
    // step from dark mane to light coat is far stronger than any strand, and
    // runs along the mane, which would turn the hair to lie along the neck.
    std::vector<float> inner = mask;
    blur(inner, w, h, std::max(2, w / 150));
    std::vector<float> lxx(count), lxy(count), lyy(count);
    double inner_strength = 0, inner_area = 0;
    for (size_t i = 0; i < count; i++) {
        float in = smoothstep(0.8f, 0.97f, inner[i]) * mask[i];
        lxx[i] = xx[i] * in, lxy[i] = xy[i] * in, lyy[i] = yy[i] * in;
        inner_strength += (double)(lxx[i] + lyy[i]), inner_area += in;
    }
    // And a little of "hanging down the picture", which decides where a strip of hair is too thin to have an inside.
    float hang = inner_area > 0 ? (float)(0.08 * inner_strength / inner_area) : 1.0f;
    // (Much more of it where there is no inside to read at all, as in a forelock.)
    for (size_t i = 0; i < count; i++) lxx[i] += hang * mask[i] * (1.0f + 30.0f * (1.0f - smoothstep(0.8f, 0.97f, inner[i])));
    for (size_t i = 0; i < count; i++) xx[i] *= grows[i], xy[i] *= grows[i], yy[i] *= grows[i];
    int radius = std::max(2, w / 110);
    // The coat's strokes are short and point every way, so there the reading
    // is smoothed over a much wider area, into paths, and leant towards the
    // way a coat lies anyhow: down the picture, which on these textures is down
    // the horse. (A strand running down has all its change across: xx.)
    {
        std::vector<float> cxx = xx, cxy = xy, cyy = yy;
        double strength = 0, area = 0;
        for (size_t i = 0; i < count; i++)
            if (grows[i] > 0.25f && mask[i] <= 0.5f) strength += xx[i] + yy[i], area += 1;
        float lean = area > 0 ? (float)(0.5 * strength / area) : 0;
        for (size_t i = 0; i < count; i++) cxx[i] += lean * grows[i];
        int wide = std::max(4, w / 22);
        blur(cxx, w, h, wide), blur(cxy, w, h, wide), blur(cyy, w, h, wide);
        blur(lxx, w, h, radius), blur(lxy, w, h, radius), blur(lyy, w, h, radius);
        for (size_t i = 0; i < count; i++) {
            float m = mask[i]; // long hair keeps its own, finer reading
            xx[i] = lxx[i] * m + cxx[i] * (1 - m), xy[i] = lxy[i] * m + cxy[i] * (1 - m), yy[i] = lyy[i] * m + cyy[i] * (1 - m);
        }
    }
    std::vector<float> dx(count), dy(count), trust(count);
    for (size_t i = 0; i < count; i++) {
        float angle = 0.5f * atan2f(2 * xy[i], xx[i] - yy[i]) + 1.5707963f; // a right angle from the steepest change
        dx[i] = cosf(angle), dy[i] = sinf(angle);
        float agree = sqrtf((xx[i] - yy[i]) * (xx[i] - yy[i]) + 4 * xy[i] * xy[i]) / (xx[i] + yy[i] + 1e-9f);
        trust[i] = std::max(mask[i], std::min(1.0f, agree * 1.6f) * (grows[i] > 0.25f ? 1.0f : 0.0f));
    }
    // For telling a fault of the map from one of the renderer: WBM_HAIR_DIR="x y"
    // gives every place that direction (x to the right, y down the picture).
    if (const char *forced = getenv("WBM_HAIR_DIR")) {
        float fx = 0, fy = 1;
        sscanf(forced, "%f %f", &fx, &fy);
        float len = std::max(1e-6f, sqrtf(fx * fx + fy * fy));
        for (size_t i = 0; i < count; i++) dx[i] = fx / len, dy[i] = fy / len, trust[i] = grows[i] > 0.25f ? 1.0f : 0.0f;
    }
    // On the coat, where the strokes are short and many, each place is simply
    // pointed down the picture (or to the right, where it runs level).
    for (size_t i = 0; i < count && !getenv("WBM_HAIR_DIR"); i++)
        if (mask[i] <= 0.5f && (dy[i] < -0.05f || (fabsf(dy[i]) <= 0.05f && dx[i] < 0))) dx[i] = -dx[i], dy[i] = -dy[i];
    // A strand's run does not say which end is the tip. Within each patch of
    // hair the choice is made the same way throughout, by spreading from one
    // pixel to its neighbours; and for the patch as a whole, the way that goes
    // down the picture, or failing that to the right, since hair hangs.
    std::vector<int> patch(count, -1);
    std::vector<size_t> queue;
    for (size_t seed = 0; seed < count && !getenv("WBM_HAIR_DIR"); seed++) {
        if (mask[seed] <= 0.5f || patch[seed] >= 0) continue;
        size_t first = queue.size();
        patch[seed] = 1;
        queue.push_back(seed);
        double sum_x = 0, sum_y = 0;
        for (size_t at = first; at < queue.size(); at++) {
            size_t i = queue[at];
            sum_x += dx[i], sum_y += dy[i];
            int x = (int)(i % (size_t)w), y = (int)(i / (size_t)w);
            const int step[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (auto &d : step) {
                int X = x + d[0], Y = y + d[1];
                if (X < 0 || Y < 0 || X >= w || Y >= h) continue;
                size_t j = (size_t)Y * w + X;
                if (mask[j] <= 0.5f || patch[j] >= 0) continue;
                if (dx[i] * dx[j] + dy[i] * dy[j] < 0) dx[j] = -dx[j], dy[j] = -dy[j];
                patch[j] = 1;
                queue.push_back(j);
            }
        }
        if (sum_y < -0.05 * (double)(queue.size() - first) || (fabs(sum_y) <= 0.05 * (double)(queue.size() - first) && sum_x < 0))
            for (size_t at = first; at < queue.size(); at++) dx[queue[at]] = -dx[queue[at]], dy[queue[at]] = -dy[queue[at]];
    }
    // Whatever the patch as a whole does, no hair points up the picture: where a
    // mane curls round to the forehead, keeping one sense all the way along it
    // would end with the forelock standing on end.
    for (size_t i = 0; i < count && !getenv("WBM_HAIR_DIR"); i++)
        if (dy[i] < -0.2f) dx[i] = -dx[i], dy[i] = -dy[i];
    Image map;
    map.w = w, map.h = h;
    map.px.assign(count * 4, 255);
    size_t covered = 0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t i = (size_t)y * w + x;
            const uint8_t *c = &base.px[i * 4];
            float brightest = powf(std::max({c[0], c[1], c[2]}) / 255.0f, 2.2f);
            float s = has_spec ? sample_luminance(spec, (x + 0.5f) / w, (y + 0.5f) / h) : 0;
            float coat = coat_at(name, has_spec, s, (x + 0.5f) / w, (y + 0.5f) / h, brightest) * 0.5f;
            if (has_tack && tack_at(x, y) > 0.07f) coat = 0; // saddle, straps and the like, as the second picture shows them
            float kind = std::max(coat * (1 - mask[i]), mask[i]);
            uint8_t *o = &map.px[i * 4];
            o[0] = (uint8_t)lroundf(127.5f + 127.5f * dx[i] * trust[i]);
            o[1] = (uint8_t)lroundf(127.5f + 127.5f * dy[i] * trust[i]);
            o[2] = (uint8_t)lroundf(kind * 255);
            covered += mask[i] > 0.5f;
        }
    }
    make_folders(out + "/masks");
    {   // What was taken for long hair, white on black, to look at.
        Image shown;
        shown.w = w, shown.h = h;
        shown.px.assign(count * 4, 255);
        for (size_t i = 0; i < count; i++) shown.px[i * 4] = shown.px[i * 4 + 1] = shown.px[i * 4 + 2] = (uint8_t)lroundf(mask[i] * 255);
        write_png(out + "/masks/" + name + "_longhair.png", shown);
    }
    {   // The hair's own colour, as the middle value of what is painted where the
        // long hair is: the renderer colours long hair with it, so that a strap or
        // a patch of coat lying across the mane in the texture does not show in the hair.
        std::vector<uint8_t> channel[3];
        for (size_t i = 0; i < count; i++)
            if (mask[i] > 0.5f)
                for (int k = 0; k < 3; k++) channel[k].push_back(base.px[i * 4 + k]);
        if (FILE *f = channel[0].empty() ? nullptr : fopen((out + "/" + name + "_hair.txt").c_str(), "w")) {
            int middle[3];
            for (int k = 0; k < 3; k++) {
                std::nth_element(channel[k].begin(), channel[k].begin() + (long)channel[k].size() / 2, channel[k].end());
                middle[k] = channel[k][channel[k].size() / 2];
            }
            fprintf(f, "%d %d %d\n", middle[0], middle[1], middle[2]);
            fclose(f);
        }
    }
    std::string path = out + "/" + name + "_hair.png";
    if (!write_png(path, map)) {
        fprintf(stderr, "%s could not be written\n", path.c_str());
        return 1;
    }
    printf("%s: %dx%d, long hair on %.1f%% of it%s%s\n", path.c_str(), w, h, 100.0 * (double)covered / (double)count, has_normal ? ", normal map used" : ", no normal map found",
           has_tack ? ", gear from the second picture" : ", no picture of the gear: saddle and straps count as coat");
    return 0;
}

// ---- compressed files -------------------------------------------------------------------
//
// A texture as a .dds file in the game's own kinds of compression (DXT1, or DXT5 where alpha holds
// something), with every smaller size down to one pixel: the renderer hands such a file to the GPU as it
// is, where a PNG is unpacked to four bytes a pixel. A block of sixteen pixels becomes two colours and,
// for each pixel, which of four blends of the two it is nearest.

static uint16_t pack565(const float *c)
{
    auto q = [](float v, float top) { return (uint16_t)lroundf(std::min(255.0f, std::max(0.0f, v)) * top / 255.0f); };
    return (uint16_t)(q(c[0], 31) << 11 | q(c[1], 63) << 5 | q(c[2], 31));
}

static void unpack565(uint16_t v, float *c)
{
    c[0] = (float)((v >> 11) * 255 / 31), c[1] = (float)(((v >> 5) & 63) * 255 / 63), c[2] = (float)((v & 31) * 255 / 31);
}

// Which blend each pixel takes, and how far off the block is in all.
static float assign(const float px[16][3], uint16_t c0, uint16_t c1, uint8_t *which)
{
    float p[4][3];
    unpack565(c0, p[0]), unpack565(c1, p[1]);
    for (int k = 0; k < 3; k++) p[2][k] = (2 * p[0][k] + p[1][k]) / 3, p[3][k] = (p[0][k] + 2 * p[1][k]) / 3;
    float total = 0;
    for (int i = 0; i < 16; i++) {
        float best = 1e30f;
        for (int n = 0; n < 4; n++) {
            float d = 0;
            for (int k = 0; k < 3; k++) d += (px[i][k] - p[n][k]) * (px[i][k] - p[n][k]);
            if (d < best) best = d, which[i] = (uint8_t)n;
        }
        total += best;
    }
    return total;
}

static void encode_colour(const float px[16][3], uint8_t *out)
{
    // The two colours: the ends of the pixels' spread along the line they lie nearest to.
    float mean[3] = {0, 0, 0}, cov[6] = {0, 0, 0, 0, 0, 0}, lo[3] = {255, 255, 255}, hi[3] = {0, 0, 0};
    for (int i = 0; i < 16; i++)
        for (int k = 0; k < 3; k++) mean[k] += px[i][k] / 16, lo[k] = std::min(lo[k], px[i][k]), hi[k] = std::max(hi[k], px[i][k]);
    for (int i = 0; i < 16; i++) {
        float d[3] = {px[i][0] - mean[0], px[i][1] - mean[1], px[i][2] - mean[2]};
        cov[0] += d[0] * d[0], cov[1] += d[0] * d[1], cov[2] += d[0] * d[2], cov[3] += d[1] * d[1], cov[4] += d[1] * d[2], cov[5] += d[2] * d[2];
    }
    float axis[3] = {hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]};
    for (int n = 0; n < 4; n++) {
        float next[3] = {cov[0] * axis[0] + cov[1] * axis[1] + cov[2] * axis[2], cov[1] * axis[0] + cov[3] * axis[1] + cov[4] * axis[2],
                         cov[2] * axis[0] + cov[4] * axis[1] + cov[5] * axis[2]};
        float length = sqrtf(next[0] * next[0] + next[1] * next[1] + next[2] * next[2]);
        if (length < 1e-6f) break;
        for (int k = 0; k < 3; k++) axis[k] = next[k] / length;
    }
    float length = sqrtf(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]), least = 1e30f, most = -1e30f;
    if (length > 1e-6f)
        for (float &a : axis) a /= length;
    for (int i = 0; i < 16; i++) {
        float t = (px[i][0] - mean[0]) * axis[0] + (px[i][1] - mean[1]) * axis[1] + (px[i][2] - mean[2]) * axis[2];
        least = std::min(least, t), most = std::max(most, t);
    }
    float e0[3], e1[3];
    for (int k = 0; k < 3; k++) e0[k] = mean[k] + axis[k] * most, e1[k] = mean[k] + axis[k] * least;
    uint16_t c0 = pack565(e0), c1 = pack565(e1);
    uint8_t which[16];
    float error = assign(px, c0, c1, which);
    // Once more: with every pixel's blend fixed, the two colours that fit them best.
    for (int pass = 0; pass < 2 && c0 != c1; pass++) {
        static const float share[4] = {1.0f, 0.0f, 2.0f / 3.0f, 1.0f / 3.0f};
        float aa = 0, ab = 0, bb = 0, ax[3] = {0, 0, 0}, bx[3] = {0, 0, 0};
        for (int i = 0; i < 16; i++) {
            float a = share[which[i]], b = 1 - a;
            aa += a * a, ab += a * b, bb += b * b;
            for (int k = 0; k < 3; k++) ax[k] += a * px[i][k], bx[k] += b * px[i][k];
        }
        float det = aa * bb - ab * ab;
        if (fabsf(det) < 1e-4f) break;
        float n0[3], n1[3];
        for (int k = 0; k < 3; k++) n0[k] = (ax[k] * bb - bx[k] * ab) / det, n1[k] = (bx[k] * aa - ax[k] * ab) / det;
        uint16_t t0 = pack565(n0), t1 = pack565(n1);
        uint8_t again[16];
        float better = assign(px, t0, t1, again);
        if (better >= error) break;
        error = better, c0 = t0, c1 = t1;
        memcpy(which, again, 16);
    }
    // The first colour has to be the larger number, or the block is read as three blends and a hole.
    if (c0 < c1) {
        std::swap(c0, c1);
        static const uint8_t other[4] = {1, 0, 3, 2};
        for (uint8_t &w : which) w = other[w];
    }
    if (c0 == c1) memset(which, 0, 16);
    uint32_t bits = 0;
    for (int i = 0; i < 16; i++) bits |= (uint32_t)which[i] << (2 * i);
    memcpy(out, &c0, 2), memcpy(out + 2, &c1, 2), memcpy(out + 4, &bits, 4);
}

static void encode_alpha(const uint8_t *a, uint8_t *out)
{
    uint8_t hi = 0, lo = 255;
    for (int i = 0; i < 16; i++) hi = std::max(hi, a[i]), lo = std::min(lo, a[i]);
    out[0] = hi, out[1] = lo; // the larger first: eight blends
    uint64_t bits = 0;
    for (int i = 0; i < 16 && hi != lo; i++) {
        int best = 0, off = 1000;
        for (int n = 0; n < 8; n++) {
            int value = n == 0 ? hi : n == 1 ? lo : ((8 - n) * hi + (n - 1) * lo) / 7;
            if (abs(value - a[i]) < off) off = abs(value - a[i]), best = n;
        }
        bits |= (uint64_t)best << (3 * i);
    }
    memcpy(out + 2, &bits, 6);
}

// `colour`: the picture is colour as the eye sees it, and its smaller sizes are averaged as light.
static bool write_dds(const std::string &path, const Image &img, bool colour)
{
    bool alpha = false;
    for (size_t i = 3; i < img.px.size() && !alpha; i += 4) alpha = img.px[i] != 255;
    const size_t block = alpha ? 16 : 8;
    static float to_light[256];
    static bool ready = false;
    if (!ready) {
        for (int v = 0; v < 256; v++) to_light[v] = powf((float)v / 255.0f, 2.2f);
        ready = true;
    }
    std::vector<uint8_t> file(128, 0);
    uint32_t levels = 0;
    Image level = img;
    for (;;) {
        const int w = level.w, h = level.h, across = (w + 3) / 4, down = (h + 3) / 4;
        size_t at = file.size();
        file.resize(at + (size_t)across * (size_t)down * block);
        uint8_t *out = file.data() + at;
        const Image *from = &level;
        dispatch_apply((size_t)down, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t by) {
          for (int bx = 0; bx < across; bx++) {
              float px[16][3];
              uint8_t a[16];
              for (int i = 0; i < 16; i++) { // past the edge, the edge again
                  int x = std::min(bx * 4 + i % 4, w - 1), y = std::min((int)by * 4 + i / 4, h - 1);
                  const uint8_t *p = &from->px[((size_t)y * (size_t)w + (size_t)x) * 4];
                  px[i][0] = p[0], px[i][1] = p[1], px[i][2] = p[2], a[i] = p[3];
              }
              uint8_t *to = out + (by * (size_t)across + (size_t)bx) * block;
              if (alpha) encode_alpha(a, to);
              encode_colour(px, to + (alpha ? 8 : 0));
          }
        });
        levels++;
        if (w == 1 && h == 1) break;
        Image half;
        half.w = std::max(w / 2, 1), half.h = std::max(h / 2, 1);
        half.px.resize((size_t)half.w * (size_t)half.h * 4);
        for (int y = 0; y < half.h; y++)
            for (int x = 0; x < half.w; x++)
                for (int k = 0; k < 4; k++) {
                    float sum = 0;
                    for (int n = 0; n < 4; n++) {
                        int sx = std::min(x * 2 + n % 2, w - 1), sy = std::min(y * 2 + n / 2, h - 1);
                        uint8_t v = level.px[((size_t)sy * (size_t)w + (size_t)sx) * 4 + (size_t)k];
                        sum += colour && k < 3 ? to_light[v] : (float)v / 255.0f;
                    }
                    sum /= 4;
                    half.px[((size_t)y * (size_t)half.w + (size_t)x) * 4 + (size_t)k] = (uint8_t)lroundf((colour && k < 3 ? powf(sum, 1.0f / 2.2f) : sum) * 255.0f);
                }
        level = std::move(half);
    }
    auto put = [&](size_t at, uint32_t v) { memcpy(&file[at], &v, 4); };
    memcpy(&file[0], "DDS ", 4);
    put(4, 124), put(8, 0x000A1007); // caps, height, width, pixel format, mipmap count, linear size
    put(12, (uint32_t)img.h), put(16, (uint32_t)img.w), put(20, (uint32_t)(((size_t)img.w + 3) / 4 * (((size_t)img.h + 3) / 4) * block)), put(28, levels);
    put(76, 32), put(80, 4); // a pixel format named by four letters
    memcpy(&file[84], alpha ? "DXT5" : "DXT1", 4);
    put(108, 0x00401008); // a texture, with smaller sizes
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = fwrite(file.data(), 1, file.size(), f) == file.size();
    return fclose(f) == 0 && ok;
}

// One of a pack's files, in the kind asked for; a file of the same name in the other kind makes way.
static bool write_texture(const std::string &stem, const Image &img, bool dds, bool colour)
{
    unlink((stem + (dds ? ".png" : ".dds")).c_str());
    return dds ? write_dds(stem + ".dds", img, colour) : write_png(stem + ".png", img);
}

// One channel of a picture at a place given in shares of its width and height, between its pixels.
static float sample_channel(const Image &m, int channel, float u, float v)
{
    float x = std::min(std::max(u * (float)m.w - 0.5f, 0.0f), (float)(m.w - 1)), y = std::min(std::max(v * (float)m.h - 0.5f, 0.0f), (float)(m.h - 1));
    int x0 = (int)x, y0 = (int)y, x1 = std::min(x0 + 1, m.w - 1), y1 = std::min(y0 + 1, m.h - 1);
    float fx = x - (float)x0, fy = y - (float)y0;
    auto at = [&](int px, int py) { return (float)m.px[((size_t)py * (size_t)m.w + (size_t)px) * 4 + (size_t)channel]; };
    return (at(x0, y0) * (1 - fx) + at(x1, y0) * fx) * (1 - fy) + (at(x0, y1) * (1 - fx) + at(x1, y1) * fx) * fy;
}

// A cut-out's alpha that is all or nothing (leaves, twigs, fences) has edges in steps of whole texels, and in
// an enlarged picture in steps of the texels it was enlarged from. Cut at the shader's threshold, those steps
// are what is seen, however the renderer smooths its pixels. Softened with a tent `step` pixels to each side
// (bilinear filtering of the picture it came from, put back), the threshold instead runs straight across the
// steps. Features one source texel wide keep three quarters of their alpha, so stay above any threshold the
// game uses. Alpha with values in between (blending, or already soft) is left alone. Returns whether it acted.
static bool soften_cutout(Image &m, int step)
{
    size_t n = (size_t)m.w * (size_t)m.h, clear = 0, solid = 0;
    for (size_t i = 0; i < n; i++) {
        uint8_t a = m.px[i * 4 + 3];
        clear += a < 8, solid += a > 247;
    }
    // Something cut out, and nothing in between: an alpha that is solid all over has no edges to soften.
    if (step < 1 || clear + solid < n - n / 100 || clear < n / 200 || solid < n / 200) return false;
    std::vector<float> a(n), b(n);
    for (size_t i = 0; i < n; i++) a[i] = m.px[i * 4 + 3] > 127 ? 1.0f : 0.0f;
    auto pass = [&](const std::vector<float> &from, std::vector<float> &to, bool across) {
        int len = across ? m.w : m.h, other = across ? m.h : m.w;
        for (int o = 0; o < other; o++)
            for (int i = 0; i < len; i++) {
                float sum = 0, weight = 0;
                for (int d = -step + 1; d < step; d++) {
                    int j = std::min(std::max(i + d, 0), len - 1);
                    float w = (float)(step - abs(d));
                    sum += w * from[across ? (size_t)o * (size_t)m.w + (size_t)j : (size_t)j * (size_t)m.w + (size_t)o];
                    weight += w;
                }
                to[across ? (size_t)o * (size_t)m.w + (size_t)i : (size_t)i * (size_t)m.w + (size_t)o] = sum / weight;
            }
    };
    pass(a, b, true);
    pass(b, a, false);
    for (size_t i = 0; i < n; i++) m.px[i * 4 + 3] = (uint8_t)lroundf(a[i] * 255.0f);
    return true;
}

// How far a channel's values spread, in steps of 255: next to nothing for one that holds nothing.
static float spread(const Image &m, int channel)
{
    double sum = 0, squares = 0;
    size_t n = (size_t)m.w * (size_t)m.h, step = std::max<size_t>(1, n / 20000);
    size_t taken = 0;
    for (size_t i = 0; i < n; i += step, taken++) {
        double v = m.px[i * 4 + (size_t)channel];
        sum += v, squares += v * v;
    }
    double mean = sum / (double)taken;
    return (float)sqrt(std::max(0.0, squares / (double)taken - mean * mean));
}

// See the top of the file.
static int import_pictures(const std::string &in, const std::string &out, float gain, bool normals_up, bool dds, bool hard_alpha)
{
    std::map<std::string, std::string> game = game_textures();
    if (game.empty()) {
        fprintf(stderr, "no texture files found; set WBM_GAME to the game's folder\n");
        return 1;
    }
    // Which normal maps the game's materials put with a texture: its own word on what a picture named
    // <texture>_normal stands in for. (The module last played is read with the game's own files.)
    std::map<std::string, std::set<std::string>> normals_of;
    std::set<std::string> in_materials;
    {
        std::string root, module = "Native";
        const char *home = getenv("HOME");
        if (const char *g = getenv("WBM_GAME")) root = g;
        else if (home) root = std::string(home) + "/Library/Application Support/Steam/steamapps/common/MountBlade Warband";
        if (FILE *f = fopen((std::string(home ? home : "") + "/Library/Application Support/MBWarband/last_module_warband").c_str(), "r")) {
            char text[512];
            size_t n = fread(text, 1, sizeof text - 1, f);
            fclose(f);
            std::string last(text, n);
            while (!last.empty() && isspace((unsigned char)last.back())) last.pop_back();
            if (!last.empty()) module = last;
        }
        BrfLibrary library;
        library.scan(root + "/CommonRes");
        library.scan(root + "/Modules/" + module + "/Resource");
        auto plain = [](std::string n) {
            for (char &c : n) c = (char)tolower((unsigned char)c);
            if (n.size() > 4 && !n.compare(n.size() - 4, 4, ".dds")) n.resize(n.size() - 4);
            return n;
        };
        for (auto &m : library.materials) {
            std::string diffuse = plain(m.second.diffuse), normal = plain(m.second.normal);
            in_materials.insert(diffuse);
            if (!normal.empty() && normal != "none" && game.count(normal)) normals_of[diffuse].insert(normal);
        }
    }
    std::map<std::string, std::string> pictures; // lower-case name without extension -> path
    DIR *d = opendir(in.c_str());
    if (!d) {
        fprintf(stderr, "cannot read the folder %s\n", in.c_str());
        return 1;
    }
    while (struct dirent *e = readdir(d)) {
        std::string name = e->d_name;
        size_t dot = name.rfind('.');
        if (name[0] == '.' || dot == std::string::npos) continue;
        std::string base = name.substr(0, dot);
        for (char &c : base) c = (char)tolower((unsigned char)c);
        pictures[base] = in + "/" + name;
    }
    closedir(d);
    make_folders(out);
    // A set of pictures may go by a shorter name than the game's texture: nordic_helmets_basecolor, for the
    // game's nordic_helmets_diff. Where the game has no texture of a set's name but has one with such an
    // ending, the set is that texture's.
    {
        std::map<std::string, std::string> renamed;
        for (auto &p : pictures) {
            std::string name = p.first;
            for (const char *suffix : {"_basecolor", "_base_color", "_albedo", "_diffuse", "_normal", "_metallicroughness", "_metallic", "_metalness", "_metal",
                                       "_roughness", "_rough", "_orm", "_ao", "_ambientocclusion", "_height", "_displacement", "_opacity", "_alpha",
                                       "_anisotropy", "_sheen"}) {
                if (!ends_with(name, suffix)) continue;
                std::string owner = name.substr(0, name.size() - strlen(suffix));
                if (game.count(owner) || game.count(name)) break;
                for (const char *theirs : {"_diff", "_diffuse", "_d", "_color", "_col"})
                    if (game.count(owner + theirs)) {
                        name = owner + theirs + suffix;
                        break;
                    }
                break;
            }
            renamed.emplace(name, p.second);
        }
        pictures.swap(renamed);
    }

    struct Maps {
        std::string metal, rough, packed, occlusion;
    };
    std::map<std::string, Maps> maps; // by the texture they belong to
    std::map<std::string, std::string> heights; // by the texture they belong to
    std::map<std::string, std::string> flows, sheens; // glTF's anisotropy and sheen pictures, by the texture they belong to
    int hair_maps = 0, sheen_maps = 0;
    int textures = 0, normal_maps = 0, pbr_maps = 0, depth_maps = 0, skipped = 0, unread = 0, into_alpha = 0, softened = 0;
    // One picture may stand in for several of the game's files: a normal map shared out under more than one name.
    std::vector<std::pair<std::string, std::string>> work; // name of the game's, path
    std::set<std::string> named_normals;
    for (auto &p : pictures) {
        std::string name = p.first;
        bool taken = false, named_normal = false;
        for (const char *suffix : {"_ao", "_height", "_ambientocclusion", "_displacement"}) {
            std::string owner = ends_with(name, suffix) ? name.substr(0, name.size() - strlen(suffix)) : "";
            if (owner.empty() || !game.count(owner)) continue;
            // The game calls some of its normal maps <name>_height. Beside a base colour or a normal map
            // named the other way, a picture of that name is the height it says it is.
            if (game.count(name) && !pictures.count(owner + "_basecolor") && !pictures.count(owner + "_normal")) continue;
            taken = true;
            bool height = !strcmp(suffix, "_height") || !strcmp(suffix, "_displacement");
            if (height) heights[owner] = p.second;
            else maps[owner].occlusion = p.second;
        }
        for (const char *suffix : {"_opacity", "_alpha"}) { // goes into its texture's alpha, below
            std::string owner = ends_with(name, suffix) ? name.substr(0, name.size() - strlen(suffix)) : "";
            if (!owner.empty() && game.count(owner) && !game.count(name)) taken = true;
        }
        for (const char *suffix : {"_anisotropy", "_sheen"}) {
            std::string owner = ends_with(name, suffix) ? name.substr(0, name.size() - strlen(suffix)) : "";
            if (owner.empty() || !game.count(owner) || game.count(name)) continue;
            (suffix[1] == 'a' ? flows : sheens)[owner] = p.second;
            taken = true;
        }
        if (taken) continue;
        // A picture named for what it is, not for the game's file: the texture's own name is under it.
        for (const char *suffix : {"_basecolor", "_base_color", "_albedo", "_diffuse"}) {
            std::string owner = ends_with(name, suffix) ? name.substr(0, name.size() - strlen(suffix)) : "";
            if (!owner.empty() && game.count(owner) && !game.count(name)) name = owner;
        }
        const struct {
            const char *suffix;
            int kind;
        } kinds[] = {{"_metallicroughness", 2}, {"_metallic", 0}, {"_metalness", 0}, {"_metal", 0}, {"_roughness", 1}, {"_rough", 1}, {"_orm", 2}};
        for (auto &k : kinds) {
            if (!ends_with(name, k.suffix)) continue;
            std::string owner = name.substr(0, name.size() - strlen(k.suffix));
            if (!game.count(owner)) continue; // a texture of the game's may itself end that way
            (k.kind == 0 ? maps[owner].metal : k.kind == 1 ? maps[owner].rough : maps[owner].packed) = p.second;
            taken = true;
            break;
        }
        if (taken || ends_with(name, "_pbr")) continue; // a map in the renderer's own layout is one for the pack as it is
        if (ends_with(name, "_normal") && !game.count(name)) {
            std::string owner = name.substr(0, name.size() - 7);
            std::vector<std::string> found(normals_of[owner].begin(), normals_of[owner].end());
            if (found.empty() && in_materials.count(owner)) {
                // The shaders of the materials that wear this texture read no normal map: there is nothing to replace.
                unread++;
                continue;
            }
            for (const char *theirs : {"_normalmap", "_n", "_nm", "_bump", "_normal_map", "_height"})
                if (found.empty() && game.count(owner + theirs)) found.push_back(owner + theirs);
            if (found.empty()) {
                fprintf(stderr, "%s: the game has no normal map that goes with %s; left out\n", p.second.c_str(), owner.c_str());
                skipped++;
                continue;
            }
            for (const std::string &f : found) work.push_back({f, p.second}), named_normals.insert(f);
            continue;
        }
        work.push_back({name, p.second});
    }
    for (auto &p : work) {
        const std::string &name = p.first;
        bool named_normal = named_normals.count(name) != 0;
        auto original = game.find(name);
        Image img;
        if (original == game.end()) {
            fprintf(stderr, "%s: the game has no texture of that name; left out\n", p.second.c_str());
            skipped++;
            continue;
        }
        if (!read_picture(p.second, &img)) {
            fprintf(stderr, "%s: cannot be read; left out\n", p.second.c_str());
            skipped++;
            continue;
        }
        Image was;
        bool have_original = read_dds(original->second, &was);
        bool normal = named_normal || name.find("normalmap") != std::string::npos || name.find("bump") != std::string::npos || ends_with(name, "_n") || ends_with(name, "_nm") || ends_with(name, "_normal");
        size_t count = (size_t)img.w * (size_t)img.h;
        if (normal) {
            if (normals_up)
                for (size_t i = 0; i < count; i++) img.px[i * 4 + 1] = (uint8_t)(255 - img.px[i * 4 + 1]);
            // The textures of rooms keep a normal's x in alpha, and their shaders read it there.
            bool x_in_alpha = have_original && spread(was, 3) > 4.0f && spread(was, 0) <= 4.0f; // and nothing in red
            if (x_in_alpha && spread(img, 3) <= 4.0f) {
                for (size_t i = 0; i < count; i++) img.px[i * 4 + 3] = img.px[i * 4];
                into_alpha++;
            }
            normal_maps++;
        } else {
            if (gain != 1.0f) {
                uint8_t back[256]; // per value: decoded, divided, encoded again
                for (int v = 0; v < 256; v++) {
                    float c = (float)v / 255.0f, lin = (c <= 0.04045f ? c / 12.92f : powf((c + 0.055f) / 1.055f, 2.4f)) / gain;
                    float enc = lin <= 0.0031308f ? lin * 12.92f : 1.055f * powf(lin, 1.0f / 2.4f) - 0.055f;
                    back[v] = (uint8_t)lroundf(std::min(1.0f, std::max(0.0f, enc)) * 255.0f);
                }
                for (size_t i = 0; i < count; i++)
                    for (size_t k = 0; k < 3; k++) img.px[i * 4 + k] = back[img.px[i * 4 + k]];
            }
            // Alpha cuts leaves out and blends faces. It may come as a picture of its own;
            // a texture that has simply lost it gets the game's back.
            Image opacity;
            std::string given;
            for (const char *suffix : {"_opacity", "_alpha"})
                if (given.empty() && pictures.count(name + suffix)) given = pictures[name + suffix];
            if (!given.empty() && read_picture(given, &opacity)) {
                for (int y = 0; y < img.h; y++)
                    for (int x = 0; x < img.w; x++)
                        img.px[((size_t)y * (size_t)img.w + (size_t)x) * 4 + 3] = (uint8_t)lroundf(sample_channel(opacity, 0, ((float)x + 0.5f) / (float)img.w, ((float)y + 0.5f) / (float)img.h));
            } else if (have_original && spread(was, 3) > 1.0f && spread(img, 3) <= 1.0f) {
                for (int y = 0; y < img.h; y++)
                    for (int x = 0; x < img.w; x++)
                        img.px[((size_t)y * (size_t)img.w + (size_t)x) * 4 + 3] = (uint8_t)lroundf(sample_channel(was, 3, ((float)x + 0.5f) / (float)img.w, ((float)y + 0.5f) / (float)img.h));
                fprintf(stderr, "%s: the picture has no alpha where the game's texture has; the game's is put back\n", name.c_str());
            }
            // The enlargement's step: how many of this picture's pixels one of the game's texels became (2 at least:
            // even an alpha drawn at this size has edges a pixel at a time).
            int step = have_original && was.w > 0 ? std::max(2, (int)lroundf((float)img.w / (float)was.w)) : 2;
            if (!hard_alpha && soften_cutout(img, step)) softened++;
            textures++;
        }
        if (!write_texture(out + "/" + name, img, dds, !normal)) {
            fprintf(stderr, "cannot write %s/%s\n", out.c_str(), name.c_str());
            return 1;
        }
    }
    for (auto &m : maps) {
        const std::string &name = m.first;
        Image metal, rough, packed, old;
        bool has_metal = !m.second.metal.empty() && read_picture(m.second.metal, &metal), has_rough = !m.second.rough.empty() && read_picture(m.second.rough, &rough);
        bool has_packed = !m.second.packed.empty() && read_picture(m.second.packed, &packed);
        Image occlusion;
        bool has_occlusion = !m.second.occlusion.empty() && read_picture(m.second.occlusion, &occlusion);
        bool has_old = read_picture(out + "/" + name + "_pbr.png", &old) || read_dds(out + "/" + name + "_pbr.dds", &old);
        if (!has_metal && !has_rough && !has_packed && !has_occlusion) {
            fprintf(stderr, "%s: its maps cannot be read; left out\n", name.c_str());
            skipped++;
            continue;
        }
        Image map;
        for (const Image *i : {&metal, &rough, &packed, &occlusion})
            if (i->w > map.w) map.w = i->w, map.h = i->h;
        map.px.assign((size_t)map.w * (size_t)map.h * 4, 255);
        if (!(has_metal || has_packed) && !has_old) fprintf(stderr, "%s: no metalness given and none in the pack; taken as no metal\n", name.c_str());
        if (!(has_rough || has_packed) && !has_old) fprintf(stderr, "%s: no roughness given and none in the pack; taken as 0.7\n", name.c_str());
        // A horse's coat, where no map in the pack has it: worked out as texture_tool pbr does, from the game's own pictures.
        Image base, spec;
        bool coat = is_horse(name) && !has_old && game.count(name) && read_dds(game[name], &base);
        bool has_spec = coat && load_specular(game, name, &spec);
        for (int y = 0; y < map.h; y++)
            for (int x = 0; x < map.w; x++) {
                float u = ((float)x + 0.5f) / (float)map.w, v = ((float)y + 0.5f) / (float)map.h;
                uint8_t *px = &map.px[((size_t)y * (size_t)map.w + (size_t)x) * 4];
                px[0] = (uint8_t)lroundf(has_metal ? sample_channel(metal, 0, u, v) : has_packed ? sample_channel(packed, 2, u, v) : has_old ? sample_channel(old, 0, u, v) : 0.0f);
                px[1] = (uint8_t)lroundf(has_rough ? sample_channel(rough, 0, u, v) : has_packed ? sample_channel(packed, 1, u, v) : has_old ? sample_channel(old, 1, u, v) : 0.7f * 255.0f);
                px[2] = (uint8_t)lroundf(has_old ? sample_channel(old, 2, u, v) : 0.0f);
                // Alpha: how much of the light from all round reaches the spot. (glTF's packed map has it in red, but it is often left white there.)
                px[3] = (uint8_t)lroundf(has_occlusion ? sample_channel(occlusion, 0, u, v) : has_old ? sample_channel(old, 3, u, v) : 255.0f);
                if (coat) {
                    float brightest = powf(std::max({sample_channel(base, 0, u, v), sample_channel(base, 1, u, v), sample_channel(base, 2, u, v)}) / 255.0f, 2.2f);
                    px[2] = (uint8_t)lroundf(coat_at(name, has_spec, has_spec ? sample_luminance(spec, u, v) : 0.0f, u, v, brightest) * 255.0f);
                }
            }
        if (!write_texture(out + "/" + name + "_pbr", map, dds, false)) {
            fprintf(stderr, "cannot write %s/%s_pbr.png\n", out.c_str(), name.c_str());
            return 1;
        }
        pbr_maps++;
    }
    for (auto &h : heights) { // as they are, under a name the game has no texture by
        Image img;
        if (!read_picture(h.second, &img) || !write_texture(out + "/" + h.first + "_depth", img, dds, false)) {
            fprintf(stderr, "%s: cannot be read; left out\n", h.second.c_str());
            skipped++;
            continue;
        }
        depth_maps++;
    }
    // glTF's KHR_materials_anisotropy picture: red and green the direction along the surface (right and up the
    // picture, each 0.5 + 0.5 x), blue the strength. It becomes the renderer's hair map (see k_fur_function):
    // the same direction with green turned to point down the picture, and blue as it is, which a horse's set
    // gives as the hair map's classes (1 long hair, 0.5 coat, 0 neither). Kept as PNG: a compressed block would
    // bend the directions. Beside it, as texture_tool hair does, the long hair's colour, from the base colour.
    for (auto &f : flows) {
        const std::string &name = f.first;
        Image flow, base;
        if (!read_picture(f.second, &flow)) {
            fprintf(stderr, "%s: cannot be read; left out\n", f.second.c_str());
            skipped++;
            continue;
        }
        bool has_base = false;
        for (const char *suffix : {"_basecolor", "_base_color", "_albedo", "_diffuse", ""})
            if (!has_base && pictures.count(name + suffix)) has_base = read_picture(pictures[name + suffix], &base);
        Image hair;
        hair.w = flow.w, hair.h = flow.h;
        hair.px.assign((size_t)flow.w * (size_t)flow.h * 4, 255);
        std::vector<uint8_t> channel[3];
        for (size_t i = 0; i < (size_t)flow.w * (size_t)flow.h; i++) {
            uint8_t strength = flow.px[i * 4 + 2];
            hair.px[i * 4 + 0] = strength ? flow.px[i * 4 + 0] : 128;
            hair.px[i * 4 + 1] = strength ? (uint8_t)(255 - flow.px[i * 4 + 1]) : 128;
            hair.px[i * 4 + 2] = strength;
            if (has_base && strength > 191 && (i % 7) == 0) {
                float u = ((float)(i % (size_t)flow.w) + 0.5f) / (float)flow.w, v = ((float)(i / (size_t)flow.w) + 0.5f) / (float)flow.h;
                for (int k = 0; k < 3; k++) channel[k].push_back((uint8_t)lroundf(sample_channel(base, k, u, v)));
            }
        }
        if (!write_png(out + "/" + name + "_hair.png", hair)) {
            fprintf(stderr, "cannot write %s/%s_hair.png\n", out.c_str(), name.c_str());
            return 1;
        }
        unlink((out + "/" + name + "_hair.txt").c_str());
        if (FILE *t = channel[0].empty() ? nullptr : fopen((out + "/" + name + "_hair.txt").c_str(), "w")) {
            int middle[3];
            for (int k = 0; k < 3; k++) {
                std::nth_element(channel[k].begin(), channel[k].begin() + (long)channel[k].size() / 2, channel[k].end());
                middle[k] = channel[k][channel[k].size() / 2];
            }
            fprintf(t, "%d %d %d\n", middle[0], middle[1], middle[2]);
            fclose(t);
        }
        hair_maps++;
    }
    // glTF's KHR_materials_sheen picture: the sheen's colour in red, green and blue, its roughness in alpha.
    // Taken as it is (k_fur_function reads it), as PNG, so that the roughness keeps its precision.
    for (auto &f : sheens) {
        Image img;
        if (!read_picture(f.second, &img) || !write_png(out + "/" + f.first + "_sheen.png", img)) {
            fprintf(stderr, "%s: cannot be read or written; left out\n", f.second.c_str());
            skipped++;
            continue;
        }
        unlink((out + "/" + f.first + "_sheen.dds").c_str());
        sheen_maps++;
    }
    printf("%d textures, %d normal maps, %d maps of metalness, roughness and occlusion and %d of heights written to %s; %d pictures left out", textures, normal_maps, pbr_maps, depth_maps, out.c_str(), skipped);
    if (hair_maps || sheen_maps) printf("; %d hair maps from anisotropy and %d sheen maps", hair_maps, sheen_maps);
    if (unread) printf(", %d normal maps for textures whose materials read none", unread);
    if (into_alpha) printf("; %d normal maps written the way the game keeps them, x in alpha", into_alpha);
    if (softened) printf("; %d cut-outs given soft edges (--hard-alpha: not)", softened);
    printf("\n");
    return 0;
}

int main(int argc, char **argv)
{
    if (argc >= 4 && !strcmp(argv[1], "import")) {
        float gain = 1.0f;
        bool normals_up = true, dds = false, hard_alpha = false;
        for (int i = 4; i < argc; i++) {
            if (!strcmp(argv[i], "--gain") && i + 1 < argc) gain = std::max(0.01f, (float)atof(argv[++i]));
            else if (!strcmp(argv[i], "--normals") && i + 1 < argc) normals_up = strcmp(argv[++i], "directx") != 0;
            else if (!strcmp(argv[i], "--dds")) dds = true;
            else if (!strcmp(argv[i], "--hard-alpha")) hard_alpha = true;
        }
        return import_pictures(argv[2], argv[3], gain, normals_up, dds, hard_alpha);
    }
    if ((argc == 5 || argc == 6) && !strcmp(argv[1], "hair")) return hair_map(argv[2], argv[3], argv[4], argc == 6 ? argv[5] : nullptr);
    if (argc < 3 || (strcmp(argv[1], "export") && strcmp(argv[1], "pbr"))) {
        fprintf(stderr, "usage: texture_tool export <out-folder> [name...]\n       texture_tool pbr <out-folder> [name...]\n"
                        "       texture_tool hair <picture of the hair alone> <name> <out-folder> [picture of the saddle and straps alone]\n"
                        "       texture_tool import <folder of pictures> <pack-folder> [--gain g] [--normals directx] [--dds] [--hard-alpha]\n");
        return 2;
    }
    bool pbr = !strcmp(argv[1], "pbr");
    std::string out = argv[2];
    make_folders(out);
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
        bool horse = is_horse(name);
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
                float fur = horse ? coat_at(name, has_spec, s, (x + 0.5f) / base.w, (y + 0.5f) / base.h, hi) : 0.0f;
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
