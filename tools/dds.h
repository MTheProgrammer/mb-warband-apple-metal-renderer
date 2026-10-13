#pragma once

// Reading the game's texture files (.dds: uncompressed, DXT1, DXT3 or DXT5),
// the largest level only. Shared by the tools.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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
