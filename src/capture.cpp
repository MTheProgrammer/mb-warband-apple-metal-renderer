// Saves frames as frames/frame_<n>.png.

#include "trace.h"

#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>

#include <cstring>
#include <string>
#include <vector>

void wb_capture_frame(int w, int h, uint32_t frame)
{
    if (w <= 0 || h <= 0) return;
    size_t row = (size_t)w * 4;
    std::vector<unsigned char> gl(row * h), top_down(row * h);
    if (!wb_gl_read_backbuffer(w, h, gl.data())) {
        wb_note("  !! capture of frame %u failed in glReadPixels", frame);
        return;
    }
    for (int y = 0; y < h; y++) memcpy(&top_down[row * y], &gl[row * (h - 1 - y)], row); // GL rows start at the bottom

    wb_save_frame_png(top_down.data(), w, h, frame, false);
}

void wb_save_frame_png(const unsigned char *pixels, int w, int h, uint32_t frame, bool bgra, const char *prefix)
{
    size_t row = (size_t)w * 4;
    char name[96];
    snprintf(name, sizeof name, "/frames/%s_%06u.png", prefix, frame);
    std::string path = std::string(g_out_dir) + name;

    bool ok = false;
    CGBitmapInfo layout = bgra ? (CGBitmapInfo)kCGBitmapByteOrder32Little | (CGBitmapInfo)kCGImageAlphaNoneSkipFirst
                               : (CGBitmapInfo)kCGImageAlphaNoneSkipLast;
    CGColorSpaceRef space = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    CGDataProviderRef provider = CGDataProviderCreateWithData(nullptr, pixels, row * h, nullptr);
    CGImageRef image = CGImageCreate(w, h, 8, 32, row, space, layout, provider, nullptr, false, kCGRenderingIntentDefault);
    CFURLRef url = CFURLCreateFromFileSystemRepresentation(nullptr, (const UInt8 *)path.c_str(), (CFIndex)path.size(), false);
    if (image && url) {
        if (CGImageDestinationRef dest = CGImageDestinationCreateWithURL(url, CFSTR("public.png"), 1, nullptr)) {
            CGImageDestinationAddImage(dest, image, nullptr);
            ok = CGImageDestinationFinalize(dest);
            CFRelease(dest);
        }
    }
    if (url) CFRelease(url);
    CGImageRelease(image);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(space);
    wb_note(ok ? "  captured frame %u -> %s" : "  !! could not write frame %u to %s", frame, path.c_str());
}
