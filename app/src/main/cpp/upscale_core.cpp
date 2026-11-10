// Tiled Real-ESRGAN inference.
//
// The image is processed in tiles so that large images fit in GPU memory. Each tile
// is padded with real neighbouring pixels ("prepadding") before inference and the
// padding is cropped off afterwards, so tiles join without visible seams.

#include "upscale_core.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "mat.h"

namespace upscaler {

int upscaleRgba(Model& u, const uint8_t* in, int w, int h, int inStride,
                uint8_t* out, int outStride, int outScale, int tile, const Progress& progress) {
    if (w <= 0 || h <= 0 || tile <= 0 || outScale <= 0 || u.scale % outScale != 0) return ERR_ARGS;
    const int scale = u.scale;
    const int f = scale / outScale;  // box-filter factor applied after inference (1 = none)
    const int ow = w * outScale, oh = h * outScale;

    // Copy input to tightly packed, un-premultiplied RGB (+ alpha plane if needed).
    // Android bitmaps store premultiplied color; the model must see the true colors.
    std::vector<uint8_t> rgb((size_t)w * h * 3);
    bool hasAlpha = false;
    for (int y = 0; y < h && !hasAlpha; y++) {
        const uint8_t* row = in + (size_t)y * inStride;
        for (int x = 0; x < w; x++)
            if (row[x * 4 + 3] != 255) { hasAlpha = true; break; }
    }
    std::vector<uint8_t> alpha;
    if (hasAlpha) alpha.resize((size_t)w * h);
    for (int y = 0; y < h; y++) {
        const uint8_t* row = in + (size_t)y * inStride;
        uint8_t* dst = &rgb[(size_t)y * w * 3];
        for (int x = 0; x < w; x++) {
            int r = row[x * 4], g = row[x * 4 + 1], b = row[x * 4 + 2], a = row[x * 4 + 3];
            if (hasAlpha) {
                alpha[(size_t)y * w + x] = (uint8_t)a;
                if (a > 0 && a < 255) {
                    r = std::min(255, (r * 255 + a / 2) / a);
                    g = std::min(255, (g * 255 + a / 2) / a);
                    b = std::min(255, (b * 255 + a / 2) / a);
                }
            }
            dst[x * 3] = (uint8_t)r;
            dst[x * 3 + 1] = (uint8_t)g;
            dst[x * 3 + 2] = (uint8_t)b;
        }
    }

    // Alpha is upscaled with bilinear interpolation (the model only handles RGB).
    std::vector<uint8_t> outAlpha;
    if (hasAlpha) {
        outAlpha.resize((size_t)ow * oh);
        ncnn::resize_bilinear_c1(alpha.data(), w, h, outAlpha.data(), ow, oh);
    }

    const int pad = u.prepadding;
    const int tilesX = (w + tile - 1) / tile;
    const int tilesY = (h + tile - 1) / tile;
    const int total = tilesX * tilesY;
    const float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};

    int done = 0;
    for (int ty = 0; ty < tilesY; ty++) {
        for (int tx = 0; tx < tilesX; tx++) {
            if (u.cancel.load()) return RESULT_CANCELLED;

            const int x0 = tx * tile, x1 = std::min(x0 + tile, w);
            const int y0 = ty * tile, y1 = std::min(y0 + tile, h);
            const int px0 = std::max(x0 - pad, 0), px1 = std::min(x1 + pad, w);
            const int py0 = std::max(y0 - pad, 0), py1 = std::min(y1 + pad, h);
            const int pw = px1 - px0, ph = py1 - py0;

            ncnn::Mat inMat = ncnn::Mat::from_pixels(&rgb[((size_t)py0 * w + px0) * 3],
                                                     ncnn::Mat::PIXEL_RGB, pw, ph, w * 3);
            inMat.substract_mean_normalize(nullptr, norm);

            ncnn::Mat outMat;
            {
                ncnn::Extractor ex = u.net.create_extractor();
                if (ex.input(u.inName.c_str(), inMat) != 0) return ERR_INFERENCE;
                if (ex.extract(u.outName.c_str(), outMat) != 0) return ERR_INFERENCE;
            }
            if (outMat.w != pw * scale || outMat.h != ph * scale || outMat.c != 3) return ERR_INFERENCE;

            // Copy the un-padded part of the tile into the output.
            const int cropX = (x0 - px0) * scale;
            const int cropY = (y0 - py0) * scale;
            const int dx0 = x0 * outScale, dx1 = x1 * outScale;
            const int dy0 = y0 * outScale, dy1 = y1 * outScale;
            const float inv = 1.f / (float)(f * f);
            const float* planes[3] = {outMat.channel(0), outMat.channel(1), outMat.channel(2)};
            const int mw = outMat.w;

            for (int dy = dy0; dy < dy1; dy++) {
                uint8_t* orow = out + (size_t)dy * outStride;
                const int sy = cropY + (dy - dy0) * f;
                for (int dx = dx0; dx < dx1; dx++) {
                    const int sx = cropX + (dx - dx0) * f;
                    const int a = hasAlpha ? outAlpha[(size_t)dy * ow + dx] : 255;
                    uint8_t* px = orow + (size_t)dx * 4;
                    for (int c = 0; c < 3; c++) {
                        const float* p = planes[c];
                        float acc = 0;
                        // Clamp each sample before averaging, so 2x equals a downscale of the visible 4x image.
                        for (int yy = 0; yy < f; yy++)
                            for (int xx = 0; xx < f; xx++)
                                acc += std::min(1.f, std::max(0.f, p[(size_t)(sy + yy) * mw + sx + xx]));
                        float v = acc * inv;
                        int iv = (int)std::lround(v * 255.f);
                        if (a != 255) iv = (iv * a + 127) / 255;  // back to premultiplied
                        px[c] = (uint8_t)iv;
                    }
                    px[3] = (uint8_t)a;
                }
            }

            done++;
            if (progress && !progress(done, total)) return ERR_CALLBACK;
        }
    }
    return RESULT_OK;
}

}  // namespace upscaler
