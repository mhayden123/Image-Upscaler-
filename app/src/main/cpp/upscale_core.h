// Platform-independent tiled upscaling core (no Android/JNI code), so it can be
// unit-tested on a desktop build of NCNN.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>

#include "net.h"

namespace upscaler {

constexpr int RESULT_OK = 0;
constexpr int RESULT_CANCELLED = 1;
constexpr int ERR_BITMAP = -1;
constexpr int ERR_ARGS = -2;
constexpr int ERR_INFERENCE = -3;
constexpr int ERR_CALLBACK = -4;

struct Model {
    ncnn::Net net;
    std::string inName;
    std::string outName;
    int scale = 4;        // the network's native upscale factor
    int prepadding = 10;  // context pixels added around each tile
    bool gpu = false;
    std::atomic<bool> cancel{false};
};

/// Progress callback: (tilesDone, tilesTotal). Return false to abort with ERR_CALLBACK.
using Progress = std::function<bool(int, int)>;

/// Upscales a premultiplied RGBA8888 image.
/// `out` must be (w*outScale) x (h*outScale); model.scale must be divisible by outScale.
int upscaleRgba(Model& model, const uint8_t* in, int w, int h, int inStride,
                uint8_t* out, int outStride, int outScale, int tileSize, const Progress& progress);

/// High-quality classic enlargement (Catmull-Rom bicubic) of a premultiplied RGBA8888
/// image to any size. This is the "regular" upscale the texture slider mixes in.
void resizeBicubicRgba(const uint8_t* in, int w, int h, int inStride,
                       uint8_t* out, int ow, int oh, int outStride);

/// out = ai * (1 - t) + classic * t, per channel. All three images are the same size.
/// `out` may be the same buffer as `ai` or `classic`.
void blendRgba(const uint8_t* ai, int aiStride, const uint8_t* classic, int classicStride,
               uint8_t* out, int outStride, int w, int h, float t);

}  // namespace upscaler
