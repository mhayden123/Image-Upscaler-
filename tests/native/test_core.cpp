// Desktop test for app/src/main/cpp/upscale_core.cpp (tiling, 2x box filter, alpha, cancel).
// Run with scripts/run_native_tests.sh, which builds NCNN for the host and a random-weight model.
#include <cstring>
#include "upscale_core.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
using namespace upscaler;

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: " __VA_ARGS__); printf("\n"); fails++; } else { printf("ok: " __VA_ARGS__); printf("\n"); } } while (0)

static void load(Model& m, const char* dir, int pad) {
    m.inName = "in0"; m.outName = "out0"; m.scale = 4; m.prepadding = pad;
    m.net.opt.use_vulkan_compute = false;
    m.net.opt.use_fp16_packed = true; m.net.opt.use_fp16_storage = true; m.net.opt.use_fp16_arithmetic = false;
    char p[512], b[512]; snprintf(p, 512, "%s/test.param", dir); snprintf(b, 512, "%s/test.bin", dir);
    if (m.net.load_param(p) || m.net.load_model(b)) { printf("load failed\n"); exit(1); }
}

static std::vector<uint8_t> makeImage(int w, int h, bool withAlpha) {
    std::vector<uint8_t> img((size_t)w * h * 4);
    std::mt19937 rng(123);
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
        uint8_t* p = &img[((size_t)y * w + x) * 4];
        int a = withAlpha ? (x * 255 / (w - 1)) : 255;
        int r = (x * 7 + y * 3 + rng() % 40) % 256, g = (y * 5 + rng() % 40) % 256, b = ((x ^ y) * 4) % 256;
        // store premultiplied, like Android bitmaps
        p[0] = (uint8_t)((r * a + 127) / 255); p[1] = (uint8_t)((g * a + 127) / 255); p[2] = (uint8_t)((b * a + 127) / 255); p[3] = (uint8_t)a;
    }
    return img;
}

static int maxDiff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    int m = 0; for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::abs(a[i] - b[i])); return m;
}

int main(int argc, char** argv) {
    const char* dir = argv[1];
    const int w = 73, h = 51;
    auto img = makeImage(w, h, false);
    const int ow = w * 4, oh = h * 4;

    Model ref; load(ref, dir, 0);
    std::vector<uint8_t> outRef((size_t)ow * oh * 4);
    int calls = 0;
    int r = upscaleRgba(ref, img.data(), w, h, w * 4, outRef.data(), ow * 4, 4, 100000, [&](int d, int t) { calls++; return true; });
    CHECK(r == RESULT_OK && calls == 1, "single-tile reference runs (r=%d, progress calls=%d)", r, calls);

    // Tiles with padding larger than the network's receptive field must match exactly (up to rounding).
    Model big; load(big, dir, 40);
    std::vector<uint8_t> outTiled((size_t)ow * oh * 4);
    int lastDone = 0, lastTotal = 0;
    r = upscaleRgba(big, img.data(), w, h, w * 4, outTiled.data(), ow * 4, 4, 16, [&](int d, int t) { lastDone = d; lastTotal = t; return true; });
    int d = maxDiff(outRef, outTiled);
    CHECK(r == RESULT_OK && d <= 1, "16px tiles + 40px padding match whole-image result (max diff %d)", d);
    CHECK(lastDone == lastTotal && lastTotal == 5 * 4, "progress reports all %d tiles (got %d/%d)", 5 * 4, lastDone, lastTotal);

    // With the app's real padding, seams should be tiny.
    Model small; load(small, dir, 10);
    r = upscaleRgba(small, img.data(), w, h, w * 4, outTiled.data(), ow * 4, 4, 16, nullptr);
    double sum = 0; int mx = 0;
    for (size_t i = 0; i < outRef.size(); i++) { int e = std::abs(outRef[i] - outTiled[i]); sum += e; mx = std::max(mx, e); }
    printf("info: 10px padding vs whole image: max diff %d, mean diff %.4f\n", mx, sum / outRef.size());

    // 2x output = 2x2 box average of the 4x output.
    std::vector<uint8_t> out2((size_t)w * 2 * h * 2 * 4);
    r = upscaleRgba(ref, img.data(), w, h, w * 4, out2.data(), w * 2 * 4, 2, 100000, nullptr);
    int d2 = 0;
    for (int y = 0; y < h * 2; y++) for (int x = 0; x < w * 2; x++) for (int c = 0; c < 4; c++) {
        int s = 0; for (int yy = 0; yy < 2; yy++) for (int xx = 0; xx < 2; xx++) s += outRef[((size_t)(y * 2 + yy) * ow + x * 2 + xx) * 4 + c];
        d2 = std::max(d2, std::abs(out2[((size_t)y * w * 2 + x) * 4 + c] - (s + 2) / 4));
    }
    CHECK(r == RESULT_OK && d2 <= 1, "2x output equals 2x2 average of 4x output (max diff %d)", d2);

    // 2x output with tiling.
    std::vector<uint8_t> out2t(out2.size());
    r = upscaleRgba(big, img.data(), w, h, w * 4, out2t.data(), w * 2 * 4, 2, 16, nullptr);
    CHECK(r == RESULT_OK && maxDiff(out2, out2t) <= 1, "2x tiled matches 2x whole (max diff %d)", maxDiff(out2, out2t));

    // Alpha handling: left column fully transparent -> output alpha 0 and color 0 (premultiplied); right column opaque.
    auto aimg = makeImage(w, h, true);
    std::vector<uint8_t> outA((size_t)ow * oh * 4);
    r = upscaleRgba(ref, aimg.data(), w, h, w * 4, outA.data(), ow * 4, 4, 100000, nullptr);
    bool premulOk = true; int a0 = outA[3], aEnd = outA[(size_t)(ow - 1) * 4 + 3];
    for (size_t i = 0; i < outA.size(); i += 4) for (int c = 0; c < 3; c++) if (outA[i + c] > outA[i + 3]) premulOk = false;
    CHECK(r == RESULT_OK && a0 == 0 && aEnd == 255 && premulOk, "alpha upscaled (left=%d right=%d) and colors stay premultiplied (%d)", a0, aEnd, premulOk);

    // Rows with padding (stride > width*4) are handled.
    std::vector<uint8_t> padded((size_t)(w * 4 + 12) * h);
    for (int y = 0; y < h; y++) memcpy(&padded[(size_t)y * (w * 4 + 12)], &img[(size_t)y * w * 4], w * 4);
    std::vector<uint8_t> outS((size_t)(ow * 4 + 20) * oh);
    r = upscaleRgba(ref, padded.data(), w, h, w * 4 + 12, outS.data(), ow * 4 + 20, 4, 100000, nullptr);
    int ds = 0;
    for (int y = 0; y < oh; y++) for (int x = 0; x < ow * 4; x++) ds = std::max(ds, std::abs(outS[(size_t)y * (ow * 4 + 20) + x] - outRef[(size_t)y * ow * 4 + x]));
    CHECK(r == RESULT_OK && ds == 0, "row strides respected (max diff %d)", ds);

    // Cancellation and callback abort.
    big.cancel = true;
    r = upscaleRgba(big, img.data(), w, h, w * 4, outTiled.data(), ow * 4, 4, 16, nullptr);
    CHECK(r == RESULT_CANCELLED, "cancel flag stops processing (r=%d)", r);
    big.cancel = false;
    r = upscaleRgba(big, img.data(), w, h, w * 4, outTiled.data(), ow * 4, 4, 16, [](int d, int) { return d < 3; });
    CHECK(r == ERR_CALLBACK, "callback returning false aborts (r=%d)", r);
    r = upscaleRgba(big, img.data(), w, h, w * 4, outTiled.data(), ow * 4, 3, 16, nullptr);
    CHECK(r == ERR_ARGS, "unsupported scale 3x rejected (r=%d)", r);

    // --- Bicubic enlargement + texture blend ---
    {
        // A flat color stays exactly flat (weights sum to 1, no ringing).
        std::vector<uint8_t> flat((size_t)w * h * 4);
        for (size_t i = 0; i < flat.size(); i += 4) { flat[i] = 200; flat[i + 1] = 100; flat[i + 2] = 50; flat[i + 3] = 255; }
        std::vector<uint8_t> big((size_t)ow * oh * 4);
        resizeBicubicRgba(flat.data(), w, h, w * 4, big.data(), ow, oh, ow * 4);
        CHECK(maxDiff(std::vector<uint8_t>(big.begin(), big.begin() + flat.size()), flat) == 0, "bicubic keeps a flat color exact");

        // A linear horizontal ramp stays linear in the interior (cubic reproduces straight lines).
        std::vector<uint8_t> ramp((size_t)w * h * 4);
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
            uint8_t* p = &ramp[((size_t)y * w + x) * 4]; p[0] = p[1] = p[2] = (uint8_t)(x * 3); p[3] = 255;
        }
        resizeBicubicRgba(ramp.data(), w, h, w * 4, big.data(), ow, oh, ow * 4);
        int rampErr = 0;
        for (int x = 8; x < ow - 8; x++) {
            float expect = ((x + 0.5f) / 4 - 0.5f) * 3;
            rampErr = std::max(rampErr, (int)std::lround(std::fabs(big[((size_t)10 * ow + x) * 4] - expect)));
        }
        CHECK(rampErr <= 1, "bicubic reproduces a linear ramp (max error %d)", rampErr);

        // Premultiplied alpha: color never exceeds alpha even with overshoot at hard edges.
        resizeBicubicRgba(aimg.data(), w, h, w * 4, big.data(), ow, oh, ow * 4);
        bool pm = true;
        for (size_t i = 0; i < big.size(); i += 4) for (int c = 0; c < 3; c++) if (big[i + c] > big[i + 3]) pm = false;
        CHECK(pm, "bicubic keeps colors premultiplied");

        // Non-integer target size (used for on-screen previews).
        std::vector<uint8_t> odd((size_t)101 * 77 * 4);
        resizeBicubicRgba(flat.data(), w, h, w * 4, odd.data(), 101, 77, 101 * 4);
        CHECK(odd[0] == 200 && odd[odd.size() - 4] == 200, "bicubic works for non-integer scale");

        // Blend endpoints and midpoint.
        std::vector<uint8_t> A((size_t)ow * oh * 4, 0), B((size_t)ow * oh * 4, 200), O(A.size());
        blendRgba(A.data(), ow * 4, B.data(), ow * 4, O.data(), ow * 4, ow, oh, 0.f);
        bool e0 = maxDiff(O, A) == 0;
        blendRgba(A.data(), ow * 4, B.data(), ow * 4, O.data(), ow * 4, ow, oh, 1.f);
        bool e1 = maxDiff(O, B) == 0;
        blendRgba(A.data(), ow * 4, B.data(), ow * 4, O.data(), ow * 4, ow, oh, 0.25f);
        CHECK(e0 && e1 && O[0] == 50, "blend: 0%% = AI, 100%% = classic, 25%% = mix (got %d)", O[0]);
        blendRgba(A.data(), ow * 4, B.data(), ow * 4, A.data(), ow * 4, ow, oh, 0.5f);  // in place
        CHECK(A[0] == 100, "blend works in place (got %d)", A[0]);
    }

    printf(fails ? "\n%d FAILURES\n" : "\nALL PASSED\n", fails);
    return fails ? 1 : 0;
}
