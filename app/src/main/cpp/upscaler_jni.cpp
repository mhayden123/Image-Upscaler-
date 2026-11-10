// JNI bridge between Kotlin (app.upscaler.engine.NativeUpscaler) and the NCNN core.

#include <jni.h>
#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <mutex>

#include "cpu.h"
#include "gpu.h"
#include "upscale_core.h"

#define TAG "UpscalerNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

using upscaler::Model;

namespace {

std::once_flag gGpuOnce;
int gGpuCount = 0;

void initGpu() {
    std::call_once(gGpuOnce, [] {
        ncnn::create_gpu_instance();
        gGpuCount = ncnn::get_gpu_count();
        LOGI("Vulkan GPUs found: %d", gGpuCount);
    });
}

int defaultTileSize(const Model* m) {
    if (!m->gpu) return 200;
    // Same thresholds as realesrgan-ncnn-vulkan: more GPU memory -> bigger (faster) tiles.
    uint32_t heapMb = ncnn::get_gpu_device(0)->get_heap_budget();
    if (heapMb > 1900) return 200;
    if (heapMb > 550) return 100;
    if (heapMb > 190) return 64;
    return 32;
}

struct LockedBitmap {
    JNIEnv* env;
    jobject bmp;
    AndroidBitmapInfo info{};
    uint8_t* pixels = nullptr;

    LockedBitmap(JNIEnv* e, jobject b) : env(e), bmp(b) {
        if (AndroidBitmap_getInfo(env, bmp, &info) != ANDROID_BITMAP_RESULT_SUCCESS) return;
        if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) return;
        void* p = nullptr;
        if (AndroidBitmap_lockPixels(env, bmp, &p) == ANDROID_BITMAP_RESULT_SUCCESS)
            pixels = static_cast<uint8_t*>(p);
    }
    ~LockedBitmap() {
        if (pixels) AndroidBitmap_unlockPixels(env, bmp);
    }
};

std::string toStd(JNIEnv* env, jstring s) {
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string r(c);
    env->ReleaseStringUTFChars(s, c);
    return r;
}

}  // namespace

extern "C" {

JNIEXPORT jint JNICALL
Java_app_upscaler_engine_NativeUpscaler_nativeInit(JNIEnv*, jobject) {
    initGpu();
    return gGpuCount;
}

JNIEXPORT jstring JNICALL
Java_app_upscaler_engine_NativeUpscaler_nativeGpuName(JNIEnv* env, jobject) {
    initGpu();
    if (gGpuCount <= 0) return env->NewStringUTF("");
    return env->NewStringUTF(ncnn::get_gpu_info(0).device_name());
}

JNIEXPORT jlong JNICALL
Java_app_upscaler_engine_NativeUpscaler_nativeCreate(JNIEnv* env, jobject, jobject assetManager,
                                                     jstring paramPath, jstring binPath,
                                                     jstring inName, jstring outName,
                                                     jint scale, jint prepadding, jboolean useGpu) {
    initGpu();
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    if (!mgr) return 0;

    auto* m = new Model();
    m->scale = scale;
    m->prepadding = prepadding;
    m->gpu = useGpu && gGpuCount > 0;
    m->inName = toStd(env, inName);
    m->outName = toStd(env, outName);

    m->net.opt.use_vulkan_compute = m->gpu;
    // 16-bit storage halves memory traffic; arithmetic stays 32-bit for accuracy.
    m->net.opt.use_fp16_packed = true;
    m->net.opt.use_fp16_storage = true;
    m->net.opt.use_fp16_arithmetic = false;
    m->net.opt.num_threads = ncnn::get_big_cpu_count();

    const std::string param = toStd(env, paramPath);
    const std::string bin = toStd(env, binPath);
    int rp = m->net.load_param(mgr, param.c_str());
    int rm = rp == 0 ? m->net.load_model(mgr, bin.c_str()) : -1;
    if (rp != 0 || rm != 0) {
        LOGE("Failed to load model %s / %s (param=%d model=%d)", param.c_str(), bin.c_str(), rp, rm);
        delete m;
        return 0;
    }
    LOGI("Loaded %s (gpu=%d)", param.c_str(), m->gpu ? 1 : 0);
    return reinterpret_cast<jlong>(m);
}

JNIEXPORT jboolean JNICALL
Java_app_upscaler_engine_NativeUpscaler_nativeUsesGpu(JNIEnv*, jobject, jlong handle) {
    auto* m = reinterpret_cast<Model*>(handle);
    return m && m->gpu ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_app_upscaler_engine_NativeUpscaler_nativeCancel(JNIEnv*, jobject, jlong handle) {
    auto* m = reinterpret_cast<Model*>(handle);
    if (m) m->cancel.store(true);
}

JNIEXPORT void JNICALL
Java_app_upscaler_engine_NativeUpscaler_nativeResetCancel(JNIEnv*, jobject, jlong handle) {
    auto* m = reinterpret_cast<Model*>(handle);
    if (m) m->cancel.store(false);
}

JNIEXPORT void JNICALL
Java_app_upscaler_engine_NativeUpscaler_nativeDestroy(JNIEnv*, jobject, jlong handle) {
    delete reinterpret_cast<Model*>(handle);
}

JNIEXPORT jint JNICALL
Java_app_upscaler_engine_NativeUpscaler_nativeUpscale(JNIEnv* env, jobject, jlong handle,
                                                      jobject input, jobject output,
                                                      jint outScale, jint tileSize,
                                                      jobject listener) {
    auto* m = reinterpret_cast<Model*>(handle);
    if (!m) return upscaler::ERR_ARGS;

    jclass lcls = env->GetObjectClass(listener);
    jmethodID onProgress = env->GetMethodID(lcls, "onProgress", "(II)V");
    env->DeleteLocalRef(lcls);
    if (!onProgress) {
        env->ExceptionClear();
        return upscaler::ERR_CALLBACK;
    }

    LockedBitmap in(env, input);
    LockedBitmap out(env, output);
    if (!in.pixels || !out.pixels) return upscaler::ERR_BITMAP;
    const int w = (int)in.info.width, h = (int)in.info.height;
    if ((int)out.info.width != w * outScale || (int)out.info.height != h * outScale) return upscaler::ERR_ARGS;

    const int tile = tileSize > 0 ? tileSize : defaultTileSize(m);
    return upscaler::upscaleRgba(*m, in.pixels, w, h, (int)in.info.stride, out.pixels, (int)out.info.stride,
                                 outScale, tile, [&](int done, int total) {
                                     env->CallVoidMethod(listener, onProgress, done, total);
                                     if (env->ExceptionCheck()) {
                                         env->ExceptionClear();
                                         return false;
                                     }
                                     return true;
                                 });
}

}  // extern "C"
