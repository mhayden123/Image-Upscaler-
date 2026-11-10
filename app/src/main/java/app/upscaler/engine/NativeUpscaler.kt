package app.upscaler.engine

import android.content.res.AssetManager
import android.graphics.Bitmap

/** Thin Kotlin wrapper over the C++ code in src/main/cpp (NCNN + Vulkan). */
object NativeUpscaler {
    init {
        System.loadLibrary("upscaler")
    }

    const val RESULT_OK = 0
    const val RESULT_CANCELLED = 1

    fun interface ProgressListener {
        fun onProgress(done: Int, total: Int)
    }

    /** Initialises Vulkan once; returns the number of usable GPUs. */
    external fun nativeInit(): Int
    external fun nativeGpuName(): String
    external fun nativeCreate(
        assets: AssetManager, paramPath: String, binPath: String,
        inputBlob: String, outputBlob: String, scale: Int, prepadding: Int, useGpu: Boolean,
    ): Long
    external fun nativeUsesGpu(handle: Long): Boolean
    external fun nativeCancel(handle: Long)
    external fun nativeResetCancel(handle: Long)
    external fun nativeDestroy(handle: Long)

    /** Blocking. Both bitmaps must be ARGB_8888; output = input size * outScale. tileSize 0 = auto. */
    external fun nativeUpscale(
        handle: Long, input: Bitmap, output: Bitmap, outScale: Int, tileSize: Int, listener: ProgressListener,
    ): Int

    /** Classic (bicubic) enlargement of [src] into [dst]; any sizes, both ARGB_8888. */
    external fun nativeResizeBicubic(src: Bitmap, dst: Bitmap): Boolean

    /** out = ai * (1 - t) + classic * t. Same sizes; [out] may be [ai] or [classic]. */
    external fun nativeBlend(ai: Bitmap, classic: Bitmap, out: Bitmap, t: Float): Boolean

    fun errorMessage(code: Int): String = when (code) {
        -1 -> "Couldn't access the image pixels"
        -2 -> "Invalid size or scale"
        -3 -> "The AI model failed while processing (code $code)"
        -4 -> "Processing was interrupted"
        else -> "Unknown error ($code)"
    }
}
