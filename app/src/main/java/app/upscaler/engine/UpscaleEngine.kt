package app.upscaler.engine

import android.content.Context
import android.graphics.Bitmap
import app.upscaler.gif.GifDecoder
import app.upscaler.gif.GifEncoder
import java.io.File
import java.util.concurrent.CancellationException

/**
 * Runs upscaling jobs. All methods are blocking: call them from a background thread.
 * Only one job runs at a time (the ViewModel guarantees this).
 */
class UpscaleEngine(private val context: Context) {

    class UpscaleException(message: String) : Exception(message)

    private var handle = 0L
    private var loadedModel: UpscaleModel? = null

    val gpuCount: Int by lazy { NativeUpscaler.nativeInit() }
    val gpuName: String by lazy { if (gpuCount > 0) NativeUpscaler.nativeGpuName() else "" }

    @Volatile private var cancelled = false

    fun cancel() {
        cancelled = true
        val h = handle
        if (h != 0L) NativeUpscaler.nativeCancel(h)
    }

    @Synchronized
    private fun ensureModel(model: UpscaleModel): Long {
        if (loadedModel == model && handle != 0L) return handle
        if (handle != 0L) NativeUpscaler.nativeDestroy(handle)
        handle = 0L
        loadedModel = null
        val h = NativeUpscaler.nativeCreate(
            context.assets, model.paramAsset, model.binAsset, model.inputBlob, model.outputBlob,
            model.nativeScale, model.prepadding, useGpu = gpuCount > 0,
        )
        if (h == 0L) throw UpscaleException("Couldn't load the ${model.title} model")
        handle = h
        loadedModel = model
        return h
    }

    /** Call once at the start of each job, before [prepare]. */
    fun beginJob() {
        cancelled = false
    }

    /** Prepares the model (can take a few seconds the first time, while GPU shaders compile). */
    fun prepare(model: UpscaleModel) {
        if (cancelled) throw CancellationException()
        val h = ensureModel(model)
        NativeUpscaler.nativeResetCancel(h)
        if (cancelled) throw CancellationException()
    }

    private fun run(h: Long, input: Bitmap, output: Bitmap, scale: Int, onTiles: (Int, Int) -> Unit) {
        if (cancelled) throw CancellationException()
        val code = NativeUpscaler.nativeUpscale(h, input, output, scale, 0) { d, t -> onTiles(d, t) }
        if (cancelled) throw CancellationException()
        when (code) {
            NativeUpscaler.RESULT_OK -> Unit
            NativeUpscaler.RESULT_CANCELLED -> throw CancellationException()
            else -> throw UpscaleException(NativeUpscaler.errorMessage(code))
        }
    }

    /** Upscales a still image. [onProgress] receives 0..1. */
    fun upscaleStill(input: Bitmap, model: UpscaleModel, scale: Int, onProgress: (Float) -> Unit): Bitmap {
        prepare(model)
        val src = if (input.config == Bitmap.Config.ARGB_8888) input else input.copy(Bitmap.Config.ARGB_8888, false)
        val out = Bitmap.createBitmap(src.width * scale, src.height * scale, Bitmap.Config.ARGB_8888)
        run(handle, src, out, scale) { d, t -> onProgress(d.toFloat() / t) }
        return out
    }

    companion object {
        /** A classic (bicubic) enlargement of [src] to [width] x [height]. */
        fun classicResize(src: Bitmap, width: Int, height: Int): Bitmap {
            val out = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            check(NativeUpscaler.nativeResizeBicubic(src, out)) { "Resize failed" }
            return out
        }

        /** New bitmap: [ai] mixed with a classic enlargement of [original] ([texture] 0..1). */
        fun applyTexture(ai: Bitmap, original: Bitmap, texture: Float): Bitmap {
            val out = classicResize(original, ai.width, ai.height)
            check(NativeUpscaler.nativeBlend(ai, out, out, texture)) { "Blend failed" }
            return out
        }
    }

    class GifResult(val file: File, val width: Int, val height: Int, val frames: Int, val firstFrame: Bitmap)

    /**
     * Upscales every frame of an animated GIF and writes a new GIF to [outFile].
     * [onProgress] receives (frameIndex, frameCount, overall 0..1).
     */
    fun upscaleGif(
        bytes: ByteArray, model: UpscaleModel, scale: Int, texture: Float, outFile: File,
        onProgress: (Int, Int, Float) -> Unit,
    ): GifResult {
        prepare(model)
        val info = GifDecoder(bytes).readInfo()
        val w = info.width; val h = info.height
        val ow = w * scale; val oh = h * scale
        if (ow > 65535 || oh > 65535) throw UpscaleException("Result would exceed the GIF size limit")

        val inBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val outBmp = Bitmap.createBitmap(ow, oh, Bitmap.Config.ARGB_8888)
        val classicBmp = if (texture > 0f) Bitmap.createBitmap(ow, oh, Bitmap.Config.ARGB_8888) else null
        val outPixels = IntArray(ow * oh)
        var prevSource: IntArray? = null
        var firstFrame: Bitmap? = null
        var frames = 0
        val total = info.frameCount.coerceAtLeast(1)

        try {
            outFile.outputStream().buffered(1 shl 16).use { os ->
                val encoder = GifEncoder(os, ow, oh, info.loopCount)
                GifDecoder(bytes).decode { frame ->
                    if (cancelled) throw CancellationException()
                    val i = frame.index
                    // Identical consecutive frames (pauses) are common in memes: reuse the last result.
                    val prev = prevSource
                    if (prev == null || !prev.contentEquals(frame.argb)) {
                        inBmp.setPixels(frame.argb, 0, w, 0, 0, w, h)
                        run(handle, inBmp, outBmp, scale) { d, t ->
                            onProgress(i, total, (i + d.toFloat() / t) / total)
                        }
                        if (classicBmp != null) {
                            // "Keep original texture": mix in a classic enlargement of the source frame.
                            NativeUpscaler.nativeResizeBicubic(inBmp, classicBmp)
                            NativeUpscaler.nativeBlend(outBmp, classicBmp, outBmp, texture)
                        }
                        outBmp.getPixels(outPixels, 0, ow, 0, 0, ow, oh)
                        prevSource = frame.argb.copyOf()
                    }
                    if (firstFrame == null) firstFrame = outBmp.copy(Bitmap.Config.ARGB_8888, false)
                    encoder.addFrame(outPixels, frame.delayCs)
                    frames++
                    onProgress(i + 1, total, (i + 1f) / total)
                    true
                }
                encoder.finish()
            }
        } catch (t: Throwable) {
            outFile.delete()
            throw t
        } finally {
            inBmp.recycle()
            outBmp.recycle()
            classicBmp?.recycle()
        }
        if (frames == 0) throw UpscaleException("The GIF has no readable frames")
        return GifResult(outFile, ow, oh, frames, firstFrame!!)
    }

    @Synchronized
    fun release() {
        if (handle != 0L) NativeUpscaler.nativeDestroy(handle)
        handle = 0L
        loadedModel = null
    }
}
