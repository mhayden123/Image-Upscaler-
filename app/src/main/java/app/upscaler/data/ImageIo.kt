package app.upscaler.data

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.ColorSpace
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Environment
import android.provider.MediaStore
import app.upscaler.gif.GifDecoder
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** A picked image, ready to upscale. */
sealed class InputImage {
    abstract val uri: Uri
    abstract val width: Int
    abstract val height: Int
    /** Shown on screen (the first frame for GIFs). */
    abstract val preview: Bitmap

    class Still(override val uri: Uri, val bitmap: Bitmap) : InputImage() {
        override val width get() = bitmap.width
        override val height get() = bitmap.height
        override val preview get() = bitmap
    }

    class AnimatedGif(
        override val uri: Uri,
        val bytes: ByteArray,
        override val width: Int,
        override val height: Int,
        val frameCount: Int,
        override val preview: Bitmap,
    ) : InputImage()
}

object ImageIo {

    /** Decodes the picked image. Animated GIFs keep their raw bytes so every frame can be processed. */
    fun load(context: Context, uri: Uri): InputImage {
        val resolver = context.contentResolver
        val header = ByteArray(6)
        val headerLen = resolver.openInputStream(uri)?.use { it.read(header) } ?: throw IOException("Can't open the image")

        if (headerLen >= 6 && GifDecoder.isGif(header)) {
            val bytes = resolver.openInputStream(uri)?.use { it.readBytes() } ?: throw IOException("Can't read the GIF")
            val info = GifDecoder(bytes).readInfo()
            if (info.frameCount > 1) {
                var first: Bitmap? = null
                GifDecoder(bytes).decode { f ->
                    first = Bitmap.createBitmap(f.argb, f.width, f.height, Bitmap.Config.ARGB_8888)
                    false // only need the first frame for the preview
                }
                val preview = first ?: throw IOException("The GIF has no readable frames")
                return InputImage.AnimatedGif(uri, bytes, info.width, info.height, info.frameCount, preview)
            }
            // Single-frame GIFs are handled like any still image.
        }

        val source = ImageDecoder.createSource(resolver, uri)
        val bitmap = ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
            // Software memory so native code can read the pixels; sRGB so colors match the model's training data.
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            decoder.setTargetColorSpace(ColorSpace.get(ColorSpace.Named.SRGB))
        }
        val argb = if (bitmap.config == Bitmap.Config.ARGB_8888) bitmap
        else bitmap.copy(Bitmap.Config.ARGB_8888, false).also { bitmap.recycle() }
        return InputImage.Still(uri, argb)
    }

    /** A smaller copy for drawing on screen: Android can't draw bitmaps over ~100 MB. */
    fun previewOf(bitmap: Bitmap, maxSide: Int = 4096): Bitmap {
        val longest = maxOf(bitmap.width, bitmap.height)
        val preview = if (longest <= maxSide) bitmap else {
            val s = maxSide.toFloat() / longest
            Bitmap.createScaledBitmap(bitmap, (bitmap.width * s).toInt().coerceAtLeast(1), (bitmap.height * s).toInt().coerceAtLeast(1), true)
        }
        // Mipmaps make the big result look smooth (not jagged) when shown zoomed out.
        preview.setHasMipMap(true)
        return preview
    }

    private fun newName(ext: String): String =
        "upscaled_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date()) + "." + ext

    /** Saves a PNG (lossless) into Pictures/Upscaler. Returns the new Gallery item. */
    fun saveStill(context: Context, bitmap: Bitmap): Uri =
        insert(context, newName("png"), "image/png") { os ->
            if (!bitmap.compress(Bitmap.CompressFormat.PNG, 100, os)) throw IOException("PNG encoding failed")
        }

    /** Copies a finished GIF file into Pictures/Upscaler. */
    fun saveGif(context: Context, file: File): Uri =
        insert(context, newName("gif"), "image/gif") { os -> file.inputStream().use { it.copyTo(os, 1 shl 16) } }

    private fun insert(context: Context, name: String, mime: String, write: (java.io.OutputStream) -> Unit): Uri {
        val resolver = context.contentResolver
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, name)
            put(MediaStore.Images.Media.MIME_TYPE, mime)
            put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/Upscaler")
            put(MediaStore.Images.Media.IS_PENDING, 1) // hidden from other apps until fully written
        }
        val collection = MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
        val uri = resolver.insert(collection, values) ?: throw IOException("Couldn't create the Gallery entry")
        try {
            resolver.openOutputStream(uri)?.use(write) ?: throw IOException("Couldn't open the Gallery entry")
            values.clear()
            values.put(MediaStore.Images.Media.IS_PENDING, 0)
            resolver.update(uri, values, null, null)
        } catch (t: Throwable) {
            resolver.delete(uri, null, null)
            throw t
        }
        return uri
    }
}
