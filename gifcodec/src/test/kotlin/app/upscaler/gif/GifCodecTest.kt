package app.upscaler.gif

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.util.Random
import javax.imageio.IIOImage
import javax.imageio.ImageIO
import javax.imageio.ImageTypeSpecifier
import javax.imageio.metadata.IIOMetadataNode

class GifCodecTest {

    private fun encode(w: Int, h: Int, frames: List<IntArray>, delay: Int = 5, loop: Int = 0): ByteArray {
        val bos = ByteArrayOutputStream()
        val enc = GifEncoder(bos, w, h, loop)
        frames.forEach { enc.addFrame(it, delay) }
        enc.finish()
        return bos.toByteArray()
    }

    private fun decodeAll(bytes: ByteArray): List<GifDecoder.Frame> {
        val out = ArrayList<GifDecoder.Frame>()
        GifDecoder(bytes).decode { f -> out.add(GifDecoder.Frame(f.argb.copyOf(), f.width, f.height, f.delayCs, f.index)); true }
        return out
    }

    /** Reads raw frames with the JDK's independent GIF decoder. */
    private fun readWithImageIO(bytes: ByteArray): List<BufferedImage> {
        val reader = ImageIO.getImageReadersByFormatName("gif").next()
        reader.input = ImageIO.createImageInputStream(ByteArrayInputStream(bytes))
        val n = reader.getNumImages(true)
        return (0 until n).map { reader.read(it) }
    }

    private fun maxChannelDiff(a: Int, b: Int): Int = maxOf(
        kotlin.math.abs(((a shr 16) and 0xFF) - ((b shr 16) and 0xFF)),
        kotlin.math.abs(((a shr 8) and 0xFF) - ((b shr 8) and 0xFF)),
        kotlin.math.abs((a and 0xFF) - (b and 0xFF)),
    )

    @Test
    fun fewColorsRoundTripExactly() {
        val w = 37; val h = 23
        val colors = intArrayOf(0xFFFF0000.toInt(), 0xFF00FF00.toInt(), 0xFF0000FF.toInt(), 0xFFFFFFFF.toInt(), 0xFF123456.toInt())
        val px = IntArray(w * h) { colors[(it * 7 + it / w) % colors.size] }
        val bytes = encode(w, h, listOf(px))
        // Independent decoder agrees.
        val img = readWithImageIO(bytes).single()
        for (y in 0 until h) for (x in 0 until w) assertEquals(px[y * w + x], img.getRGB(x, y))
        // Our decoder agrees.
        val f = decodeAll(bytes).single()
        assertTrue(px.contentEquals(f.argb))
    }

    @Test
    fun gradientQuantizesClosely() {
        val w = 256; val h = 256
        val px = IntArray(w * h) { i -> val x = i % w; val y = i / w; (0xFF shl 24) or (x shl 16) or (y shl 8) or ((x + y) / 2) }
        val bytes = encode(w, h, listOf(px))
        val img = readWithImageIO(bytes).single()
        var worst = 0; var total = 0L
        for (i in px.indices) {
            val d = maxChannelDiff(px[i], img.getRGB(i % w, i / w))
            worst = maxOf(worst, d); total += d
        }
        val avg = total.toDouble() / px.size
        println("gradient: worst channel error=$worst avg=$avg")
        assertTrue("avg error too high: $avg", avg < 8.0)
        assertTrue("worst error too high: $worst", worst < 40)
    }

    @Test
    fun lzwTableResetsWithNoisyData() {
        // Random noise with 256 colors forces many LZW table-full resets.
        val w = 300; val h = 200
        val rnd = Random(42)
        val palette = IntArray(256) { (0xFF shl 24) or (it shl 16) or ((255 - it) shl 8) or ((it * 37) and 0xFF) }
        val px = IntArray(w * h) { palette[rnd.nextInt(256)] }
        val bytes = encode(w, h, listOf(px))
        val img = readWithImageIO(bytes).single()
        for (i in px.indices) assertEquals("pixel $i", px[i], img.getRGB(i % w, i / w))
        assertTrue(px.contentEquals(decodeAll(bytes).single().argb))
    }

    @Test
    fun animationWithDiffFramesComposesCorrectly() {
        val w = 64; val h = 48
        val rnd = Random(7)
        val colors = IntArray(40) { 0xFF000000.toInt() or rnd.nextInt(0xFFFFFF) }
        val frames = ArrayList<IntArray>()
        var cur = IntArray(w * h) { colors[rnd.nextInt(colors.size)] }
        repeat(6) {
            frames.add(cur.copyOf())
            cur = cur.copyOf()
            // change a moving square
            for (y in 10 until 20) for (x in (it * 5) until (it * 5 + 10)) cur[y * w + x] = colors[rnd.nextInt(colors.size)]
        }
        val bytes = encode(w, h, frames, delay = 7, loop = 3)
        val info = GifDecoder(bytes).readInfo()
        assertEquals(6, info.frameCount); assertEquals(3, info.loopCount); assertEquals(w, info.width)
        val decoded = decodeAll(bytes)
        assertEquals(6, decoded.size)
        decoded.forEachIndexed { i, f ->
            assertEquals(7, f.delayCs)
            assertTrue("frame $i differs", frames[i].contentEquals(f.argb))
        }
        // The JDK reader must accept every frame too.
        assertEquals(6, readWithImageIO(bytes).size)
    }

    @Test
    fun transparencySurvivesAndClearsBetweenFrames() {
        val w = 20; val h = 10
        val red = 0xFFFF0000.toInt(); val blue = 0xFF0000FF.toInt()
        val f1 = IntArray(w * h) { if (it % w < 10) red else 0 } // left half red, right transparent
        val f2 = IntArray(w * h) { if (it % w >= 10) blue else 0 } // right half blue, left transparent
        val decoded = decodeAll(encode(w, h, listOf(f1, f2)))
        fun norm(a: IntArray) = IntArray(a.size) { if ((a[it] ushr 24) < 128) 0 else a[it] }
        assertTrue(norm(f1).contentEquals(norm(decoded[0].argb)))
        assertTrue("frame 2 must not show frame 1's red through transparency", norm(f2).contentEquals(norm(decoded[1].argb)))
    }

    @Test
    fun decodesGifWrittenByJdkIncludingInterlaced() {
        for (interlaced in listOf(false, true)) {
            val w = 41; val h = 29
            val img = BufferedImage(w, h, BufferedImage.TYPE_BYTE_INDEXED)
            val g = img.createGraphics()
            for (y in 0 until h) for (x in 0 until w) { g.color = java.awt.Color(x * 6, y * 8, (x * y) % 256); g.fillRect(x, y, 1, 1) }
            g.dispose()
            val writer = ImageIO.getImageWritersByFormatName("gif").next()
            val bos = ByteArrayOutputStream()
            writer.output = ImageIO.createImageOutputStream(bos)
            val param = writer.defaultWriteParam
            val meta = writer.getDefaultImageMetadata(ImageTypeSpecifier.createFromRenderedImage(img), param)
            val root = meta.getAsTree("javax_imageio_gif_image_1.0") as IIOMetadataNode
            val desc = root.getElementsByTagName("ImageDescriptor").item(0) as IIOMetadataNode
            desc.setAttribute("interlaceFlag", if (interlaced) "TRUE" else "FALSE")
            meta.setFromTree("javax_imageio_gif_image_1.0", root)
            writer.write(null, IIOImage(img, null, meta), param)
            (writer.output as javax.imageio.stream.ImageOutputStream).close()
            val f = decodeAll(bos.toByteArray()).single()
            for (y in 0 until h) for (x in 0 until w) assertEquals("interlaced=$interlaced ($x,$y)", img.getRGB(x, y), f.argb[y * w + x])
        }
    }

    @Test
    fun truncatedFileKeepsCompleteFrames() {
        val w = 16; val h = 16
        val frames = (0 until 4).map { k -> IntArray(w * h) { 0xFF000000.toInt() or (k * 60 shl 16) or it } }
        val bytes = encode(w, h, frames)
        val cut = bytes.copyOf(bytes.size - 40)
        val decoded = decodeAll(cut)
        assertTrue("got ${decoded.size} frames", decoded.size in 3..4)
        assertTrue(GifDecoder(cut).readInfo().frameCount in 3..4)
    }
}
