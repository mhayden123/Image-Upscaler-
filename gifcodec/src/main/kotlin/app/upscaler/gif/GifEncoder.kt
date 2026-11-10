package app.upscaler.gif

import java.io.OutputStream

/**
 * Animated GIF89a encoder with a per-frame (local) 256-color palette.
 *
 * Size optimisation: when a frame is fully opaque, pixels that did not change
 * from the previous frame are written as "transparent", so the viewer keeps
 * showing the old pixel. Long runs of the same index compress very well with LZW,
 * which keeps upscaled GIFs much smaller. Frames that contain real transparency
 * are written in full, and the previous frame is cleared first (disposal 2), so
 * transparent areas show correctly.
 *
 * Frames are written one behind: a frame's disposal method depends on the frame
 * after it, so the encoder holds one frame until the next arrives.
 */
class GifEncoder(
    private val out: OutputStream,
    private val width: Int,
    private val height: Int,
    /** 0 = loop forever, N = loop N extra times, -1 = no loop extension. */
    loopCount: Int,
) {
    private class Pending(
        val palette: IntArray,
        val indices: ByteArray,
        val transparentIndex: Int,
        val delayCs: Int,
    )

    private val quantizer = ColorQuantizer()
    private val lzw = LzwEncoder()
    private var pending: Pending? = null
    private var prevArgb: IntArray? = null
    private var finished = false

    init {
        require(width in 1..65535 && height in 1..65535) { "GIF dimensions must be 1..65535, got ${width}x$height" }
        out.write("GIF89a".toByteArray(Charsets.US_ASCII))
        writeShort(width); writeShort(height)
        out.write(0x00) // no global color table; every frame carries its own
        out.write(0) // background color index
        out.write(0) // pixel aspect ratio
        if (loopCount >= 0) {
            out.write(byteArrayOf(0x21, 0xFF.toByte(), 11))
            out.write("NETSCAPE2.0".toByteArray(Charsets.US_ASCII))
            out.write(byteArrayOf(3, 1))
            writeShort(loopCount.coerceIn(0, 65535))
            out.write(0)
        }
    }

    /**
     * Adds a full-canvas frame (ARGB, alpha < 128 counts as transparent).
     * [argb] may be reused by the caller after this returns.
     */
    fun addFrame(argb: IntArray, delayCs: Int) {
        check(!finished) { "Encoder already finished" }
        val n = width * height
        require(argb.size >= n) { "Frame has ${argb.size} pixels, expected $n" }

        var hasTransparency = false
        for (i in 0 until n) if ((argb[i] ushr 24) < 128) { hasTransparency = true; break }

        val prev = prevArgb
        val useDiff = prev != null && !hasTransparency
        val result = if (useDiff) {
            quantizer.quantize(argb, n, 256, reserveTransparent = true) { i -> sameColor(argb[i], prev!![i]) }
        } else {
            quantizer.quantize(argb, n, 256, reserveTransparent = hasTransparency) { i -> (argb[i] ushr 24) < 128 }
        }

        // Now that we know this frame's needs, the previous frame can be written.
        pending?.let { writeFrame(it, disposal = if (hasTransparency) 2 else 1) }
        pending = Pending(result.palette, result.indices, result.transparentIndex, delayCs)

        val keep = prevArgb ?: IntArray(n)
        System.arraycopy(argb, 0, keep, 0, n)
        prevArgb = keep
    }

    /** Writes the last frame and the trailer. Does not close the stream. */
    fun finish() {
        if (finished) return
        pending?.let { writeFrame(it, disposal = 1) }
        pending = null
        out.write(0x3B)
        out.flush()
        finished = true
    }

    private fun writeFrame(f: Pending, disposal: Int) {
        // Graphic control extension.
        out.write(byteArrayOf(0x21, 0xF9.toByte(), 4))
        val transparentFlag = if (f.transparentIndex >= 0) 1 else 0
        out.write((disposal shl 2) or transparentFlag)
        writeShort(f.delayCs.coerceIn(0, 65535))
        out.write(if (f.transparentIndex >= 0) f.transparentIndex else 0)
        out.write(0)

        // Image descriptor with local color table.
        var tableBits = 1
        while ((1 shl tableBits) < f.palette.size) tableBits++ // 2..256 entries
        out.write(0x2C)
        writeShort(0); writeShort(0); writeShort(width); writeShort(height)
        out.write(0x80 or (tableBits - 1))
        val tableSize = 1 shl tableBits
        val table = ByteArray(tableSize * 3)
        for (i in f.palette.indices) {
            val c = f.palette[i]
            table[i * 3] = (c shr 16).toByte(); table[i * 3 + 1] = (c shr 8).toByte(); table[i * 3 + 2] = c.toByte()
        }
        out.write(table)

        val minCodeSize = maxOf(2, tableBits)
        out.write(minCodeSize)
        lzw.encode(f.indices, minCodeSize, out)
        out.write(0) // block terminator
    }

    private fun writeShort(v: Int) {
        out.write(v and 0xFF)
        out.write((v shr 8) and 0xFF)
    }

    /** Pixels within a tiny tolerance count as unchanged (absorbs GPU rounding noise). */
    private fun sameColor(a: Int, b: Int): Boolean {
        if ((b ushr 24) < 128) return false // previous pixel was transparent: must draw
        val dr = ((a shr 16) and 0xFF) - ((b shr 16) and 0xFF)
        val dg = ((a shr 8) and 0xFF) - ((b shr 8) and 0xFF)
        val db = (a and 0xFF) - (b and 0xFF)
        return dr in -2..2 && dg in -2..2 && db in -2..2
    }

    /** GIF-flavoured LZW compressor writing 255-byte sub-blocks. */
    private class LzwEncoder {
        private val hashKeys = IntArray(HASH_SIZE)
        private val hashCodes = ShortArray(HASH_SIZE)
        private val block = ByteArray(256)
        private var blockLen = 0
        private var bitBuf = 0
        private var bitCount = 0

        fun encode(indices: ByteArray, minCodeSize: Int, out: OutputStream) {
            val clear = 1 shl minCodeSize
            val eoi = clear + 1
            var codeSize = minCodeSize + 1
            var next = clear + 2
            blockLen = 0; bitBuf = 0; bitCount = 0
            java.util.Arrays.fill(hashKeys, -1)

            emit(clear, codeSize, out)
            if (indices.isEmpty()) {
                emit(eoi, codeSize, out); flushBits(out); flushBlock(out); return
            }
            var prefix = indices[0].toInt() and 0xFF
            for (i in 1 until indices.size) {
                val k = indices[i].toInt() and 0xFF
                val key = (prefix shl 8) or k
                var h = ((k shl 12) xor prefix) and (HASH_SIZE - 1)
                var found = -1
                while (true) {
                    val hk = hashKeys[h]
                    if (hk == -1) break
                    if (hk == key) { found = hashCodes[h].toInt() and 0xFFFF; break }
                    h = (h + 1) and (HASH_SIZE - 1)
                }
                if (found >= 0) {
                    prefix = found
                    continue
                }
                emit(prefix, codeSize, out)
                if (next < 4096) {
                    hashKeys[h] = key
                    hashCodes[h] = next.toShort()
                    // Code size grows once the code we just added no longer fits.
                    if (next == (1 shl codeSize) && codeSize < 12) codeSize++
                    next++
                } else {
                    // Table full: reset.
                    emit(clear, codeSize, out)
                    java.util.Arrays.fill(hashKeys, -1)
                    codeSize = minCodeSize + 1
                    next = clear + 2
                }
                prefix = k
            }
            emit(prefix, codeSize, out)
            emit(eoi, codeSize, out)
            flushBits(out)
            flushBlock(out)
        }

        private fun emit(code: Int, size: Int, out: OutputStream) {
            bitBuf = bitBuf or (code shl bitCount)
            bitCount += size
            while (bitCount >= 8) {
                addByte(bitBuf and 0xFF, out)
                bitBuf = bitBuf ushr 8
                bitCount -= 8
            }
        }

        private fun flushBits(out: OutputStream) {
            if (bitCount > 0) addByte(bitBuf and 0xFF, out)
            bitBuf = 0; bitCount = 0
        }

        private fun addByte(b: Int, out: OutputStream) {
            block[blockLen++] = b.toByte()
            if (blockLen == 255) flushBlock(out)
        }

        private fun flushBlock(out: OutputStream) {
            if (blockLen == 0) return
            out.write(blockLen)
            out.write(block, 0, blockLen)
            blockLen = 0
        }

        companion object { private const val HASH_SIZE = 8192 }
    }
}
