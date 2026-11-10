package app.upscaler.gif

/**
 * Streaming GIF87a/GIF89a decoder.
 *
 * It composites each frame onto a full-size canvas, the way a browser shows it,
 * and passes that canvas (ARGB, not premultiplied) to the callback one frame at a time,
 * so even long GIFs only ever hold one canvas in memory.
 *
 * Handles global/local color tables, transparency, interlacing, all disposal
 * methods and the NETSCAPE2.0 loop count. Truncated files decode up to the last complete frame.
 */
class GifDecoder(private val data: ByteArray) {

    class Info(
        val width: Int,
        val height: Int,
        val frameCount: Int,
        /** 0 = loop forever, N = play N+1 times in total, -1 = no loop extension (play once). */
        val loopCount: Int,
    )

    class Frame(
        /** Full-canvas pixels, ARGB_8888 (not premultiplied). Reused between callbacks, so copy it to keep it. */
        val argb: IntArray,
        val width: Int,
        val height: Int,
        /** Frame delay in hundredths of a second, as stored in the file. */
        val delayCs: Int,
        val index: Int,
    )

    class FormatException(msg: String) : Exception(msg)

    private var pos = 0

    private fun u8(): Int {
        if (pos >= data.size) throw EndOfData()
        return data[pos++].toInt() and 0xFF
    }

    private fun u16(): Int = u8() or (u8() shl 8)

    private class EndOfData : Exception()

    private fun readColorTable(size: Int): IntArray {
        val table = IntArray(size)
        for (i in 0 until size) {
            val r = u8(); val g = u8(); val b = u8()
            table[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        return table
    }

    private fun skipSubBlocks() {
        while (true) {
            val len = u8()
            if (len == 0) return
            pos += len
            if (pos > data.size) throw EndOfData()
        }
    }

    private var width = 0
    private var height = 0
    private var globalTable: IntArray? = null

    private fun readHeader() {
        pos = 0
        if (data.size < 13) throw FormatException("File too small to be a GIF")
        val sig = String(data, 0, 6, Charsets.US_ASCII)
        if (sig != "GIF87a" && sig != "GIF89a") throw FormatException("Not a GIF file")
        pos = 6
        width = u16()
        height = u16()
        val packed = u8()
        u8() // background color index (browsers clear to transparent instead)
        u8() // pixel aspect ratio
        globalTable = if (packed and 0x80 != 0) readColorTable(1 shl ((packed and 0x07) + 1)) else null
        if (width <= 0 || height <= 0) throw FormatException("Invalid GIF dimensions ${width}x$height")
    }

    /** Fast pass that counts frames and reads the loop count without decoding pixels. */
    fun readInfo(): Info {
        readHeader()
        var frames = 0
        var loop = -1
        try {
            loop@ while (pos < data.size) {
                when (u8()) {
                    0x21 -> {
                        val label = u8()
                        if (label == 0xFF) loop = readAppExtension(loop) else skipSubBlocks()
                    }
                    0x2C -> {
                        pos += 8
                        val packed = u8()
                        if (packed and 0x80 != 0) pos += 3 * (1 shl ((packed and 0x07) + 1))
                        u8() // LZW min code size
                        skipSubBlocks()
                        frames++
                    }
                    0x3B -> break@loop
                    else -> break@loop // garbage; stop like browsers do
                }
            }
        } catch (_: EndOfData) {
            // truncated: count only complete frames
        }
        return Info(width, height, frames, loop)
    }

    /** Reads an application extension block; returns the updated loop count. */
    private fun readAppExtension(current: Int): Int {
        var loop = current
        val blockLen = u8()
        val start = pos
        val id = if (blockLen == 11 && pos + 11 <= data.size) String(data, pos, 11, Charsets.US_ASCII) else ""
        pos = start + blockLen
        if (id == "NETSCAPE2.0" || id == "ANIMEXTS1.0") {
            while (true) {
                val len = u8()
                if (len == 0) break
                val subStart = pos
                if (len >= 3 && u8() == 1) loop = u16()
                pos = subStart + len
            }
        } else {
            skipSubBlocks()
        }
        return loop
    }

    /**
     * Decodes all frames, calling [onFrame] for each composited frame.
     * Return false from [onFrame] to stop early.
     */
    fun decode(onFrame: (Frame) -> Boolean) {
        readHeader()
        val canvas = IntArray(width * height) // starts fully transparent
        var saved: IntArray? = null
        var disposal = 0
        var delay = 0
        var transparentIndex = -1
        var frameIndex = 0
        val lzw = LzwDecoder()

        try {
            while (pos < data.size) {
                when (u8()) {
                    0x21 -> {
                        when (u8()) {
                            0xF9 -> { // graphic control extension
                                val len = u8()
                                val start = pos
                                if (len >= 4) {
                                    val packed = u8()
                                    disposal = (packed shr 2) and 0x07
                                    delay = u16()
                                    val ti = u8()
                                    transparentIndex = if (packed and 0x01 != 0) ti else -1
                                }
                                pos = start + len
                                skipSubBlocks()
                            }
                            else -> skipSubBlocks()
                        }
                    }
                    0x2C -> {
                        val fx = u16(); val fy = u16(); val fw = u16(); val fh = u16()
                        val packed = u8()
                        val interlaced = packed and 0x40 != 0
                        val table = if (packed and 0x80 != 0) readColorTable(1 shl ((packed and 0x07) + 1))
                        else globalTable ?: DEFAULT_TABLE
                        val minCodeSize = u8()
                        val pixelCount = fw * fh
                        val indices = lzw.decode(this, minCodeSize, pixelCount)

                        if (disposal == 3) saved = canvas.copyOf()

                        // Draw frame pixels onto the canvas.
                        var srcRow = 0
                        for (pass in 0 until (if (interlaced) 4 else 1)) {
                            val startRow = if (interlaced) INTERLACE_START[pass] else 0
                            val step = if (interlaced) INTERLACE_STEP[pass] else 1
                            var row = startRow
                            while (row < fh) {
                                val cy = fy + row
                                if (cy in 0 until height) {
                                    val srcOff = srcRow * fw
                                    val dstOff = cy * width
                                    for (col in 0 until fw) {
                                        val cx = fx + col
                                        if (cx < 0 || cx >= width) continue
                                        val idx = indices[srcOff + col].toInt() and 0xFF
                                        if (idx == transparentIndex) continue
                                        if (idx < table.size) canvas[dstOff + cx] = table[idx]
                                    }
                                }
                                srcRow++
                                row += step
                            }
                        }

                        val keepGoing = onFrame(Frame(canvas, width, height, delay, frameIndex++))
                        if (!keepGoing) return

                        // Apply disposal before the next frame.
                        when (disposal) {
                            2 -> for (y in fy until minOf(fy + fh, height)) {
                                if (y < 0) continue
                                val off = y * width
                                for (x in maxOf(fx, 0) until minOf(fx + fw, width)) canvas[off + x] = 0
                            }
                            3 -> saved?.copyInto(canvas)
                        }
                        // GCE values only apply to the next image.
                        disposal = 0; delay = 0; transparentIndex = -1
                    }
                    0x3B -> return
                    else -> return
                }
            }
        } catch (_: EndOfData) {
            // Truncated file: keep what we have.
        }
    }

    /** LZW decoder for GIF image data, reusing its buffers across frames. */
    private class LzwDecoder {
        private val prefix = ShortArray(4096)
        private val suffix = ByteArray(4096)
        private val stack = ByteArray(4097)
        private var out = ByteArray(0)

        fun decode(d: GifDecoder, minCodeSize: Int, pixelCount: Int): ByteArray {
            if (out.size < pixelCount) out = ByteArray(pixelCount)
            else java.util.Arrays.fill(out, 0, pixelCount, 0)
            if (minCodeSize < 1 || minCodeSize > 11) { d.skipSubBlocks(); return out }

            val clear = 1 shl minCodeSize
            val eoi = clear + 1
            var codeSize = minCodeSize + 1
            var codeMask = (1 shl codeSize) - 1
            var available = clear + 2
            var oldCode = -1
            var first = 0
            for (i in 0 until clear) { prefix[i] = 0; suffix[i] = i.toByte() }

            var bits = 0
            var datum = 0
            var blockLeft = 0
            var outPos = 0
            var finished = false

            while (true) {
                // Fill the bit buffer from sub-blocks.
                while (bits < codeSize) {
                    if (blockLeft == 0) {
                        blockLeft = d.u8()
                        if (blockLeft == 0) { // block terminator: data ended
                            return out
                        }
                    }
                    datum = datum or (d.u8() shl bits)
                    bits += 8
                    blockLeft--
                }
                val code = datum and codeMask
                datum = datum ushr codeSize
                bits -= codeSize

                if (finished) continue // drain remaining data until terminator
                if (code == clear) {
                    codeSize = minCodeSize + 1
                    codeMask = (1 shl codeSize) - 1
                    available = clear + 2
                    oldCode = -1
                    continue
                }
                if (code == eoi) {
                    finished = true
                    // skip rest of this frame's data
                    if (blockLeft > 0) { d.pos += blockLeft; blockLeft = 0 }
                    d.skipSubBlocks()
                    return out
                }
                if (oldCode == -1) {
                    if (code >= available) continue // corrupt; ignore
                    if (outPos < pixelCount) out[outPos++] = suffix[code]
                    oldCode = code
                    first = code
                    continue
                }

                var inCode = code
                var top = 0
                var c = code
                if (c >= available) { // KwKwK case
                    stack[top++] = first.toByte()
                    c = oldCode
                }
                while (c >= clear) {
                    if (top >= 4096) break
                    stack[top++] = suffix[c]
                    c = prefix[c].toInt()
                }
                first = suffix[c].toInt() and 0xFF
                stack[top++] = first.toByte()

                if (available < 4096) {
                    prefix[available] = oldCode.toShort()
                    suffix[available] = first.toByte()
                    available++
                    if (available and codeMask == 0 && available < 4096) {
                        codeSize++
                        codeMask += available
                    }
                }
                oldCode = inCode

                while (top > 0) {
                    top--
                    if (outPos < pixelCount) out[outPos++] = stack[top]
                }
            }
        }
    }

    companion object {
        private val INTERLACE_START = intArrayOf(0, 4, 2, 1)
        private val INTERLACE_STEP = intArrayOf(8, 8, 4, 2)
        private val DEFAULT_TABLE = IntArray(256) { (0xFF shl 24) or (it shl 16) or (it shl 8) or it }

        /** True if the bytes start with a GIF signature. */
        fun isGif(header: ByteArray): Boolean =
            header.size >= 6 && header[0] == 'G'.code.toByte() && header[1] == 'I'.code.toByte() &&
                header[2] == 'F'.code.toByte() && header[3] == '8'.code.toByte()
    }
}
