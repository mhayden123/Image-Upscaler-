package app.upscaler.gif

/**
 * Median-cut color quantizer.
 *
 * GIF frames can hold at most 256 colors. After AI upscaling a frame has many
 * more (smooth edges, new detail), so for each frame we pick the palette that best
 * represents its pixels:
 *  1. Build a histogram of colors at 6 bits per channel (262,144 buckets).
 *  2. Repeatedly split the "biggest" box of colors along its widest channel
 *     at the median pixel, until there are [maxColors] boxes.
 *  3. Each box's palette color is the average of the real pixels in it.
 * Pixels are then mapped to the nearest palette color (cached per bucket).
 */
class ColorQuantizer {
    private val counts = IntArray(BUCKETS)
    private val sumR = LongArray(BUCKETS)
    private val sumG = LongArray(BUCKETS)
    private val sumB = LongArray(BUCKETS)
    private val nearest = ShortArray(BUCKETS)

    class Result(val palette: IntArray, val indices: ByteArray, val transparentIndex: Int)

    /**
     * Quantizes [argb] pixels. Pixels where [isTransparent] returns true get the reserved
     * transparent index (if [reserveTransparent] is true) and are left out of the palette.
     */
    fun quantize(
        argb: IntArray,
        count: Int,
        maxColors: Int,
        reserveTransparent: Boolean,
        isTransparent: (Int) -> Boolean,
    ): Result {
        java.util.Arrays.fill(counts, 0)
        java.util.Arrays.fill(sumR, 0)
        java.util.Arrays.fill(sumG, 0)
        java.util.Arrays.fill(sumB, 0)

        for (i in 0 until count) {
            if (reserveTransparent && isTransparent(i)) continue
            val c = argb[i]
            val r = (c shr 16) and 0xFF; val g = (c shr 8) and 0xFF; val b = c and 0xFF
            val key = bucket(r, g, b)
            counts[key]++
            sumR[key] += r.toLong(); sumG[key] += g.toLong(); sumB[key] += b.toLong()
        }

        val colorSlots = if (reserveTransparent) maxColors - 1 else maxColors
        val palette = buildPalette(colorSlots.coerceAtLeast(1))
        val transparentIndex = if (reserveTransparent) palette.size else -1
        val fullPalette = if (reserveTransparent) palette + 0 else palette

        java.util.Arrays.fill(nearest, (-1).toShort())
        val indices = ByteArray(count)
        for (i in 0 until count) {
            if (reserveTransparent && isTransparent(i)) {
                indices[i] = transparentIndex.toByte()
                continue
            }
            val c = argb[i]
            val key = bucket((c shr 16) and 0xFF, (c shr 8) and 0xFF, c and 0xFF)
            var idx = nearest[key].toInt()
            if (idx < 0) {
                idx = findNearest(palette, c)
                nearest[key] = idx.toShort()
            }
            indices[i] = idx.toByte()
        }
        return Result(fullPalette, indices, transparentIndex)
    }

    private class Box(var lo: Int, var hi: Int) // range in the sorted bucket list

    private fun buildPalette(maxColors: Int): IntArray {
        // Collect non-empty buckets.
        var n = 0
        for (k in 0 until BUCKETS) if (counts[k] > 0) n++
        if (n == 0) return intArrayOf(0xFF000000.toInt())
        val keys = IntArray(n)
        n = 0
        for (k in 0 until BUCKETS) if (counts[k] > 0) keys[n++] = k

        if (n <= maxColors) {
            return IntArray(n) { averageColor(keys, it, it + 1) }
        }

        val boxes = ArrayList<Box>(maxColors)
        boxes.add(Box(0, n))
        while (boxes.size < maxColors) {
            // Pick the box with the largest (population x longest side): splits busy AND wide boxes first.
            var best = -1
            var bestScore = -1.0
            for (bi in boxes.indices) {
                val b = boxes[bi]
                if (b.hi - b.lo < 2) continue
                val score = population(keys, b).toDouble() * longestSide(keys, b).second
                if (score > bestScore) { bestScore = score; best = bi }
            }
            if (best < 0) break
            val box = boxes[best]
            val (channel, _) = longestSide(keys, box)
            sortRange(keys, box.lo, box.hi, channel)
            // Split at the pixel-weighted median.
            val total = population(keys, box)
            var acc = 0L
            var split = box.lo + 1
            for (i in box.lo until box.hi) {
                acc += counts[keys[i]]
                if (acc * 2 >= total) { split = (i + 1).coerceIn(box.lo + 1, box.hi - 1); break }
            }
            boxes.add(Box(split, box.hi))
            box.hi = split
        }
        return IntArray(boxes.size) { averageColor(keys, boxes[it].lo, boxes[it].hi) }
    }

    private fun population(keys: IntArray, b: Box): Long {
        var p = 0L
        for (i in b.lo until b.hi) p += counts[keys[i]]
        return p
    }

    /** Returns (channel, range) for the channel with the widest spread in the box. */
    private fun longestSide(keys: IntArray, b: Box): Pair<Int, Int> {
        var rMin = 63; var rMax = 0; var gMin = 63; var gMax = 0; var bMin = 63; var bMax = 0
        for (i in b.lo until b.hi) {
            val k = keys[i]
            val r = k shr 12; val g = (k shr 6) and 63; val bl = k and 63
            if (r < rMin) rMin = r; if (r > rMax) rMax = r
            if (g < gMin) gMin = g; if (g > gMax) gMax = g
            if (bl < bMin) bMin = bl; if (bl > bMax) bMax = bl
        }
        // Weight green highest: human eyes are most sensitive to it.
        val rr = (rMax - rMin) * 3; val gr = (gMax - gMin) * 4; val br = (bMax - bMin) * 2
        return when {
            gr >= rr && gr >= br -> 1 to gr
            rr >= br -> 0 to rr
            else -> 2 to br
        }
    }

    private fun sortRange(keys: IntArray, lo: Int, hi: Int, channel: Int) {
        val shift = when (channel) { 0 -> 12; 1 -> 6; else -> 0 }
        // Re-key so the chosen channel is most significant, sort, then restore.
        for (i in lo until hi) {
            val k = keys[i]
            keys[i] = (((k shr shift) and 63) shl 18) or k
        }
        java.util.Arrays.sort(keys, lo, hi)
        for (i in lo until hi) keys[i] = keys[i] and 0x3FFFF
    }

    private fun averageColor(keys: IntArray, lo: Int, hi: Int): Int {
        var n = 0L; var r = 0L; var g = 0L; var b = 0L
        for (i in lo until hi) {
            val k = keys[i]
            n += counts[k]; r += sumR[k]; g += sumG[k]; b += sumB[k]
        }
        if (n == 0L) return 0xFF000000.toInt()
        return (0xFF shl 24) or ((((r + n / 2) / n).toInt()) shl 16) or
            ((((g + n / 2) / n).toInt()) shl 8) or (((b + n / 2) / n).toInt())
    }

    private fun findNearest(palette: IntArray, c: Int): Int {
        val r = (c shr 16) and 0xFF; val g = (c shr 8) and 0xFF; val b = c and 0xFF
        var best = 0
        var bestD = Int.MAX_VALUE
        for (i in palette.indices) {
            val p = palette[i]
            val dr = ((p shr 16) and 0xFF) - r
            val dg = ((p shr 8) and 0xFF) - g
            val db = (p and 0xFF) - b
            val d = 3 * dr * dr + 4 * dg * dg + 2 * db * db
            if (d < bestD) { bestD = d; best = i; if (d == 0) break }
        }
        return best
    }

    companion object {
        private const val BUCKETS = 1 shl 18
        private fun bucket(r: Int, g: Int, b: Int) = ((r shr 2) shl 12) or ((g shr 2) shl 6) or (b shr 2)
    }
}
