package app.upscaler.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.calculateCentroid
import androidx.compose.foundation.gestures.calculatePan
import androidx.compose.foundation.gestures.calculateZoom
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.positionChanged
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlin.math.roundToInt

/**
 * Before/after comparison. "Before" is the original stretched with ordinary
 * smoothing (bilinear), which is what you'd normally see when enlarging an image;
 * "After" is the AI result.
 *
 * Gestures: one finger drags the divider, two fingers pinch-zoom/pan,
 * double-tap resets the zoom.
 */
@Composable
fun CompareView(before: ImageBitmap, after: ImageBitmap, modifier: Modifier = Modifier) {
    var divider by remember { mutableFloatStateOf(0.5f) }
    var zoom by remember { mutableFloatStateOf(1f) }
    var offset by remember { mutableStateOf(Offset.Zero) }
    val textMeasurer = rememberTextMeasurer()

    Canvas(
        modifier
            .pointerInput(Unit) {
                detectTapGestures(onDoubleTap = { zoom = 1f; offset = Offset.Zero })
            }
            .pointerInput(Unit) {
                awaitEachGesture {
                    awaitFirstDown(requireUnconsumed = false)
                    do {
                        val event = awaitPointerEvent()
                        val pressed = event.changes.count { it.pressed }
                        if (pressed >= 2) {
                            val centroid = event.calculateCentroid(useCurrent = true)
                            val pan = event.calculatePan()
                            val newZoom = (zoom * event.calculateZoom()).coerceIn(1f, 10f)
                            val c = Offset(size.width / 2f, size.height / 2f)
                            // Keep the content under the fingers in place while zooming.
                            var o = centroid + pan - c - (centroid - offset - c) * (newZoom / zoom)
                            if (newZoom == 1f) o = Offset.Zero
                            val maxX = size.width * newZoom / 2f
                            val maxY = size.height * newZoom / 2f
                            offset = Offset(o.x.coerceIn(-maxX, maxX), o.y.coerceIn(-maxY, maxY))
                            zoom = newZoom
                            event.changes.forEach { if (it.positionChanged()) it.consume() }
                        } else if (pressed == 1) {
                            val ch = event.changes.first { it.pressed }
                            if (ch.positionChanged()) {
                                divider = (ch.position.x / size.width).coerceIn(0f, 1f)
                                ch.consume()
                            }
                        }
                    } while (event.changes.any { it.pressed })
                }
            },
    ) {
        val cw = size.width; val ch = size.height
        val iw = after.width.toFloat(); val ih = after.height.toFloat()
        val fit = minOf(cw / iw, ch / ih)
        val fw = iw * fit * zoom; val fh = ih * fit * zoom
        val left = cw / 2f + offset.x - fw / 2f
        val top = ch / 2f + offset.y - fh / 2f
        val dstOff = IntOffset(left.roundToInt(), top.roundToInt())
        val dstSize = IntSize(fw.roundToInt().coerceAtLeast(1), fh.roundToInt().coerceAtLeast(1))
        val splitX = divider * cw

        clipRect(0f, 0f, splitX, ch) {
            drawImage(before, IntOffset.Zero, IntSize(before.width, before.height), dstOff, dstSize, filterQuality = FilterQuality.Low)
        }
        clipRect(splitX, 0f, cw, ch) {
            drawImage(after, IntOffset.Zero, IntSize(after.width, after.height), dstOff, dstSize, filterQuality = FilterQuality.Low)
        }

        // Divider line and handle.
        val line = 2.dp.toPx()
        drawRect(Color.White, Offset(splitX - line / 2, 0f), Size(line, ch))
        val r = 14.dp.toPx()
        drawCircle(Color.White, r, Offset(splitX, ch / 2))
        drawCircle(Color(0xFF333333), r - line, Offset(splitX, ch / 2))

        // Labels.
        val style = TextStyle(color = Color.White, fontSize = 12.sp)
        val pad = 6.dp.toPx()
        fun label(text: String, alignRight: Boolean) {
            val m = textMeasurer.measure(text, style)
            val x = if (alignRight) cw - m.size.width - pad * 3 else pad
            drawRoundRect(Color(0x99000000), Offset(x, pad), Size(m.size.width + pad * 2, m.size.height + pad), CornerRadius(pad))
            drawText(m, topLeft = Offset(x + pad, pad + pad / 2))
        }
        if (splitX > 60.dp.toPx()) label("Before", alignRight = false)
        if (cw - splitX > 60.dp.toPx()) label("After", alignRight = true)
    }
}
