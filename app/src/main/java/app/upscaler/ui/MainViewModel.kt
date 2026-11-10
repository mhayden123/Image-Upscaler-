package app.upscaler.ui

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import android.os.SystemClock
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import app.upscaler.data.ImageIo
import app.upscaler.data.InputImage
import app.upscaler.engine.NativeUpscaler
import app.upscaler.engine.UpscaleEngine
import app.upscaler.engine.UpscaleModel
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

data class UpscaleResult(
    val before: Bitmap,
    /** Display-sized copy of the raw AI result (Android can't draw huge bitmaps). */
    val aiPreview: Bitmap,
    /** Classic enlargement of the original at preview size, for the live texture slider (stills only). */
    val classicPreview: Bitmap?,
    /** The original image, kept to apply the texture setting at full size when saving (stills only). */
    val original: Bitmap?,
    /** Full-resolution raw AI result (stills only; GIFs live in [gifFile]). */
    val full: Bitmap?,
    val gifFile: File?,
    val width: Int,
    val height: Int,
    val frames: Int,
    val elapsedMs: Long,
    val model: UpscaleModel,
    val scale: Int,
    /** Texture setting baked into a GIF result (stills apply it live instead). */
    val texture: Float,
    val savedUri: Uri? = null,
)

sealed interface JobState {
    data object Idle : JobState
    data class Running(val stage: String, val progress: Float?, val startedAt: Long) : JobState
}

data class UiState(
    val input: InputImage? = null,
    val loadingInput: Boolean = false,
    val model: UpscaleModel = UpscaleModel.GENERAL,
    val scale: Int = 4,
    val job: JobState = JobState.Idle,
    val result: UpscaleResult? = null,
    val saving: Boolean = false,
    val message: String? = null,
    val gpuName: String? = null,
    /** "Keep original texture", 0..1. */
    val texture: Float = 0f,
    /** What the "After" side shows: the AI result with the texture setting applied. */
    val afterPreview: Bitmap? = null,
)

class MainViewModel(app: Application) : AndroidViewModel(app) {

    private val engine = UpscaleEngine(app)
    private val prefs = app.getSharedPreferences("settings", android.content.Context.MODE_PRIVATE)
    private val _state = MutableStateFlow(UiState(texture = prefs.getFloat(KEY_TEXTURE, 0f).coerceIn(0f, 1f)))
    val state: StateFlow<UiState> = _state.asStateFlow()
    private var job: Job? = null

    // Must be declared before init {}, which starts collecting it.
    private val blendRequests = MutableStateFlow(0L)
    private val blendBuffers = arrayOfNulls<Bitmap>(2)

    private val resultsDir = File(app.cacheDir, "results").apply {
        deleteRecursively() // leftovers from earlier sessions
        mkdirs()
    }

    init {
        viewModelScope.launch(Dispatchers.Default) {
            val name = try {
                if (engine.gpuCount > 0) engine.gpuName.ifBlank { "GPU" } else ""
            } catch (t: Throwable) { "" }
            _state.update { it.copy(gpuName = name) }
        }
        // Re-blend the preview whenever the slider or the result changes. collectLatest cancels
        // and waits for the previous blend first, so only one blend writes at a time.
        viewModelScope.launch(Dispatchers.Default) {
            blendRequests.collectLatest { updateAfterPreview() }
        }
    }

    private fun requestBlend() = blendRequests.update { it + 1 }

    private fun updateAfterPreview() {
        val s = _state.value
        val r = s.result ?: return
        val shown: Bitmap = if (r.classicPreview == null || s.texture <= 0f) r.aiPreview else {
            // Two buffers: write into the one not currently on screen, then swap.
            val w = r.aiPreview.width
            val h = r.aiPreview.height
            val idx = if (blendBuffers[0] != null && blendBuffers[0] === s.afterPreview) 1 else 0
            var target = blendBuffers[idx]
            if (target == null || target.width != w || target.height != h) {
                target = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                blendBuffers[idx] = target
            }
            NativeUpscaler.nativeBlend(r.aiPreview, r.classicPreview, target, s.texture)
            target.setHasMipMap(true)
            target
        }
        _state.update { if (it.result === r) it.copy(afterPreview = shown) else it }
    }

    fun setTexture(t: Float) {
        val v = t.coerceIn(0f, 1f)
        prefs.edit().putFloat(KEY_TEXTURE, v).apply()
        _state.update { st ->
            val r = st.result
            // A saved still no longer matches what's on screen once the texture changes.
            val newResult = if (r != null && r.gifFile == null && r.savedUri != null) r.copy(savedUri = null) else r
            st.copy(texture = v, result = newResult)
        }
        requestBlend()
    }

    fun pickImage(uri: Uri) {
        if (_state.value.job is JobState.Running) return
        _state.update { it.copy(loadingInput = true, message = null) }
        viewModelScope.launch {
            try {
                val input = withContext(Dispatchers.IO) { ImageIo.load(getApplication(), uri) }
                _state.update { it.copy(input = input, result = null, afterPreview = null, loadingInput = false) }
            } catch (t: Throwable) {
                _state.update { it.copy(loadingInput = false, message = "Couldn't open that image: ${t.message ?: t.javaClass.simpleName}") }
            }
        }
    }

    fun selectModel(model: UpscaleModel) = _state.update { it.copy(model = model) }
    fun selectScale(scale: Int) = _state.update { it.copy(scale = scale) }
    fun dismissMessage() = _state.update { it.copy(message = null) }

    /** Returns an error message if the chosen scale would make the result too large. */
    fun sizeProblem(s: UiState = _state.value): String? {
        val input = s.input ?: return null
        val outPixels = input.width.toLong() * input.height * s.scale * s.scale
        val limit = if (input is InputImage.AnimatedGif) MAX_GIF_FRAME_PIXELS else MAX_STILL_PIXELS
        if (outPixels <= limit) return null
        val mp = limit / 1_000_000
        return if (s.scale > 2 && input.width.toLong() * input.height * 4 <= limit)
            "At ${s.scale}x the result would be over $mp megapixels. Try 2x."
        else "This image is too large to upscale (result would be over $mp megapixels)."
    }

    fun upscale() {
        val s = _state.value
        val input = s.input ?: return
        if (s.job is JobState.Running) return
        sizeProblem(s)?.let { msg -> _state.update { it.copy(message = msg) }; return }

        val started = SystemClock.elapsedRealtime()
        engine.beginJob()
        _state.update { it.copy(job = JobState.Running("Loading AI model…", null, started), result = null, afterPreview = null, message = null) }

        job = viewModelScope.launch {
            try {
                val result = withContext(Dispatchers.Default) {
                    when (input) {
                        is InputImage.Still -> runStill(input, s.model, s.scale, started)
                        is InputImage.AnimatedGif -> runGif(input, s.model, s.scale, s.texture, started)
                    }
                }
                _state.update { it.copy(job = JobState.Idle, result = result, afterPreview = result.aiPreview) }
                requestBlend()
            } catch (e: CancellationException) {
                _state.update { it.copy(job = JobState.Idle, message = "Cancelled") }
            } catch (e: OutOfMemoryError) {
                _state.update { it.copy(job = JobState.Idle, message = "Ran out of memory. Try 2x or a smaller image.") }
            } catch (t: Throwable) {
                _state.update { it.copy(job = JobState.Idle, message = "Upscaling failed: ${t.message ?: t.javaClass.simpleName}") }
            }
        }
    }

    private fun progress(stage: String, p: Float?, started: Long) =
        _state.update { it.copy(job = JobState.Running(stage, p, started)) }

    private fun runStill(input: InputImage.Still, model: UpscaleModel, scale: Int, started: Long): UpscaleResult {
        engine.prepare(model)
        progress("Upscaling…", 0f, started)
        val out = engine.upscaleStill(input.bitmap, model, scale) { p -> progress("Upscaling…", p, started) }
        val aiPreview = ImageIo.previewOf(out)
        return UpscaleResult(
            before = input.bitmap, aiPreview = aiPreview,
            classicPreview = UpscaleEngine.classicResize(input.bitmap, aiPreview.width, aiPreview.height),
            original = input.bitmap, full = out, gifFile = null,
            width = out.width, height = out.height, frames = 1,
            elapsedMs = SystemClock.elapsedRealtime() - started, model = model, scale = scale, texture = 0f,
        )
    }

    private fun runGif(input: InputImage.AnimatedGif, model: UpscaleModel, scale: Int, texture: Float, started: Long): UpscaleResult {
        engine.prepare(model)
        val file = File(resultsDir, "result_${System.currentTimeMillis()}.gif")
        val r = engine.upscaleGif(input.bytes, model, scale, texture, file) { frame, total, p ->
            progress("Frame ${minOf(frame + 1, total)} of $total", p, started)
        }
        return UpscaleResult(
            before = input.preview, aiPreview = ImageIo.previewOf(r.firstFrame), classicPreview = null,
            original = null, full = null, gifFile = r.file,
            width = r.width, height = r.height, frames = r.frames,
            elapsedMs = SystemClock.elapsedRealtime() - started, model = model, scale = scale, texture = texture,
        )
    }

    fun cancel() {
        engine.cancel()
        job?.cancel()
    }

    fun save() {
        val result = _state.value.result ?: return
        if (_state.value.saving) return
        _state.update { it.copy(saving = true) }
        viewModelScope.launch {
            try {
                val texture = _state.value.texture
                val uri = withContext(Dispatchers.Default) {
                    if (result.gifFile != null) ImageIo.saveGif(getApplication(), result.gifFile)
                    else if (texture <= 0f) ImageIo.saveStill(getApplication(), result.full!!)
                    else {
                        // Apply the texture setting at full resolution for the saved file.
                        val blended = UpscaleEngine.applyTexture(result.full!!, result.original!!, texture)
                        try { ImageIo.saveStill(getApplication(), blended) } finally { blended.recycle() }
                    }
                }
                _state.update {
                    it.copy(
                        saving = false,
                        result = if (it.result === result && it.texture == texture) result.copy(savedUri = uri) else it.result,
                        message = "Saved to Pictures/Upscaler",
                    )
                }
            } catch (t: Throwable) {
                _state.update { it.copy(saving = false, message = "Save failed: ${t.message ?: t.javaClass.simpleName}") }
            }
        }
    }

    override fun onCleared() {
        engine.cancel()
        // Never free the model while the GPU may still be using it: wait for the job to finish.
        val j = job
        if (j != null && !j.isCompleted) j.invokeOnCompletion { engine.release() } else engine.release()
    }

    companion object {
        /** 64 MP: a 256 MB bitmap. Plenty for memes; keeps phones from running out of memory. */
        const val MAX_STILL_PIXELS = 64_000_000L
        /** GIF frames are re-encoded one by one; 16 MP per frame keeps that reasonable. */
        const val MAX_GIF_FRAME_PIXELS = 16_000_000L
        private const val KEY_TEXTURE = "texture"
    }
}
