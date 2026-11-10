package app.upscaler.ui

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import android.os.SystemClock
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import app.upscaler.data.ImageIo
import app.upscaler.data.InputImage
import app.upscaler.engine.UpscaleEngine
import app.upscaler.engine.UpscaleModel
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class UpscaleResult(
    val before: Bitmap,
    /** Display-sized copy of the result (Android can't draw huge bitmaps). */
    val afterPreview: Bitmap,
    /** Full-resolution still result (null for GIFs, which live in [gifFile]). */
    val full: Bitmap?,
    val gifFile: File?,
    val width: Int,
    val height: Int,
    val frames: Int,
    val elapsedMs: Long,
    val model: UpscaleModel,
    val scale: Int,
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
)

class MainViewModel(app: Application) : AndroidViewModel(app) {

    private val engine = UpscaleEngine(app)
    private val _state = MutableStateFlow(UiState())
    val state: StateFlow<UiState> = _state.asStateFlow()
    private var job: Job? = null

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
    }

    fun pickImage(uri: Uri) {
        if (_state.value.job is JobState.Running) return
        _state.update { it.copy(loadingInput = true, message = null) }
        viewModelScope.launch {
            try {
                val input = withContext(Dispatchers.IO) { ImageIo.load(getApplication(), uri) }
                _state.update { it.copy(input = input, result = null, loadingInput = false) }
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
        _state.update { it.copy(job = JobState.Running("Loading AI model…", null, started), result = null, message = null) }

        job = viewModelScope.launch {
            try {
                val result = withContext(Dispatchers.Default) {
                    when (input) {
                        is InputImage.Still -> runStill(input, s.model, s.scale, started)
                        is InputImage.AnimatedGif -> runGif(input, s.model, s.scale, started)
                    }
                }
                _state.update { it.copy(job = JobState.Idle, result = result) }
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
        return UpscaleResult(
            before = input.bitmap, afterPreview = ImageIo.previewOf(out), full = out, gifFile = null,
            width = out.width, height = out.height, frames = 1,
            elapsedMs = SystemClock.elapsedRealtime() - started, model = model, scale = scale,
        )
    }

    private fun runGif(input: InputImage.AnimatedGif, model: UpscaleModel, scale: Int, started: Long): UpscaleResult {
        engine.prepare(model)
        val file = File(resultsDir, "result_${System.currentTimeMillis()}.gif")
        val r = engine.upscaleGif(input.bytes, model, scale, file) { frame, total, p ->
            progress("Frame ${minOf(frame + 1, total)} of $total", p, started)
        }
        return UpscaleResult(
            before = input.preview, afterPreview = ImageIo.previewOf(r.firstFrame), full = null, gifFile = r.file,
            width = r.width, height = r.height, frames = r.frames,
            elapsedMs = SystemClock.elapsedRealtime() - started, model = model, scale = scale,
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
                val uri = withContext(Dispatchers.IO) {
                    if (result.gifFile != null) ImageIo.saveGif(getApplication(), result.gifFile)
                    else ImageIo.saveStill(getApplication(), result.full!!)
                }
                _state.update {
                    it.copy(
                        saving = false,
                        result = if (it.result === result) copyWithSaved(result, uri) else it.result,
                        message = "Saved to Pictures/Upscaler",
                    )
                }
            } catch (t: Throwable) {
                _state.update { it.copy(saving = false, message = "Save failed: ${t.message ?: t.javaClass.simpleName}") }
            }
        }
    }

    private fun copyWithSaved(r: UpscaleResult, uri: Uri) = UpscaleResult(
        r.before, r.afterPreview, r.full, r.gifFile, r.width, r.height, r.frames, r.elapsedMs, r.model, r.scale, uri,
    )

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
    }
}
