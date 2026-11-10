package app.upscaler.ui

import android.graphics.ImageDecoder
import android.graphics.drawable.AnimatedImageDrawable
import android.widget.ImageView
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import app.upscaler.data.InputImage
import app.upscaler.engine.UpscaleModel
import kotlinx.coroutines.delay
import java.io.File
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun UpscalerScreen(vm: MainViewModel) {
    val state by vm.state.collectAsStateWithLifecycle()
    val snackbar = remember { SnackbarHostState() }
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) vm.pickImage(uri)
    }
    val pick = { picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) }
    val running = state.job as? JobState.Running
    var showAbout by rememberSaveable { mutableStateOf(false) }
    if (showAbout) AboutDialog(onDismiss = { showAbout = false })

    LaunchedEffect(state.message) {
        state.message?.let { snackbar.showSnackbar(it); vm.dismissMessage() }
    }

    // Keep the screen on while processing so the phone doesn't sleep mid-job.
    val view = LocalView.current
    DisposableEffect(running != null) {
        view.keepScreenOn = running != null
        onDispose { view.keepScreenOn = false }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Upscaler") },
                actions = {
                    state.gpuName?.let {
                        Text(
                            if (it.isBlank()) "CPU mode" else "GPU: $it",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.width(150.dp),
                            maxLines = 1,
                        )
                    }
                    IconButton(onClick = { showAbout = true }) {
                        Icon(Icons.Filled.Info, contentDescription = "About and licenses")
                    }
                },
            )
        },
        snackbarHost = { SnackbarHost(snackbar) },
    ) { padding ->
        Column(
            Modifier
                .padding(padding)
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            val input = state.input
            if (input == null) {
                EmptyPicker(loading = state.loadingInput, onPick = pick)
                return@Column
            }

            ImageArea(state)

            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(describeInput(input), style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
                TextButton(onClick = pick, enabled = running == null) { Text("Change image") }
            }

            if (running != null) {
                ProgressPanel(running, onCancel = vm::cancel)
            } else {
                state.result?.let { ResultPanel(it, saving = state.saving, onSave = vm::save) }
                Settings(state, vm)
                val problem = vm.sizeProblem(state)
                if (problem != null) {
                    Text(problem, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
                }
                Button(
                    onClick = vm::upscale,
                    enabled = problem == null && !state.loadingInput,
                    modifier = Modifier.fillMaxWidth().height(52.dp),
                ) {
                    Text(if (state.result == null) "Upscale ${state.scale}x" else "Upscale again with these settings")
                }
            }
            Spacer(Modifier.height(16.dp))
        }
    }
}

private fun describeInput(input: InputImage): String = when (input) {
    is InputImage.Still -> "${input.width} × ${input.height}"
    is InputImage.AnimatedGif -> "${input.width} × ${input.height} · animated GIF, ${input.frameCount} frames"
}

@Composable
private fun EmptyPicker(loading: Boolean, onPick: () -> Unit) {
    Card(
        onClick = onPick,
        enabled = !loading,
        modifier = Modifier.fillMaxWidth().height(320.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
    ) {
        Column(
            Modifier.fillMaxSize(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            if (loading) {
                CircularProgressIndicator()
            } else {
                Icon(Icons.Filled.Add, contentDescription = null, modifier = Modifier.size(56.dp))
                Spacer(Modifier.height(12.dp))
                Text("Pick an image", style = MaterialTheme.typography.titleMedium)
                Spacer(Modifier.height(4.dp))
                Text(
                    "Memes, photos, screenshots, GIFs. Everything runs on your phone, offline.",
                    style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.padding(horizontal = 32.dp),
                )
            }
        }
    }
}

@Composable
private fun ImageArea(state: UiState) {
    val input = state.input ?: return
    val result = state.result
    var playGif by remember(result) { mutableStateOf(false) }
    val box = Modifier
        .fillMaxWidth()
        .heightIn(min = 240.dp)
        .height(420.dp)
        .clip(RoundedCornerShape(16.dp))
        .background(Color(0xFF111111))

    Box(box) {
        when {
            result != null && playGif && result.gifFile != null -> AnimatedGifView(result.gifFile, Modifier.fillMaxSize())
            result != null -> {
                val before = remember(result) { result.before.asImageBitmap() }
                val after = remember(result) { result.afterPreview.asImageBitmap() }
                CompareView(before, after, Modifier.fillMaxSize())
            }
            else -> Image(
                bitmap = remember(input) { input.preview.asImageBitmap() },
                contentDescription = "Selected image",
                contentScale = ContentScale.Fit,
                modifier = Modifier.fillMaxSize(),
            )
        }
    }
    if (result != null) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                if (playGif) "Playing the upscaled GIF" else "Drag to compare · pinch to zoom · double-tap to reset",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.weight(1f),
            )
            if (result.gifFile != null) {
                TextButton(onClick = { playGif = !playGif }) {
                    if (!playGif) Icon(Icons.Filled.PlayArrow, contentDescription = null)
                    Text(if (playGif) "Compare" else "Play")
                }
            }
        }
    }
}

@Composable
private fun AnimatedGifView(file: File, modifier: Modifier) {
    AndroidView(
        factory = { ctx -> ImageView(ctx).apply { scaleType = ImageView.ScaleType.FIT_CENTER } },
        update = { iv ->
            val d = ImageDecoder.decodeDrawable(ImageDecoder.createSource(file))
            iv.setImageDrawable(d)
            (d as? AnimatedImageDrawable)?.start()
        },
        modifier = modifier,
    )
}

@Composable
private fun ProgressPanel(running: JobState.Running, onCancel: () -> Unit) {
    var now by remember { mutableLongStateOf(android.os.SystemClock.elapsedRealtime()) }
    LaunchedEffect(running.startedAt) {
        while (true) { now = android.os.SystemClock.elapsedRealtime(); delay(250) }
    }
    val elapsed = (now - running.startedAt).coerceAtLeast(0) / 1000.0
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(running.stage, style = MaterialTheme.typography.titleSmall, modifier = Modifier.weight(1f))
                Text(String.format(Locale.US, "%.0fs", elapsed), style = MaterialTheme.typography.bodySmall)
            }
            val p = running.progress
            if (p == null) LinearProgressIndicator(Modifier.fillMaxWidth())
            else LinearProgressIndicator(progress = { p }, modifier = Modifier.fillMaxWidth())
            if (p != null) {
                // Rough estimate from progress so far.
                val eta = if (p > 0.03f) elapsed / p * (1 - p) else null
                Text(
                    String.format(Locale.US, "%d%%", (p * 100).toInt()) + (eta?.let { String.format(Locale.US, " · about %.0fs left", it) } ?: ""),
                    style = MaterialTheme.typography.bodySmall,
                )
            }
            OutlinedButton(onClick = onCancel, modifier = Modifier.align(Alignment.End)) { Text("Cancel") }
        }
    }
}

@Composable
private fun ResultPanel(result: UpscaleResult, saving: Boolean, onSave: () -> Unit) {
    val sizeText = result.gifFile?.let { " · " + formatBytes(it.length()) } ?: ""
    Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer)) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(
                "Done in " + String.format(Locale.US, "%.1fs", result.elapsedMs / 1000.0),
                style = MaterialTheme.typography.titleSmall,
            )
            Text(
                "${result.width} × ${result.height} · ${result.scale}x · ${result.model.title}" +
                    (if (result.frames > 1) " · ${result.frames} frames" else "") + sizeText,
                style = MaterialTheme.typography.bodySmall,
            )
            if (result.savedUri != null) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Filled.CheckCircle, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                    Spacer(Modifier.width(8.dp))
                    Text("Saved to Pictures/Upscaler", style = MaterialTheme.typography.bodyMedium)
                }
            } else {
                FilledTonalButton(onClick = onSave, enabled = !saving, modifier = Modifier.fillMaxWidth()) {
                    if (saving) {
                        CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp)
                        Spacer(Modifier.width(8.dp))
                        Text("Saving…")
                    } else {
                        Text(if (result.gifFile != null) "Save GIF to Gallery" else "Save PNG to Gallery")
                    }
                }
            }
        }
    }
}

private fun formatBytes(b: Long): String = when {
    b >= 1_000_000 -> String.format(Locale.US, "%.1f MB", b / 1_000_000.0)
    b >= 1_000 -> String.format(Locale.US, "%.0f KB", b / 1_000.0)
    else -> "$b B"
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun Settings(state: UiState, vm: MainViewModel) {
    Text("AI model", style = MaterialTheme.typography.titleSmall)
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        UpscaleModel.entries.forEach { m ->
            val selected = state.model == m
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .selectable(selected = selected, role = Role.RadioButton, onClick = { vm.selectModel(m) }),
                border = if (selected) BorderStroke(2.dp, MaterialTheme.colorScheme.primary) else null,
            ) {
                Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
                    RadioButton(selected = selected, onClick = null)
                    Spacer(Modifier.width(8.dp))
                    Column {
                        Text(m.title, fontWeight = FontWeight.SemiBold)
                        Text(m.description, style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
        }
    }

    Text("Scale", style = MaterialTheme.typography.titleSmall)
    val input = state.input
    SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
        listOf(2, 4).forEachIndexed { i, s ->
            SegmentedButton(
                selected = state.scale == s,
                onClick = { vm.selectScale(s) },
                shape = SegmentedButtonDefaults.itemShape(index = i, count = 2),
            ) {
                Text(if (input != null) "${s}x  (${input.width * s} × ${input.height * s})" else "${s}x")
            }
        }
    }
}

@Composable
private fun AboutDialog(onDismiss: () -> Unit) {
    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = { TextButton(onClick = onDismiss) { Text("Close") } },
        title = { Text("About Upscaler") },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(
                    "Everything runs on your phone. Images never leave the device and the app has no internet permission.",
                    style = MaterialTheme.typography.bodySmall,
                )
                Text("How it works", fontWeight = FontWeight.SemiBold)
                Text(
                    "A neural network trained on millions of degraded/clean image pairs predicts the missing detail " +
                        "at 4x size. The image is processed in overlapping tiles on the GPU (Vulkan). " +
                        "For 2x, each 2x2 block of the 4x result is averaged.",
                    style = MaterialTheme.typography.bodySmall,
                )
                Text("Open-source credits", fontWeight = FontWeight.SemiBold)
                Text(LICENSES, style = MaterialTheme.typography.bodySmall)
            }
        },
    )
}

private const val BSD3 = "Redistribution and use in source and binary forms, with or without modification, are permitted " +
    "provided that the following conditions are met: (1) Redistributions of source code must retain the above copyright " +
    "notice, this list of conditions and the following disclaimer. (2) Redistributions in binary form must reproduce the " +
    "above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other " +
    "materials provided with the distribution. (3) Neither the name of the copyright holder nor the names of its " +
    "contributors may be used to endorse or promote products derived from this software without specific prior written " +
    "permission. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\" AND ANY EXPRESS OR IMPLIED " +
    "WARRANTIES ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES ARISING " +
    "IN ANY WAY OUT OF THE USE OF THIS SOFTWARE."

private val LICENSES = """
Real-ESRGAN models (realesr-general-x4v3, RealESRGAN_x4plus, RealESRGAN_x4plus_anime_6B)
Copyright (c) 2021, Xintao Wang. BSD 3-Clause License.

NCNN neural network inference framework
Copyright (C) 2017 Tencent. All rights reserved. BSD 3-Clause License.
NCNN includes third-party components under their own permissive licenses
(listed in its LICENSE.txt: github.com/Tencent/ncnn).

$BSD3
""".trim()
