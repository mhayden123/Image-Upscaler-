package app.upscaler

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import app.upscaler.ui.MainViewModel
import app.upscaler.ui.UpscalerScreen
import app.upscaler.ui.theme.UpscalerTheme

class MainActivity : ComponentActivity() {
    private val vm: MainViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            UpscalerTheme { UpscalerScreen(vm) }
        }
    }
}
