import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

// Release signing comes from environment variables (set by CI from GitHub Secrets).
// Without them, release builds fall back to the debug key so they still install.
val signingStoreFile = System.getenv("UPSCALER_KEYSTORE_FILE")?.takeIf { file(it).exists() }

// CI sets GITHUB_RUN_NUMBER, so every build gets a higher version and installs as an update.
val buildNumber = System.getenv("GITHUB_RUN_NUMBER")?.toIntOrNull() ?: 1

android {
    namespace = "app.upscaler"
    compileSdk = 35

    defaultConfig {
        applicationId = "app.upscaler"
        minSdk = 29 // Android 10+: MediaStore saving without storage permission, AnimatedImageDrawable
        targetSdk = 35
        versionCode = buildNumber
        versionName = "1.0.$buildNumber"

        ndk {
            // 64-bit ARM: every modern phone, including the Galaxy S25 Ultra.
            abiFilters += "arm64-v8a"
        }
        externalNativeBuild {
            cmake {
                arguments += listOf("-DANDROID_STL=c++_static", "-DCMAKE_BUILD_TYPE=Release")
            }
        }
    }

    signingConfigs {
        create("release") {
            if (signingStoreFile != null) {
                storeFile = file(signingStoreFile)
                storePassword = System.getenv("UPSCALER_KEYSTORE_PASSWORD")
                keyAlias = System.getenv("UPSCALER_KEY_ALIAS")
                keyPassword = System.getenv("UPSCALER_KEY_PASSWORD")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = if (signingStoreFile != null) signingConfigs.getByName("release")
            else signingConfigs.getByName("debug")
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        compose = true
    }

    androidResources {
        // Store model weights uncompressed so they load fast.
        noCompress += listOf("bin", "param")
    }
}

kotlin {
    compilerOptions { jvmTarget.set(JvmTarget.JVM_17) }
}

// Fail early with a clear message if the models haven't been downloaded/converted.
val checkModels by tasks.registering {
    val dir = layout.projectDirectory.dir("src/main/assets/models")
    doLast {
        val needed = listOf(
            "realesr-general-x4v3", "realesrgan-x4plus-anime", "realesrgan-x4plus",
            "4xNomosWebPhoto_esrgan", "4x-UltraSharp",
        )
            .flatMap { listOf("$it.param", "$it.bin") }
        val missing = needed.filterNot { dir.file(it).asFile.exists() }
        if (missing.isNotEmpty()) {
            throw GradleException("Missing model files $missing in app/src/main/assets/models. Run scripts/fetch_models.sh first.")
        }
    }
}
tasks.named("preBuild") { dependsOn(checkModels) }

dependencies {
    implementation(project(":gifcodec"))

    val composeBom = platform("androidx.compose:compose-bom:2024.12.01")
    implementation(composeBom)
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-core")

    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
}
