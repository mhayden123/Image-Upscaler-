# Upscaler

An offline Android app that enlarges and enhances images with AI, including memes, photos, screenshots and **animated GIFs**.
Everything runs on the phone's GPU. No internet connection is needed, and the app doesn't even request internet permission.

## Features

- **3 AI models** to choose from (all from the open-source Real-ESRGAN project):

  | Model | Best for | Speed | Size in APK |
  |---|---|---|---|
  | **General (fast)** (`realesr-general-x4v3`) | Most memes, screenshots, photos; removes JPEG blockiness | Fastest | ~2.4 MB |
  | **Cartoon & text** (`RealESRGAN_x4plus_anime_6B`) | Drawn memes, comics, anime, text-heavy images | Medium | ~9 MB |
  | **Photo (max detail)** (`RealESRGAN_x4plus`) | Real photos and photo-based memes | Slowest | ~33 MB |

- **2x or 4x** scale.
- **Before/after slider**: drag to compare, pinch to zoom, double-tap to reset.
- **Animated GIFs**: every frame is upscaled and a new animated GIF is written, keeping frame timing and looping.
- **Save to Gallery** in `Pictures/Upscaler` (PNG for still images, GIF for animations).

Requires Android 10 or newer on a 64-bit ARM phone (practically every phone from the last several years, e.g. the Galaxy S25 Ultra).

## Installing on your phone

GitHub builds the APK automatically on every push (see `.github/workflows/build.yml`).

- **From a Release** (pushes to the default branch): open the repo's **Releases** page on your phone, download `Upscaler-v1.0.N.apk`, and tap it.
- **From any branch**: on the **Actions** tab, open the latest "Build APK" run and download the `Upscaler-v1.0.N` artifact. GitHub wraps artifacts in a `.zip`, so unzip it first.

Android will ask you to allow installing apps from your browser or Files app the first time. That's normal for apps that don't come from the Play Store.

### Signing (so updates install over the old version)

Android only lets an update replace an installed app if both are signed with the same key.
Store your permanent key in the repo's GitHub Secrets (**Settings → Secrets and variables → Actions → New repository secret**):

| Secret name | Value |
|---|---|
| `UPSCALER_KEYSTORE_BASE64` | the keystore file, base64-encoded |
| `UPSCALER_KEYSTORE_PASSWORD` | keystore password |
| `UPSCALER_KEY_ALIAS` | key alias |
| `UPSCALER_KEY_PASSWORD` | key password |

Without these secrets, CI signs with a throwaway debug key and shows a warning. Those builds still install, but you have to uninstall the previous version first.
**Back up the keystore and passwords somewhere safe.** If you lose them, future builds can't update the installed app.

To make your own key: `keytool -genkeypair -v -keystore upscaler.jks -alias upscaler -keyalg RSA -keysize 4096 -validity 10000`, then `base64 -w0 upscaler.jks`.

## How it works

1. **Decoding**: still images go through Android's `ImageDecoder` (JPEG, PNG, WebP, HEIC…). Animated GIFs go through a custom streaming decoder (`gifcodec/`) that composites frames the way a browser does.
2. **Inference**: the C++ code (`app/src/main/cpp/`) runs the model with [NCNN](https://github.com/Tencent/ncnn) on the GPU via Vulkan, falling back to the CPU if there's no GPU. The image is split into tiles so large images fit in GPU memory. Each tile is padded with neighbouring pixels and the padding is trimmed afterwards, so there are no visible seams.
3. **2x mode**: all three models work at 4x natively. For 2x, each 2×2 block of the 4x output is averaged, which keeps the AI detail at half the size.
4. **Transparency**: the models handle color only. The alpha channel is enlarged separately with bilinear interpolation.
5. **GIF encoding**: each upscaled frame gets its own optimized 256-color palette (median-cut). Unchanged pixels between frames are stored as transparent, which keeps the file much smaller.

## Building locally (Android Studio)

The NCNN library and the model files are downloaded at build time, not committed:

```bash
scripts/fetch_ncnn.sh                       # NCNN prebuilt for Android (Vulkan)
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r scripts/requirements-convert.txt
scripts/fetch_models.sh                     # downloads models, converts realesr-general-x4v3 to NCNN
```

Then open the project in Android Studio and run it, or use `./gradlew assembleRelease`.

## Tests

- `./gradlew :gifcodec:test`: GIF encoder/decoder, checked against Java's independent ImageIO GIF codec (round trips, LZW table resets, interlacing, transparency, frame diffing, truncated files).
- `scripts/run_native_tests.sh`: builds NCNN for your computer and tests the app's C++ upscaling core. Tiled output must match whole-image output, 2x must equal the 2×2 average of 4x, plus alpha, row strides, cancellation and progress.

CI runs both before building the APK.

## Credits and licenses

- **Real-ESRGAN** models: Copyright (c) 2021, Xintao Wang. [BSD 3-Clause](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE).
- **NCNN**: Copyright (C) 2017 Tencent. [BSD 3-Clause](https://github.com/Tencent/ncnn/blob/master/LICENSE.txt).

The app's About screen (ⓘ) shows these notices.
