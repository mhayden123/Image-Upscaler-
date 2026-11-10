package app.upscaler.engine

/**
 * The AI models bundled with the app. All are 4x Real-ESRGAN-family networks
 * (BSD-3-Clause) converted to NCNN format; 2x output is produced by upscaling 4x
 * and averaging each 2x2 block, which keeps the AI detail at half the size.
 */
enum class UpscaleModel(
    val title: String,
    val description: String,
    val assetName: String,
    val inputBlob: String,
    val outputBlob: String,
    val nativeScale: Int,
    /** Extra context pixels around each tile so tiles join seamlessly. */
    val prepadding: Int,
) {
    GENERAL(
        title = "General (fast)",
        description = "Good all-rounder for memes, screenshots and photos. Cleans up JPEG blockiness. Fastest.",
        assetName = "realesr-general-x4v3",
        inputBlob = "in0",
        outputBlob = "out0",
        nativeScale = 4,
        prepadding = 16,
    ),
    CARTOON(
        title = "Cartoon & text",
        description = "Crisp lines and flat colors. Best for drawn memes, comics, anime, rage faces and text-heavy images.",
        assetName = "realesrgan-x4plus-anime",
        inputBlob = "data",
        outputBlob = "output",
        nativeScale = 4,
        prepadding = 10,
    ),
    PHOTO(
        title = "Photo (max detail)",
        description = "Most detail for real photos and photo memes. Much slower: big images can take a minute or more.",
        assetName = "realesrgan-x4plus",
        inputBlob = "data",
        outputBlob = "output",
        nativeScale = 4,
        prepadding = 10,
    );

    val paramAsset get() = "models/$assetName.param"
    val binAsset get() = "models/$assetName.bin"
}
