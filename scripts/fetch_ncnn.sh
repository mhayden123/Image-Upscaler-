#!/usr/bin/env bash
# Downloads the prebuilt NCNN library for Android (with Vulkan GPU support)
# into app/src/main/cpp/ncnn-android-vulkan. NCNN is BSD-3-Clause licensed.
set -euo pipefail

NCNN_VERSION="${NCNN_VERSION:-20260526}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/app/src/main/cpp/ncnn-android-vulkan"

if [ -f "$DEST/arm64-v8a/lib/cmake/ncnn/ncnnConfig.cmake" ]; then
  echo "NCNN already present at $DEST"
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
URL="https://github.com/Tencent/ncnn/releases/download/${NCNN_VERSION}/ncnn-${NCNN_VERSION}-android-vulkan.zip"
echo "Downloading $URL"
curl -fL --retry 4 --retry-delay 5 -o "$TMP/ncnn.zip" "$URL"
unzip -q "$TMP/ncnn.zip" -d "$TMP/x"
rm -rf "$DEST"
mv "$TMP/x/ncnn-${NCNN_VERSION}-android-vulkan" "$DEST"
ls "$DEST"
echo "NCNN $NCNN_VERSION installed to $DEST"
