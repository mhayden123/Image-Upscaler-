#!/usr/bin/env bash
# Downloads the Real-ESRGAN models (BSD-3-Clause, by Xintao Wang et al.) and puts
# NCNN versions of them in app/src/main/assets/models:
#   realesrgan-x4plus        - "Photo (max detail)"   (official NCNN release)
#   realesrgan-x4plus-anime  - "Cartoon & text"       (official NCNN release)
#   realesr-general-x4v3     - "General (fast)"       (converted from PyTorch here)
# plus two community ESRGAN models, converted from PyTorch here:
#   4xNomosWebPhoto_esrgan   - "Web photo (sharp)"   by Philip Hofmann, CC BY 4.0
#   4x-UltraSharp            - "UltraSharp"          by Kim2091, CC BY-NC-SA 4.0 (non-commercial)
#
# Needs: curl, unzip, python3 with torch + pnnx (see scripts/requirements-convert.txt).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/app/src/main/assets/models"
PYTHON="${PYTHON:-python3}"
REL="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0"
mkdir -p "$DEST"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

have() { [ -s "$DEST/$1.param" ] && [ -s "$DEST/$1.bin" ]; }

if have realesrgan-x4plus && have realesrgan-x4plus-anime; then
  echo "x4plus models already present"
else
  echo "Downloading official Real-ESRGAN NCNN models"
  curl -fL --retry 4 --retry-delay 5 -o "$TMP/realesrgan.zip" "$REL/realesrgan-ncnn-vulkan-20220424-ubuntu.zip"
  unzip -q "$TMP/realesrgan.zip" -d "$TMP/realesrgan"
  echo "Archive contents:"; find "$TMP/realesrgan" -name '*.param' -o -name '*.bin' | sort
  for m in realesrgan-x4plus realesrgan-x4plus-anime; do
    p="$(find "$TMP/realesrgan" -name "$m.param" | head -1)"
    b="$(find "$TMP/realesrgan" -name "$m.bin" | head -1)"
    [ -n "$p" ] && [ -n "$b" ] || { echo "ERROR: $m not found in archive"; exit 1; }
    cp "$p" "$b" "$DEST/"
  done
fi

if have realesr-general-x4v3; then
  echo "realesr-general-x4v3 already present"
else
  echo "Downloading realesr-general-x4v3.pth and converting to NCNN"
  curl -fL --retry 4 --retry-delay 5 -o "$TMP/realesr-general-x4v3.pth" "$REL/realesr-general-x4v3.pth"
  "$PYTHON" "$ROOT/scripts/convert_to_ncnn.py" --arch srvgg --weights "$TMP/realesr-general-x4v3.pth" --out-dir "$DEST" --name realesr-general-x4v3
fi

# Downloads the first URL that works (community uploads sometimes move), then converts it.
convert_rrdb() { # name, urls...
  local name="$1"; shift
  if have "$name"; then echo "$name already present"; return; fi
  local file
  for url in "$@"; do
    file="$TMP/$name.${url##*.}"
    echo "Trying $url"
    if curl -fL --retry 3 --retry-delay 5 -o "$file" "$url"; then
      "$PYTHON" "$ROOT/scripts/convert_to_ncnn.py" --arch rrdb --weights "$file" --out-dir "$DEST" --name "$name"
      return
    fi
  done
  echo "ERROR: could not download $name from any source"; exit 1
}

convert_rrdb 4xNomosWebPhoto_esrgan \
  "https://github.com/Phhofm/models/releases/download/4xNomosWebPhoto_esrgan/4xNomosWebPhoto_esrgan.pth" \
  "https://huggingface.co/Phips/4xNomosWebPhoto_esrgan/resolve/main/4xNomosWebPhoto_esrgan.pth" \
  "https://huggingface.co/Phips/4xNomosWebPhoto_esrgan/resolve/main/4xNomosWebPhoto_esrgan.safetensors"

convert_rrdb 4x-UltraSharp \
  "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth" \
  "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.safetensors"

echo
echo "Models in $DEST:"
ls -la "$DEST"
# The app looks up input/output layers by name (see UpscaleModel.kt): verify they exist.
check_blobs() {
  local f="$DEST/$1.param"
  for blob in "$2" "$3"; do
    grep -qw -- "$blob" "$f" || { echo "ERROR: blob '$blob' not found in $1.param"; exit 1; }
  done
  echo "  $1: blobs '$2' -> '$3' OK"
}
check_blobs realesr-general-x4v3 in0 out0
check_blobs realesrgan-x4plus-anime data output
check_blobs realesrgan-x4plus data output
check_blobs 4xNomosWebPhoto_esrgan in0 out0
check_blobs 4x-UltraSharp in0 out0
sha256sum "$DEST"/*
