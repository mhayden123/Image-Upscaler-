#!/usr/bin/env bash
# Builds NCNN (CPU only) for this computer, creates a small random-weight model with
# the real conversion script, then compiles and runs tests/native/test_core.cpp
# against the app's native upscaling code.
# Needs: curl, cmake, a C++ compiler with OpenMP, python3 with torch + pnnx.
set -euo pipefail

NCNN_VERSION="${NCNN_VERSION:-20260526}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${WORK:-$ROOT/build/native-tests}"
PYTHON="${PYTHON:-python3}"
mkdir -p "$WORK"

INSTALL="$WORK/ncnn-install"
if [ ! -f "$INSTALL/lib/libncnn.a" ]; then
  echo "Building NCNN $NCNN_VERSION for the host (CPU only)…"
  URL="$(curl -fsS "https://pypi.org/pypi/ncnn/1.0.${NCNN_VERSION}/json" |
    "$PYTHON" -c 'import json,sys; print([u["url"] for u in json.load(sys.stdin)["urls"] if u["packagetype"]=="sdist"][0])')"
  curl -fL --retry 4 -o "$WORK/ncnn.tar.gz" "$URL"
  rm -rf "$WORK/ncnn-src" && mkdir -p "$WORK/ncnn-src"
  tar xzf "$WORK/ncnn.tar.gz" -C "$WORK/ncnn-src" --strip-components=1
  cmake -S "$WORK/ncnn-src" -B "$WORK/ncnn-build" -DCMAKE_BUILD_TYPE=Release \
    -DNCNN_VULKAN=OFF -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF \
    -DNCNN_BUILD_TESTS=OFF -DNCNN_PYTHON=OFF -DNCNN_SHARED_LIB=OFF -DCMAKE_INSTALL_PREFIX="$INSTALL"
  cmake --build "$WORK/ncnn-build" -j"$(nproc)"
  cmake --install "$WORK/ncnn-build"
fi

echo "Creating a random-weight test model with the real converter…"
"$PYTHON" "$ROOT/scripts/convert_to_ncnn.py" --arch srvgg --random --out-dir "$WORK/model" --name test

echo "Compiling and running native tests…"
g++ -std=c++17 -O2 -fopenmp -I"$ROOT/app/src/main/cpp" -I"$INSTALL/include/ncnn" \
  "$ROOT/tests/native/test_core.cpp" "$ROOT/app/src/main/cpp/upscale_core.cpp" \
  "$INSTALL/lib/libncnn.a" -lpthread -o "$WORK/test_core"
"$WORK/test_core" "$WORK/model"
