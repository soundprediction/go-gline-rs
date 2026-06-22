#!/usr/bin/env bash
# Build the gliner2_binding cdylib + bundle a matching CPU onnxruntime, gzip-compressed
# into pkg/gliner2/lib/ where `//go:embed lib` picks them up. Mirrors the old gline bundling.
#
# - On Linux: builds for the host arch (linux-amd64 or linux-arm64).
# - On macOS: builds a UNIVERSAL2 binary (arm64 + x86_64 via lipo) and emits it to BOTH
#   darwin-arm64 and darwin-amd64, so one Apple-Silicon runner covers both Mac arches
#   (Intel macOS runners are being retired).
#
# ort 2.0.0-rc.9 targets ONNX Runtime 1.20.0; we bundle that exact CPU build so
# load-dynamic resolves a compatible libonnxruntime out of the box.
set -euo pipefail

ONNXRUNTIME_VERSION="1.20.0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LIBDIR="$ROOT/pkg/gliner2/lib"
BINDIR="$ROOT/gliner2_binding"

# Ensure the (gitignored) upstream engine is fetched + patched before building.
if [ ! -f "$ROOT/third_party/gliner2_inference/src/lib.rs" ]; then
  echo "==> third_party/gliner2_inference missing; running setup"
  "$SCRIPT_DIR/setup_gliner2_inference.sh"
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# emit_binding <platform> <dylib_path> <dyext>
emit_binding() {
  local plat="$1" src="$2" dyext="$3"
  mkdir -p "$LIBDIR/$plat"
  gzip -9 -c "$src" > "$LIBDIR/$plat/libgliner2_binding.$dyext.gz"
  echo "    wrote $LIBDIR/$plat/libgliner2_binding.$dyext.gz"
}

# fetch_onnx <platform> <asset_basename> <lib_filename_in_tarball> <dyext>
fetch_onnx() {
  local plat="$1" asset="$2" lib="$3" dyext="$4"
  mkdir -p "$LIBDIR/onnxruntime/$plat"
  if [ ! -f "$TMP/$asset/lib/$lib" ]; then
    local url="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${asset}.tgz"
    echo "    downloading $url"
    curl -sSL -o "$TMP/$asset.tgz" "$url"
    tar xzf "$TMP/$asset.tgz" -C "$TMP"
  fi
  gzip -9 -c "$TMP/$asset/lib/$lib" > "$LIBDIR/onnxruntime/$plat/libonnxruntime.$dyext.gz"
  echo "    wrote $LIBDIR/onnxruntime/$plat/libonnxruntime.$dyext.gz"
}

OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Linux)
    case "$ARCH" in
      x86_64)  PLAT=linux-amd64; ORT_ASSET="onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}" ;;
      aarch64) PLAT=linux-arm64; ORT_ASSET="onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}" ;;
      *) echo "unsupported linux arch: $ARCH" >&2; exit 1 ;;
    esac
    echo "==> Building gliner2_binding cdylib for $PLAT"
    ( cd "$BINDIR" && cargo build --release --config net.git-fetch-with-cli=true )
    emit_binding "$PLAT" "$BINDIR/target/release/libgliner2_binding.so" so
    echo "==> Bundling onnxruntime $ONNXRUNTIME_VERSION ($ORT_ASSET)"
    fetch_onnx "$PLAT" "$ORT_ASSET" "libonnxruntime.so.${ONNXRUNTIME_VERSION}" so
    ;;

  Darwin)
    echo "==> Building UNIVERSAL2 gliner2_binding cdylib (arm64 + x86_64)"
    rustup target add aarch64-apple-darwin x86_64-apple-darwin >/dev/null 2>&1 || true
    ( cd "$BINDIR" && cargo build --release --target aarch64-apple-darwin --config net.git-fetch-with-cli=true )
    ( cd "$BINDIR" && cargo build --release --target x86_64-apple-darwin  --config net.git-fetch-with-cli=true )
    lipo -create \
      "$BINDIR/target/aarch64-apple-darwin/release/libgliner2_binding.dylib" \
      "$BINDIR/target/x86_64-apple-darwin/release/libgliner2_binding.dylib" \
      -output "$TMP/libgliner2_binding.universal.dylib"
    ORT_ASSET="onnxruntime-osx-universal2-${ONNXRUNTIME_VERSION}"
    echo "==> Bundling onnxruntime $ONNXRUNTIME_VERSION ($ORT_ASSET) for both Mac arches"
    for PLAT in darwin-arm64 darwin-amd64; do
      emit_binding "$PLAT" "$TMP/libgliner2_binding.universal.dylib" dylib
      fetch_onnx "$PLAT" "$ORT_ASSET" "libonnxruntime.${ONNXRUNTIME_VERSION}.dylib" dylib
    done
    ;;

  *) echo "unsupported OS: $OS" >&2; exit 1 ;;
esac

echo "==> Done. Bundled artifacts under $LIBDIR"
