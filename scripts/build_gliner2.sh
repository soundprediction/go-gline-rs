#!/usr/bin/env bash
# Build the gliner2_binding cdylib for the host platform and bundle it — together
# with a matching CPU onnxruntime — gzip-compressed into pkg/gliner2/lib/, where
# `//go:embed lib` picks them up. Mirrors the old gline bundling.
#
# ort 2.0.0-rc.9 targets ONNX Runtime 1.20.0; we bundle that exact CPU build so
# load-dynamic resolves a compatible libonnxruntime out of the box.
set -euo pipefail

ONNXRUNTIME_VERSION="1.20.0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LIBDIR="$ROOT/pkg/gliner2/lib"

# Ensure the (gitignored) upstream engine is fetched + patched before building.
if [ ! -f "$ROOT/third_party/gliner2_inference/src/lib.rs" ]; then
  echo "==> third_party/gliner2_inference missing; running setup"
  "$SCRIPT_DIR/setup_gliner2_inference.sh"
fi

OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS-$ARCH" in
  Linux-x86_64)   PLAT=linux-amd64;  DYEXT=so;    ORT_ASSET="onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}";       ORT_LIB="libonnxruntime.so.${ONNXRUNTIME_VERSION}" ;;
  Linux-aarch64)  PLAT=linux-arm64;  DYEXT=so;    ORT_ASSET="onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}";   ORT_LIB="libonnxruntime.so.${ONNXRUNTIME_VERSION}" ;;
  Darwin-arm64)   PLAT=darwin-arm64; DYEXT=dylib; ORT_ASSET="onnxruntime-osx-universal2-${ONNXRUNTIME_VERSION}";  ORT_LIB="libonnxruntime.${ONNXRUNTIME_VERSION}.dylib" ;;
  Darwin-x86_64)  PLAT=darwin-amd64; DYEXT=dylib; ORT_ASSET="onnxruntime-osx-universal2-${ONNXRUNTIME_VERSION}";  ORT_LIB="libonnxruntime.${ONNXRUNTIME_VERSION}.dylib" ;;
  *) echo "unsupported platform: $OS-$ARCH" >&2; exit 1 ;;
esac

echo "==> Building gliner2_binding cdylib for $PLAT"
# net.git-fetch-with-cli lets cargo fetch the git dependency via the git CLI
# (honors local auth/credential helpers); harmless when not needed.
( cd "$ROOT/gliner2_binding" && cargo build --release --config net.git-fetch-with-cli=true )

BIND_OUT="$LIBDIR/$PLAT/libgliner2_binding.$DYEXT.gz"
mkdir -p "$LIBDIR/$PLAT"
gzip -9 -c "$ROOT/gliner2_binding/target/release/libgliner2_binding.$DYEXT" > "$BIND_OUT"
echo "    wrote $BIND_OUT"

echo "==> Bundling onnxruntime $ONNXRUNTIME_VERSION ($ORT_ASSET)"
ORT_OUT_DIR="$LIBDIR/onnxruntime/$PLAT"
mkdir -p "$ORT_OUT_DIR"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ORT_ASSET}.tgz"
echo "    downloading $URL"
curl -sSL -o "$TMP/ort.tgz" "$URL"
tar xzf "$TMP/ort.tgz" -C "$TMP"
gzip -9 -c "$TMP/$ORT_ASSET/lib/$ORT_LIB" > "$ORT_OUT_DIR/libonnxruntime.$DYEXT.gz"
echo "    wrote $ORT_OUT_DIR/libonnxruntime.$DYEXT.gz"

echo "==> Done. Bundled artifacts under $LIBDIR"
