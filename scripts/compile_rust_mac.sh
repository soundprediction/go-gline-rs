#!/bin/bash

# This script compiles the Rust library as a Universal Binary for macOS (arm64 + x86_64)

set -e

export MACOSX_DEPLOYMENT_TARGET=11.0

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Confirm OS
OS=$(uname -s)
if [ "$OS" != "Darwin" ]; then
    echo "❌ This script is intended for macOS only."
    exit 1
fi

TARGET_DIR="$PROJECT_ROOT/pkg/gline/lib/darwin"
mkdir -p "$TARGET_DIR"

cd "$PROJECT_ROOT/gline_binding"

# Check if rustup is available to add targets
if command -v rustup >/dev/null 2>&1; then
    echo "🦀 Adding Rust targets for macOS..."
    rustup target add aarch64-apple-darwin x86_64-apple-darwin
else
    echo "⚠️  rustup not found. Assuming targets are already installed."
fi

echo "🍎 Building for Apple Silicon (arm64)..."
cargo build --release --target aarch64-apple-darwin 

echo "💻 Building for Intel (x86_64)..."
if cargo build --release --target x86_64-apple-darwin; then
    HAS_X86=true
else
    echo "⚠️  Failed to build for x86_64. Continuing with only arm64..."
    HAS_X86=false
fi

if [ "$HAS_X86" = true ] && command -v lipo >/dev/null 2>&1; then
    echo "🚀 Creating Universal Binary using lipo..."
    lipo -create \
        target/aarch64-apple-darwin/release/libgline_binding.dylib \
        target/x86_64-apple-darwin/release/libgline_binding.dylib \
        -output "$TARGET_DIR/libgline_binding.dylib"
    
    # Compress
    gzip -9 -f "$TARGET_DIR/libgline_binding.dylib"

    echo "✅ Universal library created at $TARGET_DIR/libgline_binding.dylib.gz"
else
    if [ "$HAS_X86" = false ]; then
         echo "⚠️  Skipping Universal Binary creation due to missing x86_64 build."
         cp target/aarch64-apple-darwin/release/libgline_binding.dylib "$TARGET_DIR/"
    else
         echo "❌ lipo not found. Unable to create universal binary."
         cp target/aarch64-apple-darwin/release/libgline_binding.dylib "$TARGET_DIR/"
    fi
    gzip -9 -f "$TARGET_DIR/libgline_binding.dylib"
     echo "✅ Single-arch library created at $TARGET_DIR/libgline_binding.dylib.gz"
fi
