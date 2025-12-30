#!/bin/bash

# This script compiles the Rust library for Linux (amd64 + arm64)
# It assumes cross-compilation tools are set up or runs on native linux

set -e

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

TARGET_DIR_AMD64="$PROJECT_ROOT/pkg/gline/lib/linux-amd64"
TARGET_DIR_ARM64="$PROJECT_ROOT/pkg/gline/lib/linux-arm64"
mkdir -p "$TARGET_DIR_AMD64"
mkdir -p "$TARGET_DIR_ARM64"

cd "$PROJECT_ROOT/gline_binding"

# Basic build for current arch (assuming running in CI that handles matrix)
# If we need cross compilation, we'd add it here.
# For now, let's look at how embedeverything handles it.
# It uses github strategy matrix. So this script might be run with specific target intent.

# But wait, local usage? 
# Let's support standard cargo build for host architecture if no args.

echo "🐧 Building for Linux..."
cargo build --release

# Determine Arch
ARCH=$(uname -m)
if [ "$ARCH" == "x86_64" ]; then
    cp target/release/libgline_binding.so "$TARGET_DIR_AMD64/libgline_binding.so"
    gzip -9 -f "$TARGET_DIR_AMD64/libgline_binding.so"
    echo "✅ AMD64 library created."
elif [ "$ARCH" == "aarch64" ]; then
    cp target/release/libgline_binding.so "$TARGET_DIR_ARM64/libgline_binding.so"
    gzip -9 -f "$TARGET_DIR_ARM64/libgline_binding.so"
    echo "✅ ARM64 library created."
else
    echo "⚠️ Unknown architecture: $ARCH"
fi
