.PHONY: all build test clean gliner2 rust-mac rust-linux

# Default target
all: build

# Build Go code
build:
	go build ./...

# Run tests
test:
	go test ./...

# Build the GLiNER2 native binding + bundle onnxruntime (host platform) into
# pkg/gliner2/lib for go:embed.
gliner2:
	./scripts/build_gliner2.sh

# --- Deprecated: GLiNER v1 (gline-rs) native library ---
# Compile Rust library for macOS
rust-mac:
	./scripts/compile_rust_mac.sh

# Compile Rust library for Linux
rust-linux:
	./scripts/compile_rust_linux.sh

# Clean build artifacts
clean:
	go clean
	rm -rf gliner2_binding/target gline_binding/target
