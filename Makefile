.PHONY: all build test clean rust-mac rust-linux

# Default target
all: build

# Build Go code
build:
	go build ./...

# Run tests
test:
	go test ./...

# Compile Rust library for macOS
rust-mac:
	./scripts/compile_rust_mac.sh

# Compile Rust library for Linux
rust-linux:
	./scripts/compile_rust_linux.sh

# Clean build artifacts
clean:
	go clean
	rm -rf gline_binding/target
