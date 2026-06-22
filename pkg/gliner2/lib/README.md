# Native gliner2_binding artifacts

The platform shared libraries (`libgliner2_binding.{so,dylib}.gz`) are built from
`gliner2_binding/` and dropped here by the Makefile (`make gliner2`) /
`scripts/compile_rust_*.sh`. They are gzip-compressed and embedded into the Go
binary, then extracted + dlopen'd at runtime by package `gliner2`.

Layout:
- `linux-amd64/libgliner2_binding.so.gz`
- `linux-arm64/libgliner2_binding.so.gz`
- `darwin-arm64/libgliner2_binding.dylib.gz`
- `darwin-amd64/libgliner2_binding.dylib.gz`

This README is a placeholder so `//go:embed lib` compiles before the artifacts
exist. The binaries themselves are gitignored (see repo .gitignore).
