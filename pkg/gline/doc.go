// Package gline provides Go bindings for GLiNER v1 span/token/relation models
// via the gline-rs Rust engine.
//
// Deprecated: package gline wraps GLiNER v1 (gline-rs). Use package gliner2
// instead — it runs GLiNER2 multi-task extraction (entities, relations, and
// classifications in a single pass) on ONNX Runtime via gliner2-rs, which is the
// faster, Python-free path and the supported direction going forward. gline is
// kept only for existing callers and will be removed in a future release.
package gline
