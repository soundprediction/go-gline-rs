// Package gliner2 provides native Go bindings for GLiNER2 multi-task extraction
// (named entities, relations, and classifications) running on ONNX Runtime via
// the gliner2_inference Rust engine (https://github.com/SemplificaAI/gliner2-rs).
//
// It loads the gliner2_binding cdylib at runtime (dlopen), so the heavy ONNX
// dependency is isolated from the Go binary. A single forward pass extracts all
// configured tasks; results cross the FFI boundary as JSON. CPU/GPU selection is
// handled inside the engine (ort execution-provider fallback chain).
//
// This supersedes package gline (GLiNER v1). New code should use gliner2.
//
// The engine downloads model weights from Hugging Face on first use (inside the
// Rust layer via hf-hub). Because ort is built with the load-dynamic feature,
// libonnxruntime must be resolvable at runtime: a CPU build is embedded and
// auto-extracted (ORT_DYLIB_PATH is set to it during Init). To use a GPU build,
// set ORT_DYLIB_PATH yourself before first use and it will be respected.
package gliner2

/*
#cgo LDFLAGS: -ldl
#include "gliner2.h"

// dlopen helpers.
static void* _g2_open_lib(const char* path) {
    return dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
}
static char* _g2_get_dlerror(void) {
    return dlerror();
}
static void* _g2_get_sym(void* handle, const char* name) {
    return dlsym(handle, name);
}

// Typed call wrappers: cast the resolved symbol to its function type and invoke.
void* _g2_call_new(void* f, const char* repo, const char* sub, int mt) {
    return ((gliner2_new_t)f)(repo, sub, mt);
}
char* _g2_call_extract(void* f, void* eng, const char* text, const char* tasks, float threshold, int flat_ner) {
    return ((gliner2_extract_t)f)(eng, text, tasks, threshold, flat_ner);
}
void _g2_call_free_engine(void* f, void* eng) {
    ((gliner2_free_engine_t)f)(eng);
}
void _g2_call_free_string(void* f, char* s) {
    ((gliner2_free_string_t)f)(s);
}
const char* _g2_call_last_error(void* f) {
    return ((gliner2_last_error_t)f)();
}
*/
import "C"

import (
	"compress/gzip"
	"embed"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"unsafe"
)

//go:embed lib
var libFS embed.FS

var (
	initOnce sync.Once
	initErr  error

	dlHandle unsafe.Pointer

	fnNew        unsafe.Pointer
	fnExtract    unsafe.Pointer
	fnFreeEngine unsafe.Pointer
	fnFreeString unsafe.Pointer
	fnLastError  unsafe.Pointer
)

// extractAndDecompress writes an embedded (optionally gzipped) file to destPath.
func extractAndDecompress(srcPath, destPath string) error {
	f, err := libFS.Open(srcPath)
	if err != nil {
		return fmt.Errorf("open embedded %s: %w", srcPath, err)
	}
	defer func() { _ = f.Close() }()

	var r io.Reader = f
	if strings.HasSuffix(srcPath, ".gz") {
		gz, err := gzip.NewReader(f)
		if err != nil {
			return fmt.Errorf("gzip reader %s: %w", srcPath, err)
		}
		defer func() { _ = gz.Close() }()
		r = gz
	}

	out, err := os.OpenFile(destPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o755)
	if err != nil {
		return fmt.Errorf("create dest %s: %w", destPath, err)
	}
	defer func() { _ = out.Close() }()

	if _, err := io.Copy(out, r); err != nil {
		return fmt.Errorf("copy %s: %w", srcPath, err)
	}
	return nil
}

// libArtifact returns the embedded library path and on-disk name for the current
// platform. ORT_DYLIB / onnxruntime resolution is handled separately at runtime.
func libArtifact() (embedPath, diskName string, err error) {
	switch runtime.GOOS {
	case "darwin":
		switch runtime.GOARCH {
		case "arm64":
			return "lib/darwin-arm64/libgliner2_binding.dylib.gz", "libgliner2_binding.dylib", nil
		case "amd64":
			return "lib/darwin-amd64/libgliner2_binding.dylib.gz", "libgliner2_binding.dylib", nil
		}
	case "linux":
		switch runtime.GOARCH {
		case "amd64":
			return "lib/linux-amd64/libgliner2_binding.so.gz", "libgliner2_binding.so", nil
		case "arm64":
			return "lib/linux-arm64/libgliner2_binding.so.gz", "libgliner2_binding.so", nil
		}
	}
	return "", "", fmt.Errorf("gliner2: unsupported platform %s/%s", runtime.GOOS, runtime.GOARCH)
}

// onnxArtifact returns the embedded CPU onnxruntime path and on-disk name for the
// current platform. ok is false when no onnxruntime is bundled for this platform
// (then we rely on a system install / a user-set ORT_DYLIB_PATH).
func onnxArtifact() (embedPath, diskName string, ok bool) {
	switch runtime.GOOS {
	case "darwin":
		switch runtime.GOARCH {
		case "arm64":
			return "lib/onnxruntime/darwin-arm64/libonnxruntime.dylib.gz", "libonnxruntime.dylib", true
		case "amd64":
			return "lib/onnxruntime/darwin-amd64/libonnxruntime.dylib.gz", "libonnxruntime.dylib", true
		}
	case "linux":
		switch runtime.GOARCH {
		case "amd64":
			return "lib/onnxruntime/linux-amd64/libonnxruntime.so.gz", "libonnxruntime.so", true
		case "arm64":
			return "lib/onnxruntime/linux-arm64/libonnxruntime.so.gz", "libonnxruntime.so", true
		}
	}
	return "", "", false
}

// ensureONNXRuntime makes libonnxruntime resolvable for ort's load-dynamic feature.
// If the caller already set ORT_DYLIB_PATH (e.g. pointing at an onnxruntime-gpu
// build), it is respected and nothing is done. Otherwise the bundled CPU build is
// extracted into tmpDir and ORT_DYLIB_PATH is pointed at it. A missing bundle is
// not fatal: ort will then fall back to the system loader's default search.
func ensureONNXRuntime(tmpDir string) {
	if os.Getenv("ORT_DYLIB_PATH") != "" {
		return
	}
	embedPath, diskName, ok := onnxArtifact()
	if !ok {
		return
	}
	dest := filepath.Join(tmpDir, diskName)
	if err := extractAndDecompress(embedPath, dest); err != nil {
		// No bundled onnxruntime for this build; rely on the system loader.
		return
	}
	_ = os.Setenv("ORT_DYLIB_PATH", dest)
}

// Init extracts and dlopens the gliner2_binding cdylib and resolves its symbols.
// It is safe to call repeatedly; the work happens once. Most callers do not need
// to call it directly — New calls it. It returns an error (rather than panicking)
// when the platform is unsupported or the library/ONNX runtime is unavailable, so
// a binary built without the native artifact still links and runs.
func Init() error {
	initOnce.Do(func() {
		embedPath, diskName, err := libArtifact()
		if err != nil {
			initErr = err
			return
		}

		tmpDir, err := os.MkdirTemp("", "go-gliner2-lib")
		if err != nil {
			initErr = fmt.Errorf("gliner2: temp dir: %w", err)
			return
		}
		dest := filepath.Join(tmpDir, diskName)
		if err := extractAndDecompress(embedPath, dest); err != nil {
			initErr = fmt.Errorf("gliner2: extract native library (build it with the Makefile's gliner2 target): %w", err)
			return
		}

		// Make libonnxruntime resolvable (ort load-dynamic) before the binding is
		// dlopen'd / first used. Respects a user-set ORT_DYLIB_PATH for GPU builds.
		ensureONNXRuntime(tmpDir)

		cDest := C.CString(dest)
		defer C.free(unsafe.Pointer(cDest))
		dlHandle = C._g2_open_lib(cDest)
		if dlHandle == nil {
			initErr = fmt.Errorf("gliner2: dlopen failed: %s", C.GoString(C._g2_get_dlerror()))
			return
		}

		loadSym := func(name string) (unsafe.Pointer, error) {
			cName := C.CString(name)
			defer C.free(unsafe.Pointer(cName))
			sym := C._g2_get_sym(dlHandle, cName)
			if sym == nil {
				return nil, fmt.Errorf("gliner2: symbol not found: %s", name)
			}
			return sym, nil
		}

		for _, s := range []struct {
			name string
			dst  *unsafe.Pointer
		}{
			{"gliner2_new", &fnNew},
			{"gliner2_extract", &fnExtract},
			{"gliner2_free_engine", &fnFreeEngine},
			{"gliner2_free_string", &fnFreeString},
			{"gliner2_last_error", &fnLastError},
		} {
			sym, e := loadSym(s.name)
			if e != nil {
				initErr = e
				return
			}
			*s.dst = sym
		}
	})
	return initErr
}

// lastError returns the engine's thread-local last error message, or "".
func lastError() string {
	if fnLastError == nil {
		return ""
	}
	c := C._g2_call_last_error(fnLastError)
	if c == nil {
		return ""
	}
	return C.GoString(c)
}
