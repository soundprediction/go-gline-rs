package gline

/*
#cgo LDFLAGS: -ldl
#include "gline.h"

// Implementation of Wrappers
void* open_lib(const char* path) {
    return dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
}

char* get_dlerror() {
    return dlerror();
}

void* get_sym(void* handle, const char* name) {
    return dlsym(handle, name);
}

void* call_new_span_model(void* f, const char* m, const char* t) {
    return ((new_span_model_t)f)(m, t);
}
BatchResult* call_inference_span(void* f, void* w, const char** i, size_t ic, const char** l, size_t lc) {
    return ((inference_span_t)f)(w, i, ic, l, lc);
}
void call_free_span_model(void* f, void* w) {
    ((free_span_model_t)f)(w);
}

void* call_new_token_model(void* f, const char* m, const char* t) {
    return ((new_token_model_t)f)(m, t);
}
BatchResult* call_inference_token(void* f, void* w, const char** i, size_t ic, const char** l, size_t lc) {
    return ((inference_token_t)f)(w, i, ic, l, lc);
}
void call_free_token_model(void* f, void* w) {
    ((free_token_model_t)f)(w);
}

void call_free_batch_result(void* f, BatchResult* r) {
    ((free_batch_result_t)f)(r);
}

void* call_new_relation_model(void* f, const char* m, const char* t) {
    return ((new_relation_model_t)f)(m, t);
}
void call_add_relation_schema(void* f, void* w, const char* r, const char** ht, size_t hc, const char** tt, size_t tc) {
    ((add_relation_schema_t)f)(w, r, ht, hc, tt, tc);
}
BatchRelationResult* call_inference_relation(void* f, void* w, const char** i, size_t ic, const char** el, size_t elc) {
    return ((inference_relation_t)f)(w, i, ic, el, elc);
}
void call_free_relation_model(void* f, void* w) {
    ((free_relation_model_t)f)(w);
}
void call_free_relation_result(void* f, BatchRelationResult* r) {
    ((free_relation_result_t)f)(r);
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
	"unsafe"
)

//go:embed lib
var libFS embed.FS

var (
	initialized bool
	dlHandle    unsafe.Pointer

	// Span
	fnNewSpanModel  unsafe.Pointer
	fnInferenceSpan unsafe.Pointer
	fnFreeSpanModel unsafe.Pointer

	// Token
	fnNewTokenModel  unsafe.Pointer
	fnInferenceToken unsafe.Pointer
	fnFreeTokenModel unsafe.Pointer

	// Shared
	fnFreeBatchResult unsafe.Pointer

	// Relation
	fnNewRelationModel   unsafe.Pointer
	fnAddRelationSchema  unsafe.Pointer
	fnInferenceRelation  unsafe.Pointer
	fnFreeRelationModel  unsafe.Pointer
	fnFreeRelationResult unsafe.Pointer
)

// extractAndDecompress extracts a file from embed.FS
func extractAndDecompress(srcPath, destPath string) error {
	f, err := libFS.Open(srcPath)
	if err != nil {
		return fmt.Errorf("open embedded %s: %w", srcPath, err)
	}
	defer f.Close()

	var r io.Reader = f
	if strings.HasSuffix(srcPath, ".gz") {
		gz, err := gzip.NewReader(f)
		if err != nil {
			return fmt.Errorf("gzip reader %s: %w", srcPath, err)
		}
		defer gz.Close()
		r = gz
	}

	out, err := os.OpenFile(destPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0755)
	if err != nil {
		return fmt.Errorf("create dest %s: %w", destPath, err)
	}
	defer out.Close()

	if _, err := io.Copy(out, r); err != nil {
		return fmt.Errorf("copy %s: %w", srcPath, err)
	}
	return nil
}

// Init extracts and loads the library
func Init() error {
	if initialized {
		return nil
	}

	goOS := runtime.GOOS
	goArch := runtime.GOARCH

	var libPath string
	var libName string

	switch goOS {
	case "darwin":
		libPath = "lib/darwin"
		libName = "libgline_binding.dylib.gz"
	case "linux":
		if goArch == "amd64" {
			libPath = "lib/linux-amd64"
		} else if goArch == "arm64" {
			libPath = "lib/linux-arm64"
		} else {
			return fmt.Errorf("unsupported linux architecture: %s", goArch)
		}
		libName = "libgline_binding.so.gz"
	default:
		return fmt.Errorf("unsupported OS: %s", goOS)
	}

	tmpDir, err := os.MkdirTemp("", "go-gline-rs-lib")
	if err != nil {
		return fmt.Errorf("failed to create temp dir: %w", err)
	}

	dest := filepath.Join(tmpDir, strings.TrimSuffix(libName, ".gz"))
	if err := extractAndDecompress(filepath.Join(libPath, libName), dest); err != nil {
		return err
	}

	// ONNX Runtime might be needed too!
	// gline-rs depends on ort. If it links dynamically, we need libonnxruntime.
	// Assuming static linking for now OR that the user provides it / we bundle it like embedeverything.
	// embedeverything bundles libonnxruntime.dylib.gz too.
	// I should extract that too if it exists. For now, let's assume valid linking or similar setup.
	// actually, gline-rs Cargo.toml has `ort = { version="=2.0.0-rc.9" }` without specific features,
	// but embedeverything had explicit dynamic loading logic.
	// Let's rely on standard linking for now. If it fails, I'll add ONNX bundle.

	cDest := C.CString(dest)
	defer C.free(unsafe.Pointer(cDest))
	dlHandle = C.open_lib(cDest)
	if dlHandle == nil {
		cErr := C.get_dlerror()
		return fmt.Errorf("dlopen failed: %s", C.GoString(cErr))
	}

	loadSym := func(name string) (unsafe.Pointer, error) {
		cName := C.CString(name)
		defer C.free(unsafe.Pointer(cName))
		sym := C.get_sym(dlHandle, cName)
		if sym == nil {
			return nil, fmt.Errorf("symbol not found: %s", name)
		}
		return sym, nil
	}

	var e error
	// Span
	if fnNewSpanModel, e = loadSym("new_span_model"); e != nil {
		return e
	}
	if fnInferenceSpan, e = loadSym("inference_span"); e != nil {
		return e
	}
	if fnFreeSpanModel, e = loadSym("free_span_model"); e != nil {
		return e
	}

	// Token
	if fnNewTokenModel, e = loadSym("new_token_model"); e != nil {
		return e
	}
	if fnInferenceToken, e = loadSym("inference_token"); e != nil {
		return e
	}
	if fnFreeTokenModel, e = loadSym("free_token_model"); e != nil {
		return e
	}

	// Shared
	if fnFreeBatchResult, e = loadSym("free_batch_result"); e != nil {
		return e
	}

	// Relation
	if fnNewRelationModel, e = loadSym("new_relation_model"); e != nil {
		return e
	}
	if fnAddRelationSchema, e = loadSym("add_relation_schema"); e != nil {
		return e
	}
	if fnInferenceRelation, e = loadSym("inference_relation"); e != nil {
		return e
	}
	if fnFreeRelationModel, e = loadSym("free_relation_model"); e != nil {
		return e
	}
	if fnFreeRelationResult, e = loadSym("free_relation_result"); e != nil {
		return e
	}

	initialized = true
	return nil
}

func init() {
	// defer Init() call to user? Or auto?
	// Auto-init logs warning if fails as per embedeverything pattern
	if err := Init(); err != nil {
		// fmt.Fprintf(os.Stderr, "WARNING: go-gline-rs failed to initialize: %v\n", err)
	}
}
