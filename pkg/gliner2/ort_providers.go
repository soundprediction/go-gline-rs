package gliner2

/*
#cgo LDFLAGS: -ldl
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct OrtStatus OrtStatus;
typedef OrtStatus* (*OrtGetAvailableProvidersFn)(char***, int*);
typedef void (*OrtReleaseAvailableProvidersFn)(char**, int);
typedef const char* (*OrtGetErrorMessageFn)(const OrtStatus*);
typedef void (*OrtReleaseStatusFn)(OrtStatus*);

static char* _g2_ort_strdup(const char* s) {
	if (s == NULL) {
		s = "unknown error";
	}
	size_t n = strlen(s) + 1;
	char* out = (char*)malloc(n);
	if (out != NULL) {
		memcpy(out, s, n);
	}
	return out;
}

static char* _g2_ort_missing_symbol(const char* name) {
	size_t n = strlen(name) + 64;
	char* out = (char*)malloc(n);
	if (out != NULL) {
		snprintf(out, n, "missing ONNXRuntime symbol: %s", name);
	}
	return out;
}

static char* _g2_ort_available_providers(const char* path, char** err_out) {
	*err_out = NULL;

	void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
	if (handle == NULL) {
		*err_out = _g2_ort_strdup(dlerror());
		return NULL;
	}

	OrtGetAvailableProvidersFn get_providers =
		(OrtGetAvailableProvidersFn)dlsym(handle, "OrtGetAvailableProviders");
	if (get_providers == NULL) {
		*err_out = _g2_ort_missing_symbol("OrtGetAvailableProviders");
		dlclose(handle);
		return NULL;
	}
	OrtReleaseAvailableProvidersFn release_providers =
		(OrtReleaseAvailableProvidersFn)dlsym(handle, "OrtReleaseAvailableProviders");
	if (release_providers == NULL) {
		*err_out = _g2_ort_missing_symbol("OrtReleaseAvailableProviders");
		dlclose(handle);
		return NULL;
	}
	OrtGetErrorMessageFn get_error =
		(OrtGetErrorMessageFn)dlsym(handle, "OrtGetErrorMessage");
	OrtReleaseStatusFn release_status =
		(OrtReleaseStatusFn)dlsym(handle, "OrtReleaseStatus");

	char** providers = NULL;
	int provider_count = 0;
	OrtStatus* status = get_providers(&providers, &provider_count);
	if (status != NULL) {
		const char* msg = get_error == NULL ? "OrtGetAvailableProviders failed" : get_error(status);
		*err_out = _g2_ort_strdup(msg);
		if (release_status != NULL) {
			release_status(status);
		}
		dlclose(handle);
		return NULL;
	}

	size_t total = 1;
	for (int i = 0; i < provider_count; i++) {
		total += strlen(providers[i]) + 1;
	}
	char* out = (char*)malloc(total);
	if (out == NULL) {
		release_providers(providers, provider_count);
		dlclose(handle);
		*err_out = _g2_ort_strdup("malloc failed");
		return NULL;
	}
	out[0] = '\0';
	for (int i = 0; i < provider_count; i++) {
		if (i > 0) {
			strcat(out, ",");
		}
		strcat(out, providers[i]);
	}

	release_providers(providers, provider_count);
	dlclose(handle);
	return out;
}
*/
import "C"

import (
	"fmt"
	"os"
	"strings"
	"unsafe"
)

// AvailableONNXProviders returns the execution providers compiled into the
// libonnxruntime selected by ORT_DYLIB_PATH. If ORT_DYLIB_PATH is unset, Init
// extracts the embedded CPU ONNXRuntime first and this returns its providers.
func AvailableONNXProviders() ([]string, error) {
	if err := Init(); err != nil {
		return nil, err
	}

	path := os.Getenv("ORT_DYLIB_PATH")
	if path == "" {
		path = "libonnxruntime.so"
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var cErr *C.char
	cProviders := C._g2_ort_available_providers(cPath, &cErr)
	if cErr != nil {
		defer C.free(unsafe.Pointer(cErr))
		return nil, fmt.Errorf("onnxruntime providers for %s: %s", path, C.GoString(cErr))
	}
	if cProviders == nil {
		return nil, fmt.Errorf("onnxruntime providers for %s: unknown error", path)
	}
	defer C.free(unsafe.Pointer(cProviders))

	raw := strings.TrimSpace(C.GoString(cProviders))
	if raw == "" {
		return nil, nil
	}
	parts := strings.Split(raw, ",")
	providers := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			providers = append(providers, p)
		}
	}
	return providers, nil
}

// HasONNXProvider reports whether providers contains name.
func HasONNXProvider(providers []string, name string) bool {
	for _, provider := range providers {
		if provider == name {
			return true
		}
	}
	return false
}
