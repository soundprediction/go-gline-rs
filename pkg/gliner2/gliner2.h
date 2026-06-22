#ifndef GLINER2_H
#define GLINER2_H

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// Function types exported by the gliner2_binding Rust cdylib (see
// gliner2_binding/src/lib.rs). The engine handle is opaque (void*); extraction
// marshals through a JSON C string the caller must free with gliner2_free_string.
typedef const char *(*gliner2_last_error_t)(void);
typedef void *(*gliner2_new_t)(const char *, const char *, int);
typedef char *(*gliner2_extract_t)(void *, const char *, const char *, float,
                                   int);
typedef void (*gliner2_free_engine_t)(void *);
typedef void (*gliner2_free_string_t)(char *);

// dlopen helpers + typed call wrappers (implemented in the gliner2.go preamble).
static void *_g2_open_lib(const char *path);
static char *_g2_get_dlerror(void);
static void *_g2_get_sym(void *handle, const char *name);

void *_g2_call_new(void *f, const char *repo, const char *sub, int mt);
char *_g2_call_extract(void *f, void *eng, const char *text, const char *tasks,
                       float threshold, int flat_ner);
void _g2_call_free_engine(void *f, void *eng);
void _g2_call_free_string(void *f, char *s);
const char *_g2_call_last_error(void *f);

#endif
