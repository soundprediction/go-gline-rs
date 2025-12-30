#ifndef GLINE_H
#define GLINE_H

#include <stdlib.h>
#include <dlfcn.h>
#include <stdio.h>

// Structs
typedef struct {
    size_t sequence_index;
    size_t start;
    size_t end;
    char* class;
    char* text;
    float prob;
} FlatSpan;

typedef struct {
    FlatSpan* spans;
    size_t count;
} BatchResult;

typedef struct {
    size_t sequence_index;
    char* source;
    char* target;
    char* relation;
    float prob;
} FlatRelation;

typedef struct {
    FlatRelation* relations;
    size_t count;
} BatchRelationResult;

// Function types from Rust library
typedef void* (*new_span_model_t)(const char*, const char*);
typedef BatchResult* (*inference_span_t)(void*, const char**, size_t, const char**, size_t);
typedef void (*free_span_model_t)(void*);

typedef void* (*new_token_model_t)(const char*, const char*);
typedef BatchResult* (*inference_token_t)(void*, const char**, size_t, const char**, size_t);
typedef void (*free_token_model_t)(void*);

typedef void (*free_batch_result_t)(BatchResult*);

typedef void* (*new_relation_model_t)(const char*, const char*);
typedef void (*add_relation_schema_t)(void*, const char*, const char**, size_t, const char**, size_t);
typedef BatchRelationResult* (*inference_relation_t)(void*, const char**, size_t, const char**, size_t);
typedef void (*free_relation_model_t)(void*);
typedef void (*free_relation_result_t)(BatchRelationResult*);

// Function Prototypes for Wrappers (implemented in gline.go preamble or c file)
static void* open_lib(const char* path);
static char* get_dlerror();
static void* get_sym(void* handle, const char* name);

void* call_new_span_model(void* f, const char* m, const char* t);
BatchResult* call_inference_span(void* f, void* w, const char** i, size_t ic, const char** l, size_t lc);
void call_free_span_model(void* f, void* w);

void* call_new_token_model(void* f, const char* m, const char* t);
BatchResult* call_inference_token(void* f, void* w, const char** i, size_t ic, const char** l, size_t lc);
void call_free_token_model(void* f, void* w);

static void call_free_batch_result(void* f, BatchResult* r);

void* call_new_relation_model(void* f, const char* m, const char* t);
void call_add_relation_schema(void* f, void* w, const char* r, const char** ht, size_t hc, const char** tt, size_t tc);
BatchRelationResult* call_inference_relation(void* f, void* w, const char** i, size_t ic, const char** el, size_t elc);
void call_free_relation_model(void* f, void* w);
void call_free_relation_result(void* f, BatchRelationResult* r);

#endif
