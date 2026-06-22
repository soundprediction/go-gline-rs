package gliner2

/*
#include "gliner2.h"
*/
import "C"

import (
	"encoding/json"
	"fmt"
	"unsafe"
)

// ModelType selects how repo weights are interpreted by the engine.
type ModelType int

const (
	// ModelTypePyTorch loads native PyTorch/safetensors weights.
	ModelTypePyTorch ModelType = 0
	// ModelTypeHuggingFace loads the ONNX-fragmented export (the gliner2-multi
	// ...-onnx repos), which is the fast, Python-free path.
	ModelTypeHuggingFace ModelType = 1
)

// Entity is a span extracted for an entity task. Char offsets index into the
// input text; token offsets index into the model's sub-word tokenization.
type Entity struct {
	Text      string  `json:"text"`
	Label     string  `json:"label"`
	Score     float32 `json:"score"`
	StartTok  int     `json:"start_tok"`
	EndTok    int     `json:"end_tok"`
	StartChar int     `json:"start_char"`
	EndChar   int     `json:"end_char"`
}

// Relation is a typed head→tail relation extracted for a relation task.
type Relation struct {
	Head         Entity `json:"head"`
	Tail         Entity `json:"tail"`
	RelationType string `json:"relation_type"`
}

// Classification is a label predicted for a (named) classification task.
type Classification struct {
	TaskName string  `json:"task_name"`
	Label    string  `json:"label"`
	Score    float32 `json:"score"`
}

// Structure is a structured/JSON extraction result: a named structure with zero
// or more extracted object instances. Each instance maps field name -> value (a
// string for dtype "str", or a []string for dtype "list").
type Structure struct {
	Name      string           `json:"name"`
	Instances []map[string]any `json:"instances"`
}

// Result is the full multi-task output of one Extract call.
type Result struct {
	Entities        []Entity         `json:"entities"`
	Relations       []Relation       `json:"relations"`
	Classifications []Classification `json:"classifications"`
	Structures      []Structure      `json:"structures"`
}

// Field is one field of a Structure task. Dtype is "str" (single value) or "list"
// (array); empty defaults to "list". Choices is informational (the engine
// span-extracts the value rather than restricting it to the choice set).
type Field struct {
	Name    string   `json:"name"`
	Dtype   string   `json:"dtype,omitempty"`
	Choices []string `json:"choices,omitempty"`
}

// Task is one extraction task in a schema. Construct tasks with Entities,
// Relations, Classifications, or Structure; the JSON shape matches the engine's
// task DTO (a "type" discriminator plus task-specific fields). Fields is []any so
// it can carry plain strings (relations) or Field objects (structures).
type Task struct {
	Type   string   `json:"type"` // "entities" | "relations" | "classifications" | "structure"
	Name   string   `json:"name,omitempty"`
	Labels []string `json:"labels,omitempty"`
	Fields []any    `json:"fields,omitempty"`
}

// Entities builds an entity-extraction task over the given entity-type labels.
func Entities(labels ...string) Task {
	return Task{Type: "entities", Labels: labels}
}

// Relations builds a relation-extraction task named name over the given relation
// types (fields).
func Relations(name string, fields ...string) Task {
	f := make([]any, len(fields))
	for i, s := range fields {
		f[i] = s
	}
	return Task{Type: "relations", Name: name, Fields: f}
}

// Classifications builds a text-classification task named name over the given
// candidate labels.
func Classifications(name string, labels ...string) Task {
	return Task{Type: "classifications", Name: name, Labels: labels}
}

// Structures builds a structured/JSON extraction task named name over the given
// fields (see Field).
func Structures(name string, fields ...Field) Task {
	f := make([]any, len(fields))
	for i, x := range fields {
		f[i] = x
	}
	return Task{Type: "structure", Name: name, Fields: f}
}

// Engine is a loaded GLiNER2 model. It is safe to reuse across many Extract
// calls. It is not guaranteed safe for concurrent Extract calls on the same
// Engine; serialize calls or use one Engine per goroutine. Call Close to free it.
type Engine struct {
	ptr unsafe.Pointer
}

// New loads a GLiNER2 engine from a Hugging Face repo (downloading weights on
// first use). subfolder selects a variant within the repo (e.g. "fp32_v2",
// "fp16_v2"); pass "" for the repo root.
func New(repoID, subfolder string, mt ModelType) (*Engine, error) {
	if err := Init(); err != nil {
		return nil, err
	}
	if repoID == "" {
		return nil, fmt.Errorf("gliner2: repoID is required")
	}

	cRepo := C.CString(repoID)
	defer C.free(unsafe.Pointer(cRepo))

	var cSub *C.char
	if subfolder != "" {
		cSub = C.CString(subfolder)
		defer C.free(unsafe.Pointer(cSub))
	}

	ptr := C._g2_call_new(fnNew, cRepo, cSub, C.int(mt))
	if ptr == nil {
		return nil, fmt.Errorf("gliner2: load %q: %s", repoID, orUnknown(lastError()))
	}
	return &Engine{ptr: ptr}, nil
}

// NewFromHuggingFace is New with ModelTypeHuggingFace — the common ONNX path.
func NewFromHuggingFace(repoID, subfolder string) (*Engine, error) {
	return New(repoID, subfolder, ModelTypeHuggingFace)
}

// Extract runs all tasks over text in a single forward pass. threshold is the
// span/label confidence cutoff (e.g. 0.5); flatNER, when true, forbids
// overlapping entity spans (greedy non-overlap) and otherwise allows them.
func (e *Engine) Extract(text string, tasks []Task, threshold float32, flatNER bool) (*Result, error) {
	if e == nil || e.ptr == nil {
		return nil, fmt.Errorf("gliner2: engine is closed")
	}
	if len(tasks) == 0 {
		return nil, fmt.Errorf("gliner2: at least one task is required")
	}

	tasksJSON, err := json.Marshal(tasks)
	if err != nil {
		return nil, fmt.Errorf("gliner2: marshal tasks: %w", err)
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))
	cTasks := C.CString(string(tasksJSON))
	defer C.free(unsafe.Pointer(cTasks))

	flat := C.int(0)
	if flatNER {
		flat = 1
	}

	cRes := C._g2_call_extract(fnExtract, e.ptr, cText, cTasks, C.float(threshold), flat)
	if cRes == nil {
		return nil, fmt.Errorf("gliner2: extract: %s", orUnknown(lastError()))
	}
	defer C._g2_call_free_string(fnFreeString, cRes)

	var out Result
	if err := json.Unmarshal([]byte(C.GoString(cRes)), &out); err != nil {
		return nil, fmt.Errorf("gliner2: decode result: %w", err)
	}
	return &out, nil
}

// Close frees the engine. After Close the Engine must not be used. Safe to call
// more than once.
func (e *Engine) Close() {
	if e != nil && e.ptr != nil {
		C._g2_call_free_engine(fnFreeEngine, e.ptr)
		e.ptr = nil
	}
}

func orUnknown(s string) string {
	if s == "" {
		return "unknown error"
	}
	return s
}
