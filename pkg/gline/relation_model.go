package gline

import (
	"errors"
	"unsafe"
)

/*
#include "gline.h"
*/
import "C"

type Relation struct {
	SequenceIndex int     `json:"sequence_index"`
	Source        string  `json:"source"`
	Target        string  `json:"target"`
	Relation      string  `json:"relation"`
	Probability   float32 `json:"probability"`
}

type RelationModel struct {
	ptr unsafe.Pointer
}

// NewRelationModel
func NewRelationModel(modelPath, tokenizerPath string) (*RelationModel, error) {
	if !initialized {
		return nil, errors.New("library not initialized")
	}

	cMod := C.CString(modelPath)
	cTok := C.CString(tokenizerPath)
	defer C.free(unsafe.Pointer(cMod))
	defer C.free(unsafe.Pointer(cTok))

	ptr := C._gl_call_new_relation_model(fnNewRelationModel, cMod, cTok)
	if ptr == nil {
		return nil, errors.New("failed to create relation model")
	}
	return &RelationModel{ptr: ptr}, nil
}

// NewRelationModelFromHF loads a Relation model directly from Hugging Face
func NewRelationModelFromHF(modelID string) (*RelationModel, error) {
	modelPath, tokenizerPath, err := DownloadModel(modelID, "")
	if err != nil {
		return nil, err
	}
	return NewRelationModel(modelPath, tokenizerPath)
}

func (r *RelationModel) Close() {
	if r.ptr != nil {
		C._gl_call_free_relation_model(fnFreeRelationModel, r.ptr)
		r.ptr = nil
	}
}

// AddRelationSchema adds a relation definition to the schema
func (r *RelationModel) AddRelationSchema(relation string, headTypes []string, tailTypes []string) error {
	if r.ptr == nil {
		return errors.New("model is closed")
	}
	if len(headTypes) == 0 || len(tailTypes) == 0 {
		return errors.New("head/tail types cannot be empty")
	}

	cRel := C.CString(relation)
	defer C.free(unsafe.Pointer(cRel))

	cHeads := make([]*C.char, len(headTypes))
	for i, s := range headTypes {
		cStr := C.CString(s)
		defer C.free(unsafe.Pointer(cStr))
		cHeads[i] = cStr
	}

	cTails := make([]*C.char, len(tailTypes))
	for i, s := range tailTypes {
		cStr := C.CString(s)
		defer C.free(unsafe.Pointer(cStr))
		cTails[i] = cStr
	}

	C._gl_call_add_relation_schema(fnAddRelationSchema, r.ptr, cRel,
		(**C.char)(unsafe.Pointer(&cHeads[0])), C.size_t(len(headTypes)),
		(**C.char)(unsafe.Pointer(&cTails[0])), C.size_t(len(tailTypes)))

	return nil
}

// Predict extracts relations. Note: It requires entity labels because it runs NER internally.
func (r *RelationModel) Predict(texts []string, entityLabels []string) ([][]Relation, error) {
	if r.ptr == nil {
		return nil, errors.New("model is closed")
	}

	cTexts := make([]*C.char, len(texts))
	for i, s := range texts {
		cStr := C.CString(s)
		defer C.free(unsafe.Pointer(cStr))
		cTexts[i] = cStr
	}

	cLabels := make([]*C.char, len(entityLabels))
	for i, s := range entityLabels {
		cStr := C.CString(s)
		defer C.free(unsafe.Pointer(cStr))
		cLabels[i] = cStr
	}

	res := C._gl_call_inference_relation(fnInferenceRelation, r.ptr,
		(**C.char)(unsafe.Pointer(&cTexts[0])), C.size_t(len(texts)),
		(**C.char)(unsafe.Pointer(&cLabels[0])), C.size_t(len(entityLabels)))

	if res == nil {
		return nil, errors.New("relation inference failed")
	}
	defer C._gl_call_free_relation_result(fnFreeRelationResult, res)

	// Parse result
	count := int(res.count)
	rels := unsafe.Slice(res.relations, count)

	out := make([][]Relation, len(texts))
	for i := range out {
		out[i] = []Relation{}
	}

	for i := 0; i < count; i++ {
		item := rels[i]
		seq := int(item.sequence_index)
		if seq >= len(out) {
			continue
		}

		rel := Relation{
			SequenceIndex: seq,
			Source:        C.GoString(item.source),
			Target:        C.GoString(item.target),
			Relation:      C.GoString(item.relation),
			Probability:   float32(item.prob),
		}
		out[seq] = append(out[seq], rel)
	}
	return out, nil
}
