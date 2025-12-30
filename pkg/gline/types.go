package gline

import (
	"errors"
	"unsafe"
)

/*
#include "gline.h"
*/
import "C"

// Entity represents an extracted entity
type Entity struct {
	Index       int     `json:"index"` // Sequence index
	Start       int     `json:"start"`
	End         int     `json:"end"`
	Label       string  `json:"label"`
	Text        string  `json:"text"`
	Probability float32 `json:"probability"`
}

type Model struct {
	ptr     unsafe.Pointer
	isToken bool
}

// NewSpanModel loads a model in Span Mode
func NewSpanModel(modelPath, tokenizerPath string) (*Model, error) {
	if !initialized {
		return nil, errors.New("library not initialized")
	}

	cMod := C.CString(modelPath)
	cTok := C.CString(tokenizerPath)
	defer C.free(unsafe.Pointer(cMod))
	defer C.free(unsafe.Pointer(cTok))

	ptr := C.call_new_span_model(fnNewSpanModel, cMod, cTok)
	if ptr == nil {
		return nil, errors.New("failed to create span model")
	}
	return &Model{ptr: ptr, isToken: false}, nil
}

// NewTokenModel loads a model in Token Mode
func NewTokenModel(modelPath, tokenizerPath string) (*Model, error) {
	if !initialized {
		return nil, errors.New("library not initialized")
	}

	cMod := C.CString(modelPath)
	cTok := C.CString(tokenizerPath)
	defer C.free(unsafe.Pointer(cMod))
	defer C.free(unsafe.Pointer(cTok))

	ptr := C.call_new_token_model(fnNewTokenModel, cMod, cTok)
	if ptr == nil {
		return nil, errors.New("failed to create token model")
	}
	return &Model{ptr: ptr, isToken: true}, nil
}

// NewSpanModelFromHF loads a Span model directly from Hugging Face
func NewSpanModelFromHF(modelID string) (*Model, error) {
	modelPath, tokenizerPath, err := DownloadModel(modelID, "")
	if err != nil {
		return nil, err
	}
	return NewSpanModel(modelPath, tokenizerPath)
}

// NewTokenModelFromHF loads a Token model directly from Hugging Face
func NewTokenModelFromHF(modelID string) (*Model, error) {
	modelPath, tokenizerPath, err := DownloadModel(modelID, "")
	if err != nil {
		return nil, err
	}
	return NewTokenModel(modelPath, tokenizerPath)
}

func (m *Model) Close() {
	if m.ptr != nil {
		if m.isToken {
			C.call_free_token_model(fnFreeTokenModel, m.ptr)
		} else {
			C.call_free_span_model(fnFreeSpanModel, m.ptr)
		}
		m.ptr = nil
	}
}

func (m *Model) Predict(texts []string, labels []string) ([][]Entity, error) {
	if m.ptr == nil {
		return nil, errors.New("model is closed")
	}

	// Prepare Inputs
	cTexts := make([]*C.char, len(texts))
	for i, s := range texts {
		cStr := C.CString(s)
		defer C.free(unsafe.Pointer(cStr))
		cTexts[i] = cStr
	}

	cLabels := make([]*C.char, len(labels))
	for i, s := range labels {
		cStr := C.CString(s)
		defer C.free(unsafe.Pointer(cStr))
		cLabels[i] = cStr
	}

	var res *C.BatchResult
	if m.isToken {
		res = C.call_inference_token(fnInferenceToken, m.ptr,
			(**C.char)(unsafe.Pointer(&cTexts[0])), C.size_t(len(texts)),
			(**C.char)(unsafe.Pointer(&cLabels[0])), C.size_t(len(labels)))
	} else {
		res = C.call_inference_span(fnInferenceSpan, m.ptr,
			(**C.char)(unsafe.Pointer(&cTexts[0])), C.size_t(len(texts)),
			(**C.char)(unsafe.Pointer(&cLabels[0])), C.size_t(len(labels)))
	}

	if res == nil {
		return nil, errors.New("inference failed")
	}
	defer C.call_free_batch_result(fnFreeBatchResult, res)

	// Parse result
	count := int(res.count)
	spans := unsafe.Slice(res.spans, count)

	// Initialize output buckets
	out := make([][]Entity, len(texts))
	for i := range out {
		out[i] = []Entity{}
	}

	for i := 0; i < count; i++ {
		s := spans[i]
		seq := int(s.sequence_index)
		if seq >= len(out) {
			continue
		} // Should not happen

		ent := Entity{
			Index:       seq,
			Start:       int(s.start),
			End:         int(s.end),
			Label:       C.GoString(s.class),
			Text:        C.GoString(s.text),
			Probability: float32(s.prob),
		}
		out[seq] = append(out[seq], ent)
	}

	return out, nil
}
