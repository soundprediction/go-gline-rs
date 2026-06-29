package gliner2

import (
	"encoding/json"
	"testing"
)

// TestTaskJSON verifies tasks marshal to the DTO shape the Rust engine expects
// (a lowercase "type" discriminator plus task-specific fields).
func TestTaskJSON(t *testing.T) {
	cases := []struct {
		name string
		task Task
		want string
	}{
		{"entities", Entities("person", "org"), `{"type":"entities","labels":["person","org"]}`},
		{"relations", Relations("links", "works_at", "located_in"), `{"type":"relations","name":"links","fields":["works_at","located_in"]}`},
		{"classifications", Classifications("sentiment", "positive", "negative"), `{"type":"classifications","name":"sentiment","labels":["positive","negative"]}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			b, err := json.Marshal(c.task)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			if string(b) != c.want {
				t.Errorf("got  %s\nwant %s", b, c.want)
			}
		})
	}
}

// TestResultDecode verifies the engine's JSON result decodes into Result with the
// Rust serde field names (text/label/score/start_tok/.../relation_type/task_name).
func TestResultDecode(t *testing.T) {
	const raw = `{
      "entities":[{"text":"Mario Rossi","label":"person","score":0.97,"start_tok":1,"end_tok":3,"start_char":0,"end_char":11}],
      "relations":[{"head":{"text":"Mario Rossi","label":"person","score":0.9,"start_tok":1,"end_tok":3,"start_char":0,"end_char":11},
                    "tail":{"text":"Apple","label":"org","score":0.9,"start_tok":6,"end_tok":7,"start_char":21,"end_char":26},
                    "relation_type":"works_at"}],
      "classifications":[{"task_name":"sentiment","label":"positive","score":0.8}]
    }`
	var r Result
	if err := json.Unmarshal([]byte(raw), &r); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(r.Entities) != 1 || r.Entities[0].Text != "Mario Rossi" || r.Entities[0].Label != "person" || r.Entities[0].EndChar != 11 {
		t.Errorf("entity decode mismatch: %+v", r.Entities)
	}
	if len(r.Relations) != 1 || r.Relations[0].RelationType != "works_at" || r.Relations[0].Tail.Label != "org" {
		t.Errorf("relation decode mismatch: %+v", r.Relations)
	}
	if len(r.Classifications) != 1 || r.Classifications[0].TaskName != "sentiment" || r.Classifications[0].Label != "positive" {
		t.Errorf("classification decode mismatch: %+v", r.Classifications)
	}
}

func TestAvailableONNXProviders(t *testing.T) {
	providers, err := AvailableONNXProviders()
	if err != nil {
		t.Skipf("native onnxruntime not available: %v", err)
	}
	if !HasONNXProvider(providers, "CPUExecutionProvider") {
		t.Fatalf("providers = %v, want CPUExecutionProvider", providers)
	}
}

// TestExtractSmoke runs a real extraction when the native library is built and
// present; otherwise it skips (the pure-Go tests above cover the marshaling).
func TestExtractSmoke(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping native smoke test in -short mode (downloads model weights)")
	}
	if err := Init(); err != nil {
		t.Skipf("native gliner2_binding not available (build with `make gliner2`): %v", err)
	}
	eng, err := NewFromHuggingFace("SemplificaAI/gliner2-multi-v1-onnx", "fp32_v2")
	if err != nil {
		t.Skipf("engine load failed (needs model download + onnxruntime): %v", err)
	}
	defer eng.Close()

	res, err := eng.Extract(
		"Mario Rossi works at Apple in Cupertino.",
		[]Task{Entities("person", "organization", "location")},
		0.5, false,
	)
	if err != nil {
		t.Fatalf("extract: %v", err)
	}
	if len(res.Entities) == 0 {
		t.Fatalf("expected at least one entity")
	}
	t.Logf("extracted %d entities (first: %q/%s)", len(res.Entities), res.Entities[0].Text, res.Entities[0].Label)
}

// TestClassifyAndStructuresSmoke exercises the patched-engine features
// (classification on fp32_v2, and structured/JSON extraction) end-to-end.
func TestClassifyAndStructuresSmoke(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping native smoke test in -short mode (downloads model weights)")
	}
	if err := Init(); err != nil {
		t.Skipf("native gliner2_binding not available: %v", err)
	}
	eng, err := NewFromHuggingFace("SemplificaAI/gliner2-multi-v1-onnx", "fp32_v2")
	if err != nil {
		t.Skipf("engine load failed: %v", err)
	}
	defer eng.Close()

	// Classification (was previously broken on fp32_v2 due to an fp16/fp32 mismatch).
	cls, err := eng.Extract("I absolutely love this product!",
		[]Task{Classifications("sentiment", "positive", "negative", "neutral")}, 0.5, false)
	if err != nil {
		t.Fatalf("classify: %v", err)
	}
	if len(cls.Classifications) == 0 || cls.Classifications[0].Label != "positive" {
		t.Errorf("expected sentiment=positive, got %+v", cls.Classifications)
	}

	// Structured/JSON extraction.
	js, err := eng.Extract(
		"The MacBook Pro costs $1999 and features M3 chip, 16GB RAM, and 512GB storage.",
		[]Task{Structures("product",
			Field{Name: "name", Dtype: "str"},
			Field{Name: "price"},
			Field{Name: "features"},
		)}, 0.3, false)
	if err != nil {
		t.Fatalf("structures: %v", err)
	}
	if len(js.Structures) != 1 || js.Structures[0].Name != "product" || len(js.Structures[0].Instances) == 0 {
		t.Fatalf("expected one 'product' structure with >=1 instance, got %+v", js.Structures)
	}
	if name, _ := js.Structures[0].Instances[0]["name"].(string); name != "MacBook Pro" {
		t.Errorf("expected name=MacBook Pro, got %v", js.Structures[0].Instances[0]["name"])
	}
	t.Logf("structures: %+v", js.Structures[0].Instances)
}
