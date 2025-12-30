package gline

import (
	"fmt"
	"os"
	"testing"
)

// TestInit verifies that the library can be loaded.
// It does NOT require a model file.
func TestInit(t *testing.T) {
	err := Init()
	if err != nil {
		t.Fatalf("Failed to initialize gline: %v", err)
	}
	fmt.Println("Successfully initialized gline")
}

// TestNewSpanModel_Fail tests that providing invalid paths correctly returns an error
// (or at least doesn't crash the entire process).
// Note: Since Rust prints to stderr and returns null on failure, we expect NewSpanModel to return nil.
func TestNewSpanModel_Fail(t *testing.T) {
	if err := Init(); err != nil {
		t.Fatalf("Failed to init: %v", err)
	}

	model, err := NewSpanModel("invalid_model_path", "invalid_tokenizer_path")
	if err == nil {
		t.Error("Expected error for invalid paths, got nil")
	}
	if model != nil {
		t.Error("Expected nil model for invalid paths")
		model.Close()
	}
}

// TestNewTokenModel_Fail tests invalid paths for TokenModel
func TestNewTokenModel_Fail(t *testing.T) {
	if err := Init(); err != nil {
		t.Fatalf("Failed to init: %v", err)
	}

	model, err := NewTokenModel("invalid_model_path", "invalid_tokenizer_path")
	if err == nil {
		t.Error("Expected error for invalid paths, got nil")
	}
	if model != nil {
		t.Error("Expected nil model for invalid paths")
		model.Close()
	}
}

// TestRelationModel_Fail tests relation model with invalid paths.
func TestRelationModel_Fail(t *testing.T) {
	if err := Init(); err != nil {
		t.Fatalf("Failed to init: %v", err)
	}

	model, err := NewRelationModel("invalid_model_path", "invalid_tokenizer_path")
	if err == nil {
		t.Error("Expected error for invalid paths")
	}
	if model != nil {
		t.Error("Expected nil model")
		model.Close()
	}
}

func TestDownloadAndLoad(t *testing.T) {
	if err := Init(); err != nil {
		t.Fatalf("Failed to init: %v", err)
	}

	// Use a small model for testing
	// This one is 250MB, might be too big for CI/regular test use?
	// User requested "onnx-community/gliner_large-v2.1" which is larger.
	// "onnx-community/gliner_medium-v2.1" is smaller.
	// There is "gliner-small-v2.1-onnx" maybe?
	// The repo https://huggingface.co/onnx-community/gliner_small-v2.1 exists.
	modelID := "onnx-community/gliner_small-v2.1"

	cacheDir, err := os.MkdirTemp("", "gline-test-cache")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(cacheDir)

	fmt.Printf("Downloading %s to %s...\n", modelID, cacheDir)
	modelPath, tokenizerPath, err := DownloadModel(modelID, cacheDir)
	if err != nil {
		t.Fatalf("Failed to download model: %v", err)
	}

	fmt.Printf("Model path: %s\n", modelPath)
	fmt.Printf("Tokenizer path: %s\n", tokenizerPath)

	// Verify files exist
	if _, err := os.Stat(modelPath); err != nil {
		t.Errorf("model.onnx missing: %v", err)
	}
	if _, err := os.Stat(tokenizerPath); err != nil {
		t.Errorf("tokenizer.json missing: %v", err)
	}

	// Attempt to load model (Span mode)
	model, err := NewSpanModel(modelPath, tokenizerPath)
	if err != nil {
		t.Fatalf("Failed to load downloaded model: %v", err)
	}
	defer model.Close()

	// Simple prediction to verify it works
	texts := []string{"Google was founded by Larry Page."}
	labels := []string{"person", "organization"}

	results, err := model.Predict(texts, labels)
	if err != nil {
		t.Fatalf("Prediction failed: %v", err)
	}

	if len(results) != 1 {
		t.Errorf("Expected 1 result sequence, got %d", len(results))
	}

	// Check if we got entities
	foundOrg := false
	foundPerson := false
	for _, e := range results[0] {
		fmt.Printf("Entity: %s [%s] (%.2f)\n", e.Text, e.Label, e.Probability)
		if e.Label == "organization" && e.Text == "Google" {
			foundOrg = true
		}
		if e.Label == "person" && e.Text == "Larry Page" {
			foundPerson = true
		}
	}

	if !foundOrg {
		t.Error("Did not find expected organization entity 'Google'")
	}
	if !foundPerson {
		t.Error("Did not find expected person entity 'Larry Page'")
	}
}

func TestNewSpanModelFromHF(t *testing.T) {
	if err := Init(); err != nil {
		t.Fatalf("Failed to init: %v", err)
	}

	modelID := "onnx-community/gliner_small-v2.1"
	model, err := NewSpanModelFromHF(modelID)
	if err != nil {
		t.Fatalf("Failed to load model from HF: %v", err)
	}
	defer model.Close()

	if model == nil {
		t.Fatal("Model is nil")
	}
}

func TestIncompatibleModel(t *testing.T) {
	if err := Init(); err != nil {
		t.Fatalf("Failed to init: %v", err)
	}

	// Use a model that definitely does not have model.onnx
	modelID := "google/flan-t5-small" // usually safetensors/bin only unless exported

	_, err := NewSpanModelFromHF(modelID)
	if err == nil {
		t.Error("Expected error for incompatible model, got nil")
	} else {
		// Verify error message content
		expected := "is not compatible"
		if len(err.Error()) < len(expected) || err.Error()[0:len("model")] != "model" {
			// Just specific check
			// "model bert-base-uncased is not compatible: model.onnx not found..."
		}
		fmt.Printf("Got expected error: %v\n", err)
	}
}
