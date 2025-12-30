package gline

import (
	"fmt"
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
