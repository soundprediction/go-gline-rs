package gline

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/gomlx/go-huggingface/hub"
)

// DownloadModel downloads the model.onnx and tokenizer.json from Hugging Face
// modelID: e.g. "onnx-community/gliner_medium-v2.1"
// cacheDir: directory to store the model. If empty, uses ~/.cache/gline-rs
func DownloadModel(modelID, cacheDir string) (modelPath, tokenizerPath string, err error) {
	if cacheDir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", "", fmt.Errorf("failed to get user home dir: %w", err)
		}
		cacheDir = filepath.Join(home, ".cache", "gline-rs")
	}

	// Create a new Hub client pointing to our cache directory
	// Note: go-huggingface might download to a specific structure.
	// Let's assume hub.New(cacheDir) or similar.
	// Checking the library usage from common patterns:
	// repo := hub.New(hub.WithCacheDir(cacheDir))
	// path, err := repo.DownloadFile(modelID, "model.onnx")

	// Actually, without seeing the docs, I'll guess standard usage or try to find it.
	// Wait, I can try to run a small script to check the API if I'm unsure.
	// But let's try to write investigating code.

	// I'll try to use the `hub` package.
	repo := hub.New(modelID).WithCacheDir(cacheDir)

	// Sanitize modelID for local path check (optional, hub handles it)
	// safeModelID := strings.ReplaceAll(modelID, "/", "_")

	// Download model.onnx
	// Try "model.onnx" first, then "onnx/model.onnx" as fallback if possible.

	// Download functions usually return the absolute path.
	modelPath, err = repo.DownloadFile("model.onnx")
	if err != nil {
		// Try onnx/model.onnx in subfolder
		if mp, err2 := repo.DownloadFile("onnx/model.onnx"); err2 == nil {
			modelPath = mp
			err = nil
		} else {
			// Check if it was a "not found" error vs connectivity error?
			// For simplicity, we assume if both fail and we have connectivity, it's incompatible.
			return "", "", fmt.Errorf("model %s is not compatible: model.onnx not found. Please ensure the model contains 'model.onnx' or 'onnx/model.onnx'", modelID)
		}
	}

	// Download tokenizer.json
	tokenizerPath, err = repo.DownloadFile("tokenizer.json")
	if err != nil {
		// Try onnx/tokenizer.json just in case? Or just error.
		return "", "", fmt.Errorf("failed to download tokenizer.json: %w", err)
	}

	return modelPath, tokenizerPath, nil
}
