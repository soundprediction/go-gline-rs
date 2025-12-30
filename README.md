# go-gline-rs

Go bindings for [GLiNER](https://github.com/urchade/GLiNER), or specifically for the excellent Rust implementation [gliner-rs](https://github.com/fbilhaut/gline-rs). This library allows you to perform Named Entity Recognition (NER) and Relation Extraction using GLiNER models directly in your Go applications.

## Features

- **Entity Extraction**: Support for both Span and Token-based GLiNER models.
- **Relation Extraction**: extract relationships between entities with customizable schemas.
- **Embedded Library**: The Rust shared library is embedded and automatically extracted at runtime, simplifying deployment (no external `.so`/`.dylib` setup required for supported platforms).
- **Efficient**: Leverages Rust's `ndarray` and `onnxruntime` for speed.

## Installation

```bash
go get github.com/soundprediction/go-gline-rs
```

## Usage

### Entity Extraction (NER)

You can use either `NewSpanModel` or `NewTokenModel` depending on your model architecture.

```go
package main

import (
	"fmt"
	"log"

	"github.com/soundprediction/go-gline-rs/pkg/gline"
)

func main() {
	// Initialize the library (optional, happens automatically on first call)
	if err := gline.Init(); err != nil {
		log.Fatalf("Failed to initialize gline: %v", err)
	}

	// You can load a model directly from Hugging Face:
	modelID := "onnx-community/gliner_small-v2.1"
	model, err := gline.NewSpanModelFromHF(modelID)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	texts := []string{
		"Cristiano Ronaldo plays for Al-Nassr FC.",
		"Apple Inc. was founded by Steve Jobs.",
	}
	labels := []string{"person", "organization", "city", "country"}

	// Predict entities
	results, err := model.Predict(texts, labels)
	if err != nil {
		log.Fatalf("Prediction failed: %v", err)
	}

	for i, entities := range results {
		fmt.Printf("Text: %s\n", texts[i])
		for _, e := range entities {
			fmt.Printf("  - [%s] %s (Prob: %.4f)\n", e.Label, e.Text, e.Probability)
		}
	}
}
```

### Relation Extraction

Extract relationships between entities by defining a schema.

```go
package main

import (
	"fmt"
	"log"

	"github.com/soundprediction/go-gline-rs/pkg/gline"
)

func main() {
	modelPath := "path/to/glirel_model.onnx"
	tokenizerPath := "path/to/tokenizer.json"

	// Load relation model
	relModel, err := gline.NewRelationModel(modelPath, tokenizerPath)
	if err != nil {
		log.Fatalf("Failed to load relation model: %v", err)
	}
	defer relModel.Close()

	// Define Relation Schema
	// Example: "founded_by" connects "organization" -> "person"
	err = relModel.AddRelationSchema("founded_by", []string{"organization"}, []string{"person"})
	if err != nil {
		log.Fatalf("Failed to add schema: %v", err)
	}
    
    // Example: "played_for" connects "person" -> "organization"
    err = relModel.AddRelationSchema("played_for", []string{"person"}, []string{"organization"})
    if err != nil {
        log.Fatalf("Failed to add schema: %v", err)
    }

	texts := []string{
		"Steve Jobs founded Apple Inc. in 1976.",
	}
	// Note: Relation extraction requires candidate entity labels to be passed
	entityLabels := []string{"person", "organization"}

	// Predict relations
	results, err := relModel.Predict(texts, entityLabels)
	if err != nil {
		log.Fatalf("Relation prediction failed: %v", err)
	}

	for i, relations := range results {
		fmt.Printf("Text: %s\n", texts[i])
		for _, r := range relations {
			fmt.Printf("  - %s [%s] -> %s (Prob: %.4f)\n", 
                r.Source, r.Relation, r.Target, r.Probability)
		}
	}
}
```

## Building from Source

To rebuild the Rust bindings:

1.  Ensure you have `cargo` (Rust) installed.
2.  Run the compile script for your platform:

```bash
# macOS
./scripts/compile_rust_mac.sh

# Linux
./scripts/compile_rust_linux.sh
```

This will compile the Rust code and place the compressed shared library (`.gz`) into the `pkg/gline/lib` directory, where `go:embed` can pick it up.

## License

[MIT](LICENSE)
