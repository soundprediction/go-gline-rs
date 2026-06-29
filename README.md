# go-gline-rs

Go bindings for **GLiNER2** — multi-task, schema-driven information extraction
(named entities, relations, and text classification) — running on **ONNX Runtime**
via the [gliner2-rs](https://github.com/SemplificaAI/gliner2-rs) (`gliner2_inference`)
engine. Inference runs on **CPU or GPU** automatically (ort execution-provider
fallback chain), with no Python dependency.

> The original GLiNER v1 bindings (package `gline`, wrapping
> [gline-rs](https://github.com/fbilhaut/gline-rs)) are **deprecated** but still
> present for existing callers. New code should use package `gliner2`.

## Features

- **Multi-task extraction** — entities, relations, classifications, and structured/JSON extraction in a single forward pass.
- **CPU or GPU** — ONNX Runtime picks the best available execution provider (CUDA/ROCm/CoreML/DirectML/…) and falls back to CPU.
- **Embedded native library** — the Rust `gliner2_binding` cdylib **and** a CPU `libonnxruntime` are gzip-embedded and extracted at runtime; no manual `.so`/`.dylib` setup for supported platforms.
- **Model auto-download** — weights are pulled from Hugging Face on first use (inside the Rust layer via `hf-hub`).
- **Drop-in HTTP microservice** — wire-compatible with the official GLiNER2 cloud API (`GLiNER2.from_api()`).

## Installation

```bash
go get github.com/soundprediction/go-gline-rs
```

cgo is required (the package `dlopen`s the embedded native library).

## Library usage

```go
package main

import (
	"encoding/json"
	"fmt"
	"log"

	"github.com/soundprediction/go-gline-rs/pkg/gliner2"
)

func main() {
	// Load a GLiNER2 ONNX model from Hugging Face. The variant subfolder selects
	// the export: "fp32_v2" (CPU-optimized) or "fp16_v2" (GPU). Weights download
	// on first use.
	eng, err := gliner2.NewFromHuggingFace("SemplificaAI/gliner2-multi-v1-onnx", "fp32_v2")
	if err != nil {
		log.Fatal(err)
	}
	defer eng.Close()

	res, err := eng.Extract(
		"Mario Rossi works at Apple in Cupertino.",
		[]gliner2.Task{
			gliner2.Entities("person", "organization", "location"),
			gliner2.Relations("works_at", "head", "tail"),
		},
		0.5,   // threshold
		false, // flatNER (forbid overlapping spans)
	)
	if err != nil {
		log.Fatal(err)
	}

	out, _ := json.MarshalIndent(res, "", "  ")
	fmt.Println(string(out))
}
```

`Extract` returns a `*Result` with `Entities`, `Relations`, `Classifications`, and
`Structures`. Build tasks with `gliner2.Entities(labels...)`,
`gliner2.Relations(name, fields...)` (use `"head"`/`"tail"` fields),
`gliner2.Classifications(task, labels...)`, and `gliner2.Structures(name, fields...)`
for structured/JSON extraction, e.g.:

```go
res, _ := eng.Extract(
	"The MacBook Pro costs $1999 and features M3 chip, 16GB RAM, and 512GB storage.",
	[]gliner2.Task{gliner2.Structures("product",
		gliner2.Field{Name: "name", Dtype: "str"},
		gliner2.Field{Name: "price"},    // dtype defaults to "list"
		gliner2.Field{Name: "features"},
	)},
	0.3, false,
)
// res.Structures[0]: {Name: "product", Instances: [{"name":"MacBook Pro",
//   "price":["$1999"], "features":["M3 chip","16GB RAM","512GB storage"]}]}
```

### CPU vs GPU

A CPU `libonnxruntime` (matching the engine's ONNX Runtime 1.20.0) is bundled and
wired up automatically — `Init` sets `ORT_DYLIB_PATH` to the extracted copy. To use
a **GPU** build, install an `onnxruntime-gpu` shared library and set `ORT_DYLIB_PATH`
to it before first use; it will be respected and the engine will use the GPU
execution provider, falling back to CPU if unavailable.

Pick the model variant to match: `fp32_v2` for CPU, `fp16_v2` for GPU. fp16 on CPU
is slow/unsupported (no native fp16 compute), so there is no default — choose per
deployment.

For GPU deployments that must not silently fall back to CPU, run the HTTP server
with `--require-gpu` or `GLINER2_REQUIRE_GPU=1`. Startup inspects the selected
`libonnxruntime` and exits unless `CUDAExecutionProvider` is available:

```bash
export ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so
export LD_LIBRARY_PATH=/opt/onnxruntime/lib:${LD_LIBRARY_PATH}
GLINER2_REQUIRE_GPU=1 \
  go run ./cmd/gliner2-server \
  --addr :8080 \
  --repo SemplificaAI/gliner2-multi-v1-onnx \
  --variant fp16_v2
```

## HTTP microservice (drop-in for `GLiNER2.from_api()`)

`cmd/gliner2-server` implements the official GLiNER2 cloud API endpoint
(`POST /gliner-2`, the one `gliner2`'s `api_client.py` talks to), so it works as a
self-hosted, Python-free replacement:

```bash
go run ./cmd/gliner2-server --addr :8080 --repo SemplificaAI/gliner2-multi-v1-onnx --variant fp32_v2
# optional: --api-key <key>  (then clients must send X-API-Key)
```

Point any GLiNER2 client at it:

```python
import os
os.environ["GLINER2_API_BASE_URL"] = "http://localhost:8080"
os.environ["PIONEER_API_KEY"] = "anything"  # or the --api-key you set
from gliner2 import GLiNER2
extractor = GLiNER2.from_api()
extractor.extract_entities("Tim Cook leads Apple.", ["person", "company"])
```

Or call it directly:

```bash
curl -X POST localhost:8080/gliner-2 -H 'Content-Type: application/json' -d '{
  "task": "extract_entities",
  "text": "Mario Rossi works at Apple in Cupertino.",
  "schema": ["person", "organization", "location"],
  "threshold": 0.5
}'
# {"result":{"entities":{"person":["Mario Rossi"],"organization":["Apple"],"location":["Cupertino"]}}}
```

Supported `task` values: `extract_entities`, `extract_relations`, `schema`
(combined entities + relations + classifications), and `classify_text`. Requests,
the `{ "result": ... }` envelope, `X-API-Key` auth, and the per-task result shapes
mirror the Python client.

## MCP server

`cmd/mcp-server` exposes `extract_entities` and `extract_relations` tools over MCP (stdio):

```bash
go run ./cmd/mcp-server --repo SemplificaAI/gliner2-multi-v1-onnx --variant fp32_v2
```

## Building the native library from source

Prebuilt, gzip-compressed artifacts live under `pkg/gliner2/lib/` and are embedded
into the Go binary. To rebuild for your host platform (requires `cargo` and `curl`):

```bash
make gliner2   # builds gliner2_binding + bundles a matching CPU onnxruntime
```

This compiles `gliner2_binding/` and writes
`pkg/gliner2/lib/<platform>/libgliner2_binding.{so,dylib}.gz` plus
`pkg/gliner2/lib/onnxruntime/<platform>/libonnxruntime.{so,dylib}.gz`.

### Patched engine

The underlying `gliner2_inference` engine is consumed from upstream
[SemplificaAI/gliner2-rs](https://github.com/SemplificaAI/gliner2-rs) at tag
`v0.5.1` with **local patches** (a classifier input-dtype fix and structured/JSON
extraction support). The upstream source is **not** committed; instead `patches/`
holds the diff and `scripts/setup_gliner2_inference.sh` fetches the pinned tag and
applies it into `third_party/gliner2_inference` (gitignored). `make gliner2` runs
this setup automatically when the source is missing.

## Limitations

- **Choice fields** in structured extraction (`field::[a|b|c]`) are span-extracted
  from the text rather than restricted to the choice set. Upstream scores choice
  fields against the schema-prefix token columns, which the V2 ONNX engine does not
  compute; values that appear verbatim in the text are still found, but the result is
  not constrained to the listed choices.
- Validation against the Python reference (`gliner2`): entities, relations,
  classification, and structured/JSON extraction (including multi-instance counting)
  match. Note our default model (`gliner2-multi-v1-onnx`) and the Python default
  (`gliner2-base-v1`) are different weights, so exact spans can differ.

Both the earlier classification dtype failure on `fp32_v2` and the lack of
structured/JSON extraction have been fixed via the patched engine (see above).

## License

[MIT](LICENSE)
