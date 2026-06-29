// Command gliner2-server is an HTTP microservice that is wire-compatible with the
// official GLiNER2 Python cloud API (the endpoint that gliner2's GLiNER2.from_api()
// / api_client.py talks to). Point a GLiNER2 client at it via GLINER2_API_BASE_URL
// and it works as a drop-in, self-hosted replacement backed by the local ONNX engine.
//
// Contract (from gliner2/api_client.py):
//
//	POST /gliner-2
//	  header: X-API-Key: <key>            (required only if -api-key / GLINER2_API_KEY is set)
//	  body:   {"task": "...", "text": <str|[]str>, "schema": <list|dict>,
//	           "threshold": 0.5, "include_confidence": false,
//	           "include_spans": false, "format_results": true}
//	  reply:  {"result": <task-shaped result>}
//
// Supported tasks: extract_entities, classify_text, extract_relations, schema.
// extract_json / structured extraction is NOT supported by the ONNX engine and
// returns HTTP 422 with a {"detail": ...} body (matching the client's error path).
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"

	"github.com/soundprediction/go-gline-rs/pkg/gliner2"
)

func main() {
	var (
		addr       = flag.String("addr", envOr("GLINER2_ADDR", ":8080"), "listen address")
		repo       = flag.String("repo", envOr("GLINER2_MODEL", "SemplificaAI/gliner2-multi-v1-onnx"), "Hugging Face model repo id")
		variant    = flag.String("variant", envOr("GLINER2_VARIANT", "fp32_v2"), "model variant subfolder (e.g. fp32_v2, fp16_v2); empty for repo root")
		modelType  = flag.String("model-type", envOr("GLINER2_MODEL_TYPE", "huggingface"), "model type: huggingface or pytorch")
		apiKey     = flag.String("api-key", firstEnv("GLINER2_API_KEY", "PIONEER_API_KEY"), "if set, require this key in the X-API-Key header")
		requireGPU = flag.Bool("require-gpu", envBool("GLINER2_REQUIRE_GPU"), "fail startup unless ONNXRuntime exposes CUDAExecutionProvider")
	)
	flag.Parse()

	mt := gliner2.ModelTypeHuggingFace
	if *modelType == "pytorch" {
		mt = gliner2.ModelTypePyTorch
	}

	providers, err := gliner2.AvailableONNXProviders()
	if err != nil {
		log.Fatalf("inspect ONNXRuntime providers: %v", err)
	}
	log.Printf("onnxruntime providers: %s", strings.Join(providers, ","))
	if *requireGPU && !gliner2.HasONNXProvider(providers, "CUDAExecutionProvider") {
		log.Fatalf("GLINER2_REQUIRE_GPU is set but CUDAExecutionProvider is unavailable; ORT_DYLIB_PATH=%q", os.Getenv("ORT_DYLIB_PATH"))
	}

	log.Printf("loading model %q (variant %q)…", *repo, *variant)
	eng, err := gliner2.New(*repo, *variant, mt)
	if err != nil {
		log.Fatalf("load model: %v", err)
	}
	defer eng.Close()
	log.Printf("model loaded")

	srv := &server{eng: eng, apiKey: *apiKey}
	mux := http.NewServeMux()
	mux.HandleFunc("/gliner-2", srv.handleExtract)
	mux.HandleFunc("/health", srv.handleHealth)

	log.Printf("listening on %s", *addr)
	if err := http.ListenAndServe(*addr, mux); err != nil {
		log.Fatalf("server: %v", err)
	}
}

type server struct {
	mu     sync.Mutex // Engine is not safe for concurrent Extract calls
	eng    *gliner2.Engine
	apiKey string
}

// apiRequest mirrors the payload built by gliner2/api_client.py._make_request.
// Pointers carry the Python-side defaults (threshold=0.5, format_results=true).
type apiRequest struct {
	Task              string          `json:"task"`
	Text              json.RawMessage `json:"text"`
	Schema            json.RawMessage `json:"schema"`
	Threshold         *float64        `json:"threshold"`
	IncludeConfidence bool            `json:"include_confidence"`
	IncludeSpans      bool            `json:"include_spans"`
	FormatResults     *bool           `json:"format_results"`
}

func (s *server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "model_loaded": true})
}

func (s *server) handleExtract(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeDetail(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if s.apiKey != "" && r.Header.Get("X-API-Key") != s.apiKey {
		writeDetail(w, http.StatusUnauthorized, "Invalid or expired API key")
		return
	}

	var req apiRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeDetail(w, http.StatusUnprocessableEntity, fmt.Sprintf("invalid request body: %v", err))
		return
	}

	threshold := float32(0.5)
	if req.Threshold != nil {
		threshold = float32(*req.Threshold)
	}

	// text is either a single string or a list of strings (batch).
	texts, batch, err := decodeText(req.Text)
	if err != nil {
		writeDetail(w, http.StatusUnprocessableEntity, err.Error())
		return
	}

	results := make([]any, 0, len(texts))
	for _, text := range texts {
		out, herr := s.runTask(text, &req, threshold)
		if herr != nil {
			writeDetail(w, herr.code, herr.msg)
			return
		}
		results = append(results, out)
	}

	// Single string in → single object out; list in → list out (matches the client,
	// which calls .get("result") and expects a dict or list accordingly).
	var result any = results[0]
	if batch {
		result = results
	}
	writeJSON(w, http.StatusOK, map[string]any{"result": result})
}

type httpError struct {
	code int
	msg  string
}

// runTask dispatches one (text, request) to the engine and formats the result to
// match the local GLiNER2 library's output shapes.
func (s *server) runTask(text string, req *apiRequest, threshold float32) (any, *httpError) {
	switch req.Task {
	case "extract_entities":
		var labels []string
		if err := json.Unmarshal(req.Schema, &labels); err != nil {
			return nil, &httpError{http.StatusUnprocessableEntity, "extract_entities: schema must be a list of entity labels"}
		}
		res, err := s.extract(text, []gliner2.Task{gliner2.Entities(labels...)}, threshold)
		if err != nil {
			return nil, &httpError{http.StatusInternalServerError, err.Error()}
		}
		return map[string]any{"entities": formatEntities(res, labels, req.IncludeConfidence, req.IncludeSpans)}, nil

	case "classify_text":
		// schema = {"categories": [labels]} (single-label, top class).
		var sc struct {
			Categories []string `json:"categories"`
		}
		if err := json.Unmarshal(req.Schema, &sc); err != nil || len(sc.Categories) == 0 {
			return nil, &httpError{http.StatusUnprocessableEntity, "classify_text: schema must be {\"categories\": [labels]}"}
		}
		res, err := s.extract(text, []gliner2.Task{gliner2.Classifications("categories", sc.Categories...)}, threshold)
		if err != nil {
			return nil, &httpError{http.StatusInternalServerError, err.Error()}
		}
		return map[string]any{"classification": topClassification(res, "categories", req.IncludeConfidence)}, nil

	case "extract_relations":
		// schema is built client-side as {"relations": [...], ...}; handle via schema path.
		return s.runSchema(text, req.Schema, threshold, req)

	case "schema":
		return s.runSchema(text, req.Schema, threshold, req)

	case "extract_json":
		// schema = {structure_name: [field_spec, ...]}, field_spec is a string
		// ("name::dtype::[choices]::desc") or an object {name,dtype,choices,...}.
		var structs map[string]json.RawMessage
		if err := json.Unmarshal(req.Schema, &structs); err != nil || len(structs) == 0 {
			return nil, &httpError{http.StatusUnprocessableEntity, "extract_json: schema must be {structure: [field specs]}"}
		}
		tasks, herr := buildStructureTasks(structs)
		if herr != nil {
			return nil, herr
		}
		res, err := s.extract(text, tasks, threshold)
		if err != nil {
			return nil, &httpError{http.StatusInternalServerError, err.Error()}
		}
		out := map[string]any{}
		for _, st := range res.Structures {
			out[st.Name] = st.Instances
		}
		return out, nil

	default:
		return nil, &httpError{http.StatusUnprocessableEntity, fmt.Sprintf("unknown task %q", req.Task)}
	}
}

// schemaDoc is the dict form of the "schema"/"extract_relations" payload.
type schemaDoc struct {
	Entities        json.RawMessage            `json:"entities"`
	Classifications map[string]json.RawMessage `json:"classifications"`
	Relations       json.RawMessage            `json:"relations"`
	Structures      map[string]json.RawMessage `json:"structures"`
}

func (s *server) runSchema(text string, raw json.RawMessage, threshold float32, req *apiRequest) (any, *httpError) {
	var doc schemaDoc
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, &httpError{http.StatusUnprocessableEntity, "schema must be an object"}
	}
	out := map[string]any{}

	// Structures.
	if len(doc.Structures) > 0 {
		tasks, herr := buildStructureTasks(doc.Structures)
		if herr != nil {
			return nil, herr
		}
		res, err := s.extract(text, tasks, threshold)
		if err != nil {
			return nil, &httpError{http.StatusInternalServerError, err.Error()}
		}
		for _, st := range res.Structures {
			out[st.Name] = st.Instances
		}
	}

	// Entities.
	entityLabels := decodeLabelList(doc.Entities)
	if len(entityLabels) > 0 {
		res, err := s.extract(text, []gliner2.Task{gliner2.Entities(entityLabels...)}, threshold)
		if err != nil {
			return nil, &httpError{http.StatusInternalServerError, err.Error()}
		}
		out["entities"] = formatEntities(res, entityLabels, req.IncludeConfidence, req.IncludeSpans)
	}

	// Classifications: one task per name; merge each as {taskName: value}.
	for task, cfgRaw := range doc.Classifications {
		labels, multi, clsThresh := decodeClassification(cfgRaw, threshold)
		if len(labels) == 0 {
			continue
		}
		res, err := s.extract(text, []gliner2.Task{gliner2.Classifications(task, labels...)}, clsThresh)
		if err != nil {
			return nil, &httpError{http.StatusInternalServerError, err.Error()}
		}
		if multi {
			out[task] = multiClassification(res, task, clsThresh, req.IncludeConfidence)
		} else {
			out[task] = topClassification(res, task, req.IncludeConfidence)
		}
	}

	// Relations: run per relation type so pairs are attributable to their type.
	relTypes := decodeLabelList(doc.Relations)
	if len(relTypes) > 0 {
		rel := map[string]any{}
		for _, rt := range relTypes {
			res, err := s.extract(text, []gliner2.Task{gliner2.Relations(rt, "head", "tail")}, threshold)
			if err != nil {
				return nil, &httpError{http.StatusInternalServerError, err.Error()}
			}
			pairs := make([][]string, 0, len(res.Relations))
			for _, r := range res.Relations {
				pairs = append(pairs, []string{r.Head.Text, r.Tail.Text})
			}
			rel[rt] = pairs // present even when empty, matching the library
		}
		out["relation_extraction"] = rel
	}

	if len(out) == 0 {
		return nil, &httpError{http.StatusUnprocessableEntity, "schema must contain at least one supported task (entities, classifications, relations)"}
	}
	return out, nil
}

func (s *server) extract(text string, tasks []gliner2.Task, threshold float32) (*gliner2.Result, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.eng.Extract(text, tasks, threshold, false)
}

// formatEntities groups entities by label (all requested labels present, possibly
// empty). Values are plain strings, or objects when confidence/spans are requested.
func formatEntities(res *gliner2.Result, labels []string, includeConf, includeSpans bool) map[string]any {
	buckets := map[string][]any{}
	for _, l := range labels {
		buckets[l] = []any{}
	}
	for _, e := range res.Entities {
		var v any = e.Text
		if includeConf || includeSpans {
			m := map[string]any{"text": e.Text}
			if includeConf {
				m["confidence"] = e.Score
			}
			if includeSpans {
				m["start"] = e.StartChar
				m["end"] = e.EndChar
			}
			v = m
		}
		buckets[e.Label] = append(buckets[e.Label], v)
	}
	out := map[string]any{}
	for k, v := range buckets {
		out[k] = v
	}
	return out
}

// topClassification returns the single highest-scoring label for a task, as a bare
// string or, with confidence, {"label":..,"confidence":..}.
func topClassification(res *gliner2.Result, task string, includeConf bool) any {
	var best *gliner2.Classification
	for i := range res.Classifications {
		c := &res.Classifications[i]
		if c.TaskName != task {
			continue
		}
		if best == nil || c.Score > best.Score {
			best = c
		}
	}
	if best == nil {
		return nil
	}
	if includeConf {
		return map[string]any{"label": best.Label, "confidence": best.Score}
	}
	return best.Label
}

// multiClassification returns all labels for a task above the threshold, sorted by
// score; bare strings or {"label","confidence"} objects with confidence.
func multiClassification(res *gliner2.Result, task string, threshold float32, includeConf bool) any {
	type lc struct {
		label string
		score float32
	}
	var picks []lc
	for _, c := range res.Classifications {
		if c.TaskName == task && c.Score >= threshold {
			picks = append(picks, lc{c.Label, c.Score})
		}
	}
	sort.SliceStable(picks, func(i, j int) bool { return picks[i].score > picks[j].score })
	if includeConf {
		out := make([]any, 0, len(picks))
		for _, p := range picks {
			out = append(out, map[string]any{"label": p.label, "confidence": p.score})
		}
		return out
	}
	out := make([]string, 0, len(picks))
	for _, p := range picks {
		out = append(out, p.label)
	}
	return out
}

// decodeText accepts a JSON string or array of strings.
func decodeText(raw json.RawMessage) (texts []string, batch bool, err error) {
	if len(raw) == 0 {
		return nil, false, fmt.Errorf("text is required")
	}
	var one string
	if err := json.Unmarshal(raw, &one); err == nil {
		return []string{one}, false, nil
	}
	var many []string
	if err := json.Unmarshal(raw, &many); err == nil {
		if len(many) == 0 {
			return nil, true, fmt.Errorf("text list is empty")
		}
		return many, true, nil
	}
	return nil, false, fmt.Errorf("text must be a string or list of strings")
}

// decodeLabelList accepts a JSON string, list of strings, or object whose keys are
// the labels (GLiNER2 allows label→description dicts); returns the label names.
func decodeLabelList(raw json.RawMessage) []string {
	if len(raw) == 0 {
		return nil
	}
	var one string
	if err := json.Unmarshal(raw, &one); err == nil {
		return []string{one}
	}
	var list []string
	if err := json.Unmarshal(raw, &list); err == nil {
		return list
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err == nil {
		keys := make([]string, 0, len(obj))
		for k := range obj {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		return keys
	}
	return nil
}

// decodeClassification parses a classification task config: either [labels] or
// {"labels":[...], "multi_label":bool, "cls_threshold":float}.
func decodeClassification(raw json.RawMessage, defaultThreshold float32) (labels []string, multi bool, threshold float32) {
	threshold = defaultThreshold
	var list []string
	if err := json.Unmarshal(raw, &list); err == nil {
		return list, false, threshold
	}
	var cfg struct {
		Labels       []string `json:"labels"`
		MultiLabel   bool     `json:"multi_label"`
		ClsThreshold *float64 `json:"cls_threshold"`
	}
	if err := json.Unmarshal(raw, &cfg); err == nil {
		if cfg.ClsThreshold != nil {
			threshold = float32(*cfg.ClsThreshold)
		}
		return cfg.Labels, cfg.MultiLabel, threshold
	}
	return nil, false, threshold
}

// buildStructureTasks converts a {structure: [field_spec,...]} map into Structures
// tasks. Each field_spec is a string ("name::dtype::[choices]::desc") or an object.
func buildStructureTasks(structs map[string]json.RawMessage) ([]gliner2.Task, *httpError) {
	// Deterministic order for stable output.
	names := make([]string, 0, len(structs))
	for n := range structs {
		names = append(names, n)
	}
	sort.Strings(names)

	tasks := make([]gliner2.Task, 0, len(names))
	for _, name := range names {
		var specs []json.RawMessage
		if err := json.Unmarshal(structs[name], &specs); err != nil {
			return nil, &httpError{http.StatusUnprocessableEntity, fmt.Sprintf("structure %q: fields must be a list", name)}
		}
		fields := make([]gliner2.Field, 0, len(specs))
		for _, sp := range specs {
			fields = append(fields, parseFieldSpec(sp))
		}
		tasks = append(tasks, gliner2.Structures(name, fields...))
	}
	return tasks, nil
}

// parseFieldSpec accepts a field spec as a JSON string or object.
func parseFieldSpec(raw json.RawMessage) gliner2.Field {
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return parseFieldSpecString(s)
	}
	var o struct {
		Name    string   `json:"name"`
		Dtype   string   `json:"dtype"`
		Choices []string `json:"choices"`
	}
	if err := json.Unmarshal(raw, &o); err == nil {
		if o.Dtype == "" {
			o.Dtype = "list"
		}
		return gliner2.Field{Name: o.Name, Dtype: o.Dtype, Choices: o.Choices}
	}
	return gliner2.Field{}
}

// parseFieldSpecString parses "name::dtype::[c1|c2]::description" (parts after name
// optional and order-independent), mirroring the Python _parse_field_spec. Default
// dtype is "list"; a choices part with no explicit dtype defaults dtype to "str".
func parseFieldSpecString(spec string) gliner2.Field {
	parts := strings.Split(spec, "::")
	f := gliner2.Field{Name: parts[0]}
	dtypeExplicit := false
	for _, p := range parts[1:] {
		switch {
		case p == "str" || p == "list":
			f.Dtype = p
			dtypeExplicit = true
		case strings.HasPrefix(p, "[") && strings.HasSuffix(p, "]"):
			for _, c := range strings.Split(p[1:len(p)-1], "|") {
				f.Choices = append(f.Choices, strings.TrimSpace(c))
			}
			if !dtypeExplicit {
				f.Dtype = "str"
			}
		default:
			// description — not used by the engine
		}
	}
	if f.Dtype == "" {
		f.Dtype = "list"
	}
	return f
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func writeDetail(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]any{"detail": msg})
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envBool(key string) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(key))) {
	case "1", "true", "yes", "y", "on":
		return true
	default:
		return false
	}
}

func firstEnv(keys ...string) string {
	for _, k := range keys {
		if v := os.Getenv(k); v != "" {
			return v
		}
	}
	return ""
}
