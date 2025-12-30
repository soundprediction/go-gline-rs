// Fix unused import
use libc::{c_char, c_float, size_t};
use std::ffi::{CStr, CString};
use gliner::model::{GLiNER, input::text::TextInput, params::Parameters};
use gliner::model::pipeline::span::SpanMode;
use gliner::model::pipeline::token::TokenMode;
use gliner::model::pipeline::relation::RelationPipeline;
use gliner::model::input::relation::schema::RelationSchema;
use orp::params::RuntimeParameters;
use orp::pipeline::Pipeline;
use composable::Composable; // Fix missing trait

// ==================================================================================
// Shared Structs
// ==================================================================================

#[repr(C)]
pub struct SpanResult {
    pub start: size_t,
    pub end: size_t,
    pub class: *mut c_char,
    pub text: *mut c_char,
    pub prob: c_float,
}

#[repr(C)]
pub struct BatchSpans {
    pub spans: *mut SpanResult,
    pub count: size_t,
    // Provide index of which input text this usage belongs to?
    // For now we assume batch size 1 or handled by caller structure.
    // Actually, gline-rs inference returns Batch<Vec<Span>>.
    // Let's flatten or make a struct that handles multiple sequences.
    // BUT simplify: The caller passes 1 text (or multiple), we return results for ALL.
    // Let's make a structure that holds a flattened list, but with `sequence_index`.
    // Gline-rs span has `sequence()`.
}

#[repr(C)]
pub struct FlatSpan {
    pub sequence_index: size_t,
    pub start: size_t,
    pub end: size_t,
    pub class: *mut c_char,
    pub text: *mut c_char,
    pub prob: c_float,
}

#[repr(C)]
pub struct BatchResult {
    pub spans: *mut FlatSpan,
    pub count: size_t,
}

// Relation Results
#[repr(C)]
pub struct FlatRelation {
    pub sequence_index: size_t,
    pub source: *mut c_char,
    pub target: *mut c_char,
    pub relation: *mut c_char,
    pub prob: c_float,
    // Add offsets if needed
}

#[repr(C)]
pub struct BatchRelationResult {
    pub relations: *mut FlatRelation,
    pub count: size_t,
}

// ==================================================================================
// Wrappers
// ==================================================================================

pub struct SpanModelWrapper {
    pub inner: GLiNER<SpanMode>,
}

pub struct TokenModelWrapper {
    pub inner: GLiNER<TokenMode>,
}

pub struct RelationModelWrapper {
    // RelationPipeline needs to be constructed with a schema.
    // The model (GLiNER struct) owns the pipeline. 
    // BUT GLiNER<P> expects P to implement Pipeline.
    // RelationPipeline has lifetime parameters and references schema.
    // This makes it hard to wrap in a struct without self-referential issues.
    // 
    // However, looking at example:
    // let pipeline = composed![... RelationPipeline ...];
    // pipeline.apply(input)
    //
    // GLiNER struct just wraps model + pipeline + params.
    // We can't easily put RelationPipeline into GLiNER<RelationPipeline> because RelationPipeline isn't 'static usually?
    // Let's check RelationPipeline definition. It has 'a lifetime for schema.
    // So we can't own the schema inside the wrapper easily unless we use unsafe or box leaking.
    // 
    // ALTERNATIVE: Accessing the model directly without GLiNER wrapper for Relation?
    // The example uses `composed!` pipeline.
    //
    // Let's stick to Span/Token first as they are standard GLiNER<P>.
    // For Relation, we might need a custom wrapper that holds Schema + Model + Pipeline constructed manually.
    
    pub model: orp::model::Model,
    pub tokenizer_path: String,
    pub schema: RelationSchema, // Owner of schema data
}


// ==================================================================================
// Span Mode
// ==================================================================================

#[no_mangle]
pub extern "C" fn new_span_model(model_path: *const c_char, tokenizer_path: *const c_char) -> *mut SpanModelWrapper {
    let m_path = unsafe { CStr::from_ptr(model_path).to_string_lossy() };
    let t_path = unsafe { CStr::from_ptr(tokenizer_path).to_string_lossy() };

    // Cow<str> implements AsRef<Path> via str? No. 
    // AsRef<Path> is implemented for str, String, Cow<'a, str>.
    // But error said `Cow<'_, str>: AsRef<...>` is not satisfied.
    // Wait, error said `AsRef<Path>` is NOT implemented for `Cow<'_, str>`.
    // It IS implemented for `str`. So we should just pass `m_path.as_ref()`.
    
    match GLiNER::<SpanMode>::new(
        Parameters::default(),
        RuntimeParameters::default(),
        t_path.as_ref(), // Pass as &str
        m_path.as_ref(), // Pass as &str
    ) {
        Ok(model) => Box::into_raw(Box::new(SpanModelWrapper { inner: model })),
        Err(e) => {
            eprintln!("Error loading span model: {:?}", e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn inference_span(
    wrapper: *mut SpanModelWrapper, 
    inputs: *const *const c_char, 
    input_count: size_t,
    labels: *const *const c_char,
    label_count: size_t
) -> *mut BatchResult {
    if wrapper.is_null() { return std::ptr::null_mut(); }
    let model = &(*wrapper).inner;

    // ... (skipping input conversion) ...
    let input_slice = std::slice::from_raw_parts(inputs, input_count as usize);
    let mut texts = Vec::with_capacity(input_count as usize);
    for &p in input_slice {
        texts.push(CStr::from_ptr(p).to_string_lossy());
    }
    let label_slice = std::slice::from_raw_parts(labels, label_count as usize);
    let mut label_strs = Vec::with_capacity(label_count as usize);
    for &p in label_slice {
        label_strs.push(CStr::from_ptr(p).to_string_lossy());
    }
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_ref()).collect();
    let label_refs: Vec<&str> = label_strs.iter().map(|s| s.as_ref()).collect();

    let text_input = match TextInput::from_str(&text_refs, &label_refs) {
        Ok(i) => i,
        Err(e) => {
             eprintln!("Input error: {:?}", e);
             return std::ptr::null_mut();
        }
    };

    // Inference
    match model.inference(text_input) {
        Ok(output) => {
            let mut flat_spans = Vec::new();
            for seq_spans in output.spans {
                for span in seq_spans {
                    let (start, end) = span.offsets(); // Use offsets() instead of start()/end()
                    flat_spans.push(FlatSpan {
                        sequence_index: span.sequence() as size_t,
                        start: start as size_t,
                        end: end as size_t,
                        class: CString::new(span.class()).unwrap().into_raw(),
                        text: CString::new(span.text()).unwrap().into_raw(),
                        prob: span.probability(),
                    });
                }
            }
            // ... (rest is same) ...
            let count = flat_spans.len();
            let ptr = flat_spans.as_mut_ptr();
            std::mem::forget(flat_spans);

            Box::into_raw(Box::new(BatchResult {
                spans: ptr,
                count: count as size_t,
            }))
        },
        Err(e) => {
            eprintln!("Inference error: {:?}", e);
            std::ptr::null_mut()
        }
    }
}

// ... Token Mode ...

#[no_mangle]
pub extern "C" fn new_token_model(model_path: *const c_char, tokenizer_path: *const c_char) -> *mut TokenModelWrapper {
    let m_path = unsafe { CStr::from_ptr(model_path).to_string_lossy() };
    let t_path = unsafe { CStr::from_ptr(tokenizer_path).to_string_lossy() };

    match GLiNER::<TokenMode>::new(
        Parameters::default(),
        RuntimeParameters::default(),
        t_path.as_ref(),
        m_path.as_ref(),
    ) {
        Ok(model) => Box::into_raw(Box::new(TokenModelWrapper { inner: model })),
        Err(e) => {
            eprintln!("Error loading token model: {:?}", e);
            std::ptr::null_mut()
        }
    }
}

// ... inference_token (apply offsets fix) ...
#[no_mangle]
pub unsafe extern "C" fn inference_token(
    wrapper: *mut TokenModelWrapper, 
    inputs: *const *const c_char, 
    input_count: size_t,
    labels: *const *const c_char,
    label_count: size_t
) -> *mut BatchResult {
      if wrapper.is_null() { return std::ptr::null_mut(); }
    let model = &(*wrapper).inner;

    // ... conversion ...
    let input_slice = std::slice::from_raw_parts(inputs, input_count as usize);
    let mut texts = Vec::with_capacity(input_count as usize);
    for &p in input_slice {
        texts.push(CStr::from_ptr(p).to_string_lossy());
    }
    let label_slice = std::slice::from_raw_parts(labels, label_count as usize);
    let mut label_strs = Vec::with_capacity(label_count as usize);
    for &p in label_slice {
        label_strs.push(CStr::from_ptr(p).to_string_lossy());
    }
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_ref()).collect();
    let label_refs: Vec<&str> = label_strs.iter().map(|s| s.as_ref()).collect();

    let text_input = match TextInput::from_str(&text_refs, &label_refs) {
        Ok(i) => i,
        Err(e) => {
             eprintln!("Input error: {:?}", e);
             return std::ptr::null_mut();
        }
    };

    match model.inference(text_input) {
        Ok(output) => {
            let mut flat_spans = Vec::new();
            for seq_spans in output.spans {
                for span in seq_spans {
                     let (start, end) = span.offsets();
                    flat_spans.push(FlatSpan {
                        sequence_index: span.sequence() as size_t,
                        start: start as size_t,
                        end: end as size_t,
                        class: CString::new(span.class()).unwrap().into_raw(),
                        text: CString::new(span.text()).unwrap().into_raw(),
                        prob: span.probability(),
                    });
                }
            }
             let count = flat_spans.len();
            let ptr = flat_spans.as_mut_ptr();
            std::mem::forget(flat_spans);

            Box::into_raw(Box::new(BatchResult {
                spans: ptr,
                count: count as size_t,
            }))
        },
        Err(e) => {
            eprintln!("Inference error: {:?}", e);
            std::ptr::null_mut()
        }
    }
}

// ==================================================================================
// Free Functions
// ==================================================================================

#[no_mangle]
pub unsafe extern "C" fn free_span_model(wrapper: *mut SpanModelWrapper) {
    if !wrapper.is_null() {
        let _ = Box::from_raw(wrapper);
    }
}

#[no_mangle]
pub unsafe extern "C" fn free_token_model(wrapper: *mut TokenModelWrapper) {
    if !wrapper.is_null() {
        let _ = Box::from_raw(wrapper);
    }
}


#[no_mangle]
pub unsafe extern "C" fn free_batch_result(result: *mut BatchResult) {
    if !result.is_null() {
        let batch = Box::from_raw(result);
        let spans = Vec::from_raw_parts(batch.spans, batch.count as usize, batch.count as usize);
        for s in spans {
             if !s.class.is_null() { let _ = CString::from_raw(s.class); }
             if !s.text.is_null() { let _ = CString::from_raw(s.text); }
        }
    }
}

// ==================================================================================
// Relation - OMITTED FOR BRIEFNESS FOR NOW, ADDING IF REQUESTED
// Since RelationPipeline requires schema lifetime games, it's safer to implement
// later or if user explicitly pushes for it. The Plan said "support all", but
// implementing it robustly in C FFI is tricky due to schemas.
// User said "support all three". I must implement it.
//
// Strategy: 
// The RelationModelWrapper will hold the schema.
// When we infer, we construct the pipeline transiently or validly.
// But wait, constructing `RelationPipeline::default` takes `&'a RelationSchema`.
// So schema must live longer than pipeline. 
// If we build pipeline on every inference, we are fine.
// ==================================================================================

#[no_mangle]
pub extern "C" fn new_relation_model(model_path: *const c_char, tokenizer_path: *const c_char) -> *mut RelationModelWrapper {
    let m_path = unsafe { CStr::from_ptr(model_path).to_string_lossy() };
    let t_path = unsafe { CStr::from_ptr(tokenizer_path).to_string_lossy() };
    
    // Load model
    // Model::new gives Result<Model>. 
    // It expects `P: AsRef<Path>`. `m_path` is Cow<str>. `&m_path` is `&Cow<str>`. 
    // We need to pass `m_path.as_ref()` or `&*m_path` if it's a string, actually `AsRef<Path>` is implemented for `str`.
    let model = match orp::model::Model::new(m_path.as_ref(), RuntimeParameters::default()) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error loading model: {:?}", e);
            return std::ptr::null_mut();
        }
    };

    let schema = RelationSchema::new();

    Box::into_raw(Box::new(RelationModelWrapper {
        model,
        tokenizer_path: t_path.to_string(),
        schema,
    }))
}

#[no_mangle]
pub unsafe extern "C" fn add_relation_schema(
    wrapper: *mut RelationModelWrapper,
    relation: *const c_char,
    head_types: *const *const c_char,
    head_count: size_t,
    tail_types: *const *const c_char,
    tail_count: size_t,
) {
    if wrapper.is_null() { return; }
    let wrapper_ref = &mut *wrapper;
    
    let rel_name = CStr::from_ptr(relation).to_string_lossy();
    
    let head_slice = std::slice::from_raw_parts(head_types, head_count as usize);
    let mut heads = Vec::new();
    for &p in head_slice { heads.push(CStr::from_ptr(p).to_string_lossy()); }
    let head_refs: Vec<&str> = heads.iter().map(|s| s.as_ref()).collect();

    let tail_slice = std::slice::from_raw_parts(tail_types, tail_count as usize);
    let mut tails = Vec::new();
    for &p in tail_slice { tails.push(CStr::from_ptr(p).to_string_lossy()); }
    let tail_refs: Vec<&str> = tails.iter().map(|s| s.as_ref()).collect();

    wrapper_ref.schema.push_with_allowed_labels(&rel_name, &head_refs, &tail_refs);
}

#[no_mangle]
pub unsafe extern "C" fn inference_relation(
    wrapper: *mut RelationModelWrapper,
    inputs: *const *const c_char,
    input_count: size_t,
    entity_labels: *const *const c_char,
    entity_label_count: size_t 
) -> *mut BatchRelationResult {
    if wrapper.is_null() { return std::ptr::null_mut(); }
    let wrapper_ref = &mut *wrapper;

    // Convert inputs
    let input_slice = std::slice::from_raw_parts(inputs, input_count as usize);
    let mut texts = Vec::with_capacity(input_count as usize);
    for &p in input_slice {
        texts.push(CStr::from_ptr(p).to_string_lossy());
    }

    // Convert entity labels
    let label_slice = std::slice::from_raw_parts(entity_labels, entity_label_count as usize);
    let mut label_strs = Vec::with_capacity(entity_label_count as usize);
    for &p in label_slice {
        label_strs.push(CStr::from_ptr(p).to_string_lossy());
    }

    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_ref()).collect();
    let label_refs: Vec<&str> = label_strs.iter().map(|s| s.as_ref()).collect();

    let text_input = match TextInput::from_str(&text_refs, &label_refs) {
        Ok(i) => i,
        Err(e) => {
             eprintln!("Input error: {:?}", e);
             return std::ptr::null_mut();
        }
    };

    // Pipeline composition for Relation Extraction
    // Logic from example:
    // let pipeline = composed![
    //     TokenPipeline::new(TOKENIZER_PATH)?.to_composable(&model, &params),
    //     RelationPipeline::default(TOKENIZER_PATH, &relation_schema)?.to_composable(&model, &params),
    // ];
    // BUT Wait. `RelationPipeline` expects `SpanOutput` as input.
    // `TokenPipeline` produces `SpanOutput`.
    // So we need to chain them!
    // The previous error was `expected `SpanOutput`, found `TextInput``. 
    // Because I was passing `text_input` (TextInput) directly to `rel_pipeline` (which expects SpanOutput).
    // I need the FULL pipeline chain: TokenPipeline -> RelationPipeline.
    
    // 1. Create Token Pipeline
    use gliner::model::pipeline::token::TokenPipeline;
    let token_pipeline_res = TokenPipeline::new(&wrapper_ref.tokenizer_path);
    let token_pipeline = match token_pipeline_res {
         Ok(p) => p,
         Err(e) => {
             eprintln!("Token pipeline error: {:?}", e);
             return std::ptr::null_mut();
         }
    };
    
    // 2. Create Relation Pipeline
    let rel_pipeline_res = RelationPipeline::default(&wrapper_ref.tokenizer_path, &wrapper_ref.schema);
    let rel_pipeline = match rel_pipeline_res {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Relation pipeline error: {:?}", e);
            return std::ptr::null_mut();
        }
    };

    let params = Parameters::default();
    
    // 3. Compose them
    // composed! macro might be hard to access if not exported or syntax weird in FFI file.
    // But `composed!` just chains them.
    // We can use `.then(...)` equivalent? 
    // `composable::Composed::new(prev, next)`
    // OR just manually chain calls? 
    // `runnable_token.apply(input) -> span_output`
    // `runnable_rel.apply(span_output) -> relation_output`
    
    let runnable_token = token_pipeline.to_composable(&wrapper_ref.model, &params);
    let runnable_rel = rel_pipeline.to_composable(&wrapper_ref.model, &params);
    
    // Run Step 1
    let span_output = match runnable_token.apply(text_input) {
        Ok(out) => out,
        Err(e) => {
             eprintln!("Token inference error: {:?}", e);
             return std::ptr::null_mut();
        }
    };
    
    // Run Step 2
    match runnable_rel.apply(span_output) {
        Ok(output) => {
            // output is RelationOutput { relations: Vec<Vec<Relation>>, ... }
            let mut flat_rels = Vec::new();
            for (seq_idx, seq_rels) in output.relations.iter().enumerate() {
                for rel in seq_rels {
                    flat_rels.push(FlatRelation {
                        sequence_index: seq_idx as size_t, // Or rel.sequence(), gline seems to carry it
                        source: CString::new(rel.subject()).unwrap().into_raw(),
                        target: CString::new(rel.object()).unwrap().into_raw(),
                        relation: CString::new(rel.class()).unwrap().into_raw(),
                        prob: rel.probability(),
                    });
                }
            }
             
            let count = flat_rels.len();
            let ptr = flat_rels.as_mut_ptr();
            std::mem::forget(flat_rels);
            
            Box::into_raw(Box::new(BatchRelationResult {
                relations: ptr,
                count: count as size_t,
            }))
        },
        Err(e) => {
            eprintln!("Inference error: {:?}", e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn free_relation_model(wrapper: *mut RelationModelWrapper) {
    if !wrapper.is_null() {
        let _ = Box::from_raw(wrapper);
    }
}

#[no_mangle]
pub unsafe extern "C" fn free_relation_result(result: *mut BatchRelationResult) {
    if !result.is_null() {
        let batch = Box::from_raw(result);
        let rels = Vec::from_raw_parts(batch.relations, batch.count as usize, batch.count as usize);
        for r in rels {
            if !r.source.is_null() { let _ = CString::from_raw(r.source); }
            if !r.target.is_null() { let _ = CString::from_raw(r.target); }
            if !r.relation.is_null() { let _ = CString::from_raw(r.relation); }
        }
    }
}
