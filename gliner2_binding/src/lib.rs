//! C FFI wrapper around the `gliner2_inference` engine (GLiNER2 over ONNX Runtime).
//!
//! The surface is deliberately tiny: build an engine from a HuggingFace repo, run a
//! multi-task `extract`, and hand results back to the caller as a JSON string. All the
//! GLiNER2 output types derive `Serialize`, so JSON is the marshaling boundary — the Go
//! side parses it. CPU/GPU selection is handled inside `gliner2_inference` (ort EP
//! fallback chain); this layer is backend-agnostic.

use libc::{c_char, c_float, c_int};
use std::cell::RefCell;
use std::ffi::{CStr, CString};

use gliner2_inference::processor::{Dtype, StructField};
use gliner2_inference::{
    ExtractedClassification, ExtractedEntity, ExtractedRelation, ExtractedStructure, Gliner2Engine,
    InferenceParams, ModelType, SchemaTask,
};
use serde::{Deserialize, Serialize};

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = RefCell::new(None);
}

fn set_last_error(msg: impl Into<String>) {
    let s = msg.into();
    eprintln!("[gliner2_binding] {s}");
    if let Ok(c) = CString::new(s) {
        LAST_ERROR.with(|e| *e.borrow_mut() = Some(c));
    }
}

/// Returns a pointer to the last error message recorded on this thread (or null).
/// The returned pointer is owned by the library and remains valid until the next
/// fallible call on the same thread; callers must NOT free it.
#[no_mangle]
pub extern "C" fn gliner2_last_error() -> *const c_char {
    LAST_ERROR.with(|e| match &*e.borrow() {
        Some(c) => c.as_ptr(),
        None => std::ptr::null(),
    })
}

/// Task description deserialized from the JSON sent by the caller.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum TaskDto {
    Entities { labels: Vec<String> },
    Relations { name: String, fields: Vec<String> },
    Classifications { name: String, labels: Vec<String> },
    Structure { name: String, fields: Vec<FieldDto> },
}

/// One structured-extraction field as sent by the caller.
#[derive(Deserialize)]
struct FieldDto {
    name: String,
    /// "str" (single value) or "list" (array). Defaults to "list" when absent.
    #[serde(default)]
    dtype: Option<String>,
    #[serde(default)]
    choices: Option<Vec<String>>,
}

impl From<TaskDto> for SchemaTask {
    fn from(t: TaskDto) -> Self {
        match t {
            TaskDto::Entities { labels } => SchemaTask::Entities(labels),
            TaskDto::Relations { name, fields } => SchemaTask::Relations(name, fields),
            TaskDto::Classifications { name, labels } => SchemaTask::Classifications(name, labels),
            TaskDto::Structure { name, fields } => SchemaTask::Structure(
                name,
                fields
                    .into_iter()
                    .map(|f| StructField {
                        dtype: match f.dtype.as_deref() {
                            Some("str") => Dtype::Str,
                            _ => Dtype::List,
                        },
                        name: f.name,
                        choices: f.choices,
                    })
                    .collect(),
            ),
        }
    }
}

/// The full multi-task result, serialized back to the caller as JSON.
#[derive(Serialize)]
struct ExtractResult {
    entities: Vec<ExtractedEntity>,
    relations: Vec<ExtractedRelation>,
    classifications: Vec<ExtractedClassification>,
    structures: Vec<ExtractedStructure>,
}

fn model_type_from_int(v: c_int) -> ModelType {
    match v {
        0 => ModelType::PyTorch,
        _ => ModelType::HuggingFace,
    }
}

/// Build an engine from a HuggingFace repo.
///
/// `subfolder` may be null (no subfolder). `model_type`: 0 = PyTorch, 1 = HuggingFace.
/// Returns null on error (see `gliner2_last_error`). The download happens here via hf-hub.
///
/// # Safety
/// `repo_id` must be a valid NUL-terminated C string; `subfolder` must be null or the same.
#[no_mangle]
pub unsafe extern "C" fn gliner2_new(
    repo_id: *const c_char,
    subfolder: *const c_char,
    model_type: c_int,
) -> *mut Gliner2Engine {
    if repo_id.is_null() {
        set_last_error("gliner2_new: repo_id is null");
        return std::ptr::null_mut();
    }
    let repo = match CStr::from_ptr(repo_id).to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("gliner2_new: invalid repo_id utf8: {e}"));
            return std::ptr::null_mut();
        }
    };
    let sub: Option<&str> = if subfolder.is_null() {
        None
    } else {
        match CStr::from_ptr(subfolder).to_str() {
            Ok(s) if s.is_empty() => None,
            Ok(s) => Some(s),
            Err(e) => {
                set_last_error(format!("gliner2_new: invalid subfolder utf8: {e}"));
                return std::ptr::null_mut();
            }
        }
    };

    match Gliner2Engine::from_pretrained(repo, sub, model_type_from_int(model_type)) {
        Ok(engine) => Box::into_raw(Box::new(engine)),
        Err(e) => {
            set_last_error(format!("gliner2_new: {e:?}"));
            std::ptr::null_mut()
        }
    }
}

/// Run a multi-task extraction. `tasks_json` is a JSON array of task objects (see TaskDto).
/// Returns a newly allocated JSON C string (free with `gliner2_free_string`), or null on error.
///
/// # Safety
/// `engine` must come from `gliner2_new`; `text`/`tasks_json` must be valid C strings.
#[no_mangle]
pub unsafe extern "C" fn gliner2_extract(
    engine: *mut Gliner2Engine,
    text: *const c_char,
    tasks_json: *const c_char,
    threshold: c_float,
    flat_ner: c_int,
) -> *mut c_char {
    if engine.is_null() {
        set_last_error("gliner2_extract: engine is null");
        return std::ptr::null_mut();
    }
    if text.is_null() || tasks_json.is_null() {
        set_last_error("gliner2_extract: text or tasks_json is null");
        return std::ptr::null_mut();
    }
    let engine = &*engine;

    let text = match CStr::from_ptr(text).to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("gliner2_extract: invalid text utf8: {e}"));
            return std::ptr::null_mut();
        }
    };
    let tasks_str = match CStr::from_ptr(tasks_json).to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("gliner2_extract: invalid tasks_json utf8: {e}"));
            return std::ptr::null_mut();
        }
    };

    let dtos: Vec<TaskDto> = match serde_json::from_str(tasks_str) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(format!("gliner2_extract: bad tasks_json: {e}"));
            return std::ptr::null_mut();
        }
    };
    let tasks: Vec<SchemaTask> = dtos.into_iter().map(Into::into).collect();

    let params = InferenceParams {
        threshold,
        // InferenceParams.flat_ner is a bool; the C ABI passes it as an int
        // (0 = allow overlapping spans, non-zero = flat/non-overlapping NER).
        flat_ner: flat_ner != 0,
    };

    match engine.extract(text, &tasks, Some(params)) {
        Ok((entities, relations, classifications, structures)) => {
            let result = ExtractResult {
                entities,
                relations,
                classifications,
                structures,
            };
            match serde_json::to_string(&result) {
                Ok(json) => match CString::new(json) {
                    Ok(c) => c.into_raw(),
                    Err(e) => {
                        set_last_error(format!("gliner2_extract: result has interior NUL: {e}"));
                        std::ptr::null_mut()
                    }
                },
                Err(e) => {
                    set_last_error(format!("gliner2_extract: serialize failed: {e}"));
                    std::ptr::null_mut()
                }
            }
        }
        Err(e) => {
            set_last_error(format!("gliner2_extract: {e:?}"));
            std::ptr::null_mut()
        }
    }
}

/// Free an engine created by `gliner2_new`.
///
/// # Safety
/// `engine` must be a pointer returned by `gliner2_new` (or null), freed at most once.
#[no_mangle]
pub unsafe extern "C" fn gliner2_free_engine(engine: *mut Gliner2Engine) {
    if !engine.is_null() {
        drop(Box::from_raw(engine));
    }
}

/// Free a string returned by `gliner2_extract`.
///
/// # Safety
/// `s` must be a pointer returned by this library (or null), freed at most once.
#[no_mangle]
pub unsafe extern "C" fn gliner2_free_string(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}
