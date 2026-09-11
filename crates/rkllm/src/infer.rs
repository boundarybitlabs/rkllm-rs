//! Per-run configuration.

use std::ffi::CString;
use std::marker::PhantomData;
use std::os::raw::c_int;

use rkllm_sys::{
    RKLLMInferMode, RKLLMInferParam, RKLLMLoraParam, RKLLMPromptCacheParam, RKLLMSamplingParam,
};

use crate::error::{Error, Result};

/// What a run should produce.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Mode {
    /// Generate text.
    #[default]
    Generate,
    /// Return the hidden states of the last layer instead of generating.
    LastHiddenLayer,
    /// Return the output logits instead of generating.
    Logits,
}

impl Mode {
    fn as_raw(self) -> RKLLMInferMode {
        match self {
            Mode::Generate => RKLLMInferMode::RKLLM_INFER_GENERATE,
            Mode::LastHiddenLayer => RKLLMInferMode::RKLLM_INFER_GET_LAST_HIDDEN_LAYER,
            Mode::Logits => RKLLMInferMode::RKLLM_INFER_GET_LOGITS,
        }
    }
}

/// Sampling settings that override the ones fixed at init, for one run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    /// Top-K sampling cutoff.
    pub top_k: i32,
    /// Top-P, or nucleus, sampling cutoff.
    pub top_p: f32,
    /// Sampling temperature.
    pub temperature: f32,
    /// Penalty applied to tokens that repeat.
    pub repeat_penalty: f32,
    /// Penalty scaled by how often a token has appeared.
    pub frequency_penalty: f32,
    /// Penalty applied to tokens already present at all.
    pub presence_penalty: f32,
    /// Mirostat strategy, where zero disables it.
    pub mirostat: i32,
    /// Mirostat target entropy.
    pub mirostat_tau: f32,
    /// Mirostat learning rate.
    pub mirostat_eta: f32,
}

impl Sampling {
    fn as_raw(self) -> RKLLMSamplingParam {
        RKLLMSamplingParam {
            top_k: self.top_k,
            top_p: self.top_p,
            temperature: self.temperature,
            repeat_penalty: self.repeat_penalty,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            mirostat: self.mirostat,
            mirostat_tau: self.mirostat_tau,
            mirostat_eta: self.mirostat_eta,
        }
    }
}

/// Settings for a single run.
///
/// The default generates text and keeps the conversation history, which is what
/// a chat session wants. Call [`InferParams::keep_history`] with `false` for
/// one-shot runs that should not see each other.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InferParams {
    mode: Mode,
    keep_history: bool,
    max_new_tokens: Option<i32>,
    lora_adapter_name: Option<CString>,
    prompt_cache: Option<PromptCache>,
    sampling: Option<Sampling>,
}

#[derive(Debug, Clone, PartialEq)]
struct PromptCache {
    path: CString,
    save: bool,
}

impl InferParams {
    /// Settings that generate text and keep history.
    pub fn new() -> Self {
        InferParams {
            keep_history: true,
            ..Default::default()
        }
    }

    /// What the run should produce.
    pub fn mode(mut self, mode: Mode) -> Self {
        self.mode = mode;
        self
    }

    /// Whether this run sees, and adds to, the conversation history.
    pub fn keep_history(mut self, keep: bool) -> Self {
        self.keep_history = keep;
        self
    }

    /// Token budget for this run, overriding the one set at init.
    pub fn max_new_tokens(mut self, tokens: i32) -> Self {
        self.max_new_tokens = Some(tokens);
        self
    }

    /// Name of a LoRA adapter to apply, previously loaded into the session.
    pub fn lora_adapter(mut self, name: impl Into<Vec<u8>>) -> Result<Self> {
        self.lora_adapter_name =
            Some(CString::new(name).map_err(|e| Error::nul("lora_adapter_name", &e))?);
        Ok(self)
    }

    /// Prompt cache file to use, and whether this run should write to it.
    pub fn prompt_cache(mut self, path: impl Into<Vec<u8>>, save: bool) -> Result<Self> {
        let path = CString::new(path).map_err(|e| Error::nul("prompt_cache_path", &e))?;
        self.prompt_cache = Some(PromptCache { path, save });
        Ok(self)
    }

    /// Sampling settings for this run, overriding the ones fixed at init.
    pub fn sampling(mut self, sampling: Sampling) -> Self {
        self.sampling = Some(sampling);
        self
    }

    /// Materializes the owned C sub-structs the run needs.
    ///
    /// They live in the returned value rather than in `self`, so that the
    /// mutable pointers `RKLLMInferParam` wants do not have to be conjured out
    /// of a shared reference.
    pub(crate) fn scratch(&self) -> Scratch<'_> {
        Scratch {
            lora: self.lora_adapter_name.as_ref().map(|name| RKLLMLoraParam {
                lora_adapter_name: name.as_ptr(),
            }),
            cache: self.prompt_cache.as_ref().map(|c| RKLLMPromptCacheParam {
                save_prompt_cache: c_int::from(c.save),
                prompt_cache_path: c.path.as_ptr(),
            }),
            sampling: self.sampling.map(Sampling::as_raw),
            mode: self.mode.as_raw(),
            keep_history: c_int::from(self.keep_history),
            max_new_tokens: self.max_new_tokens.unwrap_or(0),
            _borrow: PhantomData,
        }
    }
}

/// Owned C sub-structs, borrowing the strings they point at from [`InferParams`].
pub(crate) struct Scratch<'a> {
    lora: Option<RKLLMLoraParam>,
    cache: Option<RKLLMPromptCacheParam>,
    sampling: Option<RKLLMSamplingParam>,
    mode: RKLLMInferMode,
    keep_history: c_int,
    max_new_tokens: i32,
    _borrow: PhantomData<&'a InferParams>,
}

impl Scratch<'_> {
    pub(crate) fn as_raw(&mut self) -> RKLLMInferParam {
        RKLLMInferParam {
            mode: self.mode,
            lora_params: opt_ptr(&mut self.lora),
            prompt_cache_params: opt_ptr(&mut self.cache),
            sampling_params: opt_ptr(&mut self.sampling),
            keep_history: self.keep_history,
            max_new_tokens: self.max_new_tokens,
        }
    }
}

fn opt_ptr<T>(slot: &mut Option<T>) -> *mut T {
    slot.as_mut().map_or(std::ptr::null_mut(), |v| v as *mut T)
}
