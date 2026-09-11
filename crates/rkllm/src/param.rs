//! Model configuration, fixed when a session is created.

use std::ffi::CString;

use rkllm_sys::{RKLLMParam, RkllmApi};

use crate::error::{Error, Result};

/// Configuration for a model, passed to `rkllm_init`.
///
/// Start from [`Param::new`], which asks the runtime for its defaults, then
/// override what you care about:
///
/// ```no_run
/// # use rkllm::{Param, Result};
/// # use rkllm_sys::RkllmApi;
/// # fn f<A: RkllmApi>(api: &A) -> Result<()> {
/// let param = Param::new(api, "/data/qwen.rkllm")?
///     .max_context_len(4096)
///     .max_new_tokens(512)
///     .temperature(0.7);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct Param {
    raw: RKLLMParam,
    model_path: CString,
}

impl Param {
    /// Builds a parameter set from the runtime's own defaults, for the model at
    /// `model_path`.
    pub fn new<A: RkllmApi>(api: &A, model_path: impl Into<Vec<u8>>) -> Result<Self> {
        let model_path = CString::new(model_path).map_err(|e| Error::nul("model_path", &e))?;
        // SAFETY: the call takes no arguments and returns a plain value.
        let raw = unsafe { api.rkllm_createDefaultParam() };
        Ok(Param { raw, model_path })
    }

    /// The path this parameter set points at.
    pub fn model_path(&self) -> &CString {
        &self.model_path
    }

    /// Maximum number of tokens held in the context window.
    pub fn max_context_len(mut self, tokens: i32) -> Self {
        self.raw.max_context_len = tokens;
        self
    }

    /// Maximum number of tokens generated in one run.
    pub fn max_new_tokens(mut self, tokens: i32) -> Self {
        self.raw.max_new_tokens = tokens;
        self
    }

    /// Number of leading key-value cache entries kept when the context shifts.
    pub fn n_keep(mut self, tokens: i32) -> Self {
        self.raw.n_keep = tokens;
        self
    }

    /// Top-K sampling cutoff.
    pub fn top_k(mut self, k: i32) -> Self {
        self.raw.top_k = k;
        self
    }

    /// Top-P, or nucleus, sampling cutoff.
    pub fn top_p(mut self, p: f32) -> Self {
        self.raw.top_p = p;
        self
    }

    /// Sampling temperature.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.raw.temperature = temperature;
        self
    }

    /// Penalty applied to tokens that repeat.
    pub fn repeat_penalty(mut self, penalty: f32) -> Self {
        self.raw.repeat_penalty = penalty;
        self
    }

    /// Penalty scaled by how often a token has appeared.
    pub fn frequency_penalty(mut self, penalty: f32) -> Self {
        self.raw.frequency_penalty = penalty;
        self
    }

    /// Penalty applied to tokens already present at all.
    pub fn presence_penalty(mut self, penalty: f32) -> Self {
        self.raw.presence_penalty = penalty;
        self
    }

    /// Mirostat strategy, where zero disables it.
    pub fn mirostat(mut self, strategy: i32) -> Self {
        self.raw.mirostat = strategy;
        self
    }

    /// Mirostat target entropy.
    pub fn mirostat_tau(mut self, tau: f32) -> Self {
        self.raw.mirostat_tau = tau;
        self
    }

    /// Mirostat learning rate.
    pub fn mirostat_eta(mut self, eta: f32) -> Self {
        self.raw.mirostat_eta = eta;
        self
    }

    /// Whether special tokens are left out of the generated text.
    pub fn skip_special_token(mut self, skip: bool) -> Self {
        self.raw.skip_special_token = skip;
        self
    }

    /// Whether the end-of-sequence token stops generation.
    pub fn ignore_eos_token(mut self, ignore: bool) -> Self {
        self.raw.ignore_eos_token = ignore;
        self
    }

    /// Number of CPU cores the runtime may use, with the matching bitmask.
    ///
    /// The mask is built from the `CPU0` through `CPU7` constants.
    pub fn cpus(mut self, count: i8, mask: u32) -> Self {
        self.raw.extend_param.enabled_cpus_num = count;
        self.raw.extend_param.enabled_cpus_mask = mask;
        self
    }

    /// How many inputs one forward pass handles. Above one enables batching.
    pub fn n_batch(mut self, n_batch: u8) -> Self {
        self.raw.extend_param.n_batch = n_batch;
        self
    }

    /// Whether word embeddings are read from flash rather than memory.
    pub fn embed_flash(mut self, from_flash: bool) -> Self {
        self.raw.extend_param.embed_flash = i8::from(from_flash);
        self
    }

    /// Whether the decoder runs cross attention.
    pub fn use_cross_attn(mut self, enabled: bool) -> Self {
        self.raw.extend_param.use_cross_attn = i8::from(enabled);
        self
    }

    /// Escape hatch: mutate the underlying C struct directly.
    ///
    /// The `model_path` field is overwritten when the session is created, so
    /// setting it here has no effect.
    pub fn raw_mut(&mut self) -> &mut RKLLMParam {
        &mut self.raw
    }

    /// The C struct, with `model_path` pointed at this `Param`'s own storage.
    pub(crate) fn as_raw(&self) -> RKLLMParam {
        let mut raw = self.raw;
        raw.model_path = self.model_path.as_ptr();
        raw
    }
}
