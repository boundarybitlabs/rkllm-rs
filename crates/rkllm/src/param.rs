//! Model configuration, fixed when a session is created.

use std::ffi::CString;

use rkllm_sys::RKLLMParam;

use crate::error::{Error, Result};

/// Configuration for a model.
///
/// A `Param` is a plain value: building one needs no runtime, and nothing in it
/// is resolved until a session is created. At that point the session asks the
/// runtime for its own defaults and lays these settings over the top, so
/// anything left alone keeps whatever default that version of `librkllmrt`
/// ships.
///
/// ```no_run
/// # use rkllm::{Param, Result};
/// # fn f() -> Result<()> {
/// let param = Param::new("/data/qwen.rkllm")?
///     .max_context_len(4096)
///     .max_new_tokens(512)
///     .temperature(0.7);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Param {
    model_path: CString,
    max_context_len: Option<i32>,
    max_new_tokens: Option<i32>,
    n_keep: Option<i32>,
    top_k: Option<i32>,
    top_p: Option<f32>,
    temperature: Option<f32>,
    repeat_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    mirostat: Option<i32>,
    mirostat_tau: Option<f32>,
    mirostat_eta: Option<f32>,
    skip_special_token: Option<bool>,
    ignore_eos_token: Option<bool>,
    enabled_cpus: Option<(i8, u32)>,
    n_batch: Option<u8>,
    embed_flash: Option<bool>,
    use_cross_attn: Option<bool>,
}

impl Param {
    /// Configuration for the model at `model_path`, with every setting left at
    /// the runtime's default.
    pub fn new(model_path: impl Into<Vec<u8>>) -> Result<Self> {
        let model_path = CString::new(model_path).map_err(|e| Error::nul("model_path", &e))?;
        Ok(Param {
            model_path,
            ..Default::default()
        })
    }

    /// The path this configuration points at.
    pub fn model_path(&self) -> &CString {
        &self.model_path
    }

    /// Maximum number of tokens held in the context window.
    pub fn max_context_len(mut self, tokens: i32) -> Self {
        self.max_context_len = Some(tokens);
        self
    }

    /// Maximum number of tokens generated in one run.
    pub fn max_new_tokens(mut self, tokens: i32) -> Self {
        self.max_new_tokens = Some(tokens);
        self
    }

    /// Number of leading key-value cache entries kept when the context shifts.
    pub fn n_keep(mut self, tokens: i32) -> Self {
        self.n_keep = Some(tokens);
        self
    }

    /// Top-K sampling cutoff.
    pub fn top_k(mut self, k: i32) -> Self {
        self.top_k = Some(k);
        self
    }

    /// Top-P, or nucleus, sampling cutoff.
    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    /// Sampling temperature.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Penalty applied to tokens that repeat.
    pub fn repeat_penalty(mut self, penalty: f32) -> Self {
        self.repeat_penalty = Some(penalty);
        self
    }

    /// Penalty scaled by how often a token has appeared.
    pub fn frequency_penalty(mut self, penalty: f32) -> Self {
        self.frequency_penalty = Some(penalty);
        self
    }

    /// Penalty applied to tokens already present at all.
    pub fn presence_penalty(mut self, penalty: f32) -> Self {
        self.presence_penalty = Some(penalty);
        self
    }

    /// Mirostat strategy, where zero disables it.
    pub fn mirostat(mut self, strategy: i32) -> Self {
        self.mirostat = Some(strategy);
        self
    }

    /// Mirostat target entropy.
    pub fn mirostat_tau(mut self, tau: f32) -> Self {
        self.mirostat_tau = Some(tau);
        self
    }

    /// Mirostat learning rate.
    pub fn mirostat_eta(mut self, eta: f32) -> Self {
        self.mirostat_eta = Some(eta);
        self
    }

    /// Whether special tokens are left out of the generated text.
    pub fn skip_special_token(mut self, skip: bool) -> Self {
        self.skip_special_token = Some(skip);
        self
    }

    /// Whether the end-of-sequence token stops generation.
    pub fn ignore_eos_token(mut self, ignore: bool) -> Self {
        self.ignore_eos_token = Some(ignore);
        self
    }

    /// Number of CPU cores the runtime may use, with the matching bitmask.
    ///
    /// The mask is built from the `CPU0` through `CPU7` constants.
    pub fn cpus(mut self, count: i8, mask: u32) -> Self {
        self.enabled_cpus = Some((count, mask));
        self
    }

    /// How many inputs one forward pass handles. Above one enables batching.
    pub fn n_batch(mut self, n_batch: u8) -> Self {
        self.n_batch = Some(n_batch);
        self
    }

    /// Whether word embeddings are read from flash rather than memory.
    pub fn embed_flash(mut self, from_flash: bool) -> Self {
        self.embed_flash = Some(from_flash);
        self
    }

    /// Whether the decoder runs cross attention.
    pub fn use_cross_attn(mut self, enabled: bool) -> Self {
        self.use_cross_attn = Some(enabled);
        self
    }

    /// Lays these settings over the runtime's defaults, and points
    /// `model_path` at this `Param`'s own storage.
    pub(crate) fn apply_to(&self, raw: &mut RKLLMParam) {
        raw.model_path = self.model_path.as_ptr();

        if let Some(v) = self.max_context_len {
            raw.max_context_len = v;
        }
        if let Some(v) = self.max_new_tokens {
            raw.max_new_tokens = v;
        }
        if let Some(v) = self.n_keep {
            raw.n_keep = v;
        }
        if let Some(v) = self.top_k {
            raw.top_k = v;
        }
        if let Some(v) = self.top_p {
            raw.top_p = v;
        }
        if let Some(v) = self.temperature {
            raw.temperature = v;
        }
        if let Some(v) = self.repeat_penalty {
            raw.repeat_penalty = v;
        }
        if let Some(v) = self.frequency_penalty {
            raw.frequency_penalty = v;
        }
        if let Some(v) = self.presence_penalty {
            raw.presence_penalty = v;
        }
        if let Some(v) = self.mirostat {
            raw.mirostat = v;
        }
        if let Some(v) = self.mirostat_tau {
            raw.mirostat_tau = v;
        }
        if let Some(v) = self.mirostat_eta {
            raw.mirostat_eta = v;
        }
        if let Some(v) = self.skip_special_token {
            raw.skip_special_token = v;
        }
        if let Some(v) = self.ignore_eos_token {
            raw.ignore_eos_token = v;
        }
        if let Some((count, mask)) = self.enabled_cpus {
            raw.extend_param.enabled_cpus_num = count;
            raw.extend_param.enabled_cpus_mask = mask;
        }
        if let Some(v) = self.n_batch {
            raw.extend_param.n_batch = v;
        }
        if let Some(v) = self.embed_flash {
            raw.extend_param.embed_flash = i8::from(v);
        }
        if let Some(v) = self.use_cross_attn {
            raw.extend_param.use_cross_attn = i8::from(v);
        }
    }
}
