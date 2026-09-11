//! What gets fed into a run.

use std::ffi::CString;

use rkllm_sys::{
    RKLLMEmbedInput, RKLLMInput, RKLLMInput__bindgen_ty_1, RKLLMInputType, RKLLMTokenInput,
};

use crate::error::{Error, Result};

/// The payload of an [`Input`].
#[derive(Debug, Clone, PartialEq)]
enum Payload {
    Prompt(CString),
    Tokens(Vec<i32>),
    Embed { values: Vec<f32>, n_tokens: usize },
}

/// One message handed to the model.
///
/// Multimodal input is not wrapped yet. Reach for `rkllm-sys` directly if you
/// need it.
#[derive(Debug, Clone, PartialEq)]
pub struct Input {
    role: Option<CString>,
    enable_thinking: bool,
    payload: Payload,
}

impl Input {
    /// A text prompt.
    pub fn prompt(prompt: impl Into<Vec<u8>>) -> Result<Self> {
        let prompt = CString::new(prompt).map_err(|e| Error::nul("prompt", &e))?;
        Ok(Input::with_payload(Payload::Prompt(prompt)))
    }

    /// A sequence of token ids, bypassing the tokenizer.
    pub fn tokens(tokens: impl Into<Vec<i32>>) -> Self {
        Input::with_payload(Payload::Tokens(tokens.into()))
    }

    /// A flat embedding buffer of `n_tokens * n_embed` values.
    pub fn embeddings(values: impl Into<Vec<f32>>, n_tokens: usize) -> Self {
        Input::with_payload(Payload::Embed {
            values: values.into(),
            n_tokens,
        })
    }

    fn with_payload(payload: Payload) -> Self {
        Input {
            role: None,
            enable_thinking: false,
            payload,
        }
    }

    /// Sets the message role, such as `user` or `tool`.
    ///
    /// Left unset, the runtime applies its own default.
    pub fn role(mut self, role: impl Into<Vec<u8>>) -> Result<Self> {
        self.role = Some(CString::new(role).map_err(|e| Error::nul("role", &e))?);
        Ok(self)
    }

    /// Turns on thinking mode, which Qwen3 models understand.
    pub fn enable_thinking(mut self, enabled: bool) -> Self {
        self.enable_thinking = enabled;
        self
    }

    /// The C struct, pointing into this `Input`'s own storage.
    ///
    /// Takes `&mut self` because the token and embedding variants hand the
    /// runtime mutable pointers into these buffers.
    pub(crate) fn as_raw(&mut self) -> RKLLMInput {
        let (input_type, payload) = match &mut self.payload {
            Payload::Prompt(prompt) => (
                RKLLMInputType::RKLLM_INPUT_PROMPT,
                RKLLMInput__bindgen_ty_1 {
                    prompt_input: prompt.as_ptr(),
                },
            ),
            Payload::Tokens(tokens) => (
                RKLLMInputType::RKLLM_INPUT_TOKEN,
                RKLLMInput__bindgen_ty_1 {
                    token_input: RKLLMTokenInput {
                        input_ids: tokens.as_mut_ptr(),
                        n_tokens: tokens.len(),
                    },
                },
            ),
            Payload::Embed { values, n_tokens } => (
                RKLLMInputType::RKLLM_INPUT_EMBED,
                RKLLMInput__bindgen_ty_1 {
                    embed_input: RKLLMEmbedInput {
                        embed: values.as_mut_ptr(),
                        n_tokens: *n_tokens,
                    },
                },
            ),
        };

        RKLLMInput {
            role: self
                .role
                .as_ref()
                .map_or(std::ptr::null(), |role| role.as_ptr()),
            enable_thinking: self.enable_thinking,
            input_type,
            __bindgen_anon_1: payload,
        }
    }
}
