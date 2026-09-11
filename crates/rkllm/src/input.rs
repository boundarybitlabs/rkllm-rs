//! What gets fed into a run.

use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr;

use rkllm_sys::{
    RKLLMEmbedInput, RKLLMInput, RKLLMInput__bindgen_ty_1, RKLLMInputType, RKLLMMultiModalInput,
    RKLLMMultiModalInput__bindgen_ty_1, RKLLMMultiModalInput__bindgen_ty_2, RKLLMTokenInput,
};

use crate::error::{Error, Result};

/// The payload of an [`Input`].
#[derive(Debug, Clone, PartialEq)]
enum Payload {
    Prompt(CString),
    Tokens(Vec<i32>),
    Embed { values: Vec<f32>, n_tokens: usize },
    Multimodal { prompt: Vec<u8>, image: ImageInput },
}

/// One or more images, already turned into embeddings.
///
/// RKLLM does not encode images. The embeddings come from a separate vision
/// model, which on Rockchip hardware means an RKNN encoder run through
/// `librknnrt`. This type only carries the result across to the runtime.
///
/// # The buffer length is your responsibility
///
/// The runtime reads `n_image * n_image_tokens * embed_dim` floats. Nothing in
/// the C API carries `embed_dim`, so neither this type nor the runtime can
/// check the buffer against the model that will consume it. Too short a buffer
/// is an out-of-bounds read inside `librkllmrt`.
///
/// [`ImageInput::new`] does check that the length divides evenly by
/// `n_image * n_image_tokens`, which catches the common mistakes, and
/// [`ImageInput::embed_dim`] reports the dimension that implies so you can
/// compare it against the vision model's output.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageInput {
    embed: Vec<f32>,
    n_image: usize,
    n_image_tokens: usize,
    start: CString,
    end: CString,
    content: CString,
    width: usize,
    height: usize,
}

impl ImageInput {
    /// Carries `embed` as `n_image` images of `n_image_tokens` tokens each.
    ///
    /// Fails when the buffer does not divide evenly between them.
    pub fn new(embed: impl Into<Vec<f32>>, n_image: usize, n_image_tokens: usize) -> Result<Self> {
        let embed = embed.into();
        let per_image = n_image.checked_mul(n_image_tokens).unwrap_or(0);
        if per_image == 0 || !embed.len().is_multiple_of(per_image) {
            return Err(Error::EmbeddingNotDivisible {
                len: embed.len(),
                n_image,
                n_image_tokens,
            });
        }
        Ok(ImageInput {
            embed,
            n_image,
            n_image_tokens,
            start: CString::default(),
            end: CString::default(),
            content: CString::default(),
            width: 0,
            height: 0,
        })
    }

    /// The tags that mark where an image sits in the prompt.
    ///
    /// These are model specific. Qwen2-VL uses `<|vision_start|>`,
    /// `<|vision_end|>` and `<|image_pad|>`.
    pub fn tags(
        mut self,
        start: impl Into<Vec<u8>>,
        end: impl Into<Vec<u8>>,
        content: impl Into<Vec<u8>>,
    ) -> Result<Self> {
        self.start = CString::new(start).map_err(|e| Error::nul("image_start", &e))?;
        self.end = CString::new(end).map_err(|e| Error::nul("image_end", &e))?;
        self.content = CString::new(content).map_err(|e| Error::nul("image_content", &e))?;
        Ok(self)
    }

    /// The pixel dimensions the embeddings were produced from.
    pub fn size(mut self, width: usize, height: usize) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// The embedding dimension implied by the buffer and the token counts.
    ///
    /// Compare this against the vision model's output width. They must match.
    pub fn embed_dim(&self) -> usize {
        self.embed.len() / (self.n_image * self.n_image_tokens)
    }

    /// How many images this carries.
    pub fn n_image(&self) -> usize {
        self.n_image
    }

    /// How many tokens each image occupies.
    pub fn n_image_tokens(&self) -> usize {
        self.n_image_tokens
    }
}

/// One message handed to the model.
///
/// Video input is not wrapped. The C struct carries a video block beside the
/// image one, and this crate zeroes it. Reach for `rkllm-sys` if you need it.
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

    /// A text prompt with images attached.
    ///
    /// The images arrive already encoded. See [`ImageInput`] for where the
    /// embeddings come from and what has to line up.
    ///
    /// # The prompt must mark where the image goes
    ///
    /// The runtime requires a literal `<image>` placeholder somewhere in the
    /// prompt, which it replaces with the tags from [`ImageInput::tags`]. A
    /// prompt without one fails the run with a status rather than being
    /// rejected here, since which placeholders are accepted is the runtime's
    /// business and not fixed by this crate.
    ///
    /// ```no_run
    /// # use rkllm::{ImageInput, Input, Result};
    /// # fn f(image: ImageInput) -> Result<()> {
    /// let input = Input::multimodal("<image>What is in this picture?", image)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn multimodal(prompt: impl Into<Vec<u8>>, image: ImageInput) -> Result<Self> {
        let prompt = CString::new(prompt).map_err(|e| Error::nul("prompt", &e))?;
        Ok(Input::with_payload(Payload::Multimodal {
            // Kept as bytes rather than a CString because the C struct wants a
            // mutable pointer, and casting one out of a CString would be a lie.
            prompt: prompt.into_bytes_with_nul(),
            image,
        }))
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
            Payload::Multimodal { prompt, image } => (
                RKLLMInputType::RKLLM_INPUT_MULTIMODAL,
                RKLLMInput__bindgen_ty_1 {
                    multimodal_input: RKLLMMultiModalInput {
                        prompt: prompt.as_mut_ptr().cast::<c_char>(),
                        image: RKLLMMultiModalInput__bindgen_ty_1 {
                            image_embed: image.embed.as_mut_ptr(),
                            n_image_tokens: image.n_image_tokens,
                            n_image: image.n_image,
                            image_start: image.start.as_ptr(),
                            image_end: image.end.as_ptr(),
                            image_content: image.content.as_ptr(),
                            image_width: image.width,
                            image_height: image.height,
                        },
                        // Zeroed, matching what the upstream demo does to the
                        // half of the struct it is not using.
                        video: RKLLMMultiModalInput__bindgen_ty_2 {
                            video_embed: ptr::null_mut(),
                            n_frame_tokens: 0,
                            n_frame_per_video: 0,
                            n_video: 0,
                            video_start: ptr::null(),
                            video_end: ptr::null(),
                            video_content: ptr::null(),
                            frame_width: 0,
                            frame_height: 0,
                        },
                    },
                },
            ),
        };

        RKLLMInput {
            role: self.role.as_ref().map_or(ptr::null(), |role| role.as_ptr()),
            enable_thinking: self.enable_thinking,
            input_type,
            __bindgen_anon_1: payload,
        }
    }
}
