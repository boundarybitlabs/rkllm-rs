//! Building a session that needs callbacks.

use crate::error::Result;
use crate::hooks::Hooks;
use crate::param::Param;
use crate::session::RkllmSession;

#[cfg(feature = "libloading")]
use std::path::Path;

#[cfg(feature = "libloading")]
use rkllm_sys::RkllmRuntime;

#[cfg(feature = "link")]
use rkllm_sys::RkllmStatic;

/// Builds a session for a model that requires callbacks of its caller.
///
/// Most models need none of this, and
/// [`RkllmSession::new_with_library`](crate::RkllmSession::new_with_library) is
/// the shorter way in. Reach for a builder when the model was exported without
/// its tokenizer or its embedding layer, which the runtime reports at load
/// time by refusing to initialize.
///
/// Both callbacks are registered when the model loads rather than per run, so
/// the session owns them for as long as it lives. They must be `Fn`, `Send` and
/// `Sync`, since nothing documents which thread the runtime calls them from.
///
/// ```no_run
/// # use rkllm::{Param, Result, SessionBuilder};
/// # fn f(tokenize: impl Fn(&str) -> Vec<i32> + Send + Sync + 'static) -> Result<()> {
/// let param = Param::new("/data/model-without-tokenizer.rkllm")?;
/// let session = SessionBuilder::new(&param)
///     .tokenizer(move |text, out| {
///         let tokens = tokenize(text);
///         if tokens.len() > out.len() {
///             return None;
///         }
///         out[..tokens.len()].copy_from_slice(&tokens);
///         Some(tokens.len())
///     })
///     .open_library(rkllm_sys::LIBRARY_NAME)?;
/// # Ok(())
/// # }
/// ```
pub struct SessionBuilder<'a> {
    param: &'a Param,
    hooks: Hooks,
}

impl<'a> SessionBuilder<'a> {
    /// Starts a session for the model `param` points at.
    pub fn new(param: &'a Param) -> Self {
        SessionBuilder {
            param,
            hooks: Hooks::default(),
        }
    }

    /// Supplies the tokenizer, for a model exported without one.
    ///
    /// The closure is handed the text and a buffer holding at most as many
    /// token ids as the runtime will accept. It returns how many it wrote, or
    /// `None` to report failure. Writing is bounded by the slice, so a closure
    /// cannot overrun the runtime's buffer.
    ///
    /// A panic is caught and reported to the runtime as a failure, since
    /// unwinding into C is not allowed.
    pub fn tokenizer<F>(mut self, tokenizer: F) -> Self
    where
        F: Fn(&str, &mut [i32]) -> Option<usize> + Send + Sync + 'static,
    {
        self.hooks.tokenizer = Some(Box::new(tokenizer));
        self
    }

    /// Supplies the embedding layer, for a model exported without one.
    ///
    /// The closure is handed the token ids and a buffer of exactly
    /// `num_tokens * embedding_dim` floats, which it must fill. It returns
    /// `false` to report failure.
    ///
    /// A panic is caught and reported to the runtime as a failure.
    pub fn embedding<F>(mut self, embedding: F) -> Self
    where
        F: Fn(&[i32], &mut [f32]) -> bool + Send + Sync + 'static,
    {
        self.hooks.embedding = Some(Box::new(embedding));
        self
    }

    /// Whether any callback has been supplied.
    pub fn is_plain(&self) -> bool {
        self.hooks.is_empty()
    }

    /// Opens the RKLLM shared library at `path` and loads the model through it.
    ///
    /// See
    /// [`RkllmSession::new_with_library`](crate::RkllmSession::new_with_library)
    /// for what `path` accepts.
    #[cfg(feature = "libloading")]
    pub fn open_library(self, path: impl AsRef<Path>) -> Result<RkllmSession<RkllmRuntime>> {
        // SAFETY: loading a shared object runs its initializers. The caller
        // names the library, and is trusting it the same way they trust any
        // native dependency.
        let runtime = unsafe { RkllmRuntime::new(path.as_ref()) }?;
        RkllmSession::with_api_and_hooks(runtime, self.param, self.hooks)
    }

    /// Loads the model through the linked `librkllmrt`.
    #[cfg(feature = "link")]
    pub fn linked(self) -> Result<RkllmSession<RkllmStatic>> {
        RkllmSession::with_api_and_hooks(RkllmStatic, self.param, self.hooks)
    }
}

#[cfg(test)]
impl SessionBuilder<'_> {
    /// Builds over a stand-in implementation, for tests inside this crate.
    pub(crate) fn with_api<A: rkllm_sys::RkllmApi>(self, api: A) -> Result<RkllmSession<A>> {
        RkllmSession::with_api_and_hooks(api, self.param, self.hooks)
    }
}

impl std::fmt::Debug for SessionBuilder<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionBuilder")
            .field("param", self.param)
            .field("hooks", &self.hooks)
            .finish()
    }
}
