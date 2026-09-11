//! Safe bindings to `librkllmrt`, the Rockchip RKLLM runtime.
//!
//! A [`RkllmSession`] owns one loaded model. It is created from a [`Param`],
//! destroyed on drop, and is both [`Send`] and [`Sync`], so it can live in an
//! `Arc` and be driven from several threads.
//!
//! There are two ways to run it:
//!
//! * [`RkllmSession::run_llm`] takes a closure and blocks, calling the closure
//!   with each chunk of output as the runtime produces it.
//! * [`RkllmSession::run_llm_async`] runs the same thing on a blocking thread
//!   with a channel for a callback, and gives back a `Stream` of owned chunks.
//!   Needs the `tokio` feature.
//!
//! ```no_run
//! use rkllm::{Control, InferParams, Input, Param, RkllmSession};
//!
//! # #[cfg(feature = "libloading")]
//! # fn chat() -> rkllm::Result<()> {
//! let param = Param::new("/data/qwen.rkllm")?.max_new_tokens(256);
//! let session = RkllmSession::new_with_library(rkllm_sys::LIBRARY_NAME, &param)?;
//!
//! let mut input = Input::prompt("Why is the sky blue?")?;
//! session.run_llm(&mut input, &InferParams::new(), |out| {
//!     print!("{}", out.text().unwrap_or(""));
//!     Control::Continue
//! })?;
//! # Ok(())
//! # }
//! ```
//!
//! # Which bindings
//!
//! [`RkllmSession`] is generic over the `rkllm-sys` binding flavour, but you
//! never have to name one. Each flavour brings its own constructor, and which
//! ones exist depends on the features:
//!
//! * `RkllmSession::new_with_library(path, &param)` opens the shared library
//!   with `dlopen`. Needs the default `libloading` feature.
//! * `RkllmSession::new(&param)` uses the symbols the linker resolved, so the
//!   library has to be present at build time. Needs the `link` feature.
//!
//! A [`Param`] is a plain value either way. Building one needs no runtime, and
//! the settings you leave alone keep whatever defaults that build of
//! `librkllmrt` ships.
//!
//! # Not wrapped yet
//!
//! Video input, cross-attention parameters, and the tokenizer and embedding
//! callbacks are reachable through `rkllm-sys` but have no safe wrapper here.
//! Image input is wrapped, see [`ImageInput`].
//!
//! Multi-batch inference is wrapped. Set [`Param::n_batch`] and run with
//! [`RkllmSession::run_llm_batch`], which takes one input per entry and reports
//! one output per entry. Streaming stays single-input, since one `Stream`
//! cannot carry several independent generations.

#![warn(missing_docs)]

// Without a flavour there is no way to reach librkllmrt, and no constructor
// exists, so every path through this crate is unreachable.
#[cfg(not(any(feature = "libloading", feature = "link")))]
compile_error!("rkllm needs one of its `libloading` (default) or `link` features enabled");

pub mod error;
pub mod infer;
pub mod input;
pub mod output;
pub mod param;
pub mod session;

#[cfg(feature = "tokio")]
pub mod stream;

#[cfg(test)]
mod fake_runtime;

pub use crate::error::{Error, Result};
pub use crate::infer::{InferParams, Mode, Sampling};
pub use crate::input::{ImageInput, Input};
pub use crate::output::{CallState, Chunk, Control, Output, PerfStat};
pub use crate::param::Param;
pub use crate::session::RkllmSession;

#[cfg(feature = "tokio")]
pub use crate::stream::ChunkStream;

/// Locating `librkllmrt.so` on disk, re-exported from `rkllm-sys`.
pub use rkllm_sys::{LIBRARY_NAME, find_library_path};

#[doc(no_inline)]
pub use rkllm_sys;
