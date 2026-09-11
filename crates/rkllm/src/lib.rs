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
//! use rkllm_sys::RkllmApi;
//!
//! # fn chat<A: RkllmApi>(api: A) -> rkllm::Result<()> {
//! let param = Param::new(&api, "/data/qwen.rkllm")?.max_new_tokens(256);
//! let session = RkllmSession::new(api, &param)?;
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
//! With the default `link` feature, `api` above is `rkllm_sys::Linked`, a
//! zero-sized value. With `libloading` it is an `RkllmRuntime` you loaded.
//!
//! # Which bindings
//!
//! [`RkllmSession`] is generic over the `rkllm-sys` binding flavour. With the
//! default `link` feature that is `Linked`, a zero-sized type over symbols the
//! linker resolved. With the `libloading` feature it can instead be an
//! `RkllmRuntime` loaded at run time, and `Arc<RkllmRuntime>` works too, so
//! several sessions can share one loaded library.
//!
//! # Not wrapped yet
//!
//! Multimodal input, cross-attention parameters, and the tokenizer and
//! embedding callbacks are reachable through `rkllm-sys` but have no safe
//! wrapper here.

#![warn(missing_docs)]

pub mod error;
pub mod infer;
pub mod input;
pub mod output;
pub mod param;
pub mod session;

#[cfg(feature = "tokio")]
pub mod stream;

pub use crate::error::{Error, Result};
pub use crate::infer::{InferParams, Mode, Sampling};
pub use crate::input::Input;
pub use crate::output::{CallState, Chunk, Control, Output, PerfStat};
pub use crate::param::Param;
pub use crate::session::RkllmSession;

#[cfg(feature = "tokio")]
pub use crate::stream::ChunkStream;

#[doc(no_inline)]
pub use rkllm_sys;
