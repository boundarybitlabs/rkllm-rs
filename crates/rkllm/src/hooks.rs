//! Callbacks a model can require of its caller.
//!
//! Most `.rkllm` files carry their own tokenizer and embedding layer. A model
//! exported without one asks the caller to supply it, and refuses to load
//! otherwise:
//!
//! ```text
//! E rkllm: Tokenizer is unavailable: model has no internal tokenizer and no
//!          external tokenizer_callback provided.
//! E rkllm: Model has no embedding layer and no embed callback provided.
//! ```
//!
//! Both are registered at `rkllm_init`, not per run, so they are supplied
//! through [`SessionBuilder`](crate::SessionBuilder) and owned by the session
//! for as long as it lives.

use std::os::raw::{c_char, c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::slice;

/// Turns text into token ids.
///
/// The slice is the largest number of tokens the runtime will accept. Return
/// how many were written, or `None` to report failure.
pub type TokenizerHook = dyn Fn(&str, &mut [i32]) -> Option<usize> + Send + Sync;

/// Turns token ids into their embeddings.
///
/// The slice is exactly `num_tokens * embedding_dim` long and must be filled.
/// Return `false` to report failure.
pub type EmbeddingHook = dyn Fn(&[i32], &mut [f32]) -> bool + Send + Sync;

/// What a session hands the runtime for the callbacks it registers.
///
/// Boxed and owned by the session, so its address stays put after `rkllm_init`
/// has been told where to find it.
#[derive(Default)]
pub(crate) struct Hooks {
    pub(crate) tokenizer: Option<Box<TokenizerHook>>,
    pub(crate) embedding: Option<Box<EmbeddingHook>>,
}

impl Hooks {
    pub(crate) fn is_empty(&self) -> bool {
        self.tokenizer.is_none() && self.embedding.is_none()
    }
}

impl std::fmt::Debug for Hooks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hooks")
            .field("tokenizer", &self.tokenizer.is_some())
            .field("embedding", &self.embedding.is_some())
            .finish()
    }
}

/// The C tokenizer callback, which finds the Rust closure behind `userdata`.
///
/// Returns the token count, or a negative value on failure. A panic in the
/// closure is caught and reported as failure, since unwinding into the runtime
/// is not allowed.
pub(crate) unsafe extern "C" fn tokenizer_trampoline(
    userdata: *mut c_void,
    text: *const c_char,
    text_len: i32,
    tokens: *mut i32,
    n_tokens_max: i32,
) -> c_int {
    const FAILED: c_int = -1;

    if userdata.is_null() || text.is_null() || tokens.is_null() {
        return FAILED;
    }
    let (Ok(text_len), Ok(max)) = (usize::try_from(text_len), usize::try_from(n_tokens_max)) else {
        return FAILED;
    };

    // SAFETY: the session owns these hooks and outlives every call the runtime
    // makes through the handle it registered them on.
    let hooks = unsafe { &*userdata.cast::<Hooks>() };
    let Some(tokenizer) = hooks.tokenizer.as_ref() else {
        return FAILED;
    };

    // SAFETY: the runtime passes `text_len` readable bytes, valid for this call.
    let bytes = unsafe { slice::from_raw_parts(text.cast::<u8>(), text_len) };
    let Ok(text) = std::str::from_utf8(bytes) else {
        return FAILED;
    };
    // SAFETY: the runtime passes a buffer with room for `n_tokens_max` ids.
    let out = unsafe { slice::from_raw_parts_mut(tokens, max) };

    match catch_unwind(AssertUnwindSafe(|| tokenizer(text, out))) {
        // Bounds are the slice's, so a closure cannot overrun the buffer, but
        // it can still claim to have written more than it did.
        Ok(Some(written)) if written <= max => c_int::try_from(written).unwrap_or(FAILED),
        _ => FAILED,
    }
}

/// The C embedding callback, which finds the Rust closure behind `userdata`.
///
/// Returns 0 on success and a negative value on failure.
pub(crate) unsafe extern "C" fn embedding_trampoline(
    userdata: *mut c_void,
    tokens: *mut i32,
    num_tokens: u64,
    embed: *mut c_void,
    len: u64,
) -> c_int {
    const OK: c_int = 0;
    const FAILED: c_int = -1;

    if userdata.is_null() || tokens.is_null() || embed.is_null() {
        return FAILED;
    }
    let (Ok(num_tokens), Ok(len)) = (usize::try_from(num_tokens), usize::try_from(len)) else {
        return FAILED;
    };

    // The buffer is handed over as bytes. Refuse rather than build a misaligned
    // float slice, which would be undefined behaviour.
    if !embed.cast::<f32>().is_aligned() || !len.is_multiple_of(size_of::<f32>()) {
        return FAILED;
    }

    // SAFETY: the session owns these hooks and outlives every call the runtime
    // makes through the handle it registered them on.
    let hooks = unsafe { &*userdata.cast::<Hooks>() };
    let Some(embedding) = hooks.embedding.as_ref() else {
        return FAILED;
    };

    // SAFETY: the runtime passes `num_tokens` readable ids, valid for this call.
    let tokens = unsafe { slice::from_raw_parts(tokens, num_tokens) };
    // SAFETY: `len` bytes are writable, the pointer is aligned for f32, and the
    // length divides evenly, both checked above.
    let out = unsafe { slice::from_raw_parts_mut(embed.cast::<f32>(), len / size_of::<f32>()) };

    match catch_unwind(AssertUnwindSafe(|| embedding(tokens, out))) {
        Ok(true) => OK,
        _ => FAILED,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::c_void;

    /// Calls the tokenizer trampoline the way the runtime would.
    fn tokenize(hooks: &Hooks, text: &str, buffer: &mut [i32]) -> c_int {
        let userdata = (&raw const *hooks).cast_mut().cast::<c_void>();
        // SAFETY: the pointers are to live local values for the call's duration.
        unsafe {
            tokenizer_trampoline(
                userdata,
                text.as_ptr().cast::<c_char>(),
                text.len() as i32,
                buffer.as_mut_ptr(),
                buffer.len() as i32,
            )
        }
    }

    /// Calls the embedding trampoline the way the runtime would.
    fn embed(hooks: &Hooks, tokens: &mut [i32], buffer: &mut [f32]) -> c_int {
        let userdata = (&raw const *hooks).cast_mut().cast::<c_void>();
        // SAFETY: the pointers are to live local values for the call's duration.
        unsafe {
            embedding_trampoline(
                userdata,
                tokens.as_mut_ptr(),
                tokens.len() as u64,
                buffer.as_mut_ptr().cast::<c_void>(),
                size_of_val(buffer) as u64,
            )
        }
    }

    #[test]
    fn the_tokenizer_sees_the_text_and_reports_what_it_wrote() {
        let hooks = Hooks {
            tokenizer: Some(Box::new(|text: &str, out: &mut [i32]| {
                assert_eq!(text, "hello");
                for (slot, byte) in out.iter_mut().zip(text.bytes()) {
                    *slot = i32::from(byte);
                }
                Some(text.len())
            })),
            embedding: None,
        };

        let mut buffer = [0; 16];
        assert_eq!(tokenize(&hooks, "hello", &mut buffer), 5);
        assert_eq!(&buffer[..5], &[104, 101, 108, 108, 111]);
    }

    #[test]
    fn the_tokenizer_buffer_is_bounded_by_what_the_runtime_offered() {
        let hooks = Hooks {
            tokenizer: Some(Box::new(|_: &str, out: &mut [i32]| {
                // The slice is the bound, so this cannot overrun whatever the
                // runtime passed, however much the closure writes.
                assert_eq!(out.len(), 4);
                out.fill(7);
                Some(out.len())
            })),
            embedding: None,
        };

        let mut buffer = [0; 4];
        assert_eq!(tokenize(&hooks, "anything", &mut buffer), 4);
        assert_eq!(buffer, [7; 4]);
    }

    #[test]
    fn a_tokenizer_claiming_too_many_tokens_is_refused() {
        let hooks = Hooks {
            tokenizer: Some(Box::new(|_: &str, out: &mut [i32]| Some(out.len() + 1))),
            embedding: None,
        };

        let mut buffer = [0; 4];
        assert_eq!(
            tokenize(&hooks, "x", &mut buffer),
            -1,
            "the runtime would otherwise read tokens that were never written"
        );
    }

    #[test]
    fn a_tokenizer_reporting_failure_is_passed_through() {
        let hooks = Hooks {
            tokenizer: Some(Box::new(|_: &str, _: &mut [i32]| None)),
            embedding: None,
        };
        assert_eq!(tokenize(&hooks, "x", &mut [0; 4]), -1);
    }

    #[test]
    fn a_panicking_tokenizer_becomes_a_failure_rather_than_an_unwind() {
        let hooks = Hooks {
            tokenizer: Some(Box::new(|_: &str, _: &mut [i32]| panic!("from the hook"))),
            embedding: None,
        };
        assert_eq!(tokenize(&hooks, "x", &mut [0; 4]), -1);
    }

    #[test]
    fn text_that_is_not_utf8_is_refused() {
        let hooks = Hooks {
            tokenizer: Some(Box::new(|_: &str, _: &mut [i32]| Some(0))),
            embedding: None,
        };

        let invalid = [0xffu8, 0xfe];
        let userdata = (&raw const hooks).cast_mut().cast::<c_void>();
        let mut buffer = [0i32; 4];
        // SAFETY: the pointers are to live local values for the call.
        let code = unsafe {
            tokenizer_trampoline(
                userdata,
                invalid.as_ptr().cast::<c_char>(),
                invalid.len() as i32,
                buffer.as_mut_ptr(),
                buffer.len() as i32,
            )
        };
        assert_eq!(
            code, -1,
            "building a &str from those bytes would be unsound"
        );
    }

    #[test]
    fn the_embedding_sees_the_tokens_and_fills_the_buffer() {
        let hooks = Hooks {
            tokenizer: None,
            embedding: Some(Box::new(|tokens: &[i32], out: &mut [f32]| {
                assert_eq!(tokens, [11, 22, 33]);
                // Three tokens of width two, which is what the lengths imply.
                assert_eq!(out.len(), 6);
                for (slot, value) in out.iter_mut().zip(1..) {
                    *slot = f32::from(value as u8);
                }
                true
            })),
        };

        let mut buffer = [0.0; 6];
        assert_eq!(embed(&hooks, &mut [11, 22, 33], &mut buffer), 0);
        assert_eq!(buffer, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn an_embedding_reporting_failure_is_passed_through() {
        let hooks = Hooks {
            tokenizer: None,
            embedding: Some(Box::new(|_: &[i32], _: &mut [f32]| false)),
        };
        assert_eq!(embed(&hooks, &mut [1], &mut [0.0; 2]), -1);
    }

    #[test]
    fn a_panicking_embedding_becomes_a_failure_rather_than_an_unwind() {
        let hooks = Hooks {
            tokenizer: None,
            embedding: Some(Box::new(|_: &[i32], _: &mut [f32]| panic!("from the hook"))),
        };
        assert_eq!(embed(&hooks, &mut [1], &mut [0.0; 2]), -1);
    }

    #[test]
    fn a_misaligned_embedding_buffer_is_refused() {
        let hooks = Hooks {
            tokenizer: None,
            embedding: Some(Box::new(|_: &[i32], _: &mut [f32]| true)),
        };

        let mut bytes = [0u8; 32];
        let userdata = (&raw const hooks).cast_mut().cast::<c_void>();
        let mut tokens = [1i32];
        // One byte into the array, so it cannot be f32-aligned.
        // SAFETY: the offset stays inside the array.
        let misaligned = unsafe { bytes.as_mut_ptr().add(1) };
        // SAFETY: the pointers are to live local values for the call.
        let code = unsafe {
            embedding_trampoline(
                userdata,
                tokens.as_mut_ptr(),
                1,
                misaligned.cast::<c_void>(),
                8,
            )
        };
        assert_eq!(
            code, -1,
            "a misaligned f32 slice would be undefined behaviour"
        );
    }

    #[test]
    fn a_hook_that_was_never_supplied_reports_failure() {
        let hooks = Hooks::default();
        assert!(hooks.is_empty());
        assert_eq!(tokenize(&hooks, "x", &mut [0; 4]), -1);
        assert_eq!(embed(&hooks, &mut [1], &mut [0.0; 2]), -1);
    }
}
