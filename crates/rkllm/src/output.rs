//! What the runtime hands back during a run.

use std::ffi::CStr;
use std::os::raw::c_int;
use std::slice;

use rkllm_sys::{LLMCallState, RKLLMPerfStat, RKLLMResult};

/// Where a run is in its lifecycle when the callback fires.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CallState {
    /// A chunk of output is ready.
    Normal,
    /// The runtime is holding back bytes until a UTF-8 character completes.
    Waiting,
    /// The run finished. This is the last call, and carries the perf stats.
    Finish,
    /// The run failed. This is the last call.
    Error,
    /// A state this crate does not know about.
    Unknown(u32),
}

impl From<LLMCallState> for CallState {
    fn from(raw: LLMCallState) -> Self {
        match raw {
            LLMCallState::RKLLM_RUN_NORMAL => CallState::Normal,
            LLMCallState::RKLLM_RUN_WAITING => CallState::Waiting,
            LLMCallState::RKLLM_RUN_FINISH => CallState::Finish,
            LLMCallState::RKLLM_RUN_ERROR => CallState::Error,
            other => CallState::Unknown(other.0),
        }
    }
}

impl CallState {
    /// Whether no further callback will fire for this run.
    pub fn is_final(self) -> bool {
        matches!(self, CallState::Finish | CallState::Error)
    }
}

/// What the callback tells the runtime to do next.
///
/// Ignored once the state is final.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Control {
    /// Keep generating.
    #[default]
    Continue,
    /// Suspend this run and return from `run_llm`.
    ///
    /// Generation can be resumed by calling `run_llm` again with the content
    /// to continue from. This is also how a caller stops a run from inside the
    /// callback, since the C API has no separate stop code.
    Pause,
    /// Keep generating, but drop the buffer backing the current output.
    ///
    /// Worth returning when running in a mode that produces hidden states or
    /// logits, where each buffer can be large and is not always wanted.
    ReleaseBuffer,
}

impl Control {
    pub(crate) fn as_code(self) -> c_int {
        match self {
            Control::Continue => 0,
            Control::Pause => 1,
            Control::ReleaseBuffer => 2,
        }
    }
}

/// Timings and memory use for a finished run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerfStat {
    /// Wall time spent in the prefill stage, in milliseconds.
    pub prefill_time_ms: f32,
    /// Tokens consumed during prefill.
    pub prefill_tokens: i32,
    /// Wall time spent generating, in milliseconds.
    pub generate_time_ms: f32,
    /// Tokens produced during generation.
    pub generate_tokens: i32,
    /// Peak resident memory during inference, in megabytes.
    pub memory_usage_mb: f32,
}

impl From<RKLLMPerfStat> for PerfStat {
    fn from(raw: RKLLMPerfStat) -> Self {
        PerfStat {
            prefill_time_ms: raw.prefill_time_ms,
            prefill_tokens: raw.prefill_tokens,
            generate_time_ms: raw.generate_time_ms,
            generate_tokens: raw.generate_tokens,
            memory_usage_mb: raw.memory_usage_mb,
        }
    }
}

/// One callback's worth of output, borrowed for the length of the call.
///
/// Everything here points into buffers the runtime owns and reuses, so nothing
/// outlives the callback. Use [`Output::to_chunk`] to keep a copy.
#[derive(Debug, Clone, Copy)]
pub struct Output<'a> {
    state: CallState,
    result: Option<&'a RKLLMResult>,
}

impl<'a> Output<'a> {
    pub(crate) fn new(state: CallState, result: Option<&'a RKLLMResult>) -> Self {
        Output { state, result }
    }

    /// Where the run is in its lifecycle.
    pub fn state(&self) -> CallState {
        self.state
    }

    /// The generated text, if this callback carries valid UTF-8 text.
    ///
    /// Returns `None` when the runtime passed no text, or passed bytes that are
    /// not UTF-8. Reach for [`Output::text_bytes`] to see them regardless.
    pub fn text(&self) -> Option<&'a str> {
        self.text_bytes().and_then(|b| std::str::from_utf8(b).ok())
    }

    /// The generated text as raw bytes, without the trailing NUL.
    pub fn text_bytes(&self) -> Option<&'a [u8]> {
        let text = self.result?.text;
        if text.is_null() {
            return None;
        }
        // SAFETY: non-null and NUL-terminated by the C API contract, and valid
        // for the duration of the callback, which bounds 'a.
        Some(unsafe { CStr::from_ptr(text) }.to_bytes())
    }

    /// The id of the token that produced this output.
    pub fn token_id(&self) -> Option<i32> {
        self.result.map(|r| r.token_id)
    }

    /// The last layer's hidden states, when running in that inference mode.
    pub fn hidden_states(&self) -> Option<&'a [f32]> {
        let layer = &self.result?.last_hidden_layer;
        let len = checked_len(layer.num_tokens, layer.embd_size)?;
        if layer.hidden_states.is_null() {
            return None;
        }
        // SAFETY: the C API documents this buffer as num_tokens * embd_size
        // floats, live for the duration of the callback.
        Some(unsafe { slice::from_raw_parts(layer.hidden_states, len) })
    }

    /// The output logits, when running in that inference mode.
    pub fn logits(&self) -> Option<&'a [f32]> {
        let logits = &self.result?.logits;
        let len = checked_len(logits.num_tokens, logits.vocab_size)?;
        if logits.logits.is_null() {
            return None;
        }
        // SAFETY: the C API documents this buffer as num_tokens * vocab_size
        // floats, live for the duration of the callback.
        Some(unsafe { slice::from_raw_parts(logits.logits, len) })
    }

    /// Timings for the run. Only meaningful once the state is [`CallState::Finish`].
    pub fn perf(&self) -> Option<PerfStat> {
        self.result.map(|r| PerfStat::from(r.perf))
    }

    /// Copies this output into an owned [`Chunk`].
    ///
    /// Only the text, token id and perf stats are copied. Hidden states and
    /// logits are left behind, since they are large and usually consumed in
    /// place.
    pub fn to_chunk(&self) -> Chunk {
        Chunk {
            state: self.state,
            text: self.text().map(str::to_owned),
            token_id: self.token_id(),
            perf: self.perf(),
        }
    }
}

fn checked_len(num_tokens: c_int, width: c_int) -> Option<usize> {
    let tokens = usize::try_from(num_tokens).ok()?;
    let width = usize::try_from(width).ok()?;
    let len = tokens.checked_mul(width)?;
    (len > 0).then_some(len)
}

/// An owned copy of one callback's output, for sending across threads.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Chunk {
    /// Where the run was when this chunk was produced.
    pub state: CallState,
    /// The generated text, when there was valid UTF-8 text to copy.
    pub text: Option<String>,
    /// The id of the token that produced this chunk.
    pub token_id: Option<i32>,
    /// Timings, populated on the final chunk.
    pub perf: Option<PerfStat>,
}

impl Chunk {
    /// The text, or the empty string when this chunk carried none.
    pub fn text(&self) -> &str {
        self.text.as_deref().unwrap_or("")
    }
}
