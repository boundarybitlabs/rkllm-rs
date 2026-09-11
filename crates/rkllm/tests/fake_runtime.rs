//! Drives `RkllmSession` against a stand-in for `librkllmrt`.
//!
//! The fake implements `RkllmApi` by storing the trampoline that `rkllm_init`
//! registers and calling it back from `rkllm_run`, which is exactly what the
//! real runtime does. That exercises the callback plumbing, the control codes
//! and the stream without an NPU or a `librkllmrt` to link against.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;
use std::sync::Arc;

use rkllm_sys::{
    LLMCallState, LLMHandle, LLMResultCallback, RKLLMCallback, RKLLMCrossAttnParam,
    RKLLMInferParam, RKLLMInput, RKLLMLoraAdapter, RKLLMParam, RKLLMPerfStat, RKLLMResult,
    RKLLMResultLastHiddenLayer, RKLLMResultLogits, RkllmApi,
};

use rkllm::{CallState, Control, InferParams, Input, Param, RkllmSession};

/// What the fake hands back, one call to the trampoline per entry.
const CHUNKS: [&str; 3] = ["Hello", ", ", "world"];

struct FakeState {
    callback: LLMResultCallback,
}

#[derive(Debug, Clone, Copy, Default)]
struct FakeApi;

fn empty_result() -> RKLLMResult {
    RKLLMResult {
        text: ptr::null(),
        token_id: 0,
        last_hidden_layer: RKLLMResultLastHiddenLayer {
            hidden_states: ptr::null(),
            embd_size: 0,
            num_tokens: 0,
        },
        logits: RKLLMResultLogits {
            logits: ptr::null(),
            vocab_size: 0,
            num_tokens: 0,
        },
        perf: RKLLMPerfStat {
            prefill_time_ms: 0.0,
            prefill_tokens: 0,
            generate_time_ms: 0.0,
            generate_tokens: 0,
            memory_usage_mb: 0.0,
        },
    }
}

unsafe impl RkllmApi for FakeApi {
    unsafe fn rkllm_createDefaultParam(&self) -> RKLLMParam {
        // SAFETY: RKLLMParam is a plain C struct of scalars and a pointer, so
        // an all-zero value is a valid one.
        unsafe { std::mem::zeroed() }
    }

    unsafe fn rkllm_init(
        &self,
        handle: *mut LLMHandle,
        _param: *mut RKLLMParam,
        callback: *mut RKLLMCallback,
    ) -> c_int {
        let state = Box::new(FakeState {
            callback: unsafe { (*callback).result_callback },
        });
        unsafe { *handle = Box::into_raw(state).cast::<c_void>() };
        0
    }

    unsafe fn rkllm_run(
        &self,
        handle: LLMHandle,
        _input: *mut RKLLMInput,
        _params: *mut RKLLMInferParam,
        userdata: *mut c_void,
    ) -> c_int {
        let state = unsafe { &*handle.cast::<FakeState>() };
        let Some(callback) = state.callback else {
            return 0;
        };

        for (index, chunk) in CHUNKS.iter().enumerate() {
            let text = CString::new(*chunk).unwrap();
            let mut result = empty_result();
            result.text = text.as_ptr();
            result.token_id = index as i32 + 1;

            let code = unsafe { callback(&mut result, userdata, LLMCallState::RKLLM_RUN_NORMAL) };
            // 1 means "pause", which is how a caller stops a run early.
            if code == 1 {
                return 0;
            }
        }

        let mut result = empty_result();
        result.perf.generate_tokens = CHUNKS.len() as i32;
        result.perf.generate_time_ms = 12.5;
        unsafe { callback(&mut result, userdata, LLMCallState::RKLLM_RUN_FINISH) };
        0
    }

    unsafe fn rkllm_destroy(&self, handle: LLMHandle) -> c_int {
        drop(unsafe { Box::from_raw(handle.cast::<FakeState>()) });
        0
    }

    unsafe fn rkllm_abort(&self, _handle: LLMHandle) -> c_int {
        0
    }

    unsafe fn rkllm_is_running(&self, _handle: LLMHandle) -> c_int {
        0
    }

    unsafe fn rkllm_load_lora(&self, _h: LLMHandle, _a: *mut RKLLMLoraAdapter) -> c_int {
        0
    }

    unsafe fn rkllm_load_prompt_cache(&self, _h: LLMHandle, _p: *const c_char) -> c_int {
        0
    }

    unsafe fn rkllm_release_prompt_cache(&self, _h: LLMHandle) -> c_int {
        0
    }

    unsafe fn rkllm_run_async(
        &self,
        _h: LLMHandle,
        _i: *mut RKLLMInput,
        _p: *mut RKLLMInferParam,
        _u: *mut c_void,
    ) -> c_int {
        0
    }

    unsafe fn rkllm_clear_kv_cache(
        &self,
        _h: LLMHandle,
        _keep: c_int,
        _start: *mut c_int,
        _end: *mut c_int,
    ) -> c_int {
        0
    }

    unsafe fn rkllm_get_kv_cache_size(&self, _h: LLMHandle, sizes: *mut c_int) -> c_int {
        unsafe { *sizes = 7 };
        0
    }

    unsafe fn rkllm_set_chat_template(
        &self,
        _h: LLMHandle,
        system: *const c_char,
        _prefix: *const c_char,
        _postfix: *const c_char,
    ) -> c_int {
        // Prove the string survived the trip across the boundary intact.
        assert_eq!(unsafe { CStr::from_ptr(system) }.to_bytes(), b"be brief");
        0
    }

    unsafe fn rkllm_set_function_tools(
        &self,
        _h: LLMHandle,
        _s: *const c_char,
        _t: *const c_char,
        _r: *const c_char,
    ) -> c_int {
        0
    }

    unsafe fn rkllm_set_cross_attn_params(
        &self,
        _h: LLMHandle,
        _p: *mut RKLLMCrossAttnParam,
    ) -> c_int {
        0
    }
}

fn session() -> RkllmSession<FakeApi> {
    let param = Param::new(&FakeApi, "/dev/null").unwrap();
    RkllmSession::new(FakeApi, &param).unwrap()
}

#[test]
fn run_llm_delivers_every_chunk_then_finishes() {
    let session = session();
    let mut input = Input::prompt("hi").unwrap();
    let mut seen = Vec::new();

    session
        .run_llm(&mut input, &InferParams::new(), |out| {
            seen.push((out.state(), out.text().map(str::to_owned), out.token_id()));
            Control::Continue
        })
        .unwrap();

    assert_eq!(seen.len(), CHUNKS.len() + 1);
    let text: String = seen.iter().filter_map(|(_, t, _)| t.clone()).collect();
    assert_eq!(text, "Hello, world");
    assert_eq!(seen[0].2, Some(1));
    assert_eq!(seen.last().unwrap().0, CallState::Finish);
}

#[test]
fn pause_stops_the_run_early() {
    let session = session();
    let mut input = Input::prompt("hi").unwrap();
    let mut count = 0;

    session
        .run_llm(&mut input, &InferParams::new(), |_| {
            count += 1;
            Control::Pause
        })
        .unwrap();

    assert_eq!(count, 1, "the run should stop at the first Pause");
}

#[test]
fn perf_stats_arrive_with_the_final_chunk() {
    let session = session();
    let mut input = Input::prompt("hi").unwrap();
    let mut perf = None;

    session
        .run_llm(&mut input, &InferParams::new(), |out| {
            if out.state() == CallState::Finish {
                perf = out.perf();
            }
            Control::Continue
        })
        .unwrap();

    let perf = perf.expect("the final chunk carries perf stats");
    assert_eq!(perf.generate_tokens, 3);
    assert_eq!(perf.generate_time_ms, 12.5);
}

#[test]
fn a_panicking_callback_resumes_on_the_calling_thread() {
    let session = session();
    let mut input = Input::prompt("hi").unwrap();

    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        session
            .run_llm(&mut input, &InferParams::new(), |_| panic!("from callback"))
            .unwrap();
    }));

    let payload = panicked.expect_err("the panic should reach the caller");
    assert_eq!(payload.downcast_ref::<&str>(), Some(&"from callback"));
}

#[test]
fn strings_reach_the_runtime_intact() {
    let session = session();
    session.set_chat_template("be brief", "", "").unwrap();
    assert_eq!(session.kv_cache_size(1).unwrap(), vec![7]);
}

#[test]
fn interior_nul_is_rejected_rather_than_truncated() {
    let err = Input::prompt("bad\0prompt").unwrap_err();
    assert!(matches!(
        err,
        rkllm::Error::InteriorNul {
            field: "prompt",
            position: 3
        }
    ));
}

#[test]
fn the_session_is_send_and_sync() {
    fn require<T: Send + Sync>() {}
    require::<RkllmSession<FakeApi>>();
    require::<Arc<RkllmSession<FakeApi>>>();
}

#[cfg(feature = "tokio")]
#[tokio::test]
async fn run_llm_async_streams_the_same_chunks() {
    use futures_util::StreamExt as _;

    let session = Arc::new(session());
    let mut stream = session.run_llm_async(Input::prompt("hi").unwrap(), InferParams::new());

    let mut text = String::new();
    let mut states = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        text.push_str(chunk.text());
        states.push(chunk.state);
    }

    assert_eq!(text, "Hello, world");
    assert_eq!(states.last(), Some(&CallState::Finish));
}

#[cfg(feature = "tokio")]
#[tokio::test]
async fn dropping_the_stream_cancels_the_run() {
    use futures_util::StreamExt as _;

    let session = Arc::new(session());
    let mut stream = session.run_llm_async(Input::prompt("hi").unwrap(), InferParams::new());

    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(first.text(), "Hello");
    drop(stream);

    // The cancelled run must release the session's lock, so a second run on the
    // same session goes through. A deadlock here would hang the test.
    let mut again = session.run_llm_async(Input::prompt("hi").unwrap(), InferParams::new());
    let mut text = String::new();
    while let Some(chunk) = again.next().await {
        text.push_str(chunk.unwrap().text());
    }
    assert_eq!(text, "Hello, world");
}
