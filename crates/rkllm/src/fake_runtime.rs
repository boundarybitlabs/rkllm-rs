//! Drives [`RkllmSession`] against a stand-in for `librkllmrt`.
//!
//! The fake implements `RkllmApi` by storing the trampoline that `rkllm_init`
//! registers and calling it back from `rkllm_run`, which is exactly what the
//! real runtime does. That exercises the callback plumbing, the control codes
//! and the stream without an NPU or a `librkllmrt` to link against.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;
use std::slice;
use std::sync::Arc;

use rkllm_sys::{
    LLMCallState, LLMHandle, LLMResultCallback, RKLLMCallback, RKLLMCrossAttnParam,
    RKLLMInferParam, RKLLMInput, RKLLMInputType, RKLLMLoraAdapter, RKLLMParam, RKLLMPerfStat,
    RKLLMResult, RKLLMResultLastHiddenLayer, RKLLMResultLogits, RkllmApi,
};

use crate::{CallState, Control, ImageInput, InferParams, Input, Param, RkllmSession};

/// What the fake hands back, one call to the trampoline per entry.
const CHUNKS: [&str; 3] = ["Hello", ", ", "world"];

struct FakeState {
    callback: LLMResultCallback,
    /// What `rkllm_init` was told the batch size would be.
    n_batch: usize,
    seen: std::sync::Mutex<Vec<SeenInput>>,
    cleared: std::sync::Mutex<Option<Vec<(c_int, c_int)>>>,
    cleared_keep: std::sync::Mutex<c_int>,
}

/// What the fake read back out of the `RKLLMInput` it was handed.
#[derive(Debug, Clone, PartialEq)]
struct SeenInput {
    input_type: u32,
    role: Option<String>,
    prompt: String,
    embed: Vec<f32>,
    n_image: usize,
    n_image_tokens: usize,
    tags: (String, String, String),
    size: (usize, usize),
    video_zeroed: bool,
}

/// Reads a NUL-terminated C string, or the empty string when null.
unsafe fn text(ptr: *const c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }
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
        param: *mut RKLLMParam,
        callback: *mut RKLLMCallback,
    ) -> c_int {
        let state = Box::new(FakeState {
            callback: unsafe { (*callback).result_callback },
            n_batch: usize::from(unsafe { (*param).extend_param.n_batch }).max(1),
            seen: std::sync::Mutex::new(Vec::new()),
            cleared: std::sync::Mutex::new(None),
            cleared_keep: std::sync::Mutex::new(0),
        });
        unsafe { *handle = Box::into_raw(state).cast::<c_void>() };
        0
    }

    unsafe fn rkllm_run(
        &self,
        handle: LLMHandle,
        input: *mut RKLLMInput,
        _params: *mut RKLLMInferParam,
        userdata: *mut c_void,
    ) -> c_int {
        let state = unsafe { &*handle.cast::<FakeState>() };
        let n_batch = state.n_batch;

        // Read every input the runtime would, which is what catches a caller
        // that passed fewer than n_batch.
        *state.seen.lock().unwrap() = (0..n_batch)
            .map(|i| unsafe { read_input(input.add(i)) })
            .collect();

        let Some(callback) = state.callback else {
            return 0;
        };

        // Every entry walks the same chunks, but later entries stop sooner, so
        // the batches finish at staggered steps and the run ends only when the
        // last one does.
        for (step, chunk) in CHUNKS.iter().enumerate() {
            // Held for the duration of the call, since the results borrow them.
            let texts: Vec<Option<CString>> = (0..n_batch)
                .map(|b| (step + b < CHUNKS.len()).then(|| CString::new(*chunk).unwrap()))
                .collect();

            let mut results: Vec<RKLLMResult> = texts
                .iter()
                .enumerate()
                .map(|(b, text)| {
                    let mut result = empty_result();
                    match text {
                        Some(text) => {
                            result.text = text.as_ptr();
                            result.token_id = step as i32 + 1;
                        }
                        // A negative id is how the runtime says this entry is
                        // done, and its text is null.
                        None => {
                            result.token_id = -(b as i32 + 1);
                        }
                    }
                    result
                })
                .collect();

            let code = unsafe {
                callback(
                    results.as_mut_ptr(),
                    userdata,
                    LLMCallState::RKLLM_RUN_NORMAL,
                )
            };
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
        handle: LLMHandle,
        keep: c_int,
        start: *mut c_int,
        end: *mut c_int,
    ) -> c_int {
        let state = unsafe { &*handle.cast::<FakeState>() };
        *state.cleared_keep.lock().unwrap() = keep;
        // Reads one entry per batch from each array, exactly as the runtime
        // does. A wrapper passing a shorter array would be read out of bounds.
        *state.cleared.lock().unwrap() = (!start.is_null() && !end.is_null()).then(|| {
            (0..state.n_batch)
                .map(|i| unsafe { (*start.add(i), *end.add(i)) })
                .collect()
        });
        0
    }

    unsafe fn rkllm_get_kv_cache_size(&self, handle: LLMHandle, sizes: *mut c_int) -> c_int {
        let state = unsafe { &*handle.cast::<FakeState>() };
        // Writes one entry per batch, as the runtime does.
        for i in 0..state.n_batch {
            unsafe { *sizes.add(i) = 7 + i as c_int };
        }
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

/// Reads an `RKLLMInput` the way the real runtime would.
unsafe fn read_input(input: *mut RKLLMInput) -> SeenInput {
    let input = unsafe { &*input };
    let mut seen = SeenInput {
        input_type: input.input_type.0,
        role: {
            let role = unsafe { text(input.role) };
            (!input.role.is_null()).then_some(role)
        },
        prompt: String::new(),
        embed: Vec::new(),
        n_image: 0,
        n_image_tokens: 0,
        tags: (String::new(), String::new(), String::new()),
        size: (0, 0),
        video_zeroed: true,
    };

    if input.input_type == RKLLMInputType::RKLLM_INPUT_PROMPT {
        seen.prompt = unsafe { text(input.__bindgen_anon_1.prompt_input) };
    } else if input.input_type == RKLLMInputType::RKLLM_INPUT_MULTIMODAL {
        let mm = unsafe { &input.__bindgen_anon_1.multimodal_input };
        seen.prompt = unsafe { text(mm.prompt) };
        seen.n_image = mm.image.n_image;
        seen.n_image_tokens = mm.image.n_image_tokens;
        let len = mm.image.n_image * mm.image.n_image_tokens;
        // Read exactly what a model with an embedding width of 1 would, which
        // is enough to prove the pointer and counts arrived intact.
        seen.embed = unsafe { slice::from_raw_parts(mm.image.image_embed, len) }.to_vec();
        seen.tags = unsafe {
            (
                text(mm.image.image_start),
                text(mm.image.image_end),
                text(mm.image.image_content),
            )
        };
        seen.size = (mm.image.image_width, mm.image.image_height);
        let v = &mm.video;
        seen.video_zeroed = v.video_embed.is_null()
            && v.video_start.is_null()
            && v.video_end.is_null()
            && v.video_content.is_null()
            && v.n_video == 0
            && v.n_frame_tokens == 0
            && v.n_frame_per_video == 0
            && v.frame_width == 0
            && v.frame_height == 0;
    }

    seen
}

/// Everything the fake read out of the input array on the last run.
fn last_inputs(session: &RkllmSession<FakeApi>) -> Vec<SeenInput> {
    // SAFETY: the handle is the `FakeState` this fake allocated in rkllm_init.
    let state = unsafe { &*session.handle().cast::<FakeState>() };
    let seen = state.seen.lock().unwrap().clone();
    assert!(!seen.is_empty(), "a run happened");
    seen
}

/// What the fake read out of the first input on the last run.
fn last_input(session: &RkllmSession<FakeApi>) -> SeenInput {
    last_inputs(session).swap_remove(0)
}

fn session() -> RkllmSession<FakeApi> {
    batched_session(1)
}

fn batched_session(n_batch: u8) -> RkllmSession<FakeApi> {
    let param = Param::new("/dev/null").unwrap().n_batch(n_batch).unwrap();
    RkllmSession::with_api(FakeApi, &param).unwrap()
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
    assert_eq!(session.kv_cache_size().unwrap(), vec![7]);
}

#[test]
fn interior_nul_is_rejected_rather_than_truncated() {
    let err = Input::prompt("bad\0prompt").unwrap_err();
    assert!(matches!(
        err,
        crate::Error::InteriorNul {
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

#[test]
fn a_prompt_arrives_as_a_prompt() {
    let session = session();
    let mut input = Input::prompt("hello").unwrap().role("user").unwrap();

    session
        .run_llm(&mut input, &InferParams::new(), |_| Control::Pause)
        .unwrap();

    let seen = last_input(&session);
    assert_eq!(seen.input_type, RKLLMInputType::RKLLM_INPUT_PROMPT.0);
    assert_eq!(seen.prompt, "hello");
    assert_eq!(seen.role.as_deref(), Some("user"));
}

#[test]
fn multimodal_input_crosses_the_boundary_intact() {
    let session = session();

    // Two images, three tokens each, embedding width of one, so the values are
    // readable back one per token.
    let embed: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let image = ImageInput::new(embed.clone(), 2, 3)
        .unwrap()
        .tags("<|vision_start|>", "<|vision_end|>", "<|image_pad|>")
        .unwrap()
        .size(392, 392);
    assert_eq!(image.embed_dim(), 1);

    let mut input = Input::multimodal("describe these", image).unwrap();
    session
        .run_llm(&mut input, &InferParams::new(), |_| Control::Pause)
        .unwrap();

    let seen = last_input(&session);
    assert_eq!(seen.input_type, RKLLMInputType::RKLLM_INPUT_MULTIMODAL.0);
    assert_eq!(seen.prompt, "describe these");
    assert_eq!(seen.embed, embed, "the embedding pointer survived");
    assert_eq!((seen.n_image, seen.n_image_tokens), (2, 3));
    assert_eq!(
        seen.tags,
        (
            "<|vision_start|>".to_owned(),
            "<|vision_end|>".to_owned(),
            "<|image_pad|>".to_owned()
        )
    );
    assert_eq!(seen.size, (392, 392));
}

#[test]
fn the_unused_video_half_is_zeroed() {
    let session = session();
    let image = ImageInput::new(vec![0.0; 4], 1, 4).unwrap();
    let mut input = Input::multimodal("x", image).unwrap();

    session
        .run_llm(&mut input, &InferParams::new(), |_| Control::Pause)
        .unwrap();

    assert!(
        last_input(&session).video_zeroed,
        "leaving the video block uninitialized would hand the runtime garbage"
    );
}

#[test]
fn an_embedding_that_does_not_divide_is_rejected() {
    // Seven floats cannot be two images of three tokens.
    let err = ImageInput::new(vec![0.0; 7], 2, 3).unwrap_err();
    assert!(matches!(
        err,
        crate::Error::EmbeddingNotDivisible {
            len: 7,
            n_image: 2,
            n_image_tokens: 3
        }
    ));

    // Zero of either would divide by zero.
    assert!(ImageInput::new(vec![0.0; 4], 0, 4).is_err());
    assert!(ImageInput::new(vec![0.0; 4], 4, 0).is_err());
}

#[test]
fn tags_default_to_empty_rather_than_null() {
    let session = session();
    let image = ImageInput::new(vec![0.0; 2], 1, 2).unwrap();
    let mut input = Input::multimodal("x", image).unwrap();

    session
        .run_llm(&mut input, &InferParams::new(), |_| Control::Pause)
        .unwrap();

    let seen = last_input(&session);
    assert_eq!(seen.tags.0, "", "an unset tag must still be a valid string");
}

/// What the fake recorded about the last cache clear.
fn last_clear(session: &RkllmSession<FakeApi>) -> (c_int, Option<Vec<(c_int, c_int)>>) {
    // SAFETY: the handle is the `FakeState` this fake allocated.
    let state = unsafe { &*session.handle().cast::<FakeState>() };
    let keep = *state.cleared_keep.lock().unwrap();
    (keep, state.cleared.lock().unwrap().clone())
}

#[test]
fn clearing_ranges_passes_one_entry_per_batch() {
    let session = batched_session(3);
    session
        .clear_kv_cache_ranges(&[(1, 2), (3, 4), (5, 6)])
        .unwrap();

    let (keep, ranges) = last_clear(&session);
    assert_eq!(ranges, Some(vec![(1, 2), (3, 4), (5, 6)]));
    assert_eq!(keep, 0, "a range overrides the keep-system-prompt flag");
}

#[test]
fn clearing_everything_passes_null_ranges() {
    let session = session();
    session.clear_kv_cache(true).unwrap();

    let (keep, ranges) = last_clear(&session);
    assert_eq!(keep, 1, "the system prompt is kept");
    assert_eq!(ranges, None, "null arrays mean the whole cache");
}

#[test]
fn the_wrong_number_of_ranges_is_rejected() {
    let session = batched_session(2);
    let err = session.clear_kv_cache_ranges(&[(1, 2)]).unwrap_err();
    assert!(matches!(
        err,
        crate::Error::BatchSizeMismatch {
            expected: 2,
            given: 1
        }
    ));
}

#[test]
fn the_cache_size_has_one_entry_per_batch() {
    let session = batched_session(3);
    assert_eq!(session.kv_cache_size().unwrap(), vec![7, 8, 9]);
}

#[test]
fn a_batch_run_passes_every_input() {
    let session = batched_session(3);
    let mut inputs = [
        Input::prompt("first").unwrap(),
        Input::prompt("second").unwrap(),
        Input::prompt("third").unwrap(),
    ];

    session
        .run_llm_batch(&mut inputs, &InferParams::new(), |_| Control::Continue)
        .unwrap();

    let prompts: Vec<String> = last_inputs(&session)
        .into_iter()
        .map(|i| i.prompt)
        .collect();
    assert_eq!(prompts, vec!["first", "second", "third"]);
}

#[test]
fn a_batch_run_reports_one_output_per_entry() {
    let session = batched_session(3);
    let mut inputs = [
        Input::prompt("a").unwrap(),
        Input::prompt("b").unwrap(),
        Input::prompt("c").unwrap(),
    ];

    let mut widths = Vec::new();
    let mut finished_at = vec![None; 3];
    let mut step = 0usize;

    session
        .run_llm_batch(&mut inputs, &InferParams::new(), |outputs| {
            widths.push((outputs[0].state(), outputs.len()));
            for (entry, output) in outputs.iter().enumerate() {
                if output.is_finished() && finished_at[entry].is_none() {
                    finished_at[entry] = Some(step);
                }
            }
            step += 1;
            Control::Continue
        })
        .unwrap();

    // Three generating callbacks of three outputs each, then a final one.
    assert_eq!(
        widths,
        vec![
            (CallState::Normal, 3),
            (CallState::Normal, 3),
            (CallState::Normal, 3),
            (CallState::Finish, 1),
        ]
    );
    // Later entries stop sooner, so the batches finish at staggered steps.
    assert_eq!(finished_at, vec![None, Some(2), Some(1)]);
}

#[test]
fn a_single_input_call_is_refused_on_a_batched_session() {
    let session = batched_session(2);
    let mut input = Input::prompt("only one").unwrap();

    let err = session
        .run_llm(&mut input, &InferParams::new(), |_| Control::Continue)
        .unwrap_err();

    assert!(
        matches!(err, crate::Error::NotSingleBatch { n_batch: 2 }),
        "passing one input would have the runtime read past its end"
    );
}

#[test]
fn the_wrong_number_of_inputs_is_rejected() {
    let session = batched_session(3);
    let mut inputs = [Input::prompt("a").unwrap(), Input::prompt("b").unwrap()];

    let err = session
        .run_llm_batch(&mut inputs, &InferParams::new(), |_| Control::Continue)
        .unwrap_err();

    assert!(matches!(
        err,
        crate::Error::BatchSizeMismatch {
            expected: 3,
            given: 2
        }
    ));
}

#[test]
fn a_batch_size_of_zero_is_rejected() {
    assert!(matches!(
        Param::new("/dev/null").unwrap().n_batch(0).unwrap_err(),
        crate::Error::ZeroBatch
    ));
}

#[cfg(feature = "tokio")]
#[tokio::test]
async fn the_stream_refuses_a_batched_session() {
    use futures_util::StreamExt as _;

    let session = Arc::new(batched_session(2));
    let mut stream = session.run_llm_async(Input::prompt("hi").unwrap(), InferParams::new());

    let first = stream.next().await.expect("an item");
    assert!(
        matches!(first, Err(crate::Error::NotSingleBatch { n_batch: 2 })),
        "one stream cannot carry two independent generations"
    );
    assert!(stream.next().await.is_none(), "and then it ends");
}
