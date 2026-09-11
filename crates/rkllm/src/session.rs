//! The model handle, and the two ways to run it.

use std::any::Any;
use std::ffi::CString;
use std::os::raw::{c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::ptr;
use std::sync::{Mutex, MutexGuard, PoisonError};

#[cfg(feature = "libloading")]
use std::path::Path;

#[cfg(feature = "libloading")]
use rkllm_sys::RkllmRuntime;
#[cfg(feature = "link")]
use rkllm_sys::RkllmStatic;
use rkllm_sys::{LLMCallState, LLMHandle, RKLLMCallback, RKLLMLoraAdapter, RKLLMResult, RkllmApi};

use crate::error::{Error, Result};
use crate::infer::InferParams;
use crate::input::Input;
use crate::output::{CallState, Control, Output};
use crate::param::Param;

/// A loaded model, and the runtime handle that owns it.
///
/// Destroyed on drop. The type parameter is which set of bindings to dispatch
/// through, and each flavour brings its own constructor:
///
/// * [`RkllmSession::<RkllmRuntime>::new_with_library`] opens `librkllmrt.so`
///   with `dlopen`. Needs the default `libloading` feature.
/// * [`RkllmSession::<RkllmStatic>::new`] uses the symbols the linker
///   resolved. Needs the `link` feature.
///
/// # Threads
///
/// The session is [`Send`] and [`Sync`]. Only one run may be in flight at a
/// time, which an internal lock enforces, so concurrent calls to
/// [`RkllmSession::run_llm`] queue up rather than racing.
///
/// [`RkllmSession::abort`] and [`RkllmSession::is_running`] deliberately skip
/// that lock, since their whole purpose is to be called from another thread
/// while a run is blocked inside the runtime.
pub struct RkllmSession<A: RkllmApi> {
    api: A,
    handle: LLMHandle,
    run_lock: Mutex<()>,
}

// SAFETY: the handle is an opaque pointer the runtime owns. Nothing in this
// type is read or written without going through `&self`, concurrent runs are
// serialized by `run_lock`, and the runtime supports `rkllm_abort` and
// `rkllm_is_running` from a thread other than the one inside `rkllm_run`.
unsafe impl<A: RkllmApi + Send> Send for RkllmSession<A> {}
// SAFETY: as above.
unsafe impl<A: RkllmApi + Sync> Sync for RkllmSession<A> {}

impl<A: RkllmApi> RkllmSession<A> {
    /// Loads a model through `api`, and returns the session that owns it.
    ///
    /// Private on purpose. Callers reach a session through the constructor for
    /// the flavour they want, so the bindings stay an implementation detail
    /// rather than something every call site has to name.
    pub(crate) fn with_api(api: A, param: &Param) -> Result<Self> {
        // Start from this runtime's own defaults, then lay the caller's
        // settings over the top.
        // SAFETY: the call takes no arguments and returns a plain value.
        let mut raw_param = unsafe { api.rkllm_createDefaultParam() };
        param.apply_to(&mut raw_param);

        let mut handle: LLMHandle = ptr::null_mut();
        let mut callback = RKLLMCallback {
            result_callback: Some(trampoline),
            // Left null on purpose. The per-run `userdata` argument of
            // `rkllm_run` takes precedence, and that is where the closure goes.
            result_userdata: ptr::null_mut(),
            tokenizer_callback: None,
            tokenizer_userdata: ptr::null_mut(),
            embed_callback: None,
            embed_userdata: ptr::null_mut(),
        };

        // SAFETY: all three pointers are to live, initialized values, and the
        // runtime copies what it needs out of them during the call.
        let code = unsafe { api.rkllm_init(&mut handle, &mut raw_param, &mut callback) };
        Error::check("rkllm_init", code)?;
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(RkllmSession {
            api,
            handle,
            run_lock: Mutex::new(()),
        })
    }

    /// The bindings this session dispatches through.
    pub fn api(&self) -> &A {
        &self.api
    }

    /// The raw handle, for reaching past this wrapper into `rkllm-sys`.
    ///
    /// # Safety
    ///
    /// The handle is destroyed when the session drops, and calling into the
    /// runtime with it concurrently with [`RkllmSession::run_llm`] is only
    /// sound for the operations the C API documents as safe to interleave.
    pub unsafe fn handle(&self) -> LLMHandle {
        self.handle
    }

    fn lock_run(&self) -> MutexGuard<'_, ()> {
        // A panic in a callback leaves the handle alone, so poisoning carries
        // no information worth propagating.
        self.run_lock.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Runs inference, calling `callback` for every chunk the runtime produces.
    ///
    /// This blocks until the run finishes, fails, or the callback returns
    /// [`Control::Pause`]. The callback fires on this thread.
    ///
    /// Each [`Output`] borrows buffers the runtime owns and reuses, so it
    /// cannot escape the callback. Call [`Output::to_chunk`] to keep a copy.
    ///
    /// A panic inside the callback is caught, the run is stopped, and the panic
    /// resumes on this thread once the runtime has returned.
    ///
    /// ```no_run
    /// # use rkllm::{Control, InferParams, Input, RkllmSession, Result};
    /// # use rkllm_sys::RkllmApi;
    /// # fn f<A: RkllmApi>(session: &RkllmSession<A>) -> Result<()> {
    /// let mut input = Input::prompt("What colour is the sky?")?;
    /// session.run_llm(&mut input, &InferParams::new(), |out| {
    ///     print!("{}", out.text().unwrap_or(""));
    ///     Control::Continue
    /// })?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn run_llm<F>(&self, input: &mut Input, params: &InferParams, callback: F) -> Result<()>
    where
        F: FnMut(Output<'_>) -> Control,
    {
        let _guard = self.lock_run();

        let mut callback = callback;
        let mut state = CallbackState {
            callback: &mut callback,
            panic: None,
        };
        let mut raw_input = input.as_raw();
        let mut scratch = params.scratch();
        let mut raw_params = scratch.as_raw();
        let userdata = (&raw mut state).cast::<c_void>();

        // SAFETY: the handle is live, both raw structs point at storage owned
        // by `input`, `scratch` and `params`, all of which outlive the call,
        // and `userdata` is the `CallbackState` the trampoline expects.
        let code = unsafe {
            self.api
                .rkllm_run(self.handle, &mut raw_input, &mut raw_params, userdata)
        };

        if let Some(panic) = state.panic {
            resume_unwind(panic);
        }
        Error::check("rkllm_run", code)
    }

    /// Stops the run currently in flight, if there is one.
    ///
    /// Safe to call from another thread while [`RkllmSession::run_llm`] blocks,
    /// which is the point of it.
    pub fn abort(&self) -> Result<()> {
        // SAFETY: the handle is live for as long as `&self` is.
        Error::check("rkllm_abort", unsafe { self.api.rkllm_abort(self.handle) })
    }

    /// Whether a run is in flight.
    pub fn is_running(&self) -> bool {
        // SAFETY: the handle is live for as long as `&self` is.
        unsafe { self.api.rkllm_is_running(self.handle) == 1 }
    }

    /// Loads a LoRA adapter, which later runs can select by name.
    pub fn load_lora(
        &self,
        path: impl Into<Vec<u8>>,
        name: impl Into<Vec<u8>>,
        scale: f32,
    ) -> Result<()> {
        let path = CString::new(path).map_err(|e| Error::nul("lora_adapter_path", &e))?;
        let name = CString::new(name).map_err(|e| Error::nul("lora_adapter_name", &e))?;
        let mut adapter = RKLLMLoraAdapter {
            lora_adapter_path: path.as_ptr(),
            lora_adapter_name: name.as_ptr(),
            scale,
        };
        let _guard = self.lock_run();
        // SAFETY: the handle is live and `adapter` outlives the call.
        let code = unsafe { self.api.rkllm_load_lora(self.handle, &mut adapter) };
        Error::check("rkllm_load_lora", code)
    }

    /// Loads a prompt cache from disk.
    pub fn load_prompt_cache(&self, path: impl Into<Vec<u8>>) -> Result<()> {
        let path = CString::new(path).map_err(|e| Error::nul("prompt_cache_path", &e))?;
        let _guard = self.lock_run();
        // SAFETY: the handle is live and `path` outlives the call.
        let code = unsafe { self.api.rkllm_load_prompt_cache(self.handle, path.as_ptr()) };
        Error::check("rkllm_load_prompt_cache", code)
    }

    /// Drops the loaded prompt cache.
    pub fn release_prompt_cache(&self) -> Result<()> {
        let _guard = self.lock_run();
        // SAFETY: the handle is live for as long as `&self` is.
        let code = unsafe { self.api.rkllm_release_prompt_cache(self.handle) };
        Error::check("rkllm_release_prompt_cache", code)
    }

    /// Clears the whole key-value cache, optionally keeping the system prompt.
    pub fn clear_kv_cache(&self, keep_system_prompt: bool) -> Result<()> {
        let _guard = self.lock_run();
        // SAFETY: the handle is live, and null position arrays mean "all of it".
        let code = unsafe {
            self.api.rkllm_clear_kv_cache(
                self.handle,
                c_int::from(keep_system_prompt),
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        Error::check("rkllm_clear_kv_cache", code)
    }

    /// Clears one half-open range of the key-value cache.
    ///
    /// # The runtime ignores this unless the run is paused
    ///
    /// A range only takes effect when the run was configured with
    /// [`InferParams::keep_history`] set to `false`
    /// and is currently suspended, which means the callback returned
    /// [`Control::Pause`]. Outside that, the call
    /// succeeds and changes nothing.
    ///
    /// Use [`RkllmSession::clear_kv_cache`] to clear the whole cache instead.
    pub fn clear_kv_cache_range(&self, start: c_int, end: c_int) -> Result<()> {
        let _guard = self.lock_run();
        // One entry each, because this crate runs one input at a time. See the
        // note on `Param` about why batching is not exposed.
        let mut start = [start];
        let mut end = [end];
        // SAFETY: the handle is live, and both arrays hold the one entry the
        // runtime reads for a batch size of one.
        let code = unsafe {
            self.api
                .rkllm_clear_kv_cache(self.handle, 0, start.as_mut_ptr(), end.as_mut_ptr())
        };
        Error::check("rkllm_clear_kv_cache", code)
    }

    /// Reads how many positions the key-value cache currently holds.
    pub fn kv_cache_size(&self) -> Result<c_int> {
        // Sized here rather than by the caller: a length shorter than the
        // runtime's batch size would have it write out of bounds.
        let mut sizes = [0 as c_int];
        let _guard = self.lock_run();
        // SAFETY: the handle is live, and the array holds the one entry the
        // runtime writes for a batch size of one.
        let code = unsafe {
            self.api
                .rkllm_get_kv_cache_size(self.handle, sizes.as_mut_ptr())
        };
        Error::check("rkllm_get_kv_cache_size", code)?;
        Ok(sizes[0])
    }

    /// Sets the chat template framing each turn.
    pub fn set_chat_template(
        &self,
        system_prompt: impl Into<Vec<u8>>,
        prompt_prefix: impl Into<Vec<u8>>,
        prompt_postfix: impl Into<Vec<u8>>,
    ) -> Result<()> {
        let system = CString::new(system_prompt).map_err(|e| Error::nul("system_prompt", &e))?;
        let prefix = CString::new(prompt_prefix).map_err(|e| Error::nul("prompt_prefix", &e))?;
        let postfix = CString::new(prompt_postfix).map_err(|e| Error::nul("prompt_postfix", &e))?;
        let _guard = self.lock_run();
        // SAFETY: the handle is live and all three strings outlive the call.
        let code = unsafe {
            self.api.rkllm_set_chat_template(
                self.handle,
                system.as_ptr(),
                prefix.as_ptr(),
                postfix.as_ptr(),
            )
        };
        Error::check("rkllm_set_chat_template", code)
    }

    /// Configures function calling, with the tools given as a JSON string.
    pub fn set_function_tools(
        &self,
        system_prompt: impl Into<Vec<u8>>,
        tools: impl Into<Vec<u8>>,
        tool_response_str: impl Into<Vec<u8>>,
    ) -> Result<()> {
        let system = CString::new(system_prompt).map_err(|e| Error::nul("system_prompt", &e))?;
        let tools = CString::new(tools).map_err(|e| Error::nul("tools", &e))?;
        let response =
            CString::new(tool_response_str).map_err(|e| Error::nul("tool_response_str", &e))?;
        let _guard = self.lock_run();
        // SAFETY: the handle is live and all three strings outlive the call.
        let code = unsafe {
            self.api.rkllm_set_function_tools(
                self.handle,
                system.as_ptr(),
                tools.as_ptr(),
                response.as_ptr(),
            )
        };
        Error::check("rkllm_set_function_tools", code)
    }
}

/// Sessions over symbols the linker resolved.
#[cfg(feature = "link")]
impl RkllmSession<RkllmStatic> {
    /// Loads a model, dispatching through the linked `librkllmrt`.
    ///
    /// ```no_run
    /// # use rkllm::{Param, RkllmSession, Result};
    /// # fn f() -> Result<()> {
    /// let param = Param::new("/data/qwen.rkllm")?.max_new_tokens(256);
    /// let session = RkllmSession::new(&param)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(param: &Param) -> Result<Self> {
        RkllmSession::with_api(RkllmStatic, param)
    }
}

/// Sessions over a runtime opened with `dlopen`.
#[cfg(feature = "libloading")]
impl RkllmSession<RkllmRuntime> {
    /// Opens the RKLLM shared library at `path`, then loads a model through it.
    ///
    /// `path` may be a full path or a bare name such as `librkllmrt.so`, which
    /// the dynamic loader resolves against the usual search path.
    /// [`rkllm_sys::LIBRARY_NAME`] holds that name.
    ///
    /// The library stays open for as long as the session lives. Opening the
    /// same library for several sessions is cheap, since the loader reference
    /// counts it rather than mapping it twice.
    ///
    /// ```no_run
    /// # use rkllm::{Param, RkllmSession, Result};
    /// # fn f() -> Result<()> {
    /// let param = Param::new("/data/qwen.rkllm")?.max_new_tokens(256);
    /// let session = RkllmSession::new_with_library(rkllm_sys::LIBRARY_NAME, &param)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_with_library(path: impl AsRef<Path>, param: &Param) -> Result<Self> {
        // SAFETY: loading a shared object runs its initializers. The caller
        // names the library, and is trusting it the same way they trust any
        // native dependency.
        let runtime = unsafe { RkllmRuntime::new(path.as_ref()) }?;
        RkllmSession::with_api(runtime, param)
    }
}

impl<A: RkllmApi> Drop for RkllmSession<A> {
    fn drop(&mut self) {
        // SAFETY: `&mut self` means no run can be in flight, and the handle is
        // not touched again.
        unsafe {
            let _ = self.api.rkllm_destroy(self.handle);
        }
    }
}

impl<A: RkllmApi> std::fmt::Debug for RkllmSession<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RkllmSession")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

/// What the trampoline finds behind the `userdata` pointer.
struct CallbackState<'a> {
    callback: &'a mut dyn FnMut(Output<'_>) -> Control,
    panic: Option<Box<dyn Any + Send>>,
}

/// The one C callback registered for every session, which finds the Rust
/// closure for the current run behind `userdata`.
unsafe extern "C" fn trampoline(
    result: *mut RKLLMResult,
    userdata: *mut c_void,
    state: LLMCallState,
) -> c_int {
    if userdata.is_null() {
        return Control::Pause.as_code();
    }

    // SAFETY: `run_llm` is the only caller that reaches here, and it always
    // passes a live `CallbackState` that outlives the run.
    let CallbackState { callback, panic } = unsafe { &mut *userdata.cast::<CallbackState<'_>>() };

    // A previous callback already panicked, so stop rather than run more code.
    if panic.is_some() {
        return Control::Pause.as_code();
    }

    // SAFETY: the runtime passes either null or a valid result that lives for
    // the duration of this call, which bounds the `Output` the closure sees.
    let output = Output::new(CallState::from(state), unsafe { result.as_ref() });

    match catch_unwind(AssertUnwindSafe(|| callback(output))) {
        Ok(control) => control.as_code(),
        Err(unwound) => {
            *panic = Some(unwound);
            Control::Pause.as_code()
        }
    }
}
