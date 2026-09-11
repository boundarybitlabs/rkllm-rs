//! Generated bindings to `librkllmrt`, in two interchangeable flavours.
//!
//! * [`link`] resolves the symbols at link time, against a `librkllmrt` the
//!   linker can see. Enabled by the default `link` feature.
//! * [`runtime`] resolves them at run time with `dlopen`, through `libloading`.
//!   Enabled by the `libloading` feature.
//!
//! Both expose the same surface through the [`RkllmApi`] trait, so code can be
//! written once and run against either. The C types in [`types`] are shared.

use std::os::raw::{c_char, c_int, c_void};

pub mod types;

#[cfg(feature = "link")]
pub mod link;

#[cfg(feature = "libloading")]
pub mod runtime;

pub use types::*;

#[cfg(feature = "libloading")]
pub use runtime::RkllmRuntime;

/// The full `librkllmrt` C API, as implemented by both binding flavours.
///
/// Method names, argument order and types mirror `rkllm.h` exactly. This is a
/// `-sys` level trait: it performs no validation, and every method is `unsafe`
/// for the same reasons the underlying C function is.
///
/// # Safety
///
/// An implementor must dispatch each method to the corresponding `librkllmrt`
/// symbol without altering its arguments or return value.
#[allow(non_snake_case)]
pub unsafe trait RkllmApi {
    /// See [`rkllm_createDefaultParam`](link::rkllm_createDefaultParam).
    ///
    /// # Safety
    /// Calls into `librkllmrt`.
    unsafe fn rkllm_createDefaultParam(&self) -> RKLLMParam;

    /// Initializes a model instance and writes its handle to `handle`.
    ///
    /// # Safety
    /// `handle`, `param` and `callback` must be valid, non-null, aligned
    /// pointers to initialized values for the duration of the call.
    unsafe fn rkllm_init(
        &self,
        handle: *mut LLMHandle,
        param: *mut RKLLMParam,
        callback: *mut RKLLMCallback,
    ) -> c_int;

    /// Loads a LoRA adapter into the model.
    ///
    /// # Safety
    /// `handle` must come from a successful `rkllm_init` and not yet have been
    /// destroyed. `lora_adapter` must be a valid pointer to an initialized value.
    unsafe fn rkllm_load_lora(
        &self,
        handle: LLMHandle,
        lora_adapter: *mut RKLLMLoraAdapter,
    ) -> c_int;

    /// Loads a prompt cache from a file.
    ///
    /// # Safety
    /// `handle` must be live. `prompt_cache_path` must be a valid pointer to a
    /// NUL-terminated string.
    unsafe fn rkllm_load_prompt_cache(
        &self,
        handle: LLMHandle,
        prompt_cache_path: *const c_char,
    ) -> c_int;

    /// Releases the loaded prompt cache.
    ///
    /// # Safety
    /// `handle` must be live.
    unsafe fn rkllm_release_prompt_cache(&self, handle: LLMHandle) -> c_int;

    /// Destroys the model instance and frees its resources.
    ///
    /// # Safety
    /// `handle` must be live, and must not be used again after this returns.
    unsafe fn rkllm_destroy(&self, handle: LLMHandle) -> c_int;

    /// Runs inference synchronously, reporting results through the callback.
    ///
    /// # Safety
    /// `handle` must be live. `rkllm_input` and `rkllm_infer_params` must be
    /// valid pointers to initialized values that stay valid for the call, and
    /// any pointers they carry must satisfy the same requirement. `userdata` is
    /// passed to the callback untouched and may be null.
    unsafe fn rkllm_run(
        &self,
        handle: LLMHandle,
        rkllm_input: *mut RKLLMInput,
        rkllm_infer_params: *mut RKLLMInferParam,
        userdata: *mut c_void,
    ) -> c_int;

    /// Runs inference asynchronously.
    ///
    /// # Safety
    /// As [`RkllmApi::rkllm_run`], and additionally every pointer reachable
    /// from the arguments must stay valid until the run finishes or is aborted,
    /// since the call returns before the work does.
    unsafe fn rkllm_run_async(
        &self,
        handle: LLMHandle,
        rkllm_input: *mut RKLLMInput,
        rkllm_infer_params: *mut RKLLMInferParam,
        userdata: *mut c_void,
    ) -> c_int;

    /// Aborts the task currently running on `handle`.
    ///
    /// # Safety
    /// `handle` must be live.
    unsafe fn rkllm_abort(&self, handle: LLMHandle) -> c_int;

    /// Returns 1 while a task is running on `handle`.
    ///
    /// # Safety
    /// `handle` must be live.
    unsafe fn rkllm_is_running(&self, handle: LLMHandle) -> c_int;

    /// Clears all or part of the key-value cache.
    ///
    /// # Safety
    /// `handle` must be live. `start_pos` and `end_pos` are either both null or
    /// both valid arrays of at least `n_batch` elements.
    unsafe fn rkllm_clear_kv_cache(
        &self,
        handle: LLMHandle,
        keep_system_prompt: c_int,
        start_pos: *mut c_int,
        end_pos: *mut c_int,
    ) -> c_int;

    /// Writes the per-batch key-value cache sizes into `cache_sizes`.
    ///
    /// # Safety
    /// `handle` must be live. `cache_sizes` must point to a writable array of
    /// at least `n_batch` elements.
    unsafe fn rkllm_get_kv_cache_size(&self, handle: LLMHandle, cache_sizes: *mut c_int) -> c_int;

    /// Sets the chat template used to frame user input.
    ///
    /// # Safety
    /// `handle` must be live. Each string pointer must be null or point to a
    /// NUL-terminated string.
    unsafe fn rkllm_set_chat_template(
        &self,
        handle: LLMHandle,
        system_prompt: *const c_char,
        prompt_prefix: *const c_char,
        prompt_postfix: *const c_char,
    ) -> c_int;

    /// Configures function calling, with tools given as a JSON string.
    ///
    /// # Safety
    /// `handle` must be live. Each string pointer must be null or point to a
    /// NUL-terminated string.
    unsafe fn rkllm_set_function_tools(
        &self,
        handle: LLMHandle,
        system_prompt: *const c_char,
        tools: *const c_char,
        tool_response_str: *const c_char,
    ) -> c_int;

    /// Sets the cross-attention inputs for the decoder.
    ///
    /// # Safety
    /// `handle` must be live. `cross_attn_params` must be a valid pointer to an
    /// initialized value, and the buffers it points at must outlive the call.
    unsafe fn rkllm_set_cross_attn_params(
        &self,
        handle: LLMHandle,
        cross_attn_params: *mut RKLLMCrossAttnParam,
    ) -> c_int;
}

/// The link-time API: a zero-sized handle onto the symbols the linker resolved.
///
/// Available with the default `link` feature.
#[cfg(feature = "link")]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Linked;

#[cfg(feature = "link")]
impl Linked {
    /// Returns the link-time API handle.
    pub const fn new() -> Self {
        Self
    }
}

#[cfg(feature = "link")]
#[allow(non_snake_case)]
unsafe impl RkllmApi for Linked {
    unsafe fn rkllm_createDefaultParam(&self) -> RKLLMParam {
        unsafe { link::rkllm_createDefaultParam() }
    }

    unsafe fn rkllm_init(
        &self,
        handle: *mut LLMHandle,
        param: *mut RKLLMParam,
        callback: *mut RKLLMCallback,
    ) -> c_int {
        unsafe { link::rkllm_init(handle, param, callback) }
    }

    unsafe fn rkllm_load_lora(
        &self,
        handle: LLMHandle,
        lora_adapter: *mut RKLLMLoraAdapter,
    ) -> c_int {
        unsafe { link::rkllm_load_lora(handle, lora_adapter) }
    }

    unsafe fn rkllm_load_prompt_cache(
        &self,
        handle: LLMHandle,
        prompt_cache_path: *const c_char,
    ) -> c_int {
        unsafe { link::rkllm_load_prompt_cache(handle, prompt_cache_path) }
    }

    unsafe fn rkllm_release_prompt_cache(&self, handle: LLMHandle) -> c_int {
        unsafe { link::rkllm_release_prompt_cache(handle) }
    }

    unsafe fn rkllm_destroy(&self, handle: LLMHandle) -> c_int {
        unsafe { link::rkllm_destroy(handle) }
    }

    unsafe fn rkllm_run(
        &self,
        handle: LLMHandle,
        rkllm_input: *mut RKLLMInput,
        rkllm_infer_params: *mut RKLLMInferParam,
        userdata: *mut c_void,
    ) -> c_int {
        unsafe { link::rkllm_run(handle, rkllm_input, rkllm_infer_params, userdata) }
    }

    unsafe fn rkllm_run_async(
        &self,
        handle: LLMHandle,
        rkllm_input: *mut RKLLMInput,
        rkllm_infer_params: *mut RKLLMInferParam,
        userdata: *mut c_void,
    ) -> c_int {
        unsafe { link::rkllm_run_async(handle, rkllm_input, rkllm_infer_params, userdata) }
    }

    unsafe fn rkllm_abort(&self, handle: LLMHandle) -> c_int {
        unsafe { link::rkllm_abort(handle) }
    }

    unsafe fn rkllm_is_running(&self, handle: LLMHandle) -> c_int {
        unsafe { link::rkllm_is_running(handle) }
    }

    unsafe fn rkllm_clear_kv_cache(
        &self,
        handle: LLMHandle,
        keep_system_prompt: c_int,
        start_pos: *mut c_int,
        end_pos: *mut c_int,
    ) -> c_int {
        unsafe { link::rkllm_clear_kv_cache(handle, keep_system_prompt, start_pos, end_pos) }
    }

    unsafe fn rkllm_get_kv_cache_size(&self, handle: LLMHandle, cache_sizes: *mut c_int) -> c_int {
        unsafe { link::rkllm_get_kv_cache_size(handle, cache_sizes) }
    }

    unsafe fn rkllm_set_chat_template(
        &self,
        handle: LLMHandle,
        system_prompt: *const c_char,
        prompt_prefix: *const c_char,
        prompt_postfix: *const c_char,
    ) -> c_int {
        unsafe {
            link::rkllm_set_chat_template(handle, system_prompt, prompt_prefix, prompt_postfix)
        }
    }

    unsafe fn rkllm_set_function_tools(
        &self,
        handle: LLMHandle,
        system_prompt: *const c_char,
        tools: *const c_char,
        tool_response_str: *const c_char,
    ) -> c_int {
        unsafe { link::rkllm_set_function_tools(handle, system_prompt, tools, tool_response_str) }
    }

    unsafe fn rkllm_set_cross_attn_params(
        &self,
        handle: LLMHandle,
        cross_attn_params: *mut RKLLMCrossAttnParam,
    ) -> c_int {
        unsafe { link::rkllm_set_cross_attn_params(handle, cross_attn_params) }
    }
}

#[cfg(feature = "libloading")]
#[allow(non_snake_case)]
unsafe impl RkllmApi for RkllmRuntime {
    unsafe fn rkllm_createDefaultParam(&self) -> RKLLMParam {
        unsafe { RkllmRuntime::rkllm_createDefaultParam(self) }
    }

    unsafe fn rkllm_init(
        &self,
        handle: *mut LLMHandle,
        param: *mut RKLLMParam,
        callback: *mut RKLLMCallback,
    ) -> c_int {
        unsafe { RkllmRuntime::rkllm_init(self, handle, param, callback) }
    }

    unsafe fn rkllm_load_lora(
        &self,
        handle: LLMHandle,
        lora_adapter: *mut RKLLMLoraAdapter,
    ) -> c_int {
        unsafe { RkllmRuntime::rkllm_load_lora(self, handle, lora_adapter) }
    }

    unsafe fn rkllm_load_prompt_cache(
        &self,
        handle: LLMHandle,
        prompt_cache_path: *const c_char,
    ) -> c_int {
        unsafe { RkllmRuntime::rkllm_load_prompt_cache(self, handle, prompt_cache_path) }
    }

    unsafe fn rkllm_release_prompt_cache(&self, handle: LLMHandle) -> c_int {
        unsafe { RkllmRuntime::rkllm_release_prompt_cache(self, handle) }
    }

    unsafe fn rkllm_destroy(&self, handle: LLMHandle) -> c_int {
        unsafe { RkllmRuntime::rkllm_destroy(self, handle) }
    }

    unsafe fn rkllm_run(
        &self,
        handle: LLMHandle,
        rkllm_input: *mut RKLLMInput,
        rkllm_infer_params: *mut RKLLMInferParam,
        userdata: *mut c_void,
    ) -> c_int {
        unsafe { RkllmRuntime::rkllm_run(self, handle, rkllm_input, rkllm_infer_params, userdata) }
    }

    unsafe fn rkllm_run_async(
        &self,
        handle: LLMHandle,
        rkllm_input: *mut RKLLMInput,
        rkllm_infer_params: *mut RKLLMInferParam,
        userdata: *mut c_void,
    ) -> c_int {
        unsafe {
            RkllmRuntime::rkllm_run_async(self, handle, rkllm_input, rkllm_infer_params, userdata)
        }
    }

    unsafe fn rkllm_abort(&self, handle: LLMHandle) -> c_int {
        unsafe { RkllmRuntime::rkllm_abort(self, handle) }
    }

    unsafe fn rkllm_is_running(&self, handle: LLMHandle) -> c_int {
        unsafe { RkllmRuntime::rkllm_is_running(self, handle) }
    }

    unsafe fn rkllm_clear_kv_cache(
        &self,
        handle: LLMHandle,
        keep_system_prompt: c_int,
        start_pos: *mut c_int,
        end_pos: *mut c_int,
    ) -> c_int {
        unsafe {
            RkllmRuntime::rkllm_clear_kv_cache(self, handle, keep_system_prompt, start_pos, end_pos)
        }
    }

    unsafe fn rkllm_get_kv_cache_size(&self, handle: LLMHandle, cache_sizes: *mut c_int) -> c_int {
        unsafe { RkllmRuntime::rkllm_get_kv_cache_size(self, handle, cache_sizes) }
    }

    unsafe fn rkllm_set_chat_template(
        &self,
        handle: LLMHandle,
        system_prompt: *const c_char,
        prompt_prefix: *const c_char,
        prompt_postfix: *const c_char,
    ) -> c_int {
        unsafe {
            RkllmRuntime::rkllm_set_chat_template(
                self,
                handle,
                system_prompt,
                prompt_prefix,
                prompt_postfix,
            )
        }
    }

    unsafe fn rkllm_set_function_tools(
        &self,
        handle: LLMHandle,
        system_prompt: *const c_char,
        tools: *const c_char,
        tool_response_str: *const c_char,
    ) -> c_int {
        unsafe {
            RkllmRuntime::rkllm_set_function_tools(
                self,
                handle,
                system_prompt,
                tools,
                tool_response_str,
            )
        }
    }

    unsafe fn rkllm_set_cross_attn_params(
        &self,
        handle: LLMHandle,
        cross_attn_params: *mut RKLLMCrossAttnParam,
    ) -> c_int {
        unsafe { RkllmRuntime::rkllm_set_cross_attn_params(self, handle, cross_attn_params) }
    }
}

/// Forwards every [`RkllmApi`] method to `**self`.
macro_rules! forward_rkllm_api {
    () => {
        unsafe fn rkllm_createDefaultParam(&self) -> RKLLMParam {
            unsafe { (**self).rkllm_createDefaultParam() }
        }

        unsafe fn rkllm_init(
            &self,
            handle: *mut LLMHandle,
            param: *mut RKLLMParam,
            callback: *mut RKLLMCallback,
        ) -> c_int {
            unsafe { (**self).rkllm_init(handle, param, callback) }
        }

        unsafe fn rkllm_load_lora(
            &self,
            handle: LLMHandle,
            lora_adapter: *mut RKLLMLoraAdapter,
        ) -> c_int {
            unsafe { (**self).rkllm_load_lora(handle, lora_adapter) }
        }

        unsafe fn rkllm_load_prompt_cache(
            &self,
            handle: LLMHandle,
            prompt_cache_path: *const c_char,
        ) -> c_int {
            unsafe { (**self).rkllm_load_prompt_cache(handle, prompt_cache_path) }
        }

        unsafe fn rkllm_release_prompt_cache(&self, handle: LLMHandle) -> c_int {
            unsafe { (**self).rkllm_release_prompt_cache(handle) }
        }

        unsafe fn rkllm_destroy(&self, handle: LLMHandle) -> c_int {
            unsafe { (**self).rkllm_destroy(handle) }
        }

        unsafe fn rkllm_run(
            &self,
            handle: LLMHandle,
            rkllm_input: *mut RKLLMInput,
            rkllm_infer_params: *mut RKLLMInferParam,
            userdata: *mut c_void,
        ) -> c_int {
            unsafe { (**self).rkllm_run(handle, rkllm_input, rkllm_infer_params, userdata) }
        }

        unsafe fn rkllm_run_async(
            &self,
            handle: LLMHandle,
            rkllm_input: *mut RKLLMInput,
            rkllm_infer_params: *mut RKLLMInferParam,
            userdata: *mut c_void,
        ) -> c_int {
            unsafe { (**self).rkllm_run_async(handle, rkllm_input, rkllm_infer_params, userdata) }
        }

        unsafe fn rkllm_abort(&self, handle: LLMHandle) -> c_int {
            unsafe { (**self).rkllm_abort(handle) }
        }

        unsafe fn rkllm_is_running(&self, handle: LLMHandle) -> c_int {
            unsafe { (**self).rkllm_is_running(handle) }
        }

        unsafe fn rkllm_clear_kv_cache(
            &self,
            handle: LLMHandle,
            keep_system_prompt: c_int,
            start_pos: *mut c_int,
            end_pos: *mut c_int,
        ) -> c_int {
            unsafe { (**self).rkllm_clear_kv_cache(handle, keep_system_prompt, start_pos, end_pos) }
        }

        unsafe fn rkllm_get_kv_cache_size(
            &self,
            handle: LLMHandle,
            cache_sizes: *mut c_int,
        ) -> c_int {
            unsafe { (**self).rkllm_get_kv_cache_size(handle, cache_sizes) }
        }

        unsafe fn rkllm_set_chat_template(
            &self,
            handle: LLMHandle,
            system_prompt: *const c_char,
            prompt_prefix: *const c_char,
            prompt_postfix: *const c_char,
        ) -> c_int {
            unsafe {
                (**self).rkllm_set_chat_template(
                    handle,
                    system_prompt,
                    prompt_prefix,
                    prompt_postfix,
                )
            }
        }

        unsafe fn rkllm_set_function_tools(
            &self,
            handle: LLMHandle,
            system_prompt: *const c_char,
            tools: *const c_char,
            tool_response_str: *const c_char,
        ) -> c_int {
            unsafe {
                (**self).rkllm_set_function_tools(handle, system_prompt, tools, tool_response_str)
            }
        }

        unsafe fn rkllm_set_cross_attn_params(
            &self,
            handle: LLMHandle,
            cross_attn_params: *mut RKLLMCrossAttnParam,
        ) -> c_int {
            unsafe { (**self).rkllm_set_cross_attn_params(handle, cross_attn_params) }
        }
    };
}

// SAFETY: each impl only dereferences to the inner `RkllmApi` and forwards.
#[allow(non_snake_case)]
unsafe impl<T: RkllmApi + ?Sized> RkllmApi for &T {
    forward_rkllm_api!();
}

#[allow(non_snake_case)]
unsafe impl<T: RkllmApi + ?Sized> RkllmApi for std::boxed::Box<T> {
    forward_rkllm_api!();
}

#[allow(non_snake_case)]
unsafe impl<T: RkllmApi + ?Sized> RkllmApi for std::sync::Arc<T> {
    forward_rkllm_api!();
}
