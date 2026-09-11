//! One prompt, one answer, streamed to stdout as the NPU produces it.
//!
//! Build and run on an RK3588 board with `librkllmrt.so` installed:
//!
//! ```text
//! cargo run --release --example chat -- model.rkllm "Why is the sky blue?"
//! ```
//!
//! Pass `--chatml` to frame the prompt with the ChatML tags that MiniCPM4 and
//! the Qwen family expect. Leave it off unless you need it. A converted model
//! usually carries its own template, and the runtime warns that setting one
//! here turns the built-in parsing off, thinking mode included.

use std::io::Write;
use std::process::ExitCode;
use std::time::Instant;

use rkllm::{CallState, Control, InferParams, Input, Param, PerfStat, RkllmSession};
use rkllm_sys::Linked;

const DEFAULT_PROMPT: &str = "Explain who Napoleon Bonaparte is in two or three sentences.";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut chatml = false;
    let mut positional = Vec::new();
    for arg in std::env::args().skip(1) {
        if arg == "--chatml" {
            chatml = true;
        } else {
            positional.push(arg);
        }
    }

    let Some((model_path, prompt)) = positional.split_first() else {
        eprintln!("usage: chat [--chatml] <model.rkllm> [prompt ...]");
        return Err("no model path given".into());
    };
    let prompt = if prompt.is_empty() {
        DEFAULT_PROMPT.to_owned()
    } else {
        prompt.join(" ")
    };

    let param = Param::new(&Linked, model_path.as_str())?
        .max_context_len(4096)
        .max_new_tokens(256)
        .temperature(0.7)
        .top_k(40)
        .top_p(0.9);

    eprintln!("loading {model_path}");
    let started = Instant::now();
    let session = RkllmSession::new(Linked, &param)?;
    eprintln!("loaded in {:.1}s", started.elapsed().as_secs_f32());

    if chatml {
        session.set_chat_template(
            "",
            "<|im_start|>user\n",
            "<|im_end|>\n<|im_start|>assistant\n",
        )?;
    }

    eprintln!("\n> {prompt}\n");

    let mut input = Input::prompt(prompt.as_str())?;
    let mut perf = None;
    let mut stdout = std::io::stdout();

    session.run_llm(&mut input, &InferParams::new(), |output| {
        if let Some(text) = output.text() {
            print!("{text}");
            let _ = stdout.flush();
        }
        if output.state() == CallState::Finish {
            perf = output.perf();
        }
        Control::Continue
    })?;

    println!();
    if let Some(perf) = perf {
        report(&perf);
    }
    Ok(())
}

fn report(perf: &PerfStat) {
    let rate = |tokens: i32, ms: f32| {
        if ms > 0.0 {
            f64::from(tokens) / f64::from(ms) * 1000.0
        } else {
            0.0
        }
    };
    eprintln!();
    eprintln!(
        "prefill   {:>5} tokens  {:>8.1} ms  {:>6.1} tok/s",
        perf.prefill_tokens,
        perf.prefill_time_ms,
        rate(perf.prefill_tokens, perf.prefill_time_ms)
    );
    eprintln!(
        "generate  {:>5} tokens  {:>8.1} ms  {:>6.1} tok/s",
        perf.generate_tokens,
        perf.generate_time_ms,
        rate(perf.generate_tokens, perf.generate_time_ms)
    );
    eprintln!("memory    {:>5.0} MB", perf.memory_usage_mb);
}
