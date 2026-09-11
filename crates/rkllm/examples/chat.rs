//! One prompt, one answer, streamed to stdout as the NPU produces it.
//!
//! Build and run on an RK3588 board with `librkllmrt.so` installed:
//!
//! ```text
//! cargo run --release --example chat -- model.rkllm "Why is the sky blue?"
//! ```
//!
//! The runtime is opened with `dlopen` at startup, so nothing needs to be
//! linked at build time.

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use clap::Parser;
use rkllm::find_library_path;
use rkllm::{CallState, Control, InferParams, Input, Param, PerfStat, RkllmSession};

const DEFAULT_PROMPT: &str = "Explain who Napoleon Bonaparte is in two or three sentences.";

/// Run one prompt against a .rkllm model on a Rockchip NPU.
#[derive(Debug, Parser)]
#[command(version)]
struct Args {
    /// Path to the .rkllm model file.
    model: String,

    /// The prompt. Omit it for a built-in one.
    prompt: Vec<String>,

    /// Frame the prompt with ChatML tags.
    ///
    /// Leave this off unless you need it. A converted model usually carries its
    /// own template, and the runtime warns that setting one here turns the
    /// built-in parsing off, thinking mode included.
    #[arg(long)]
    chatml: bool,

    /// The RKLLM shared library to open.
    ///
    /// Defaults to the first one found, which honours RKLLM_LIB and
    /// RKLLM_LIB_DIR before falling back to the usual system directories.
    #[arg(long)]
    library: Option<PathBuf>,

    /// Tokens the context window holds.
    #[arg(long, default_value_t = 4096)]
    max_context_len: i32,

    /// Tokens to generate before stopping.
    #[arg(long, default_value_t = 256)]
    max_new_tokens: i32,

    /// Sampling temperature.
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    /// Top-K sampling cutoff.
    #[arg(long, default_value_t = 40)]
    top_k: i32,

    /// Top-P, or nucleus, sampling cutoff.
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,
}

fn main() -> ExitCode {
    match run(Args::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let prompt = if args.prompt.is_empty() {
        DEFAULT_PROMPT.to_owned()
    } else {
        args.prompt.join(" ")
    };

    let library = match args.library {
        Some(path) => path,
        None => find_library_path()
            .next()
            .ok_or("no librkllmrt.so found; pass --library or set RKLLM_LIB")?,
    };

    let param = Param::new(args.model.as_str())?
        .max_context_len(args.max_context_len)
        .max_new_tokens(args.max_new_tokens)
        .temperature(args.temperature)
        .top_k(args.top_k)
        .top_p(args.top_p);

    eprintln!("loading {} via {}", args.model, library.display());
    let started = Instant::now();
    let session = RkllmSession::new_with_library(&library, &param)?;
    eprintln!("loaded in {:.1}s", started.elapsed().as_secs_f32());

    if args.chatml {
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
