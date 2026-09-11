//! The same model, driven through the async API instead of a closure.
//!
//! ```text
//! cargo run --release --features tokio --example stream -- model.rkllm "Why is the sky blue?"
//! ```
//!
//! `run_llm_async` puts the blocking run on its own thread and hands back a
//! `Stream`, so the rest of the program stays responsive while the NPU works.
//! Dropping the stream cancels the run, which is what `--stop-after` shows.

use std::io::Write;
use std::process::ExitCode;
use std::sync::Arc;

use clap::Parser;
use futures_util::StreamExt as _;
use rkllm::{CallState, InferParams, Input, Param, RkllmSession};
use rkllm_sys::{LIBRARY_NAME, RkllmRuntime};

const DEFAULT_PROMPT: &str = "Explain who Napoleon Bonaparte is in two or three sentences.";

/// Stream one prompt against a .rkllm model on a Rockchip NPU.
#[derive(Debug, Parser)]
#[command(version)]
struct Args {
    /// Path to the .rkllm model file.
    model: String,

    /// The prompt. Omit it for a built-in one.
    prompt: Vec<String>,

    /// Drop the stream after this many chunks, cancelling the run.
    #[arg(long, value_name = "N")]
    stop_after: Option<usize>,

    /// The RKLLM shared library to open.
    #[arg(long, env = "RKLLM_LIB", default_value = LIBRARY_NAME)]
    library: String,

    /// Tokens the context window holds.
    #[arg(long, default_value_t = 4096)]
    max_context_len: i32,

    /// Tokens to generate before stopping.
    #[arg(long, default_value_t = 256)]
    max_new_tokens: i32,

    /// Sampling temperature.
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,
}

#[tokio::main]
async fn main() -> ExitCode {
    match run(Args::parse()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let prompt = if args.prompt.is_empty() {
        DEFAULT_PROMPT.to_owned()
    } else {
        args.prompt.join(" ")
    };

    eprintln!("loading {}", args.library);
    // SAFETY: loading a shared object runs its initializers. This one is the
    // RKLLM runtime, whose symbols the bindings were generated from.
    let runtime = unsafe { RkllmRuntime::new(&args.library) }?;

    let param = Param::new(&runtime, args.model.as_str())?
        .max_context_len(args.max_context_len)
        .max_new_tokens(args.max_new_tokens)
        .temperature(args.temperature);

    eprintln!("loading {}", args.model);
    let session = Arc::new(RkllmSession::new(runtime, &param)?);
    eprintln!("\n> {prompt}\n");

    let mut stream = session.run_llm_async(Input::prompt(prompt.as_str())?, InferParams::new());
    let mut stdout = std::io::stdout();
    let mut chunks = 0usize;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        print!("{}", chunk.text());
        let _ = stdout.flush();
        chunks += 1;

        if chunk.state == CallState::Finish
            && let Some(perf) = chunk.perf
        {
            eprintln!(
                "\n\n{} tokens in {:.0} ms",
                perf.generate_tokens, perf.generate_time_ms
            );
        }

        if args.stop_after.is_some_and(|limit| chunks >= limit) {
            // Dropping the stream is the cancellation: the next chunk the
            // runtime produces finds no receiver, and the run stops there.
            drop(stream);
            eprintln!("\n\ncancelled after {chunks} chunks");
            return Ok(());
        }
    }

    println!();
    Ok(())
}
