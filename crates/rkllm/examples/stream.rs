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

use futures_util::StreamExt as _;
use rkllm::{CallState, InferParams, Input, Param, RkllmSession};
use rkllm_sys::{LIBRARY_NAME, RkllmRuntime};

const DEFAULT_PROMPT: &str = "Explain who Napoleon Bonaparte is in two or three sentences.";

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut stop_after: Option<usize> = None;
    let mut positional = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--stop-after" {
            let n = args.next().ok_or("--stop-after needs a chunk count")?;
            stop_after = Some(n.parse()?);
        } else {
            positional.push(arg);
        }
    }

    let Some((model_path, prompt)) = positional.split_first() else {
        eprintln!("usage: stream [--stop-after N] <model.rkllm> [prompt ...]");
        return Err("no model path given".into());
    };
    let prompt = if prompt.is_empty() {
        DEFAULT_PROMPT.to_owned()
    } else {
        prompt.join(" ")
    };

    let library = std::env::var("RKLLM_LIB").unwrap_or_else(|_| LIBRARY_NAME.to_owned());
    eprintln!("loading {library}");
    // SAFETY: loading a shared object runs its initializers. This one is the
    // RKLLM runtime, whose symbols the bindings were generated from.
    let runtime = unsafe { RkllmRuntime::new(&library) }?;

    let param = Param::new(&runtime, model_path.as_str())?
        .max_context_len(4096)
        .max_new_tokens(256)
        .temperature(0.7);

    eprintln!("loading {model_path}");
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
        if stop_after.is_some_and(|limit| chunks >= limit) {
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
