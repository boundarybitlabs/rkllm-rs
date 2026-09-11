//! Several prompts answered side by side in one batch.
//!
//! ```text
//! cargo run --release --example batch -- model.rkllm "Name a colour." "Name a fruit."
//! ```
//!
//! The runtime decodes every prompt in the same forward pass. Each callback
//! carries one output per prompt, and a prompt that has stopped generating
//! reports [`rkllm::Output::is_finished`] while the others keep going.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use clap::Parser;
use rkllm::{CallState, Control, InferParams, Input, Param, RkllmSession, find_library_path};

/// Answer several prompts at once on a Rockchip NPU.
#[derive(Debug, Parser)]
#[command(version)]
struct Args {
    /// Path to the .rkllm model file.
    model: String,

    /// The prompts. One batch entry each.
    #[arg(required = true)]
    prompts: Vec<String>,

    /// The RKLLM shared library to open. Defaults to the first one found.
    #[arg(long)]
    library: Option<PathBuf>,

    /// Tokens the context window holds.
    #[arg(long, default_value_t = 4096)]
    max_context_len: i32,

    /// Tokens to generate before stopping.
    #[arg(long, default_value_t = 128)]
    max_new_tokens: i32,
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
    let n_batch = u8::try_from(args.prompts.len())?;
    let library = match args.library {
        Some(path) => path,
        None => find_library_path()
            .next()
            .ok_or("no librkllmrt.so found; pass --library or set RKLLM_LIB")?,
    };

    let param = Param::new(args.model.as_str())?
        .max_context_len(args.max_context_len)
        .max_new_tokens(args.max_new_tokens)
        .n_batch(n_batch)?;

    eprintln!("loading {} via {}", args.model, library.display());
    let started = Instant::now();
    let session = RkllmSession::new_with_library(&library, &param)?;
    eprintln!(
        "loaded in {:.1}s, batch of {}",
        started.elapsed().as_secs_f32(),
        session.n_batch()
    );

    let mut inputs = args
        .prompts
        .iter()
        .map(|p| Input::prompt(p.as_str()))
        .collect::<Result<Vec<_>, _>>()?;

    for (entry, prompt) in args.prompts.iter().enumerate() {
        eprintln!("[{entry}] > {prompt}");
    }
    eprintln!();

    // Collected rather than printed as they arrive, since interleaving several
    // generations on one terminal is unreadable.
    let mut answers = vec![String::new(); inputs.len()];
    let mut finished = vec![false; inputs.len()];
    let mut perf = None;

    session.run_llm_batch(&mut inputs, &InferParams::new(), |outputs| {
        for (entry, output) in outputs.iter().enumerate() {
            if let Some(text) = output.text() {
                answers[entry].push_str(text);
            }
            if output.is_finished() && !finished[entry] {
                finished[entry] = true;
                eprintln!("[{entry}] finished");
            }
            if output.state() == CallState::Finish {
                perf = output.perf();
            }
        }
        Control::Continue
    })?;

    for (entry, answer) in answers.iter().enumerate() {
        println!("\n[{entry}] {}", answer.trim());
    }

    if let Some(perf) = perf {
        eprintln!(
            "\nprefill {} tokens in {:.0} ms, generate {} tokens in {:.0} ms, {:.0} MB",
            perf.prefill_tokens,
            perf.prefill_time_ms,
            perf.generate_tokens,
            perf.generate_time_ms,
            perf.memory_usage_mb
        );
    }
    Ok(())
}
