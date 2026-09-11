//! Running a `.rkllm` model that was exported without its tokenizer.
//!
//! Most models carry their own. One exported with `export_tokenizer=False`
//! does not, and the runtime refuses to load it:
//!
//! ```text
//! E rkllm: Tokenizer is unavailable: model has no internal tokenizer and no
//!          external tokenizer_callback provided.
//! ```
//!
//! Supplying one through `SessionBuilder::tokenizer` is what makes that model
//! usable. This example uses the Hugging Face tokenizer loaded from the
//! original model's `tokenizer.json`.
//!
//! Decoding is the caller's job too. A model with no tokenizer cannot turn its
//! own output back into text, so the results carry token ids and an empty
//! `text`, and this example decodes them at the end.
//!
//! ```text
//! cargo run --release -- \
//!   --model minicpm4-no-tokenizer.rkllm \
//!   --tokenizer MiniCPM4-0.5B/tokenizer.json \
//!   "Name a colour."
//! ```

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use clap::Parser;
use rkllm::{CallState, Control, InferParams, Input, Param, SessionBuilder, find_library_path};
use tokenizers::Tokenizer;

/// Run a model whose tokenizer lives outside it.
#[derive(Debug, Parser)]
#[command(version)]
struct Args {
    /// The .rkllm model, exported without its tokenizer.
    #[arg(long)]
    model: String,

    /// The tokenizer.json to use in its place.
    #[arg(long)]
    tokenizer: PathBuf,

    /// The prompt.
    prompt: Vec<String>,

    /// The RKLLM shared library to open. Defaults to the first one found.
    #[arg(long)]
    library: Option<PathBuf>,

    /// Tokens to generate before stopping.
    #[arg(long, default_value_t = 64)]
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
    let prompt = if args.prompt.is_empty() {
        "Name a colour.".to_owned()
    } else {
        args.prompt.join(" ")
    };

    // The tokenizer crate's error is not Sized behind the trait object this
    // returns, so it is flattened into a message here.
    let tokenizer = Tokenizer::from_file(&args.tokenizer).map_err(|e| e.to_string())?;
    eprintln!(
        "loaded {} with a vocabulary of {}",
        args.tokenizer.display(),
        tokenizer.get_vocab_size(true)
    );

    let library = match args.library {
        Some(path) => path,
        None => find_library_path()
            .next()
            .ok_or("no librkllmrt.so found; pass --library or set RKLLM_LIB")?,
    };

    // Counted so the run can report whether the runtime actually used it.
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    let tokenizer = Arc::new(tokenizer);
    // The same tokenizer is needed after the run, to turn the generated ids
    // back into text.
    let decoder = Arc::clone(&tokenizer);

    let param = Param::new(args.model.as_str())?.max_new_tokens(args.max_new_tokens);

    eprintln!("loading {}", args.model);
    let session = SessionBuilder::new(&param)
        .tokenizer(move |text, out| {
            counter.fetch_add(1, Ordering::Relaxed);
            let encoded = tokenizer.encode(text, false).ok()?;
            let ids = encoded.get_ids();
            if ids.len() > out.len() {
                // The runtime's buffer is the bound, and saying so is better
                // than silently truncating the prompt.
                return None;
            }
            for (slot, id) in out.iter_mut().zip(ids) {
                *slot = i32::try_from(*id).ok()?;
            }
            Some(ids.len())
        })
        .open_library(&library)?;

    eprintln!("\n> {prompt}\n");

    let mut input = Input::prompt(prompt.as_str())?;
    let mut generated: Vec<u32> = Vec::new();
    let mut any_text = false;
    let mut perf = None;

    session.run_llm(&mut input, &InferParams::new(), |output| {
        if let Some(text) = output.text() {
            any_text = true;
            print!("{text}");
            let _ = std::io::stdout().flush();
        }
        // Without an internal tokenizer the runtime cannot detokenize either,
        // so the ids are what comes back.
        if let Some(id) = output.token_id().filter(|id| *id >= 0) {
            generated.push(id as u32);
        }
        if output.state() == CallState::Finish {
            perf = output.perf();
        }
        Control::Continue
    })?;

    if !any_text {
        let decoded = decoder
            .decode(&generated, true)
            .map_err(|e| e.to_string())?;
        println!("{decoded}");
    }
    println!();
    eprintln!(
        "the tokenizer callback ran {} times, for {} generated tokens",
        calls.load(Ordering::Relaxed),
        generated.len()
    );
    if let Some(perf) = perf {
        eprintln!(
            "prefill {} tokens in {:.0} ms, generate {} tokens in {:.0} ms",
            perf.prefill_tokens, perf.prefill_time_ms, perf.generate_tokens, perf.generate_time_ms
        );
    }
    Ok(())
}
