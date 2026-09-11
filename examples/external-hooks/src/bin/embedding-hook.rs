//! Running a `.rkllm` model that was exported without its embedding layer.
//!
//! A model exported with `export_embedding=False` has no embedding table, and
//! the runtime refuses to load it:
//!
//! ```text
//! E rkllm: Model has no embedding layer and no embed callback provided.
//! ```
//!
//! Supplying one through `SessionBuilder::embedding` is what makes it usable.
//! The table has to come from the original model. Dump it as flat
//! little-endian `f32`, row-major and vocabulary by hidden size, which is what
//! this reads.
//!
//! ```text
//! cargo run --release --bin embedding-hook -- \
//!   --model minicpm4-no-embedding.rkllm \
//!   --embeddings embed_tokens.f32 \
//!   --hidden-size 1024 \
//!   --scale 12 \
//!   "Name a colour."
//! ```
//!
//! # Supply the raw rows
//!
//! MiniCPM's architecture multiplies embeddings by `scale_emb`, which is 12 in
//! this model's config, and it would be reasonable to think the callback should
//! apply it. It should not. The runtime does its own scaling, and supplying
//! pre-scaled rows produces fluent nonsense rather than an error. That is what
//! `--scale` is for: the default of 1 is correct, and anything else shows what
//! the wrong answer looks like.

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use clap::Parser;
use rkllm::{CallState, Control, InferParams, Input, Param, SessionBuilder, find_library_path};

/// Run a model whose embedding table lives outside it.
#[derive(Debug, Parser)]
#[command(version)]
struct Args {
    /// The .rkllm model, exported without its embedding layer.
    #[arg(long)]
    model: String,

    /// Flat little-endian f32 embeddings, vocabulary by hidden size.
    #[arg(long)]
    embeddings: PathBuf,

    /// The model's hidden size, which is the width of one row.
    #[arg(long, default_value_t = 1024)]
    hidden_size: usize,

    /// Factor applied to every embedding row before handing it over.
    ///
    /// The default of 1 is correct. The runtime applies whatever scaling the
    /// architecture needs, so this exists to demonstrate that.
    #[arg(long, default_value_t = 1.0)]
    scale: f32,

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

    let table = Arc::new(EmbeddingTable::load(&args.embeddings, args.hidden_size)?);
    eprintln!(
        "loaded {} rows of {} from {}",
        table.rows(),
        args.hidden_size,
        args.embeddings.display()
    );

    let library = match args.library {
        Some(path) => path,
        None => find_library_path()
            .next()
            .ok_or("no librkllmrt.so found; pass --library or set RKLLM_LIB")?,
    };

    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    let scale = args.scale;

    let param = Param::new(args.model.as_str())?.max_new_tokens(args.max_new_tokens);

    eprintln!("loading {}", args.model);
    let session = SessionBuilder::new(&param)
        .embedding(move |tokens, out| {
            counter.fetch_add(1, Ordering::Relaxed);
            // The runtime sizes the buffer, so this is the width it wants.
            let width = match out.len().checked_div(tokens.len()) {
                Some(width) if width * tokens.len() == out.len() => width,
                _ => return false,
            };
            if width != table.width {
                return false;
            }
            for (token, slot) in tokens.iter().zip(out.chunks_mut(width)) {
                match table.row(*token) {
                    Some(row) => {
                        for (out, value) in slot.iter_mut().zip(row) {
                            *out = value * scale;
                        }
                    }
                    None => return false,
                }
            }
            true
        })
        .open_library(&library)?;

    eprintln!("\n> {prompt}\n");

    let mut input = Input::prompt(prompt.as_str())?;
    let mut stdout = std::io::stdout();
    let mut perf = None;

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
    eprintln!(
        "\nthe embedding callback ran {} times",
        calls.load(Ordering::Relaxed)
    );
    if let Some(perf) = perf {
        eprintln!(
            "prefill {} tokens in {:.0} ms, generate {} tokens in {:.0} ms",
            perf.prefill_tokens, perf.prefill_time_ms, perf.generate_tokens, perf.generate_time_ms
        );
    }
    Ok(())
}

/// The model's embedding table, read from a flat f32 dump.
struct EmbeddingTable {
    values: Vec<f32>,
    width: usize,
}

impl EmbeddingTable {
    fn load(path: &PathBuf, width: usize) -> Result<Self, Box<dyn std::error::Error>> {
        if width == 0 {
            return Err("a hidden size of zero has no rows".into());
        }
        let bytes = std::fs::read(path)?;
        if !bytes.len().is_multiple_of(size_of::<f32>() * width) {
            return Err(format!(
                "{} does not divide into rows of {width} f32 values",
                path.display()
            )
            .into());
        }
        let values = bytes
            .as_chunks::<{ size_of::<f32>() }>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        Ok(EmbeddingTable { values, width })
    }

    fn rows(&self) -> usize {
        self.values.len() / self.width
    }

    fn row(&self, token: i32) -> Option<&[f32]> {
        let start = usize::try_from(token).ok()?.checked_mul(self.width)?;
        self.values.get(start..start + self.width)
    }
}
