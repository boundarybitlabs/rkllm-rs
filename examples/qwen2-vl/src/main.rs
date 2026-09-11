//! Image question answering with Qwen2-VL on an RK3588 NPU.
//!
//! Qwen2-VL is two models. A vision encoder turns an image into embeddings, and
//! a language model consumes those alongside the text. They run on two
//! different Rockchip runtimes, so this example drives both:
//!
//! * the `.rknn` encoder through `rknpu2`, which binds `librknnrt`
//! * the `.rkllm` language model through `rkllm`, which binds `librkllmrt`
//!
//! RKLLM itself never touches the image. It receives a float buffer and the
//! tags that say where the image sits in the prompt.
//!
//! ```text
//! cargo run --release -- \
//!   --encoder Qwen2-VL-2B-Instruct.rknn \
//!   --llm Qwen2-VL-2B-Instruct.rkllm \
//!   --image photo.jpg \
//!   "What is in this picture?"
//! ```

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use image::imageops::FilterType;
use rkllm::{CallState, Control, ImageInput, InferParams, Input, Param, RkllmSession};
use rknpu2::api::RknnInitFlags;
use rknpu2::api::runtime::RuntimeAPI;
use rknpu2::io::buffer::{BufMutView, BufView};
use rknpu2::io::input::Input as RknnInput;
use rknpu2::io::output::{Output, OutputKind};
use rknpu2::query::{InputAttr, OutputAttr, TensorAttrView};
use rknpu2::tensor::{TensorFormat, TensorFormatKind};
use rknpu2::utils::find_rknn_library;
use rknpu2::{RKNN, rknpu2_sys};

/// The runtime looks for this in the prompt and replaces it with the tags
/// below. A multimodal run without it fails.
const IMAGE_PLACEHOLDER: &str = "<image>";

/// Qwen2-VL marks an image in the prompt with these, per its tokenizer.
const IMAGE_START: &str = "<|vision_start|>";
const IMAGE_END: &str = "<|vision_end|>";
const IMAGE_CONTENT: &str = "<|image_pad|>";

/// CLIP normalization, from Qwen2-VL's preprocessor config.
const MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
const STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

/// Ask Qwen2-VL about an image, with the vision and language halves on their
/// own runtimes.
#[derive(Debug, Parser)]
#[command(version)]
struct Args {
    /// The .rknn vision encoder.
    #[arg(long)]
    encoder: PathBuf,

    /// The .rkllm language model.
    #[arg(long)]
    llm: PathBuf,

    /// The image to ask about.
    #[arg(long)]
    image: PathBuf,

    /// The question. Omit it for a built-in one.
    prompt: Vec<String>,

    /// Skip CLIP normalization and hand the encoder plain 0..1 values.
    ///
    /// Right when the conversion baked the normalization into the model.
    #[arg(long)]
    no_normalize: bool,

    /// Print the encoder's tensor shapes and stop.
    #[arg(long)]
    describe: bool,

    /// Frame the prompt with the ChatML tags Qwen2-VL was trained on.
    #[arg(long)]
    chatml: bool,

    /// Tokens to generate before stopping.
    #[arg(long, default_value_t = 256)]
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

type Fallible<T> = Result<T, Box<dyn std::error::Error>>;

fn run(args: Args) -> Fallible<()> {
    let question = if args.prompt.is_empty() {
        "Describe this image in two or three sentences.".to_owned()
    } else {
        args.prompt.join(" ")
    };
    // Put the placeholder in front unless the question already places it.
    let prompt = if question.contains(IMAGE_PLACEHOLDER) {
        question
    } else {
        format!("{IMAGE_PLACEHOLDER}{question}")
    };

    let encoder = load_encoder(&args.encoder)?;
    let shape = EncoderShape::query(&encoder)?;
    eprintln!("{shape}");
    if args.describe {
        return Ok(());
    }

    let pixels = prepare_image(&args.image, &shape, !args.no_normalize)?;
    let embeddings = encode(&encoder, &pixels, &shape)?;
    eprintln!(
        "encoded {} into {} tokens of {} values",
        args.image.display(),
        shape.n_tokens,
        shape.embed_dim
    );

    let image = ImageInput::new(embeddings, 1, shape.n_tokens)?
        .tags(IMAGE_START, IMAGE_END, IMAGE_CONTENT)?
        .size(shape.width, shape.height);

    if image.embed_dim() != shape.embed_dim {
        return Err(format!(
            "the buffer implies an embedding width of {} but the encoder reports {}",
            image.embed_dim(),
            shape.embed_dim
        )
        .into());
    }

    let param = Param::new(args.llm.to_string_lossy().as_ref())?
        .max_context_len(4096)
        .max_new_tokens(args.max_new_tokens);

    eprintln!("loading {}", args.llm.display());
    let session = RkllmSession::new_with_library(
        rkllm::find_library_path()
            .next()
            .ok_or("no librkllmrt.so found")?,
        &param,
    )?;

    if args.chatml {
        session.set_chat_template(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
            "<|im_start|>user\n",
            "<|im_end|>\n<|im_start|>assistant\n",
        )?;
    }

    eprintln!("\n> {prompt}\n");

    let mut input = Input::multimodal(prompt.as_str(), image)?;
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

fn load_encoder(path: &PathBuf) -> Fallible<RKNN<RuntimeAPI>> {
    // The buffer is dropped when this returns, which is fine: without the
    // zero-copy init flag the runtime takes its own copy of the model.
    let mut model = std::fs::read(path)?;
    let library = find_rknn_library()
        .next()
        .ok_or("no librknnrt.so found; install the RKNN runtime")?;
    Ok(RKNN::new_with_library(
        library,
        &mut model,
        RknnInitFlags::empty(),
    )?)
}

/// What the encoder expects in and produces out.
struct EncoderShape {
    width: usize,
    height: usize,
    channels: usize,
    nhwc: bool,
    n_tokens: usize,
    embed_dim: usize,
}

impl EncoderShape {
    fn query(model: &RKNN<RuntimeAPI>) -> Fallible<Self> {
        let input: InputAttr = model.query_with_input(0)?;
        let output: OutputAttr = model.query_with_input(0)?;

        let dims = input.dims();
        let nhwc = matches!(input.format(), TensorFormatKind::NHWC(_));
        // NHWC is [n, h, w, c]; NCHW is [n, c, h, w].
        let (height, width, channels) = match (nhwc, dims.len()) {
            (true, 4) => (dims[1], dims[2], dims[3]),
            (false, 4) => (dims[2], dims[3], dims[1]),
            _ => return Err(format!("unexpected encoder input shape {dims:?}").into()),
        };

        // The embedding width is the last dimension; everything before it is
        // the token count.
        let out_dims = output.dims();
        let embed_dim = *out_dims.last().ok_or("encoder output has no dimensions")? as usize;
        let n_tokens = out_dims.iter().map(|d| *d as usize).product::<usize>() / embed_dim;

        Ok(EncoderShape {
            width: width as usize,
            height: height as usize,
            channels: channels as usize,
            nhwc,
            n_tokens,
            embed_dim,
        })
    }
}

impl std::fmt::Display for EncoderShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "encoder input {}x{}x{} ({}), output {} tokens of {}",
            self.width,
            self.height,
            self.channels,
            if self.nhwc { "NHWC" } else { "NCHW" },
            self.n_tokens,
            self.embed_dim
        )
    }
}

/// Resizes and normalizes the image into the layout the encoder wants.
fn prepare_image(path: &PathBuf, shape: &EncoderShape, normalize: bool) -> Fallible<Vec<f32>> {
    let img = image::open(path)?
        .resize_exact(
            shape.width as u32,
            shape.height as u32,
            FilterType::CatmullRom,
        )
        .to_rgb8();

    let value = |pixel: &image::Rgb<u8>, channel: usize| {
        let raw = f32::from(pixel[channel]) / 255.0;
        if normalize {
            (raw - MEAN[channel]) / STD[channel]
        } else {
            raw
        }
    };

    let mut out = Vec::with_capacity(shape.width * shape.height * shape.channels);
    if shape.nhwc {
        for y in 0..shape.height as u32 {
            for x in 0..shape.width as u32 {
                let pixel = img.get_pixel(x, y);
                for c in 0..shape.channels {
                    out.push(value(pixel, c));
                }
            }
        }
    } else {
        for c in 0..shape.channels {
            for y in 0..shape.height as u32 {
                for x in 0..shape.width as u32 {
                    out.push(value(img.get_pixel(x, y), c));
                }
            }
        }
    }
    Ok(out)
}

/// Runs the encoder and returns the embeddings as floats.
fn encode(model: &RKNN<RuntimeAPI>, pixels: &[f32], shape: &EncoderShape) -> Fallible<Vec<f32>> {
    let format = if shape.nhwc {
        TensorFormatKind::NHWC(rknpu2_sys::_rknn_tensor_format::RKNN_TENSOR_NHWC)
    } else {
        TensorFormatKind::NCHW(TensorFormat::NCHW)
    };

    // pass_through false lets the runtime quantize into whatever the model
    // wants, so this code does not have to know the tensor's dtype.
    model.set_inputs(RknnInput::new(0, BufView::F32(pixels), false, format))?;
    model.run()?;

    let mut embeddings = vec![0.0f32; shape.n_tokens * shape.embed_dim];
    model.get_outputs(&mut [Output {
        index: 0,
        kind: OutputKind::Preallocated {
            buf: BufMutView::F32(&mut embeddings),
            want_float: true,
        },
    }])?;
    Ok(embeddings)
}
