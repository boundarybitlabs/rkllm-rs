# qwen2-vl-example

Image question answering with Qwen2-VL on an RK3588 NPU.

Qwen2-VL is two models. A vision encoder turns an image into embeddings, and a
language model consumes those alongside the text. They run on two different
Rockchip runtimes, so this example drives both:

| Half | Runtime | Crate |
| --- | --- | --- |
| `.rknn` vision encoder | `librknnrt` | [`rknpu2`](https://github.com/boundarybitlabs/rknpu2-rs) |
| `.rkllm` language model | `librkllmrt` | `rkllm` |

RKLLM never touches the image. It receives a float buffer and the tags that say
where the image sits in the prompt.

This package is excluded from the workspace, because it needs a second NPU
runtime and a git dependency that the library crates do not. Build it from this
directory.

## Requirements

- An RK3588 board with both `librknnrt.so` and `librkllmrt.so` installed.
- The two model files, for example from
  [thanhtantran/Qwen2-VL-2B-RKLLM](https://huggingface.co/thanhtantran/Qwen2-VL-2B-RKLLM),
  which ships the encoder and the language model together.

## Running

```sh
cargo run --release -- \
  --encoder Qwen2-VL-2B-Instruct.rknn \
  --llm Qwen2-VL-2B-Instruct.rkllm \
  --image photo.jpg \
  "What is in this picture?"
```

`--describe` prints the encoder's tensor shapes and stops, which is the quickest
way to see what a given `.rknn` expects.

The runtime requires a literal `<image>` placeholder in the prompt, which it
replaces with the tags above. This example prepends one when the question does
not already contain it.

## Verified

On an Orange Pi 5 Max with `librkllmrt` 1.3.0 and NPU driver 0.9.8, against the
Qwen2-VL-2B files linked above:

| | |
| --- | --- |
| Encoder | 392x392x3 NHWC in, 196 tokens of 1536 out |
| Prefill | 224 tokens, about 1.1 s |
| Generate | about 14 tokens/s |
| Resident | 3.2 GB, both models loaded at once |

Asked to list what it saw in a photo of a street, it answered with the people,
the building, the bus in the background and the dark lighting. Different images
produce different and accurate descriptions, which is what shows the embeddings
are really reaching the language model.

Answers are short unless the prompt asks for more. That is the model, not the
plumbing.

## Preprocessing

The image is resized to whatever the encoder reports as its input shape, then
normalized with the CLIP mean and standard deviation from Qwen2-VL's
preprocessor config. `--no-normalize` skips that step, which is correct when the
normalization was baked into the model at conversion time.

Nothing here is hardcoded to a particular encoder. The input and output shapes,
the channel layout, and the token and embedding counts all come from querying
the `.rknn` at load time.

## The invariant that cannot be checked

RKLLM reads `n_image * n_image_tokens * embed_dim` floats from the buffer. The C
API does not carry `embed_dim`, so neither `rkllm` nor the runtime can verify
the buffer against the model that will consume it. This example compares
`ImageInput::embed_dim` against the encoder's reported output width and refuses
to continue when they disagree, which is the only check available.
