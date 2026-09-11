# external-hooks-example

Running `.rkllm` models that were exported without their tokenizer or their
embedding layer.

Most models carry both, and the runtime uses them. A model exported without one
asks the caller to supply it, and refuses to load otherwise:

```text
E rkllm: Tokenizer is unavailable: model has no internal tokenizer and no
         external tokenizer_callback provided.
E rkllm: Model has no embedding layer and no embed callback provided.
```

`SessionBuilder` is how those get supplied. This package has one binary for
each.

This package is excluded from the workspace, because the Hugging Face tokenizer
crate is a heavier dependency than the library crates should carry. Build it
from this directory.

## Making the models

No published model needs these callbacks, so the test models have to be made.
The RKLLM toolkit's `export_rkllm` takes the two flags that leave the pieces
out. See `tools/` at the repository root for the scripts, which convert
MiniCPM4-0.5B three ways: normal, without the tokenizer, and without the
embedding layer.

## Tokenizer

```sh
cargo run --release --bin tokenizer-hook -- \
  --model minicpm4-no-tokenizer.rkllm \
  --tokenizer MiniCPM4-0.5B/tokenizer.json \
  "Name a colour."
```

Decoding is the caller's job too. A model with no tokenizer cannot turn its own
output back into text, so the results carry token ids and an empty `text`, and
this binary decodes them at the end.

## Embeddings

```sh
cargo run --release --bin embedding-hook -- \
  --model minicpm4-no-embedding.rkllm \
  --embeddings embed_tokens.f32 \
  --hidden-size 1024 \
  "Name a colour."
```

The table is the original model's, dumped as flat little-endian `f32`,
row-major and vocabulary by hidden size.

**Supply the raw rows.** MiniCPM's architecture multiplies embeddings by
`scale_emb`, which is 12 for this model, and it would be reasonable to think the
callback should apply it. It should not: the runtime does its own scaling. With
`--scale 12` the model still runs, and still returns timings, and produces
nothing but blank lines. That failure is silent, which is the reason to say so
here.

## Verified

On an Orange Pi 5 Max with `librkllmrt` 1.3.0 and NPU driver 0.9.8, against
MiniCPM4-0.5B converted locally with toolkit 1.3.0. Both models load only when
their callback is supplied, and the runtime says which it is using:

```text
I rkllm: Model has no tokenizer, but tokenizer_callback is provided, so it will
         be used for tokenization.
I rkllm: Model has no embedding layer, but embed_callback is provided, so it
         will be used for embedding.
```
