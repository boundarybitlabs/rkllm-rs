# tools

Scripts for building the models that exercise the RKLLM callbacks.

No published `.rkllm` needs the tokenizer or embedding callbacks, because a
normally exported model carries both. Testing them means converting a model
with the pieces left out, which the RKLLM toolkit's `export_rkllm` supports
through `export_tokenizer` and `export_embedding`.

## Setup

The toolkit is an x86_64 Linux wheel, shipped in the upstream repository rather
than on PyPI. It is not needed to build or use this crate, only to make these
test models.

```sh
python3 -m venv venv
./venv/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
./venv/bin/pip install "transformers==5.8.0" "numpy<=1.26.4" "datasets==4.1.1" \
    "pyarrow==21.0.0" safetensors "sentencepiece==0.2.0" accelerate tqdm \
    "Jinja2==3.1.4" "protobuf<=4.25.4" "tiktoken==0.9.0" colorlog "einops==0.4.1" \
    scipy "tabulate==0.9.0" flatbuffers easydict addict jsonlines
./venv/bin/pip install --no-deps rkllm_toolkit-1.3.0-cp312-cp312-linux_x86_64.whl
```

The wheel declares `auto_gptq`, `optimum`, `matplotlib`, `jsonschema` and
`datamodel_code_generator` as well. They are only needed for the GPTQ and
plotting paths, and `auto_gptq` wants CUDA to build, so they are left out.

## Converting

```sh
./venv/bin/python convert-test-models.py /path/to/MiniCPM4-0.5B ./out
```

Writes three files from one build, so a difference in behaviour cannot be
blamed on the conversion:

| File | Exported with | Needs |
| --- | --- | --- |
| `minicpm4-normal.rkllm` | everything | nothing |
| `minicpm4-no-tokenizer.rkllm` | `export_tokenizer=False` | a tokenizer callback |
| `minicpm4-no-embedding.rkllm` | `export_embedding=False` | an embedding callback |

On a 16-core machine with no GPU this takes a few minutes and about 11 GB of
memory.

## The embedding table

A model without its embedding layer needs the table supplied by token id. This
dumps it from the original weights as flat little-endian `f32`, row-major and
vocabulary by hidden size:

```sh
./venv/bin/python dump-embeddings.py /path/to/MiniCPM4-0.5B/model.safetensors embed_tokens.f32
```

For MiniCPM4-0.5B that is 73448 by 1024, about 287 MB.

Supply those rows unscaled. See `examples/external-hooks` for why.
