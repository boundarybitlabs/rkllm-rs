"""Dump a model's embedding table as flat little-endian f32.

A .rkllm exported without its embedding layer asks the caller for embeddings by
token id, so the caller needs the table the model was trained with. This writes
it in the simplest shape something else can read: row-major, vocab by hidden.

Supply the rows unscaled. The runtime applies whatever scaling the architecture
needs, so pre-scaling them produces silent nonsense.
"""

import sys

import numpy as np
import torch
from safetensors import safe_open

model_path, out_path = sys.argv[1], sys.argv[2]

with safe_open(model_path, "pt") as f:
    weight = f.get_tensor("model.embed_tokens.weight")

print(f"{tuple(weight.shape)} {weight.dtype}", flush=True)
array = weight.to(torch.float32).numpy().astype("<f4", copy=False)
assert array.flags["C_CONTIGUOUS"], "rows must be contiguous for a flat dump"
array.tofile(out_path)
print(f"wrote {array.size * 4} bytes to {out_path}", flush=True)
