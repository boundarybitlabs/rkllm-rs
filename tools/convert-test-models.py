"""Convert MiniCPM4-0.5B three ways: normal, without the tokenizer, and
without the embedding layer.

The last two are what make the RKLLM tokenizer and embedding callbacks fire.
Nothing else does, which is why an ordinary model cannot exercise them.
"""

import sys
import time

from rkllm.api import RKLLM

model_dir, out_dir = sys.argv[1], sys.argv[2]


def step(label, fn):
    started = time.time()
    print(f"--- {label}", flush=True)
    result = fn()
    print(f"--- {label}: {result!r} in {time.time() - started:.1f}s", flush=True)
    if result not in (0, None):
        sys.exit(f"{label} failed: {result!r}")
    return result


llm = RKLLM()

step("load", lambda: llm.load_huggingface(model=model_dir, device="cpu", dtype="float32"))

step(
    "build",
    lambda: llm.build(
        do_quantization=True,
        optimization_level=1,
        quantized_dtype="w8a8",
        quantized_algorithm="normal",
        target_platform="rk3588",
        num_npu_core=3,
        max_context=4096,
    ),
)

# A control, converted by the same toolkit as the other two so a difference in
# behaviour cannot be blamed on the conversion.
step(
    "export normal",
    lambda: llm.export_rkllm(
        f"{out_dir}/minicpm4-normal.rkllm", export_tokenizer=True, export_embedding=True
    ),
)

step(
    "export without tokenizer",
    lambda: llm.export_rkllm(
        f"{out_dir}/minicpm4-no-tokenizer.rkllm",
        export_tokenizer=False,
        export_embedding=True,
    ),
)

step(
    "export without embedding",
    lambda: llm.export_rkllm(
        f"{out_dir}/minicpm4-no-embedding.rkllm",
        export_tokenizer=True,
        export_embedding=False,
    ),
)

print("done", flush=True)
