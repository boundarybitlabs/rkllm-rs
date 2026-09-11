# Vendored RKLLM header

`rkllm.h` is copied verbatim from the Rockchip `rknn-llm` repository:

<https://github.com/airockchip/rknn-llm/blob/main/rkllm-runtime/Linux/librkllm_api/include/rkllm.h>

`wrapper.h` is ours. It exists because `rkllm.h` uses `size_t` while including
only `<cstdint>`, which does not declare it, so the header does not parse on its
own. The wrapper pulls in `<cstddef>` first and leaves `rkllm.h` untouched.

The header is C++ (`#include <cstdint>`, despite the `extern "C"` block), so
bindgen is run with `-x c++ -std=c++17`.

To refresh: replace `rkllm.h` from upstream, then run `../regen-bindings.sh`.
