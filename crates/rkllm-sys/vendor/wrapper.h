// rkllm.h uses size_t but only includes <cstdint>, which does not declare it.
// Pull in <cstddef> first so the vendored header parses unmodified.
#include <cstddef>
#include <cstdint>

#include "rkllm.h"
