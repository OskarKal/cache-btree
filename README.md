# Cache-line-sized B+ tree with SIMD node search

A small C++17 library built around `hft::CacheTree`, an insertion-only B+ tree
whose node layout is sized to one cache line, plus runtime-dispatched SIMD
search kernels. This is an experimental systems-programming project. It makes
no claims about production readiness, trading suitability, tail latency, or
speedups over other data structures.

## Overview

- **`hft::CacheTree`** — insertion-only B+ tree. Inserting an existing key
  updates its value. Keys and values are signed 32-bit integers.
- **Cache-line-sized nodes** — node capacity is derived at compile time from
  the configured line size: `(line_bytes - 8) / 8`. A 32-byte line holds 3
  keys, 64 bytes holds 7, and 128 bytes holds 15.
- **Runtime SIMD dispatch** — scalar, SSE2, and AVX2 node-search kernels are
  selected at runtime via `__builtin_cpu_supports`, with a scalar fallback.
  Comparisons use signed `int32_t` ordering.
- **Configurable target** — `HFT_CACHE_LINE_SIZE` is fixed when CMake
  configures the build. It is detected on native Linux or supplied explicitly
  for cross-compilation.

Public API (`include/hft/cache_tree.hpp`):

```cpp
namespace hft {
class CacheTree {
public:
    explicit CacheTree(Backend backend = Backend::Auto);

    void insert(std::int32_t key, std::int32_t value);           // upsert
    std::optional<std::int32_t> find(std::int32_t key) const;
    std::vector<std::int32_t> range_search(std::int32_t start,
                                           std::int32_t max_count) const;
    std::size_t size() const;
    Backend backend() const;
    bool validate() const;  // structural invariants and alignment
};
}
```

## Layout

The intended integrated layout is:

```
CMakeLists.txt
cmake/config.hpp.in
include/hft/cache_tree.hpp
include/hft/simd_search.hpp
src/cache_tree.cpp
src/simd_search.cpp
tests/test_cache_tree.cpp
benchmark_both.cpp

ordinalB++tree.h       experimental reference
```

`include/hft/config.hpp` is generated into the build tree at
`generated/hft/config.hpp` during CMake configuration and defines
`HFT_CACHE_LINE_SIZE`. The source tree is assembled from feature branches, so
an individual branch may contain only a subset of these paths.

## Build

Requirements: CMake 3.16 or newer and a C++17 compiler. Automatic cache-line
detection is native-Linux only; cross-compilation requires an explicit value.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Targets:

- `hft_cache` — the library.
- `hft_tests` — the CTest suite.
- `hft_bench` — the benchmark.

## Tests

`tests/test_cache_tree.cpp` is a dependency-free CTest suite. It compares the
tree differentially against `std::map`, checks structural invariants through
`validate()`, exercises forced backends, and covers boundary keys and range
edges.

```bash
ctest --test-dir build --output-on-failure
```

Verified on this machine (Intel i5-10210U, 64-byte L1 line, GCC 16.2.1):
`1/1` test passed. Separate builds with `-DHFT_CACHE_LINE_SIZE=32` and
`-DHFT_CACHE_LINE_SIZE=128`, and a build with
`-DHFT_ENABLE_SANITIZERS=ON`, also passed. These results are specific to this
machine and toolchain, not a permanent guarantee.

## Benchmark

`hft_bench` compares the scalar, SSE2, and AVX2 backends on identical,
deterministically generated workloads. Correctness is verified before timing
and again after updates; unavailable backends are skipped.

```bash
./build/hft_bench --output results.csv \
  --keys 20000 --lookups 40000 --ranges 2000 --updates 2000 --runs 3 --seed 42
```

Usage:

```
hft_bench --output PATH [--distribution uniform|zipf] [--keys N]
          [--lookups N] [--ranges N] [--updates N] [--runs N] [--seed N]
          [--build-flags TEXT]
```

The CSV written to `--output` records machine and build metadata (CPU,
compiler, supplied build flags, cache-line size, SSE2/AVX2 availability),
workload parameters, backend, run number, per-operation timings, and a
checksum. The file is created exclusively; the benchmark refuses to overwrite
an existing path. Timings are per-machine observations, not performance claims.

## Configuration

| Option | Meaning |
| --- | --- |
| `HFT_CACHE_LINE_SIZE` | Empty (default) detects the native Linux L1 data/unified line size. An explicit byte count is required when cross-compiling. Must be a power of two between 32 and 268435456 and supported by the compiler's alignment. |
| `HFT_ENABLE_SANITIZERS` | Builds with AddressSanitizer and UndefinedBehaviorSanitizer (GCC/Clang only). |
| `HFT_BUILD_BENCHMARK` | Builds `hft_bench` when its sources are present (default `ON`). |

The cache-line size is a compile-time choice: it determines node layout and is
baked into the generated header and binary. A binary must be reconfigured and
rebuilt for a different target line size. SIMD support, by contrast, is
selected at runtime, with scalar fallback when a vector backend is unavailable.

## Limitations

- Insertion-only: there is no erase, rebalancing policy beyond splits, or
  concurrency support.
- Keys and values are signed 32-bit integers; the implementation is
  single-threaded.
- Correctness is covered by the test suite on the tested configurations; there
  is no fuzzing campaign or formal verification.
- No latency, throughput, or suitability claims are made for any workload.
- The cache-line size is compile-time; rebuild for each target. SIMD
  availability depends on the compiler, architecture, CPU, and OS-enabled
  register state.
- The ordinal tree is not part of the maintained library and should not be
  relied upon.
