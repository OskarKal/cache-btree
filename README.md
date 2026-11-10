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
  Comparisons use signed `int32_t` ordering. Each vector kernel issues a masked
  full-width load, so AVX2 still vectorizes a 64-byte node (7 keys, 8 lanes):
  lanes at or beyond the node's key count are masked out, and the load stays
  inside the node object.
- **Configurable target** — `HFT_CACHE_LINE_SIZE` is fixed when CMake
  configures the build. It is detected on native Linux or supplied explicitly
  for cross-compilation. `HFT_SIMD` sets a compile-time SIMD ceiling
  (`SCALAR`, `SSE2`, `AVX2`, or `AUTO`); runtime detection still picks among
  the compiled backends.

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

Repository layout:

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
`HFT_CACHE_LINE_SIZE`. The `integration` branch contains all of these paths;
other branches may contain only a subset.

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
edges. It also tests the search primitive directly on standalone sorted arrays
of lengths 0..33 for every available backend, including padded `room` arrays
whose extra lanes must be masked; this executes the SSE2 and AVX2 vector loops
even when a tree node holds fewer keys than a vector. `validate()` is checked
against deliberately corrupted trees, so it cannot pass vacuously. The suite
asserts that `Auto` resolves to the highest available backend and that forced
unavailable backends are rejected.

```bash
ctest --test-dir build --output-on-failure
```

Verified on this machine (Intel i5-10210U, 64-byte L1 line, GCC 16.2.1): tests
passed at `HFT_CACHE_LINE_SIZE` 32, 64, and 128; with `HFT_SIMD` set to
`SCALAR` and `SSE2`; and with `-DHFT_ENABLE_SANITIZERS=ON`. Sanitizer failures
abort (UBSan is built with `-fno-sanitize-recover=all`), so they fail CTest.
These results are specific to this machine and toolchain, not a permanent
guarantee.

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
hft_bench --output PATH [--distribution uniform|zipf]
          [--backend all|scalar|sse2|avx2] [--keys N] [--lookups N]
          [--ranges N] [--updates N] [--runs N] [--seed N]
          [--build-flags TEXT]
```

`--help` prints the same summary. Inserts always cover every `--keys` key
exactly once, so tree size is the same for both distributions; `--distribution`
biases lookups, ranges, and updates. Zipf uses a fixed exponent `alpha = 1.5`.
`--backend` selects one kernel (default `all`), which is useful with external
profilers. `--seed` accepts 0. The output path is opened exclusively before the
timed runs, so an existing path or missing parent directory fails immediately
and is never overwritten.

The CSV written to `--output` records machine and build metadata (CPU,
compiler, supplied build flags, cache-line size, SSE2/AVX2 availability),
workload parameters, backend, run number, phase-total timings in milliseconds,
and a checksum. Backend order rotates between runs to reduce drift bias.
Timings are per-machine observations, not performance claims; whether a backend
vectorizes depends on the node capacity and the number of keys in the node.

## Configuration

| Option | Meaning |
| --- | --- |
| `HFT_CACHE_LINE_SIZE` | Empty (default) detects the native Linux L1 data/unified line size. An explicit byte count is required when cross-compiling. Must be a power of two between 32 and 268435456 and supported by the compiler's alignment. |
| `HFT_ENABLE_SANITIZERS` | Builds with AddressSanitizer and UndefinedBehaviorSanitizer (GCC/Clang only). UBSan is non-recoverable, so findings fail the test run. |
| `HFT_SIMD` | Compile-time SIMD ceiling: `AUTO`/`AVX2` keep SSE2 and AVX2, `SSE2` disables AVX2, `SCALAR` disables both. Runtime detection still chooses among the compiled backends. Default `AUTO`. |
| `HFT_BUILD_BENCHMARK` | Builds `hft_bench` when its sources are present (default `ON`). |

The cache-line size is a compile-time choice: it determines node layout and is
baked into the generated header and binary. A binary must be reconfigured and
rebuilt for a different target line size. SIMD has both a compile-time ceiling
(`HFT_SIMD`) and runtime selection, with scalar fallback when a vector backend
is unavailable on the CPU or was compiled out.

## Limitations

- Insertion-only: there is no erase, rebalancing policy beyond splits, or
  concurrency support.
- Keys and values are signed 32-bit integers; the implementation is
  single-threaded.
- Correctness is covered by the test suite on the tested configurations; there
  is no fuzzing campaign or formal verification.
- No latency, throughput, or suitability claims are made for any workload.
- The cache-line size is compile-time; rebuild for each target. SIMD
  availability depends on the compiler, architecture, `HFT_SIMD` ceiling, CPU,
  and OS-enabled register state.
- The ordinal tree is not part of the maintained library and should not be
  relied upon.
