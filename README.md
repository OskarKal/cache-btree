# High-Frequency Trading & Cache-Optimized Data Structures (`inf/hft`)

This repository contains high-performance, low-latency C++ implementations of cache-optimized data structures tailored for high-frequency trading (HFT) and systems programming environments. 

## Overview of Components

- **`benchmark_both.cpp`**: 
  - Comprehensive benchmarking suite comparing B+ tree performance across uniform, skewed, and Zipfian distributions with metrics for insertions, point lookups, updates, and range searches. Results output to `benchmark_results.csv` and `benchmark_comparison.csv`.
- **`temp.cpp` / `temp.s`**:
  - Experimental scratchpads and assembly outputs for studying generated instructions and micro-optimizations.

## Build and Run

To compile and execute the benchmark suite:
```bash
g++ -O3 -std=c++17 benchmark_both.cpp -o benchmark_both
./benchmark_both
```

## Testing Status
Currently, validation is driven primarily through the benchmark suite (`benchmark_both.cpp`). A formalized test suite (e.g., GoogleTest or Catch2) is recommended for production hardening.
