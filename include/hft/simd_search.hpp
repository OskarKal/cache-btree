#ifndef HFT_SIMD_SEARCH_HPP
#define HFT_SIMD_SEARCH_HPP

#include <cstddef>
#include <cstdint>

namespace hft {

enum class Backend { Auto, Scalar, SSE2, AVX2 };

// Auto and Scalar are always available. Availability of SIMD backends depends
// on the compiler, target architecture, CPU, and OS-enabled register state.
bool backend_available(Backend backend);

// Resolve Auto to the best available backend. An unavailable or invalid forced
// backend throws std::invalid_argument. Resolve once when constructing a tree.
Backend resolve_backend(Backend backend);

// Returns "unknown" for an invalid enumerator.
const char* backend_name(Backend backend);

// keys[0..cnt) must be initialized, sorted in signed int32_t order and
// accessible. keys may be null when cnt is zero. Returns cnt if no match exists.
// These functions accept Auto or a resolved backend; forcing an unavailable or
// invalid backend throws std::invalid_argument rather than executing it.
std::size_t lower_bound(const std::int32_t* keys, std::size_t cnt,
                        std::int32_t key, Backend backend);
std::size_t upper_bound(const std::int32_t* keys, std::size_t cnt,
                        std::int32_t key, Backend backend);

}  // namespace hft

#endif  // HFT_SIMD_SEARCH_HPP
