#include "hft/simd_search.hpp"

#include <stdexcept>
#include <string>

#if defined(HFT_SIMD_SCALAR_ONLY)
#define HFT_SIMD_USE_SSE2 0
#define HFT_SIMD_USE_AVX2 0
#elif defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))
// GCC and Clang both provide GNU-style function targeting. Clang defines
// __GNUC__ too. AVX needs OS-enabled XMM/YMM state, SSE2 needs XMM state.
#define HFT_SIMD_USE_SSE2 1
#define HFT_SIMD_USE_AVX2 1
#define HFT_SIMD_TARGET(isa) __attribute__((target(isa)))
#include <immintrin.h>
#else
#define HFT_SIMD_USE_SSE2 0
#define HFT_SIMD_USE_AVX2 0
#endif

namespace hft {
namespace {

std::size_t scalar_lower_bound(const std::int32_t* keys, std::size_t cnt,
                               std::int32_t key) {
    std::size_t first = 0;
    while (first < cnt) {
        const std::size_t middle = first + (cnt - first) / 2;
        if (keys[middle] < key) {
            first = middle + 1;
        } else {
            cnt = middle;
        }
    }
    return first;
}

std::size_t scalar_upper_bound(const std::int32_t* keys, std::size_t cnt,
                               std::int32_t key) {
    std::size_t first = 0;
    while (first < cnt) {
        const std::size_t middle = first + (cnt - first) / 2;
        if (key < keys[middle]) {
            cnt = middle;
        } else {
            first = middle + 1;
        }
    }
    return first;
}

#if HFT_SIMD_USE_SSE2

bool cpu_supports_sse2() {
    __builtin_cpu_init();
#if defined(__x86_64__)
    // The x86-64 ABI requires OS support for SSE2 register state.
    return __builtin_cpu_supports("sse2");
#else
    // On 32-bit x86, conservatively require OS-enabled AVX state as proof
    // that XMM state is saved as well; this may skip SSE2 on older systems.
    return __builtin_cpu_supports("sse2") && __builtin_cpu_supports("avx");
#endif
}

bool cpu_os_support_avx2() {
    __builtin_cpu_init();
    // GCC/Clang CPU feature detection includes OSXSAVE/XCR0 checks for AVX.
    return __builtin_cpu_supports("avx2");
}

HFT_SIMD_TARGET("sse2")
std::size_t sse2_lower_bound(const std::int32_t* keys, std::size_t cnt,
                             std::int32_t key) {
    const __m128i needle = _mm_set1_epi32(key);
    std::size_t i = 0;
    for (; cnt - i >= 4; i += 4) {
        const __m128i values = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(keys + i));
        const int mask = _mm_movemask_ps(
            _mm_castsi128_ps(_mm_cmpgt_epi32(needle, values)));
        if (mask != 15) {
            return i + static_cast<std::size_t>(__builtin_ctz(~mask));
        }
    }
    for (; i < cnt && keys[i] < key; ++i) {
    }
    return i;
}

HFT_SIMD_TARGET("sse2")
std::size_t sse2_upper_bound(const std::int32_t* keys, std::size_t cnt,
                             std::int32_t key) {
    const __m128i needle = _mm_set1_epi32(key);
    std::size_t i = 0;
    for (; cnt - i >= 4; i += 4) {
        const __m128i values = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(keys + i));
        const int mask = _mm_movemask_ps(
            _mm_castsi128_ps(_mm_cmpgt_epi32(values, needle)));
        if (mask != 0) {
            return i + static_cast<std::size_t>(__builtin_ctz(mask));
        }
    }
    for (; i < cnt && keys[i] <= key; ++i) {
    }
    return i;
}

#endif  // HFT_SIMD_USE_SSE2

#if HFT_SIMD_USE_AVX2

HFT_SIMD_TARGET("avx2")
std::size_t avx2_lower_bound(const std::int32_t* keys, std::size_t cnt,
                             std::int32_t key) {
    const __m256i needle = _mm256_set1_epi32(key);
    std::size_t i = 0;
    for (; cnt - i >= 8; i += 8) {
        const __m256i values =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(keys + i));
        const int mask = _mm256_movemask_ps(
            _mm256_castsi256_ps(_mm256_cmpgt_epi32(needle, values)));
        if (mask != 255) {
            return i + static_cast<std::size_t>(__builtin_ctz(~mask));
        }
    }
    for (; i < cnt && keys[i] < key; ++i) {
    }
    return i;
}

HFT_SIMD_TARGET("avx2")
std::size_t avx2_upper_bound(const std::int32_t* keys, std::size_t cnt,
                             std::int32_t key) {
    const __m256i needle = _mm256_set1_epi32(key);
    std::size_t i = 0;
    for (; cnt - i >= 8; i += 8) {
        const __m256i values =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(keys + i));
        const int mask = _mm256_movemask_ps(
            _mm256_castsi256_ps(_mm256_cmpgt_epi32(values, needle)));
        if (mask != 0) {
            return i + static_cast<std::size_t>(__builtin_ctz(mask));
        }
    }
    for (; i < cnt && keys[i] <= key; ++i) {
    }
    return i;
}

#endif  // HFT_SIMD_USE_AVX2

bool supported(Backend backend) {
    switch (backend) {
        case Backend::Auto:
        case Backend::Scalar:
            return true;
        case Backend::SSE2:
#if HFT_SIMD_USE_SSE2
            {
                static const bool available = cpu_supports_sse2();
                return available;
            }
#else
            return false;
#endif
        case Backend::AVX2:
#if HFT_SIMD_USE_AVX2
            {
                static const bool available = cpu_os_support_avx2();
                return available;
            }
#else
            return false;
#endif
    }
    return false;
}

[[noreturn]] void unsupported(Backend backend) {
    throw std::invalid_argument("hft: requested search backend is unavailable: " +
                                std::string(backend_name(backend)));
}

std::size_t dispatch_lower(const std::int32_t* keys, std::size_t cnt,
                           std::int32_t key, Backend backend) {
    switch (backend) {
        case Backend::Scalar:
            return scalar_lower_bound(keys, cnt, key);
#if HFT_SIMD_USE_SSE2
        case Backend::SSE2:
            return sse2_lower_bound(keys, cnt, key);
#endif
#if HFT_SIMD_USE_AVX2
        case Backend::AVX2:
            return avx2_lower_bound(keys, cnt, key);
#endif
        default:
            unsupported(backend);
    }
}

std::size_t dispatch_upper(const std::int32_t* keys, std::size_t cnt,
                           std::int32_t key, Backend backend) {
    switch (backend) {
        case Backend::Scalar:
            return scalar_upper_bound(keys, cnt, key);
#if HFT_SIMD_USE_SSE2
        case Backend::SSE2:
            return sse2_upper_bound(keys, cnt, key);
#endif
#if HFT_SIMD_USE_AVX2
        case Backend::AVX2:
            return avx2_upper_bound(keys, cnt, key);
#endif
        default:
            unsupported(backend);
    }
}

}  // namespace

bool backend_available(Backend backend) { return supported(backend); }

Backend resolve_backend(Backend backend) {
    if (backend == Backend::Auto) {
#if HFT_SIMD_USE_AVX2
        if (backend_available(Backend::AVX2)) {
            return Backend::AVX2;
        }
#endif
#if HFT_SIMD_USE_SSE2
        if (backend_available(Backend::SSE2)) {
            return Backend::SSE2;
        }
#endif
        return Backend::Scalar;
    }
    if (!supported(backend)) {
        unsupported(backend);
    }
    return backend;
}

const char* backend_name(Backend backend) {
    switch (backend) {
        case Backend::Auto:
            return "auto";
        case Backend::Scalar:
            return "scalar";
        case Backend::SSE2:
            return "sse2";
        case Backend::AVX2:
            return "avx2";
    }
    return "unknown";
}

std::size_t lower_bound(const std::int32_t* keys, std::size_t cnt,
                        std::int32_t key, Backend backend) {
    return dispatch_lower(keys, cnt, key, resolve_backend(backend));
}

std::size_t upper_bound(const std::int32_t* keys, std::size_t cnt,
                        std::int32_t key, Backend backend) {
    return dispatch_upper(keys, cnt, key, resolve_backend(backend));
}

}  // namespace hft
