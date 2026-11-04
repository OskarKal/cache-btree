cmake_minimum_required(VERSION 3.16)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "In-source builds are not supported. Configure with cmake -S . -B <build-directory>.")
endif()

project(hft_cache LANGUAGES CXX)

set(HFT_CACHE_LINE_SIZE "" CACHE STRING
    "Target L1 data-cache line size in bytes (required for cross-compilation)")
option(HFT_BUILD_BENCHMARK "Build the benchmark when its sources are available" ON)
option(HFT_ENABLE_SANITIZERS "Enable address and undefined-behavior sanitizers" OFF)
include(CTest)

# Detection is deliberately native-only: host cache information is not evidence
# about a cross-compilation target. An empty override requests detection.
if(NOT HFT_CACHE_LINE_SIZE STREQUAL "")
  set(_hft_line_size "${HFT_CACHE_LINE_SIZE}")
  set(_hft_line_source "explicit HFT_CACHE_LINE_SIZE override")
elseif(CMAKE_CROSSCOMPILING)
  message(FATAL_ERROR
    "Cannot detect the target cache line size while cross-compiling. "
    "Set -DHFT_CACHE_LINE_SIZE=<target L1 data-cache line size in bytes>.")
elseif(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR
    "Automatic L1 data-cache line detection is only supported on native Linux. "
    "Set -DHFT_CACHE_LINE_SIZE=<target L1 data-cache line size in bytes>.")
else()
  # Check all available CPU cache records, not just CPU 0, so that a machine
  # reporting different L1 line sizes does not silently select one of them.
  file(GLOB _hft_cache_dirs LIST_DIRECTORIES true
       "/sys/devices/system/cpu/cpu[0-9]*/cache/index[0-9]*")
  set(_hft_detected_line_size "")
  foreach(_hft_cache_dir IN LISTS _hft_cache_dirs)
    if(NOT EXISTS "${_hft_cache_dir}/level" OR NOT EXISTS "${_hft_cache_dir}/type")
      continue()
    endif()
    file(READ "${_hft_cache_dir}/level" _hft_cache_level)
    file(READ "${_hft_cache_dir}/type" _hft_cache_type)
    string(STRIP "${_hft_cache_level}" _hft_cache_level)
    string(STRIP "${_hft_cache_type}" _hft_cache_type)
    if(NOT _hft_cache_level STREQUAL "1" OR
       NOT _hft_cache_type MATCHES "^(Data|Unified)$")
      continue()
    endif()
    if(NOT EXISTS "${_hft_cache_dir}/coherency_line_size")
      message(FATAL_ERROR
        "L1 cache line size is unavailable at ${_hft_cache_dir}. "
        "Set -DHFT_CACHE_LINE_SIZE=<target L1 data-cache line size in bytes>.")
    endif()
    file(READ "${_hft_cache_dir}/coherency_line_size" _hft_cache_line)
    string(STRIP "${_hft_cache_line}" _hft_cache_line)
    if(_hft_cache_line STREQUAL "")
      message(FATAL_ERROR
        "L1 cache line size is empty at ${_hft_cache_dir}. "
        "Set -DHFT_CACHE_LINE_SIZE=<target L1 data-cache line size in bytes>.")
    endif()
    if(_hft_detected_line_size STREQUAL "")
      set(_hft_detected_line_size "${_hft_cache_line}")
    elseif(NOT _hft_cache_line STREQUAL _hft_detected_line_size)
      message(FATAL_ERROR
        "Linux sysfs reports inconsistent L1 data-cache line sizes "
        "(${_hft_detected_line_size} and ${_hft_cache_line}). "
        "Set -DHFT_CACHE_LINE_SIZE=<target L1 data-cache line size in bytes> explicitly.")
    endif()
  endforeach()
  if(_hft_detected_line_size STREQUAL "")
    message(FATAL_ERROR
      "Could not detect an L1 data-cache line size from Linux sysfs. "
      "Set -DHFT_CACHE_LINE_SIZE=<target L1 data-cache line size in bytes>.")
  endif()
  set(_hft_line_size "${_hft_detected_line_size}")
  set(_hft_line_source "native Linux sysfs L1 data/unified cache")
endif()

# Bound the value before CMake arithmetic and the compiler alignment check.
# The minimum permits at least three keys in the planned node layout. The upper
# bound is the maximum extended alignment accepted by common C++ compilers.
if(NOT _hft_line_size MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR
    "Invalid cache line size '${_hft_line_size}' from ${_hft_line_source}: "
    "set -DHFT_CACHE_LINE_SIZE to a decimal power of two between 32 and 268435456 bytes.")
endif()
string(LENGTH "${_hft_line_size}" _hft_line_length)
if(_hft_line_length GREATER 9 OR _hft_line_size LESS 32 OR
   _hft_line_size GREATER 268435456)
  message(FATAL_ERROR
    "Unsupported cache line size '${_hft_line_size}' from ${_hft_line_source}: "
    "set -DHFT_CACHE_LINE_SIZE to a decimal power of two between 32 and 268435456 bytes.")
endif()
math(EXPR _hft_power_check "${_hft_line_size} & (${_hft_line_size} - 1)")
if(NOT _hft_power_check EQUAL 0)
  message(FATAL_ERROR
    "Cache line size '${_hft_line_size}' from ${_hft_line_source} is not a power of two. "
    "Set -DHFT_CACHE_LINE_SIZE to a supported power of two in bytes.")
endif()

include(CheckCXXSourceCompiles)
set(_hft_alignment_test "HFT_ALIGNMENT_${_hft_line_size}_SUPPORTED")
check_cxx_source_compiles(
  "struct alignas(${_hft_line_size}) Aligned { char value; };\nstatic_assert(alignof(Aligned) == ${_hft_line_size}, \"unsupported alignment\");\nint main() { return 0; }"
  "${_hft_alignment_test}")
if(NOT ${_hft_alignment_test})
  message(FATAL_ERROR
    "The C++ compiler does not support ${_hft_line_size}-byte alignment. "
    "Set -DHFT_CACHE_LINE_SIZE to a supported target cache line size.")
endif()
message(STATUS "HFT cache line size: ${_hft_line_size} bytes (${_hft_line_source})")

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated/hft")
configure_file(cmake/config.hpp.in
               "${CMAKE_CURRENT_BINARY_DIR}/generated/hft/config.hpp" @ONLY)

add_library(hft_build_options INTERFACE)
target_compile_features(hft_build_options INTERFACE cxx_std_17)
if(HFT_ENABLE_SANITIZERS)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    message(FATAL_ERROR "HFT_ENABLE_SANITIZERS requires a GCC- or Clang-compatible compiler.")
  endif()
  # -fno-sanitize-recover=all makes UBSan terminate instead of printing a
  # diagnostic and continuing, so a sanitizer finding fails ctest.
  target_compile_options(hft_build_options INTERFACE
                         -fsanitize=address,undefined -fno-sanitize-recover=all
                         -fno-omit-frame-pointer)
  target_link_options(hft_build_options INTERFACE -fsanitize=address,undefined)
endif()

# SIMD compile-time ceiling. This never adds -march flags. HFT_SIMD=SCALAR
# compiles the library without SSE2/AVX2 code, HFT_SIMD=SSE2 compiles SSE2 but
# disables AVX2, and AUTO/AVX2 keep both. Runtime CPU/OS detection still decides
# which compiled backend Auto selects. HFT_SIMD_DEFAULT=<TOKEN> is also exported
# publicly for diagnostics.
set(HFT_SIMD "AUTO" CACHE STRING
    "SIMD compile-time ceiling and default preference: AUTO, SCALAR, SSE2 or AVX2")
set_property(CACHE HFT_SIMD PROPERTY STRINGS AUTO SCALAR SSE2 AVX2)
if(NOT HFT_SIMD MATCHES "^(AUTO|SCALAR|SSE2|AVX2)$")
  message(FATAL_ERROR
    "Invalid HFT_SIMD value '${HFT_SIMD}'. Valid values are: AUTO, SCALAR, SSE2, AVX2.")
endif()
set(_hft_simd_definitions)
if(HFT_SIMD STREQUAL "SCALAR")
  list(APPEND _hft_simd_definitions HFT_SIMD_SCALAR_ONLY)
elseif(HFT_SIMD STREQUAL "SSE2")
  list(APPEND _hft_simd_definitions HFT_SIMD_DISABLE_AVX2)
endif()
message(STATUS "HFT SIMD ceiling: ${HFT_SIMD} (runtime backend selection remains authoritative)")

set(_hft_tree_source "${CMAKE_CURRENT_SOURCE_DIR}/src/cache_tree.cpp")
set(_hft_simd_source "${CMAKE_CURRENT_SOURCE_DIR}/src/simd_search.cpp")
if(EXISTS "${_hft_tree_source}" AND EXISTS "${_hft_simd_source}")
  add_library(hft_cache "${_hft_tree_source}" "${_hft_simd_source}")
  target_link_libraries(hft_cache PUBLIC hft_build_options)
  target_compile_definitions(hft_cache PUBLIC "HFT_SIMD_DEFAULT=${HFT_SIMD}"
                             PRIVATE ${_hft_simd_definitions})
  target_include_directories(hft_cache PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_BINARY_DIR}/generated")
  set_target_properties(hft_cache PROPERTIES CXX_EXTENSIONS OFF)
elseif(BUILD_TESTING)
  message(FATAL_ERROR
    "BUILD_TESTING is enabled but the library sources are missing: "
    "${_hft_tree_source} and/or ${_hft_simd_source}.")
else()
  message(STATUS "Skipping hft_cache: src/cache_tree.cpp and/or src/simd_search.cpp not yet available")
endif()

if(BUILD_TESTING)
  if(NOT TARGET hft_cache)
    message(FATAL_ERROR "BUILD_TESTING is enabled but the hft_cache library is unavailable.")
  endif()
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cache_tree.cpp")
    message(FATAL_ERROR
      "BUILD_TESTING is enabled but tests/test_cache_tree.cpp is missing.")
  endif()
  add_executable(hft_tests tests/test_cache_tree.cpp)
  target_link_libraries(hft_tests PRIVATE hft_cache)
  set_target_properties(hft_tests PROPERTIES CXX_EXTENSIONS OFF)
  add_test(NAME hft_tests COMMAND hft_tests)
  if(HFT_ENABLE_SANITIZERS)
    set_tests_properties(hft_tests PROPERTIES
      ENVIRONMENT "UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1;ASAN_OPTIONS=detect_leaks=1")
  endif()
endif()

if(HFT_BUILD_BENCHMARK)
  if(TARGET hft_cache AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_both.cpp")
    add_executable(hft_bench benchmark_both.cpp)
    target_link_libraries(hft_bench PRIVATE hft_cache)
    set_target_properties(hft_bench PROPERTIES CXX_EXTENSIONS OFF)
  else()
    message(STATUS "Skipping hft_bench: library and/or benchmark_both.cpp not yet available")
  endif()
endif()
