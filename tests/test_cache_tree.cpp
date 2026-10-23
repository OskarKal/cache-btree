#include <hft/cache_tree.hpp>
#include <hft/simd_search.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Run this same executable in separate 32-, 64-, and 128-byte CMake
// configurations. These are synthetic layouts, not detected CPU properties.
// Node internals deliberately remain private: validate() is responsible for
// capacities, separators, child relationships, and the ordered leaf chain.
// Actual node-address alignment needs an implementation-side check; the public
// API does not expose node storage, so alignof(CacheTree) would not test it.
#ifndef HFT_CACHE_LINE_SIZE
#error "The library configuration must provide HFT_CACHE_LINE_SIZE"
#endif

namespace {
using Key = std::int32_t;
using Value = std::int32_t;
using Model = std::map<Key, Value>;
using hft::Backend;
using hft::CacheTree;

constexpr Key kMin = std::numeric_limits<Key>::min();
constexpr Key kMax = std::numeric_limits<Key>::max();
constexpr std::size_t kCapacity = (HFT_CACHE_LINE_SIZE - 8U) / 8U;
static_assert(HFT_CACHE_LINE_SIZE >= 32, "Tests require at least a 32-byte layout");
static_assert(kCapacity <= static_cast<std::size_t>(kMax) / 32,
              "Test key generation must fit int32_t");

std::string context;
std::size_t checks = 0;

void require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        throw std::runtime_error(context + ": " + message);
    }
}

const char* name(Backend backend) {
    const char* result = hft::backend_name(backend);
    require(result != nullptr && result[0] != '\0', "backend has no name");
    return result;
}

std::vector<Backend> available_modes() {
    require(hft::backend_available(Backend::Scalar), "Scalar must be available");
    std::vector<Backend> modes{Backend::Scalar, Backend::Auto};
    for (Backend backend : {Backend::SSE2, Backend::AVX2}) {
        if (hft::backend_available(backend)) {
            modes.push_back(backend);
            continue;
        }
        bool rejected = false;
        try {
            CacheTree unsupported(backend);
            (void)unsupported;
        } catch (...) {
            // The public API promises clean rejection, not an exception type.
            rejected = true;
        }
        require(rejected, std::string("unavailable forced backend accepted: ") + name(backend));
        std::cout << "SKIP " << name(backend)
                  << " execution: unavailable; forced construction rejected\n";
    }
    return modes;
}

std::vector<Value> expected_range(const Model& model, Key start, Key length) {
    std::vector<Value> result;
    if (length <= 0) {
        return result;
    }
    for (auto it = model.lower_bound(start);
         it != model.end() && result.size() < static_cast<std::size_t>(length); ++it) {
        result.push_back(it->second);
    }
    return result;
}

class Trees {
public:
    explicit Trees(const std::vector<Backend>& modes) {
        for (Backend mode : modes) {
            trees_.push_back(std::make_unique<CacheTree>(mode));
            const Backend actual = trees_.back()->backend();
            if (mode != Backend::Auto) {
                require(actual == mode, std::string("forced backend changed: ") + name(mode));
            } else {
                require(actual == Backend::Auto || hft::backend_available(actual),
                        "Auto selected an unavailable backend");
            }
        }
    }

    void insert(Key key, Value value) {
        model_[key] = value;
        for (auto& tree : trees_) {
            tree->insert(key, value);
        }
        size();
    }

    void size() const {
        for (const auto& tree : trees_) {
            require(tree->size() == model_.size(),
                    std::string(name(tree->backend())) + " size differs from std::map");
        }
    }

    void structure() const {
        size();
        for (const auto& tree : trees_) {
            require(tree->validate(), std::string(name(tree->backend())) + " validate() failed");
        }
    }

    void find(Key key) const {
        const auto it = model_.find(key);
        const std::optional<Value> expected = it == model_.end()
                                                 ? std::nullopt
                                                 : std::optional<Value>(it->second);
        const auto scalar = trees_.front()->find(key);
        for (const auto& tree : trees_) {
            const auto actual = tree->find(key);
            require(actual == expected && actual == scalar,
                    std::string(name(tree->backend())) + " find(" + std::to_string(key) + ") mismatch");
        }
    }

    void range(Key start, Key length) const {
        const auto expected = expected_range(model_, start, length);
        const auto scalar = trees_.front()->range_search(start, length);
        for (const auto& tree : trees_) {
            const auto actual = tree->range_search(start, length);
            require(actual == expected && actual == scalar,
                    std::string(name(tree->backend())) + " range(" + std::to_string(start) + ", " +
                        std::to_string(length) + ") mismatch");
        }
    }

    void complete() const {
        structure();
        range(kMin, kMax); // Values in key order, including a possible INT32_MIN key.
        for (const auto& entry : model_) {
            find(entry.first);
        }
    }

    void edge_ranges(Key start) const {
        for (Key length : {kMin, Key{-1}, Key{0}, Key{1}, Key{2}, Key{3}, Key{4},
                           Key{7}, Key{8}, Key{9}, Key{15}, Key{16}, Key{17}, kMax}) {
            range(start, length);
        }
    }

private:
    Model model_;
    std::vector<std::unique_ptr<CacheTree>> trees_;
};

void default_constructor() {
    context = "default constructor";
    const CacheTree tree;
    require(tree.size() == 0 && tree.validate(), "default tree is not valid and empty");
    require(!tree.find(0).has_value(), "default empty tree contains a key");
    require(tree.range_search(kMin, kMax).empty(), "default empty range is nonempty");
    std::cout << "Default backend: " << name(tree.backend()) << '\n';
}

void empty_and_extrema(const std::vector<Backend>& modes) {
    context = "empty and signed extrema";
    Trees trees(modes);
    trees.complete();
    for (Key key : {kMin, Key{-1}, Key{0}, Key{1}, kMax}) {
        trees.find(key);
        trees.edge_ranges(key);
    }
    trees.insert(0, -1);
    trees.complete(); // Root-leaf regression: leaf index zero must be traversed.
    trees.find(-1);
    trees.find(1);
    trees.edge_ranges(-1);
    trees.edge_ranges(0);
    trees.edge_ranges(1);
    trees.insert(0, kMin);
    trees.complete();
    trees.insert(0, -1);
    trees.insert(kMin, kMax);
    trees.insert(kMax, kMin);
    trees.insert(-1, 42);
    trees.insert(1, -42);
    trees.insert(kMin + 1, -1);
    trees.insert(kMax - 1, 0);
    trees.complete();
    for (Key key : {kMin, Key{kMin + 1}, Key{kMin + 2}, Key{-2}, Key{-1}, Key{0},
                    Key{1}, Key{2}, Key{kMax - 2}, Key{kMax - 1}, kMax}) {
        trees.find(key);
        trees.edge_ranges(key);
    }
    // Duplicate updates of boundary keys must not grow size or change order.
    trees.insert(kMin, -1);
    trees.insert(kMax, 123);
    trees.insert(0, 99);
    trees.complete();
    trees.edge_ranges(kMax);
}

void partial_nodes_and_splits(const std::vector<Backend>& modes) {
    context = "partial nodes and split boundaries";
    Trees trees(modes);
    // Before the first split each occupancy 1..capacity is a root leaf. This
    // exercises every attainable SIMD tail for the compiled layout, not padded
    // metadata or unused key slots. Continue through multiple leaf splits.
    const auto count = static_cast<Key>(4 * kCapacity + 5);
    for (Key i = 0; i < count; ++i) {
        context = "partial nodes, insert=" + std::to_string(i);
        trees.insert(-1000 + 10 * i, 500 - 3 * i);
        trees.structure();
        trees.range(kMin, kMax);
        for (Key j = 0; j <= i; ++j) {
            const Key key = -1000 + 10 * j;
            trees.find(key);
            trees.find(key - 1);
            trees.find(key + 1);
            // Starting inside a leaf must not count skipped entries toward
            // length. Cover all small result lengths, not only SIMD multiples.
            for (Key length = 0; length <= static_cast<Key>(kCapacity + 2); ++length) {
                trees.range(key, length);
                trees.range(key + 1, length);
            }
        }
        trees.edge_ranges(-1001);
        trees.edge_ranges(-1000 + 10 * i + 1); // Beyond the largest key.
    }
    trees.complete();
}

void ordered_insertions(const std::vector<Backend>& modes) {
    // 6000 keys force several internal levels for all requested layouts,
    // repeated root splits, and enough vector growth to move node storage.
    constexpr Key count = 6000;
    constexpr std::uint32_t seed = 0x4b1d9023U;
    for (int order = 0; order != 3; ++order) {
        std::vector<Key> keys(static_cast<std::size_t>(count));
        std::iota(keys.begin(), keys.end(), Key{0});
        const std::string label = order == 0 ? "ascending" : order == 1 ? "descending" : "shuffled";
        if (order == 1) {
            std::reverse(keys.begin(), keys.end());
        } else if (order == 2) {
            std::mt19937 rng(seed);
            std::shuffle(keys.begin(), keys.end(), rng);
        }
        std::cout << "Insertion order=" << label << " count=" << count << " seed=" << seed << '\n';
        Trees trees(modes);
        for (std::size_t i = 0; i < keys.size(); ++i) {
            context = label + " insertion=" + std::to_string(i) + " seed=" + std::to_string(seed);
            const Key ordinal = keys[i];
            trees.insert(2 * ordinal - count, ordinal * 17 - 50000);
            trees.find(2 * ordinal - count);
            if (i < kCapacity + 2 || i % 127 == 0) {
                trees.structure();
                trees.range(kMin, kMax);
            }
        }
        trees.complete();
        for (Key i = 0; i < count; ++i) {
            const Key key = 2 * i - count;
            trees.find(key + 1); // Every inter-key gap and beyond the last key.
            if (i % 31 == 0) {
                trees.edge_ranges(key);
                trees.edge_ranges(key + 1);
            }
        }
        // Upsert every key, including keys promoted to internal separators.
        // Reverse traversal also catches accidental insertion on equality.
        for (Key i = count; i-- > 0;) {
            trees.insert(2 * i - count, i % 2 == 0 ? -1 : count - i);
            if (i % 127 == 0) {
                trees.structure();
            }
        }
        trees.complete();
        trees.edge_ranges(-count - 1);
        trees.edge_ranges(count);
    }
}

void randomized_operations(const std::vector<Backend>& modes, std::uint32_t seed) {
    constexpr std::size_t operations = 12000;
    std::cout << "Random differential seed=" << seed << " operations=" << operations << '\n';
    std::mt19937 rng(seed);
    std::uniform_int_distribution<Key> any_key(kMin, kMax);
    std::uniform_int_distribution<Key> dense_key(-1500, 1500);
    std::uniform_int_distribution<Key> small_length(-8, 100);
    Trees trees(modes);
    for (std::size_t i = 0; i < operations; ++i) {
        context = "random seed=" + std::to_string(seed) + " operation=" + std::to_string(i);
        // A dense domain supplies many updates/hits; full-width keys stress
        // signed comparison and sparse ranges. Explicit extrema avoid relying
        // on a random generator to happen to produce boundary values.
        Key key = i % 3 == 0 ? any_key(rng) : dense_key(rng);
        if (i % 101 == 0) {
            key = kMin;
        } else if (i % 103 == 0) {
            key = kMax;
        }
        const auto operation = rng() % 10U;
        if (operation < 6) {
            trees.insert(key, i % 7 == 0 ? -1 : any_key(rng));
            trees.find(key);
        } else if (operation < 8) {
            trees.find(key);
        } else {
            trees.range(key, i % 13 == 0 ? kMax : small_length(rng));
        }
        if (i % 251 == 0) {
            trees.complete();
            trees.edge_ranges(dense_key(rng));
            trees.edge_ranges(kMin);
            trees.edge_ranges(kMax);
        }
    }
    trees.complete();
}
} // namespace

int main() {
    try {
        // unitbuf ensures the failing seed/operation's workload is visible even
        // if a sanitizer terminates execution before C++ exception handling.
        std::cout << std::unitbuf;
        std::cout << "HFT_CACHE_LINE_SIZE=" << HFT_CACHE_LINE_SIZE
                  << " derived_capacity=" << kCapacity << '\n';
        context = "backend selection";
        const auto modes = available_modes();
        for (Backend mode : modes) {
            std::cout << "Testing backend=" << name(mode) << '\n';
        }
        default_constructor();
        empty_and_extrema(modes);
        partial_nodes_and_splits(modes);
        ordered_insertions(modes);
        randomized_operations(modes, 0xc0ffee12U);
        randomized_operations(modes, 0x7e57a11cU);
        std::cout << "PASS " << checks << " checks\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
    } catch (...) {
        std::cerr << "FAIL " << context << ": unexpected non-standard exception\n";
    }
    return 1;
}
