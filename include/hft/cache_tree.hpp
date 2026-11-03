#pragma once

#include <hft/config.hpp>
#include <hft/simd_search.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace hft {

// An insertion-only B+ tree. Inserting an existing key replaces its value.
// Range searches return up to length values for keys >= start_key, in key order.
class CacheTree {
public:
    explicit CacheTree(Backend backend = Backend::Auto);

    void insert(std::int32_t key, std::int32_t value);
    std::optional<std::int32_t> find(std::int32_t key) const;
    std::vector<std::int32_t> range_search(std::int32_t start_key,
                                           std::int32_t length) const;
    std::size_t size() const noexcept { return size_; }
    Backend backend() const noexcept { return backend_; }

    // Checks node alignment, reachability, occupancy, separator bounds,
    // uniform leaf depth, leaf links, and the stored entry count.
    bool validate() const;

private:
    static constexpr std::size_t line_size_ = HFT_CACHE_LINE_SIZE;
    static_assert(line_size_ >= 32 && (line_size_ & (line_size_ - 1)) == 0,
                  "HFT_CACHE_LINE_SIZE must be a power of two of at least 32");
    static_assert(sizeof(std::int32_t) == 4 && sizeof(std::uint32_t) == 4,
                  "CacheTree requires 32-bit keys, slots and metadata");
    static constexpr std::size_t capacity_ = (line_size_ - 8) / 8;
    static_assert(capacity_ >= 3 && capacity_ <= 0x7fffffffu,
                  "CacheTree node capacity is not representable");
    // keys starts after the 4-byte metadata and runs to the end of the node, so
    // this many int32_t are readable (and zero-initialized) from keys onward.
    // SIMD searches may load a full vector past count() as long as lanes at or
    // beyond count() are masked, which is what makes vectorization possible
    // when capacity_ is smaller than the vector width (e.g. 7 keys at 64 B).
    static constexpr std::size_t readable_keys_ = (line_size_ - 4) / 4;
    static_assert(readable_keys_ >= capacity_,
                  "Node key storage must cover the key capacity");

    static constexpr std::uint32_t leaf_flag_ = 0x80000000u;
    static constexpr std::uint32_t count_mask_ = 0x7fffffffu;

    // Slot capacity_ is the next-leaf index; zero terminates the chain.
    // Internal nodes use all capacity_ + 1 slots for child indices.
    struct alignas(line_size_) Node {
        std::uint32_t metadata{};
        std::int32_t keys[capacity_]{};
        std::int32_t slots[capacity_ + 1]{};

        bool is_leaf() const noexcept { return (metadata & leaf_flag_) != 0; }
        std::size_t count() const noexcept { return metadata & count_mask_; }
        void set_count(std::size_t count) noexcept {
            metadata = (metadata & leaf_flag_) | static_cast<std::uint32_t>(count);
        }
    };
    static_assert(sizeof(Node) == line_size_, "CacheTree node must occupy one cache line");
    static_assert(alignof(Node) == line_size_, "CacheTree node alignment must match the cache line");
    static_assert(offsetof(Node, keys) == 4,
                  "SIMD readable_keys_ assumes keys begins after the 4-byte metadata");
    static_assert(offsetof(Node, keys) + readable_keys_ * sizeof(std::int32_t) <= sizeof(Node),
                  "SIMD loads must stay inside the node");

    // Test-only access to node storage for validate() fault injection.
    friend struct CacheTreeTestAccess;

    struct PathEntry {
        std::int32_t index;
        std::size_t child;
    };
    struct Split {
        std::int32_t right;
        std::int32_t separator;
    };

    Backend backend_;
    std::vector<Node> nodes_;
    std::size_t size_ = 0;

    std::int32_t allocate_node(bool leaf);
    std::int32_t find_leaf(std::int32_t key) const;
    Split split_leaf(std::int32_t index, std::size_t position,
                     std::int32_t key, std::int32_t value);
    Split split_internal(std::int32_t index, std::size_t position, Split split);
    void insert_internal(std::int32_t index, std::size_t position, Split split);
    void split_root(Split split);
};

} // namespace hft
