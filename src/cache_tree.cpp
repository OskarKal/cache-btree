#include <hft/cache_tree.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace hft {

CacheTree::CacheTree(Backend backend) : backend_(resolve_backend(backend)), nodes_(1) {
    nodes_[0].metadata = leaf_flag_;
}

std::int32_t CacheTree::allocate_node(bool leaf) {
    // Zero is reserved for the root and the end of a leaf chain.
    if (nodes_.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::length_error("CacheTree node index overflow");
    }
    const auto index = static_cast<std::int32_t>(nodes_.size());
    nodes_.emplace_back();
    if (leaf) nodes_.back().metadata = leaf_flag_;
    return index;
}

std::int32_t CacheTree::find_leaf(std::int32_t key) const {
    std::int32_t index = 0;
    while (!nodes_[index].is_leaf()) {
        const Node& node = nodes_[index];
        index = node.slots[hft::upper_bound_resolved(node.keys, node.count(), key,
                                                     backend_, readable_keys_)];
    }
    return index;
}

void CacheTree::insert_internal(std::int32_t index, std::size_t position, Split split) {
    Node& node = nodes_[index];
    const std::size_t count = node.count();
    for (std::size_t i = count; i > position; --i) {
        node.keys[i] = node.keys[i - 1];
        node.slots[i + 1] = node.slots[i];
    }
    node.keys[position] = split.separator;
    node.slots[position + 1] = split.right;
    node.set_count(count + 1);
}

CacheTree::Split CacheTree::split_leaf(std::int32_t index, std::size_t position,
                                       std::int32_t key, std::int32_t value) {
    std::array<std::int32_t, capacity_ + 1> keys{};
    std::array<std::int32_t, capacity_ + 1> values{};
    const Node& old = nodes_[index];
    for (std::size_t i = 0; i < capacity_ + 1; ++i) {
        if (i == position) {
            keys[i] = key;
            values[i] = value;
        } else {
            const std::size_t source = i < position ? i : i - 1;
            keys[i] = old.keys[source];
            values[i] = old.slots[source];
        }
    }

    const std::int32_t right_index = allocate_node(true);
    Node& left = nodes_[index]; // Allocation may have moved the vector.
    Node& right = nodes_[right_index];
    const std::size_t left_count = (capacity_ + 1) / 2;
    const std::size_t right_count = capacity_ + 1 - left_count;
    for (std::size_t i = 0; i < left_count; ++i) {
        left.keys[i] = keys[i];
        left.slots[i] = values[i];
    }
    for (std::size_t i = 0; i < right_count; ++i) {
        right.keys[i] = keys[left_count + i];
        right.slots[i] = values[left_count + i];
    }
    left.set_count(left_count);
    right.set_count(right_count);
    right.slots[capacity_] = left.slots[capacity_];
    left.slots[capacity_] = right_index;
    return {right_index, right.keys[0]};
}

CacheTree::Split CacheTree::split_internal(std::int32_t index, std::size_t position,
                                           Split split) {
    std::array<std::int32_t, capacity_ + 1> keys{};
    std::array<std::int32_t, capacity_ + 2> children{};
    const Node& old = nodes_[index];
    for (std::size_t i = 0; i < capacity_ + 1; ++i) {
        keys[i] = i == position ? split.separator : old.keys[i < position ? i : i - 1];
    }
    for (std::size_t i = 0; i < capacity_ + 2; ++i) {
        children[i] = i == position + 1 ? split.right : old.slots[i <= position ? i : i - 1];
    }

    const std::int32_t right_index = allocate_node(false);
    Node& left = nodes_[index];
    Node& right = nodes_[right_index];
    const std::size_t middle = (capacity_ + 1) / 2;
    const std::size_t right_count = capacity_ - middle;
    for (std::size_t i = 0; i < middle; ++i) left.keys[i] = keys[i];
    for (std::size_t i = 0; i <= middle; ++i) left.slots[i] = children[i];
    for (std::size_t i = 0; i < right_count; ++i) right.keys[i] = keys[middle + 1 + i];
    for (std::size_t i = 0; i <= right_count; ++i) right.slots[i] = children[middle + 1 + i];
    left.set_count(middle);
    right.set_count(right_count);
    return {right_index, keys[middle]};
}

void CacheTree::split_root(Split split) {
    const bool leaf = nodes_[0].is_leaf();
    const std::int32_t left_index = allocate_node(leaf);
    nodes_[left_index] = nodes_[0];
    nodes_[0] = Node{};
    nodes_[0].keys[0] = split.separator;
    nodes_[0].slots[0] = left_index;
    nodes_[0].slots[1] = split.right;
    nodes_[0].set_count(1);
}

void CacheTree::insert(std::int32_t key, std::int32_t value) {
    // The root-to-leaf path is stored on the stack: a tree with int32 node
    // indices and capacity_ >= 3 has minimum fanout 2, so the path is at most
    // 32 entries. Avoiding a per-insert heap allocation matters on the hot path.
    std::array<PathEntry, 64> path;
    std::size_t depth = 0;
    std::int32_t index = 0;
    while (!nodes_[index].is_leaf()) {
        const Node& node = nodes_[index];
        const std::size_t child = hft::upper_bound_resolved(node.keys, node.count(), key,
                                                            backend_, readable_keys_);
        if (depth == path.size()) throw std::length_error("CacheTree depth exceeded");
        path[depth++] = {index, child};
        index = node.slots[child];
    }
    const std::size_t count = nodes_[index].count();
    const std::size_t position = hft::lower_bound_resolved(nodes_[index].keys, count, key,
                                                           backend_, readable_keys_);
    if (position < count && nodes_[index].keys[position] == key) {
        nodes_[index].slots[position] = value;
        return;
    }
    if (size_ == std::numeric_limits<std::size_t>::max()) {
        throw std::length_error("CacheTree size overflow");
    }
    if (count < capacity_) {
        Node& leaf = nodes_[index];
        for (std::size_t i = count; i > position; --i) {
            leaf.keys[i] = leaf.keys[i - 1];
            leaf.slots[i] = leaf.slots[i - 1];
        }
        leaf.keys[position] = key;
        leaf.slots[position] = value;
        leaf.set_count(count + 1);
        ++size_;
        return;
    }

    // Reserve every node needed by the cascade before changing the tree. This
    // also makes allocation failure or index exhaustion leave the tree intact.
    std::size_t needed = 1;
    bool reaches_root = true;
    for (std::size_t i = depth; i > 0; --i) {
        if (nodes_[path[i - 1].index].count() < capacity_) {
            reaches_root = false;
            break;
        }
        ++needed;
    }
    if (reaches_root) ++needed; // evacuated old root
    const std::size_t max_nodes = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) + 1;
    if (needed > max_nodes - nodes_.size() || needed > nodes_.max_size() - nodes_.size()) {
        throw std::length_error("CacheTree node index overflow");
    }
    const std::size_t required = nodes_.size() + needed;
    if (nodes_.capacity() < required) {
        const std::size_t limit = std::min(max_nodes, nodes_.max_size());
        const std::size_t current = nodes_.capacity();
        const std::size_t grown = current <= limit - current ? current * 2 : limit;
        nodes_.reserve(std::max(required, grown));
    }

    Split split = split_leaf(index, position, key, value);
    for (std::size_t i = depth; i > 0; --i) {
        const PathEntry& entry = path[i - 1];
        if (nodes_[entry.index].count() < capacity_) {
            insert_internal(entry.index, entry.child, split);
            ++size_;
            return;
        }
        split = split_internal(entry.index, entry.child, split);
    }
    split_root(split);
    ++size_;
}

std::optional<std::int32_t> CacheTree::find(std::int32_t key) const {
    const Node& leaf = nodes_[find_leaf(key)];
    const std::size_t position = hft::lower_bound_resolved(leaf.keys, leaf.count(), key,
                                                           backend_, readable_keys_);
    if (position < leaf.count() && leaf.keys[position] == key) return leaf.slots[position];
    return std::nullopt;
}

std::vector<std::int32_t> CacheTree::range_search(std::int32_t start_key,
                                                   std::int32_t length) const {
    std::vector<std::int32_t> result;
    if (length <= 0) return result;
    const std::size_t limit = static_cast<std::size_t>(length);
    // Reserve only a small prefix: a range starting past the last key would
    // otherwise allocate O(size_) for a result that is usually empty or short.
    result.reserve(std::min<std::size_t>(limit, 64));
    std::int32_t index = find_leaf(start_key);
    std::size_t position = hft::lower_bound_resolved(nodes_[index].keys,
                                                     nodes_[index].count(), start_key,
                                                     backend_, readable_keys_);
    do {
        const Node& leaf = nodes_[index];
        while (position < leaf.count() && result.size() < limit) {
            result.push_back(leaf.slots[position++]);
        }
        if (result.size() == limit) break;
        index = leaf.slots[capacity_];
        position = 0;
    } while (index != 0);
    return result;
}

bool CacheTree::validate() const {
    if (nodes_.empty()) return false;
    for (const Node& node : nodes_) {
        if (reinterpret_cast<std::uintptr_t>(&node) % line_size_ != 0) return false;
    }
    std::vector<bool> visited(nodes_.size(), false);
    std::vector<std::int32_t> leaves;
    std::size_t leaf_depth = 0;
    bool have_leaf_depth = false;
    std::size_t entries = 0;
    struct Bounds { std::int32_t min; std::int32_t max; };

    const auto visit = [&](const auto& self, std::int32_t index, std::size_t depth,
                           Bounds& bounds) -> bool {
        if (index < 0 || static_cast<std::size_t>(index) >= nodes_.size() || visited[index]) return false;
        visited[index] = true;
        const Node& node = nodes_[index];
        const std::size_t count = node.count();
        if (count > capacity_) return false;
        if (index != 0 && count < (node.is_leaf() ? (capacity_ + 1) / 2 : capacity_ / 2)) return false;
        if (!node.is_leaf() && count == 0) return false;
        for (std::size_t i = 1; i < count; ++i) {
            if (node.keys[i - 1] >= node.keys[i]) return false;
        }
        if (node.is_leaf()) {
            if (!have_leaf_depth) { leaf_depth = depth; have_leaf_depth = true; }
            if (depth != leaf_depth || entries > std::numeric_limits<std::size_t>::max() - count) return false;
            entries += count;
            leaves.push_back(index);
            if (count) bounds = {node.keys[0], node.keys[count - 1]};
            return count != 0 || index == 0;
        }
        Bounds child{};
        if (node.slots[0] <= 0 || !self(self, node.slots[0], depth + 1, child)) return false;
        bounds = child;
        for (std::size_t i = 0; i < count; ++i) {
            Bounds right{};
            if (node.slots[i + 1] <= 0 || !self(self, node.slots[i + 1], depth + 1, right)) return false;
            if (child.max >= right.min || node.keys[i] != right.min) return false;
            child = right;
            bounds.max = right.max;
        }
        return true;
    };
    Bounds root{};
    if (!visit(visit, 0, 0, root) || entries != size_ || leaves.empty()) return false;
    if (std::find(visited.begin(), visited.end(), false) != visited.end()) return false;
    for (std::size_t i = 0; i < leaves.size(); ++i) {
        const std::int32_t next = i + 1 < leaves.size() ? leaves[i + 1] : 0;
        if (nodes_[leaves[i]].slots[capacity_] != next) return false;
    }
    return true;
}

} // namespace hft
