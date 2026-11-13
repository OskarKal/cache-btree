#include <hft/cache_tree.hpp>
#include <hft/simd_search.hpp>
#include <hft/config.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
struct Options {
    std::string output, distribution = "uniform", backend = "all", flags = "not supplied";
    std::size_t keys = 10000, lookups = 20000, ranges = 300, updates = 2000, runs = 5;
    std::uint32_t seed = 42;
    bool reserve = false;
};
struct Workload {
    std::vector<int32_t> inserts, lookups, updates;
    std::vector<std::pair<int32_t, int32_t>> ranges;
    std::map<int32_t, int32_t> initial, updated;
};
struct Sample {
    double insert = 0, lookup = 0, range = 0, update = 0;
    bool has_range = false;
    std::uint64_t checksum = 0;
};

// One benchmark target: a CacheTree SIMD backend or an STL baseline.
enum class Target { Scalar, SSE2, AVX2, StdMap, SortedVector, StdUnorderedMap };

const char* target_name(Target target) {
    switch (target) {
        case Target::Scalar: return "scalar";
        case Target::SSE2: return "sse2";
        case Target::AVX2: return "avx2";
        case Target::StdMap: return "std::map";
        case Target::SortedVector: return "sorted_vector";
        case Target::StdUnorderedMap: return "std::unordered_map";
    }
    return "unknown";
}
bool target_has_range(Target target) { return target != Target::StdUnorderedMap; }
hft::Backend target_backend(Target target) {
    switch (target) {
        case Target::Scalar: return hft::Backend::Scalar;
        case Target::SSE2: return hft::Backend::SSE2;
        case Target::AVX2: return hft::Backend::AVX2;
        case Target::StdMap:
        case Target::SortedVector:
        case Target::StdUnorderedMap: break;
    }
    throw std::runtime_error("target is not a CacheTree backend");
}
bool target_available(Target target) {
    switch (target) {
        case Target::Scalar: return hft::backend_available(hft::Backend::Scalar);
        case Target::SSE2: return hft::backend_available(hft::Backend::SSE2);
        case Target::AVX2: return hft::backend_available(hft::Backend::AVX2);
        case Target::StdMap:
        case Target::SortedVector:
        case Target::StdUnorderedMap: return true;
    }
    return false;
}
Target target_from_name(const std::string& name) {
    if (name == "scalar") return Target::Scalar;
    if (name == "sse2") return Target::SSE2;
    if (name == "avx2") return Target::AVX2;
    if (name == "std_map") return Target::StdMap;
    if (name == "sorted_vector") return Target::SortedVector;
    if (name == "std_unordered_map") return Target::StdUnorderedMap;
    throw std::invalid_argument(
        "backend must be all, scalar, sse2, avx2, std_map, sorted_vector, or std_unordered_map");
}

std::size_t number(const std::string& s, std::size_t max, bool allow_zero = false) {
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("expected an unsigned decimal integer: " + s);
    const auto n = std::stoull(s);
    if ((!allow_zero && n == 0) || n > max) throw std::invalid_argument("number out of range: " + s);
    return static_cast<std::size_t>(n);
}
Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            std::cout << "Usage: hft_bench --output PATH [--distribution uniform|zipf] [--backend all|scalar|sse2|avx2|std_map|sorted_vector|std_unordered_map] [--keys N] [--lookups N] [--ranges N] [--updates N] [--runs N] [--seed N] [--build-flags TEXT] [--reserve]\n"
                         "\n"
                         "--distribution zipf uses a fixed exponent alpha = 1.5 and biases\n"
                         "lookups, ranges, and updates; inserts always cover every --keys key.\n"
                         "--backend all (default) benchmarks every available SIMD backend plus\n"
                         "the std::map, sorted_vector, and std::unordered_map baselines.\n"
                         "sorted_vector is bulk-loaded in the insert phase (append + sort), so\n"
                         "its insert time is not comparable to incremental insertion.\n"
                         "std::unordered_map has no ordered range query, so its range phase is\n"
                         "reported as not measured (range_measured=0).\n"
                         "--reserve pre-sizes the std::unordered_map baseline with reserve(keys).\n"
                         "--seed accepts 0. Timings are phase totals in milliseconds, not per-operation.\n";
            std::exit(0);
        }
        if (arg == "--reserve") { o.reserve = true; continue; }
        if (++i == argc) throw std::invalid_argument("missing value for " + arg);
        std::string val = argv[i];
        if (arg == "--output") o.output = val;
        else if (arg == "--distribution") o.distribution = val;
        else if (arg == "--backend") o.backend = val;
        else if (arg == "--keys") o.keys = number(val, 10000000);
        else if (arg == "--lookups") o.lookups = number(val, 100000000);
        else if (arg == "--ranges") o.ranges = number(val, 10000000);
        else if (arg == "--updates") o.updates = number(val, 100000000);
        else if (arg == "--runs") o.runs = number(val, 10000);
        else if (arg == "--seed") o.seed = static_cast<std::uint32_t>(number(val, UINT32_MAX, true));
        else if (arg == "--build-flags") o.flags = val;
        else throw std::invalid_argument("unknown option: " + arg);
    }
    if (o.output.empty()) throw std::invalid_argument("--output is required");
    if (o.distribution != "uniform" && o.distribution != "zipf")
        throw std::invalid_argument("distribution must be uniform or zipf");
    if (o.backend != "all")
        (void)target_from_name(o.backend); // validates the name and throws otherwise
    return o;
}

Workload generate(const Options& o) {
    std::mt19937 rng(o.seed);
    // Every key is inserted exactly once, so the tree size is --keys for every
    // distribution. Shuffling the insertion order avoids sorted-input bias.
    std::vector<int32_t> keys(o.keys);
    std::iota(keys.begin(), keys.end(), 0);
    std::shuffle(keys.begin(), keys.end(), rng);
    for (int32_t& key : keys) key *= 2; // even keys; key + 1 is a guaranteed miss
    std::vector<double> cdf;
    if (o.distribution == "zipf") {
        cdf.resize(o.keys);
        double total = 0;
        for (std::size_t i = 0; i < o.keys; ++i) { total += 1.0 / std::pow(double(i + 1), 1.5); cdf[i] = total; }
        for (double& x : cdf) x /= total;
        cdf.back() = 1.0;
    }
    // The distribution biases lookups, ranges, and updates only, not the
    // inserted key set.
    auto sample = [&]() -> int32_t {
        std::size_t rank;
        if (cdf.empty()) rank = std::uniform_int_distribution<std::size_t>(0, o.keys - 1)(rng);
        else rank = std::lower_bound(cdf.begin(), cdf.end(), std::generate_canonical<double, 53>(rng)) - cdf.begin();
        return keys[rank];
    };
    Workload w;
    for (int32_t key : keys) {
        w.inserts.push_back(key); w.initial[key] = key + 1;
    }
    for (std::size_t i = 0; i < o.lookups; ++i) {
        w.lookups.push_back(sample() + static_cast<int32_t>(i % 2));
    }
    constexpr int32_t lengths[] = {1, 8, 64};
    for (std::size_t i = 0; i < o.ranges; ++i)
        w.ranges.emplace_back(sample() + static_cast<int32_t>(i % 2), lengths[i % 3]);
    w.updated = w.initial;
    for (std::size_t i = 0; i < o.updates; ++i) {
        const int32_t k = sample(); w.updates.push_back(k); w.updated[k] = -k - 1;
    }
    return w;
}
std::vector<int32_t> expected_range(const std::map<int32_t, int32_t>& m, int32_t start, int32_t length) {
    std::vector<int32_t> result;
    for (auto it = m.lower_bound(start); it != m.end() && result.size() < static_cast<std::size_t>(length); ++it)
        result.push_back(it->second);
    return result;
}

using SortedVector = std::vector<std::pair<int32_t, int32_t>>;

bool entry_less(const std::pair<int32_t, int32_t>& a, const std::pair<int32_t, int32_t>& b) {
    return a.first < b.first;
}
bool key_less(const std::pair<int32_t, int32_t>& entry, int32_t key) {
    return entry.first < key;
}
void verify_hft(const hft::CacheTree& tree, const Workload& w, const std::map<int32_t, int32_t>& reference) {
    if (tree.size() != reference.size() || !tree.validate()) throw std::runtime_error("tree size or structure mismatch");
    for (const auto& kv : reference)
        if (tree.find(kv.first) != kv.second) throw std::runtime_error("stored value mismatch");
    for (int32_t k : w.lookups) {
        auto it = reference.find(k); auto actual = tree.find(k);
        if ((it == reference.end()) != !actual || (actual && *actual != it->second))
            throw std::runtime_error("lookup mismatch");
    }
    for (const auto& r : w.ranges)
        if (tree.range_search(r.first, r.second) != expected_range(reference, r.first, r.second))
            throw std::runtime_error("range mismatch");
}
void verify_std_map(const std::map<int32_t, int32_t>& tree, const Workload& w,
                    const std::map<int32_t, int32_t>& reference) {
    if (!(tree == reference)) throw std::runtime_error("std::map contents mismatch");
    for (int32_t k : w.lookups) {
        auto it = reference.find(k); auto actual = tree.find(k);
        if ((it == reference.end()) != (actual == tree.end()) ||
            (actual != tree.end() && actual->second != it->second))
            throw std::runtime_error("std::map lookup mismatch");
    }
    for (const auto& r : w.ranges) {
        std::vector<int32_t> result;
        for (auto it = tree.lower_bound(r.first);
             it != tree.end() && result.size() < static_cast<std::size_t>(r.second); ++it)
            result.push_back(it->second);
        if (result != expected_range(reference, r.first, r.second))
            throw std::runtime_error("std::map range mismatch");
    }
}
void verify_std_unordered(const std::unordered_map<int32_t, int32_t>& tree,
                          const Workload& w,
                          const std::map<int32_t, int32_t>& reference) {
    if (tree.size() != reference.size())
        throw std::runtime_error("std::unordered_map size mismatch");
    for (const auto& kv : reference) {
        auto it = tree.find(kv.first);
        if (it == tree.end() || it->second != kv.second)
            throw std::runtime_error("std::unordered_map contents mismatch");
    }
    for (int32_t k : w.lookups) {
        auto it = reference.find(k); auto actual = tree.find(k);
        if ((it == reference.end()) != (actual == tree.end()) ||
            (actual != tree.end() && actual->second != it->second))
            throw std::runtime_error("std::unordered_map lookup mismatch");
    }
}
void verify_sorted_vector(const SortedVector& tree, const Workload& w,
                          const std::map<int32_t, int32_t>& reference) {
    if (tree.size() != reference.size())
        throw std::runtime_error("sorted_vector size mismatch");
    auto entry = tree.begin();
    for (const auto& kv : reference) {
        if (entry == tree.end() || entry->first != kv.first || entry->second != kv.second)
            throw std::runtime_error("sorted_vector contents mismatch");
        ++entry;
    }
    for (int32_t k : w.lookups) {
        auto it = reference.find(k);
        auto actual = std::lower_bound(tree.begin(), tree.end(), k, key_less);
        const bool found = actual != tree.end() && actual->first == k;
        if ((it == reference.end()) != !found || (found && actual->second != it->second))
            throw std::runtime_error("sorted_vector lookup mismatch");
    }
    for (const auto& r : w.ranges) {
        std::vector<int32_t> result;
        for (auto it = std::lower_bound(tree.begin(), tree.end(), r.first, key_less);
             it != tree.end() && result.size() < static_cast<std::size_t>(r.second); ++it)
            result.push_back(it->second);
        if (result != expected_range(reference, r.first, r.second))
            throw std::runtime_error("sorted_vector range mismatch");
    }
}
template<class F> double timed(F f) {
    auto start = Clock::now(); f();
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
Sample run_hft(hft::Backend backend, const Workload& w) {
    hft::CacheTree tree(backend);
    if (tree.backend() != backend) throw std::runtime_error("requested backend was not selected");
    Sample s;
    s.has_range = true;
    s.insert = timed([&] { for (int32_t k : w.inserts) tree.insert(k, k + 1); });
    verify_hft(tree, w, w.initial);
    s.lookup = timed([&] { for (int32_t k : w.lookups) { auto v = tree.find(k); if (v) s.checksum += static_cast<std::uint32_t>(*v); } });
    s.range = timed([&] { for (const auto& r : w.ranges) for (int32_t v : tree.range_search(r.first, r.second)) s.checksum += static_cast<std::uint32_t>(v); });
    s.update = timed([&] { for (int32_t k : w.updates) tree.insert(k, -k - 1); });
    verify_hft(tree, w, w.updated);
    return s;
}
Sample run_std_map(const Workload& w) {
    std::map<int32_t, int32_t> tree;
    Sample s;
    s.has_range = true;
    s.insert = timed([&] { for (int32_t k : w.inserts) tree[k] = k + 1; });
    verify_std_map(tree, w, w.initial);
    s.lookup = timed([&] { for (int32_t k : w.lookups) { auto it = tree.find(k); if (it != tree.end()) s.checksum += static_cast<std::uint32_t>(it->second); } });
    s.range = timed([&] {
        for (const auto& r : w.ranges) {
            std::size_t count = 0;
            for (auto it = tree.lower_bound(r.first);
                 it != tree.end() && count < static_cast<std::size_t>(r.second);
                 ++it, ++count)
                s.checksum += static_cast<std::uint32_t>(it->second);
        }
    });
    s.update = timed([&] { for (int32_t k : w.updates) tree[k] = -k - 1; });
    verify_std_map(tree, w, w.updated);
    return s;
}
Sample run_sorted_vector(const Workload& w) {
    SortedVector tree;
    Sample s;
    s.has_range = true;
    // Bulk load: append every entry, then sort once. This is the static-data
    // baseline; it is not comparable to incremental insertion.
    s.insert = timed([&] {
        tree.reserve(w.inserts.size());
        for (int32_t k : w.inserts) tree.emplace_back(k, k + 1);
        std::sort(tree.begin(), tree.end(), entry_less);
    });
    verify_sorted_vector(tree, w, w.initial);
    s.lookup = timed([&] {
        for (int32_t k : w.lookups) {
            auto it = std::lower_bound(tree.begin(), tree.end(), k, key_less);
            if (it != tree.end() && it->first == k)
                s.checksum += static_cast<std::uint32_t>(it->second);
        }
    });
    s.range = timed([&] {
        for (const auto& r : w.ranges) {
            std::size_t count = 0;
            for (auto it = std::lower_bound(tree.begin(), tree.end(), r.first, key_less);
                 it != tree.end() && count < static_cast<std::size_t>(r.second);
                 ++it, ++count)
                s.checksum += static_cast<std::uint32_t>(it->second);
        }
    });
    s.update = timed([&] {
        for (int32_t k : w.updates) {
            auto it = std::lower_bound(tree.begin(), tree.end(), k, key_less);
            if (it != tree.end() && it->first == k) it->second = -k - 1;
            else tree.insert(it, {k, -k - 1});
        }
    });
    verify_sorted_vector(tree, w, w.updated);
    return s;
}
Sample run_std_unordered(const Workload& w, bool reserve) {
    std::unordered_map<int32_t, int32_t> tree;
    if (reserve) tree.reserve(w.inserts.size());
    Sample s;
    s.has_range = false;
    s.insert = timed([&] { for (int32_t k : w.inserts) tree[k] = k + 1; });
    verify_std_unordered(tree, w, w.initial);
    s.lookup = timed([&] { for (int32_t k : w.lookups) { auto it = tree.find(k); if (it != tree.end()) s.checksum += static_cast<std::uint32_t>(it->second); } });
    // No ordered range query exists for unordered_map; the phase is skipped and
    // reported as not measured rather than emulated by sorting.
    s.update = timed([&] { for (int32_t k : w.updates) tree[k] = -k - 1; });
    verify_std_unordered(tree, w, w.updated);
    return s;
}
Sample run(Target target, const Workload& w, bool reserve_unordered) {
    switch (target) {
        case Target::Scalar:
        case Target::SSE2:
        case Target::AVX2: return run_hft(target_backend(target), w);
        case Target::StdMap: return run_std_map(w);
        case Target::SortedVector: return run_sorted_vector(w);
        case Target::StdUnorderedMap: return run_std_unordered(w, reserve_unordered);
    }
    throw std::runtime_error("unknown benchmark target");
}
double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    auto n = v.size();
    return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
}
std::string csv(const std::string& s) {
    std::string out = "\"";
    for (char c : s) { if (c == '"') out += '"'; out += c; }
    return out + '"';
}
std::string cpu() {
    std::ifstream in("/proc/cpuinfo"); std::string line;
    while (std::getline(in, line)) {
        auto p = line.find(':');
        if (p != std::string::npos && (line.find("model name") == 0 || line.find("Hardware") == 0)) return line.substr(p + 1);
    }
    return "unknown";
}
std::string compiler() {
#if defined(__clang__)
    return std::string("Clang ") + __clang_version__;
#elif defined(__GNUC__)
    return std::string("GCC ") + __VERSION__;
#elif defined(_MSC_VER)
    return "MSVC " + std::to_string(_MSC_VER);
#else
    return "unknown";
#endif
}
} // namespace

int main(int argc, char** argv) {
    std::FILE* file = nullptr;
    std::string output;
    bool created = false;
    try {
        const Options o = parse(argc, argv);
        output = o.output;
        // C11 exclusive create: fail before the long run if the path exists or
        // the parent directory is missing, and never truncate an artifact.
        file = std::fopen(o.output.c_str(), "wx");
        if (!file) throw std::runtime_error("cannot create output (already exists or parent directory missing): " + o.output);
        created = true;
        const Workload w = generate(o);
        std::ostringstream out;
        out << std::setprecision(12);
        out << "record,cpu,compiler,build_flags,line_bytes,available_sse2,available_avx2,distribution,zipf_alpha,seed,insert_ops,distinct_keys,lookup_ops,range_ops,update_ops,backend,range_measured,run,insert_ms,lookup_ms,range_ms,update_ms,checksum\n";
        const auto prefix = csv(cpu()) + "," + csv(compiler()) + "," + csv(o.flags) + "," + std::to_string(HFT_CACHE_LINE_SIZE) + "," + std::to_string(hft::backend_available(hft::Backend::SSE2)) + "," + std::to_string(hft::backend_available(hft::Backend::AVX2)) + "," + o.distribution + "," + (o.distribution == "zipf" ? "1.5" : "") + "," + std::to_string(o.seed) + "," + std::to_string(o.keys) + "," + std::to_string(w.initial.size()) + "," + std::to_string(o.lookups) + "," + std::to_string(o.ranges) + "," + std::to_string(o.updates) + ",";
        std::vector<Target> modes;
        if (o.backend == "all") {
            for (Target target : {Target::Scalar, Target::SSE2, Target::AVX2}) {
                if (!target_available(target)) { std::cout << "Skipping unavailable " << target_name(target) << '\n'; continue; }
                modes.push_back(target);
            }
            // STL baselines are always available and use the same workload.
            modes.push_back(Target::StdMap);
            modes.push_back(Target::SortedVector);
            modes.push_back(Target::StdUnorderedMap);
        } else {
            const Target target = target_from_name(o.backend);
            if (!target_available(target))
                throw std::runtime_error("requested backend is unavailable: " + o.backend);
            modes.push_back(target);
        }
        for (auto mode : modes) run(mode, w, o.reserve); // untimed warmup, including correctness checks
        const std::size_t count = modes.size();
        std::vector<std::vector<double>> ins(count), look(count), ranges(count), updates(count);
        // Rotate the backend order each run so drift does not consistently favor
        // one backend; report the median of each backend's runs.
        for (std::size_t i = 0; i < o.runs; ++i) {
            for (std::size_t k = 0; k < count; ++k) {
                const std::size_t index = (i + k) % count;
                const auto s = run(modes[index], w, o.reserve);
                ins[index].push_back(s.insert); look[index].push_back(s.lookup); ranges[index].push_back(s.range); updates[index].push_back(s.update);
                out << "sample," << prefix << csv(target_name(modes[index])) << ',' << (s.has_range ? 1 : 0) << ',' << i + 1 << ',' << s.insert << ',' << s.lookup << ',';
                if (s.has_range) out << s.range;
                out << ',' << s.update << ',' << s.checksum << '\n';
            }
        }
        for (std::size_t index = 0; index < count; ++index) {
            const bool has_range = target_has_range(modes[index]);
            const double insert = median(ins[index]);
            const double lookup = median(look[index]);
            const double update = median(updates[index]);
            out << "median," << prefix << csv(target_name(modes[index])) << ',' << (has_range ? 1 : 0) << ",," << insert << ',' << lookup << ',';
            if (has_range) out << median(ranges[index]);
            out << ',' << update << ",\n";
            std::cout << target_name(modes[index]) << " median ms: insert " << insert << ", lookup " << lookup << ", range ";
            if (has_range) std::cout << median(ranges[index]);
            else std::cout << "n/a";
            std::cout << ", update " << update << '\n';
        }
        const std::string data = out.str();
        const bool ok = std::fwrite(data.data(), 1, data.size(), file) == data.size();
        const bool closed = std::fclose(file) == 0;
        file = nullptr;
        if (!ok || !closed) throw std::runtime_error("failed writing output: " + o.output);
        std::cout << "Saved " << o.output << '\n';
    } catch (const std::exception& e) {
        if (file) { std::fclose(file); file = nullptr; }
        if (created) std::remove(output.c_str());
        std::cerr << "benchmark: " << e.what() << '\n'; return 1;
    }
}
