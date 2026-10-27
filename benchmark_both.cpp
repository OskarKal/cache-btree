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
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
struct Options {
    std::string output, distribution = "uniform", flags = "not supplied";
    std::size_t keys = 10000, lookups = 20000, ranges = 300, updates = 2000, runs = 5;
    std::uint32_t seed = 42;
};
struct Workload {
    std::vector<int32_t> inserts, lookups, updates;
    std::vector<std::pair<int32_t, int32_t>> ranges;
    std::map<int32_t, int32_t> initial, updated;
};
struct Sample { double insert, lookup, range, update; std::uint64_t checksum; };

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
            std::cout << "Usage: hft_bench --output PATH [--distribution uniform|zipf] [--keys N] [--lookups N] [--ranges N] [--updates N] [--runs N] [--seed N] [--build-flags TEXT]\n";
            std::exit(0);
        }
        if (++i == argc) throw std::invalid_argument("missing value for " + arg);
        std::string val = argv[i];
        if (arg == "--output") o.output = val;
        else if (arg == "--distribution") o.distribution = val;
        else if (arg == "--keys") o.keys = number(val, 10000000);
        else if (arg == "--lookups") o.lookups = number(val, 100000000);
        else if (arg == "--ranges") o.ranges = number(val, 10000000);
        else if (arg == "--updates") o.updates = number(val, 100000000);
        else if (arg == "--runs") o.runs = number(val, 10000);
        else if (arg == "--seed") o.seed = static_cast<std::uint32_t>(number(val, UINT32_MAX));
        else if (arg == "--build-flags") o.flags = val;
        else throw std::invalid_argument("unknown option: " + arg);
    }
    if (o.output.empty()) throw std::invalid_argument("--output is required");
    if (o.distribution != "uniform" && o.distribution != "zipf")
        throw std::invalid_argument("distribution must be uniform or zipf");
    return o;
}

Workload generate(const Options& o) {
    std::mt19937 rng(o.seed);
    // Shuffle Zipf ranks so popular keys are not clustered in key order.
    std::vector<int32_t> ranks(o.keys);
    std::iota(ranks.begin(), ranks.end(), 0);
    std::shuffle(ranks.begin(), ranks.end(), rng);
    std::vector<double> cdf;
    if (o.distribution == "zipf") {
        cdf.resize(o.keys);
        double total = 0;
        for (std::size_t i = 0; i < o.keys; ++i) { total += 1.0 / std::pow(double(i + 1), 1.5); cdf[i] = total; }
        for (double& x : cdf) x /= total;
        cdf.back() = 1.0;
    }
    auto key = [&]() -> int32_t {
        std::size_t rank;
        if (cdf.empty()) rank = std::uniform_int_distribution<std::size_t>(0, o.keys - 1)(rng);
        else rank = std::lower_bound(cdf.begin(), cdf.end(), std::generate_canonical<double, 53>(rng)) - cdf.begin();
        return ranks[rank] * 2; // odd keys are guaranteed misses; bounded by --keys
    };
    Workload w;
    for (std::size_t i = 0; i < o.keys; ++i) {
        int32_t k = key(); w.inserts.push_back(k); w.initial[k] = k + 1;
    }
    std::uniform_int_distribution<std::size_t> pick(0, w.inserts.size() - 1);
    for (std::size_t i = 0; i < o.lookups; ++i) {
        const int32_t k = w.inserts[pick(rng)];
        w.lookups.push_back(k + static_cast<int32_t>(i % 2));
    }
    constexpr int32_t lengths[] = {1, 8, 64};
    for (std::size_t i = 0; i < o.ranges; ++i)
        w.ranges.emplace_back(w.inserts[pick(rng)] + static_cast<int32_t>(i % 2), lengths[i % 3]);
    w.updated = w.initial;
    for (std::size_t i = 0; i < o.updates; ++i) {
        const int32_t k = w.inserts[pick(rng)]; w.updates.push_back(k); w.updated[k] = -k - 1;
    }
    return w;
}
std::vector<int32_t> expected_range(const std::map<int32_t, int32_t>& m, int32_t start, int32_t length) {
    std::vector<int32_t> result;
    for (auto it = m.lower_bound(start); it != m.end() && result.size() < static_cast<std::size_t>(length); ++it)
        result.push_back(it->second);
    return result;
}
void verify(const hft::CacheTree& tree, const Workload& w, const std::map<int32_t, int32_t>& reference) {
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
template<class F> double timed(F f) {
    auto start = Clock::now(); f();
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
Sample run(hft::Backend backend, const Workload& w) {
    hft::CacheTree tree(backend);
    if (tree.backend() != backend) throw std::runtime_error("requested backend was not selected");
    Sample s{};
    s.insert = timed([&] { for (int32_t k : w.inserts) tree.insert(k, k + 1); });
    verify(tree, w, w.initial);
    s.lookup = timed([&] { for (int32_t k : w.lookups) { auto v = tree.find(k); if (v) s.checksum += static_cast<std::uint32_t>(*v); } });
    s.range = timed([&] { for (const auto& r : w.ranges) for (int32_t v : tree.range_search(r.first, r.second)) s.checksum += static_cast<std::uint32_t>(v); });
    s.update = timed([&] { for (int32_t k : w.updates) tree.insert(k, -k - 1); });
    verify(tree, w, w.updated);
    return s;
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
    try {
        const Options o = parse(argc, argv);
        const Workload w = generate(o);
        std::ostringstream out;
        out << std::setprecision(12);
        out << "record,cpu,compiler,build_flags,line_bytes,available_sse2,available_avx2,distribution,zipf_alpha,seed,insert_ops,distinct_keys,lookup_ops,range_ops,update_ops,backend,run,insert_ms,lookup_ms,range_ms,update_ms,checksum\n";
        const auto prefix = csv(cpu()) + "," + csv(compiler()) + "," + csv(o.flags) + "," + std::to_string(HFT_CACHE_LINE_SIZE) + "," + std::to_string(hft::backend_available(hft::Backend::SSE2)) + "," + std::to_string(hft::backend_available(hft::Backend::AVX2)) + "," + o.distribution + "," + (o.distribution == "zipf" ? "1.5" : "") + "," + std::to_string(o.seed) + "," + std::to_string(o.keys) + "," + std::to_string(w.initial.size()) + "," + std::to_string(o.lookups) + "," + std::to_string(o.ranges) + "," + std::to_string(o.updates) + ",";
        for (auto mode : {hft::Backend::Scalar, hft::Backend::SSE2, hft::Backend::AVX2}) {
            if (!hft::backend_available(mode)) { std::cout << "Skipping unavailable " << hft::backend_name(mode) << '\n'; continue; }
            run(mode, w); // untimed warmup, including correctness checks
            std::vector<double> ins, look, ranges, updates;
            for (std::size_t i = 0; i < o.runs; ++i) {
                const auto s = run(mode, w);
                ins.push_back(s.insert); look.push_back(s.lookup); ranges.push_back(s.range); updates.push_back(s.update);
                out << "sample," << prefix << csv(hft::backend_name(mode)) << ',' << i + 1 << ',' << s.insert << ',' << s.lookup << ',' << s.range << ',' << s.update << ',' << s.checksum << '\n';
            }
            out << "median," << prefix << csv(hft::backend_name(mode)) << ",," << median(ins) << ',' << median(look) << ',' << median(ranges) << ',' << median(updates) << ",\n";
            std::cout << hft::backend_name(mode) << " median ms: insert " << median(ins) << ", lookup " << median(look) << ", range " << median(ranges) << ", update " << median(updates) << '\n';
        }
        // C11 exclusive create: never truncate an existing artifact, even after a race.
        std::FILE* file = std::fopen(o.output.c_str(), "wx");
        if (!file) throw std::runtime_error("cannot create output (already exists or parent directory missing): " + o.output);
        const std::string data = out.str();
        const bool ok = std::fwrite(data.data(), 1, data.size(), file) == data.size();
        const bool closed = std::fclose(file) == 0;
        if (!ok || !closed) throw std::runtime_error("failed writing output: " + o.output);
        std::cout << "Saved " << o.output << '\n';
    } catch (const std::exception& e) {
        std::cerr << "benchmark: " << e.what() << '\n'; return 1;
    }
}
