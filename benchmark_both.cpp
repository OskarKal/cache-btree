#include "cacheB+tree.cpp"
#include "ordinalB++tree.h"
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <string>

using namespace std;
using namespace std::chrono;

// Zipf distribution generator with rank-to-value mapping
class ZipfDistribution {
private:
    double alpha;
    int n;
    double c;  // normalization constant
    vector<int32_t> rankMap;  // maps rank -> actual key value
    vector<double> cdf;  // cumulative distribution function for sampling
    
public:
    ZipfDistribution(int _n, double _alpha, mt19937& rng) : n(_n), alpha(_alpha) {
        // Create rank map: values 0 to n-1 shuffled
        rankMap.resize(n);
        for (int i = 0; i < n; i++) {
            rankMap[i] = i;
        }
        shuffle(rankMap.begin(), rankMap.end(), rng);
        
        // Compute normalization constant
        c = 0.0;
        for (int i = 1; i <= n; i++) {
            c += 1.0 / pow(i, alpha);
        }
        c = 1.0 / c;
        // Precompute CDF for sampling
        cdf.resize(n);
        double cumulative = 0.0;
        for (int i = 1; i <= n; i++) {
            cumulative += c / pow(i, alpha);
            cdf[i - 1] = cumulative;
        }

    }
    
    int32_t operator()(mt19937& rng) {
        uniform_real_distribution<double> uniform(0.0, 1.0);
        double z = uniform(rng);

        
 
        // Binary search
        int l = 0, r = n - 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (cdf[mid] >= z) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return rankMap[l];

    }
};

struct BenchResult {
    double insertTime;
    double searchTime;
    double rangeSearchTime;
    double updateTime;
};

// 1~chmark CacheFriendlyBTree
BenchResult benchmarkBPlusTree(const vector<int32_t>& insertKeys, 
                               const vector<int32_t>& searchKeys,
                               const vector<pair<int32_t, int32_t>>& rangeSearches,
                               const vector<int32_t>& updateKeys) {
    BPlusTree::CacheFriendlyBTree tree;
    
    // Insert benchmark
    auto start = high_resolution_clock::now();
    for (int32_t key : insertKeys) {
        tree.insert(key, key * 2);
    }
    auto end = high_resolution_clock::now();
    double insertTime = duration<double, milli>(end - start).count();
    
    // Search benchmark
    start = high_resolution_clock::now();
    volatile int32_t sum = 0;
    for (int32_t key : searchKeys) {
        int32_t val = tree.search(key);
        if (val != -1) sum += val;
    }
    end = high_resolution_clock::now();
    double searchTime = duration<double, milli>(end - start).count();
    
    // Range search benchmark
    start = high_resolution_clock::now();
    volatile int32_t rangeSum = 0;
    for (const auto& range : rangeSearches) {
        auto results = tree.range_search(range.first, range.second);
        for (int32_t val : results) {
            rangeSum += val;
        }
    }
    end = high_resolution_clock::now();
    double rangeSearchTime = duration<double, milli>(end - start).count();
    
    // Update benchmark
    start = high_resolution_clock::now();
    for (int32_t key : updateKeys) {
        tree.insert(key, key * 3);
    }
    end = high_resolution_clock::now();
    double updateTime = duration<double, milli>(end - start).count();
    
    return {insertTime, searchTime, rangeSearchTime, updateTime};
}

// Benchmark Ordinal BPlusTree
BenchResult benchmarkOrdinalTree(const vector<int32_t>& insertKeys, 
                                 const vector<int32_t>& searchKeys,
                                 const vector<pair<int32_t, int32_t>>& rangeSearches,
                                 const vector<int32_t>& updateKeys) {
    BPlusTree<int32_t> tree(7);  // degree 4 (similar branching to cache tree)
    
    // Insert benchmark
    auto start = high_resolution_clock::now();
    for (int32_t key : insertKeys) {
        tree.insert(key);
    }
    auto end = high_resolution_clock::now();
    double insertTime = duration<double, milli>(end - start).count();
    
    // Search benchmark
    start = high_resolution_clock::now();
    volatile bool found = false;
    for (int32_t key : searchKeys) {
        found = tree.search(key);
    }
    end = high_resolution_clock::now();
    double searchTime = duration<double, milli>(end - start).count();
    
    // Range search benchmark
    start = high_resolution_clock::now();
    volatile int32_t rangeSum = 0;
    for (const auto& range : rangeSearches) {
            // Allocate enough space for results
            auto results = tree.range_search(range.first, range.second, 10000);
        for (int32_t val : results) {
            rangeSum += val;
        }
    }
    end = high_resolution_clock::now();
    double rangeSearchTime = duration<double, milli>(end - start).count();
    
    // Update benchmark
    start = high_resolution_clock::now();
    for (int32_t key : updateKeys) {
        tree.insert(key);
    }
    end = high_resolution_clock::now();
    double updateTime = duration<double, milli>(end - start).count();
    
    return {insertTime, searchTime, rangeSearchTime, updateTime};
}

int main(int argc, char** argv) {
    string distribution = "uniform";  // default distribution
    double zipf_alpha = 1.5;  // default Zipf exponent
    
    // Parse command-line arguments
    if (argc > 1) {
        distribution = argv[1];
    }
    if (argc > 2) {
        zipf_alpha = stod(argv[2]);
    }
    
    // Validate distribution
    if (distribution != "uniform" && distribution != "zipf") {
        cout << "Usage: " << argv[0] << " [uniform|zipf] [zipf_alpha]" << endl;
        cout << "  uniform  - uniform distribution (default)" << endl;
        cout << "  zipf     - Zipf distribution (use with alpha parameter)" << endl;
        cout << "  alpha    - Zipf exponent (default: 1.5, higher = more skewed)" << endl;
        return 1;
    }
    
    const vector<int> OP_SIZES = {10000, 50000, 100000, 250000, 500000};
    const int NUM_RUNS = 10;
    
    // Open CSV file
    ofstream csvFile("benchmark_comparison.csv");
    csvFile << "distribution,alpha,operations,tree_type,inserts,searches,range_searches,updates,insert_ms,search_ms,range_search_ms,update_ms,total_ms,insert_Mops_sec,search_Mops_sec,range_search_Mops_sec,update_Mops_sec\n";
    
    cout << "=== B+ Tree Comparison: Cache vs Ordinal ===" << endl;
    cout << "Distribution: " << distribution;
    if (distribution == "zipf") {
        cout << " (alpha=" << fixed << setprecision(2) << zipf_alpha << ")";
    }
    cout << endl;
    cout << "Runs per config: " << NUM_RUNS << " iterations (median reported)" << endl;
    cout << "Results will be saved to: benchmark_comparison.csv" << endl;
    cout << "\nWarming up CPU..." << endl;
    
    // Warmup
    BPlusTree::CacheFriendlyBTree warmBPlusTree;
    for (int j = 0; j < 5000; j++) {
        warmBPlusTree.insert(j, j);
        warmBPlusTree.search(j / 2);
    }
    
    BPlusTree<int32_t> warmOrdinalTree(4);
    for (int j = 0; j < 5000; j++) {
        warmOrdinalTree.insert(j);
        warmOrdinalTree.search(j / 2);
    }
    
    // Benchmark different sizes
    for (int NUM_OPS : OP_SIZES) {
        int NUM_SEARCHES = NUM_OPS * 5;
        int NUM_RANGE_SEARCHES = NUM_OPS / 20;  // Fewer range searches as they're more expensive
        int NUM_UPDATES = NUM_OPS / 10;
        
        cout << "\n========================================" << endl;
        cout << "Testing with " << NUM_OPS << " operations" << endl;
        cout << "  Inserts: " << NUM_OPS << endl;
        cout << "  Searches: " << NUM_SEARCHES << endl;
        cout << "  Range Searches: " << NUM_RANGE_SEARCHES << endl;
        cout << "  Updates: " << NUM_UPDATES << endl;
        cout << "========================================" << endl;
        
        // Generate test data (same for both trees)
        mt19937 rng(42);
        uniform_int_distribution<int32_t> uniform_dist(0, NUM_OPS * 10);
        ZipfDistribution zipf_dist(NUM_OPS * 10, zipf_alpha, rng);
        
        vector<int32_t> insertKeys(NUM_OPS);
        for (int i = 0; i < NUM_OPS; i++) {
            if (distribution == "uniform") {
                insertKeys[i] = uniform_dist(rng);
            } else {
                insertKeys[i] = zipf_dist(rng);
            }
        }
        
        vector<int32_t> searchKeys(NUM_SEARCHES);
        for (int i = 0; i < NUM_SEARCHES; i++) {
            searchKeys[i] = insertKeys[rng() % NUM_OPS];
        }
        
        // Generate range search pairs with geometric distribution for range sizes
        vector<pair<int32_t, int32_t>> rangeSearches(NUM_RANGE_SEARCHES);
        geometric_distribution<int> geom_dist(0.5);  // Expected range size ~1 item
        for (int i = 0; i < NUM_RANGE_SEARCHES; i++) {
            int32_t start = insertKeys[rng() % NUM_OPS];
            int32_t length = geom_dist(rng) + 1;  // At least 1
            rangeSearches[i] = {start, length};
        }
        
        vector<int32_t> updateKeys(NUM_UPDATES);
        for (int i = 0; i < NUM_UPDATES; i++) {
            updateKeys[i] = insertKeys[rng() % NUM_OPS];
        }
        
        // Benchmark BPlusTree
        cout << "\nBenchmarking CacheFriendlyBTree..." << endl;
        vector<BenchResult> cacheResults;
        for (int run = 0; run < NUM_RUNS; run++) {
            auto result = benchmarkBPlusTree(insertKeys, searchKeys, rangeSearches, updateKeys);
            cacheResults.push_back(result);
            cout << "  Run " << setw(2) << (run + 1) << ": Insert=" << fixed << setprecision(2) 
                 << setw(8) << result.insertTime << "ms, Search=" << setw(8) << result.searchTime 
                 << "ms, RangeSearch=" << setw(8) << result.rangeSearchTime << "ms, Update=" << setw(7) << result.updateTime << "ms" << endl;
        }
        
        // Benchmark OrdinalTree
        // cout << "\nBenchmarking Ordinal BPlusTree..." << endl;
        // vector<BenchResult> ordinalResults;
        // for (int run = 0; run < NUM_RUNS; run++) {
        //     auto result = benchmarkOrdinalTree(insertKeys, searchKeys, rangeSearches, updateKeys);
        //     ordinalResults.push_back(result);
        //     cout << "  Run " << setw(2) << (run + 1) << ": Insert=" << fixed << setprecision(2) 
        //          << setw(8) << result.insertTime << "ms, Search=" << setw(8) << result.searchTime 
        //          << "ms, RangeSearch=" << setw(8) << result.rangeSearchTime << "ms, Update=" << setw(7) << result.updateTime << "ms" << endl;
        // }
        
        // Calculate medians
        auto getMedian = [](vector<double>& v) {
            sort(v.begin(), v.end());
            return v[v.size() / 2];
        };
        
        // Cache tree medians
        vector<double> cacheInserts, cacheSearches, cacheRangeSearches, cacheUpdates;
        for (const auto& r : cacheResults) {
            cacheInserts.push_back(r.insertTime);
            cacheSearches.push_back(r.searchTime);
            cacheRangeSearches.push_back(r.rangeSearchTime);
            cacheUpdates.push_back(r.updateTime);
        }
        
        // Ordinal tree medians
        // vector<double> ordinalInserts, ordinalSearches, ordinalRangeSearches, ordinalUpdates;
        // for (const auto& r : ordinalResults) {
        //     ordinalInserts.push_back(r.insertTime);
        //     ordinalSearches.push_back(r.searchTime);
        //     ordinalRangeSearches.push_back(r.rangeSearchTime);
        //     ordinalUpdates.push_back(r.updateTime);
        // }
        
        double cacheInsertMedian = getMedian(cacheInserts);
        double cacheSearchMedian = getMedian(cacheSearches);
        double cacheRangeSearchMedian = getMedian(cacheRangeSearches);
        double cacheUpdateMedian = getMedian(cacheUpdates);
        
        // double ordinalInsertMedian = getMedian(ordinalInserts);
        // double ordinalSearchMedian = getMedian(ordinalSearches);
        // double ordinalRangeSearchMedian = getMedian(ordinalRangeSearches);
        // double ordinalUpdateMedian = getMedian(ordinalUpdates);
        
        // Write to CSV
        auto writeCSV = [&](const string& treeType, double insertMs, double searchMs, double rangeSearchMs, double updateMs) {
            csvFile << distribution << ",";
            if (distribution == "zipf") {
                csvFile << fixed << setprecision(2) << zipf_alpha << ",";
            } else {
                csvFile << "N/A,";
            }
            csvFile << NUM_OPS << "," 
                    << treeType << "," 
                    << NUM_OPS << "," 
                    << NUM_SEARCHES << "," 
                    << NUM_RANGE_SEARCHES << "," 
                    << NUM_UPDATES << ","
                    << fixed << setprecision(3) << insertMs << ","
                    << searchMs << ","
                    << rangeSearchMs << ","
                    << updateMs << ","
                    << (insertMs + searchMs + rangeSearchMs + updateMs) << ","
                    << setprecision(4) << (NUM_OPS / insertMs * 1000.0 / 1000000.0) << ","
                    << (NUM_SEARCHES / searchMs * 1000.0 / 1000000.0) << ","
                    << (NUM_RANGE_SEARCHES / rangeSearchMs * 1000.0 / 1000000.0) << ","
                    << (NUM_UPDATES / updateMs * 1000.0 / 1000000.0) << "\n";
        };
        
        writeCSV("CacheFriendly", cacheInsertMedian, cacheSearchMedian, cacheRangeSearchMedian, cacheUpdateMedian);
        // writeCSV("Ordinal", ordinalInsertMedian, ordinalSearchMedian, ordinalRangeSearchMedian, ordinalUpdateMedian);
        
        // Print results
        cout << "\n--- MEDIAN RESULTS ---" << endl;
        cout << fixed << setprecision(2);
        cout << "\nCacheFriendlyBTree:" << endl;
        cout << "  Insert:       " << setw(8) << cacheInsertMedian << " ms (" 
             << setw(6) << setprecision(3) << (NUM_OPS / cacheInsertMedian * 1000.0 / 1000000.0) << " M ops/sec)" << endl;
        cout << "  Search:       " << setw(8) << setprecision(2) << cacheSearchMedian << " ms (" 
             << setw(6) << setprecision(3) << (NUM_SEARCHES / cacheSearchMedian * 1000.0 / 1000000.0) << " M ops/sec)" << endl;
        cout << "  Range Search: " << setw(8) << setprecision(2) << cacheRangeSearchMedian << " ms (" 
             << setw(6) << setprecision(3) << (NUM_RANGE_SEARCHES / cacheRangeSearchMedian * 1000.0 / 1000000.0) << " M ops/sec)" << endl;
        cout << "  Update:       " << setw(8) << setprecision(2) << cacheUpdateMedian << " ms (" 
             << setw(6) << setprecision(3) << (NUM_UPDATES / cacheUpdateMedian * 1000.0 / 1000000.0) << " M ops/sec)" << endl;
        cout << "  Total:        " << setw(8) << setprecision(2) << (cacheInsertMedian + cacheSearchMedian + cacheRangeSearchMedian + cacheUpdateMedian) << " ms" << endl;
        
        // cout << "\nOrdinal BPlusTree:" << endl;
        // cout << "  Insert:       " << setw(8) << ordinalInsertMedian << " ms (" 
        //      << setw(6) << setprecision(3) << (NUM_OPS / ordinalInsertMedian * 1000.0 / 1000000.0) << " M ops/sec)" << endl;
        // cout << "  Search:       " << setw(8) << setprecision(2) << ordinalSearchMedian << " ms (" 
        //      << setw(6) << setprecision(3) << (NUM_SEARCHES / ordinalSearchMedian * 1000.0 / 1000000.0) << " M ops/sec)" << endl;
        // cout << "  Range Search: " << setw(8) << setprecision(2) << ordinalRangeSearchMedian << " ms (" 
        //      << setw(6) << setprecision(3) << (NUM_RANGE_SEARCHES / ordinalRangeSearchMedian * 1000.0 / 1000000.0) << " M ops/sec)" << endl;
        // cout << "  Update:       " << setw(8) << setprecision(2) << ordinalUpdateMedian << " ms (" 
        //      << setw(6) << setprecision(3) << (NUM_UPDATES / ordinalUpdateMedian * 1000.0 / 1000000.0) << " M ops/sec)" << endl;
        // cout << "  Total:        " << setw(8) << setprecision(2) << (ordinalInsertMedian + ordinalSearchMedian + ordinalRangeSearchMedian + ordinalUpdateMedian) << " ms" << endl;
        
        // cout << "\n--- SPEEDUP (Cache vs Ordinal) ---" << endl;
        // cout << "  Insert:       " << setprecision(2) << (ordinalInsertMedian / cacheInsertMedian) << "x faster" << endl;
        // cout << "  Search:       " << (ordinalSearchMedian / cacheSearchMedian) << "x faster" << endl;
        // cout << "  Range Search: " << (ordinalRangeSearchMedian / cacheRangeSearchMedian) << "x faster" << endl;
        // cout << "  Update:       " << (ordinalUpdateMedian / cacheUpdateMedian) << "x faster" << endl;
        // cout << "  Overall:      " << ((ordinalInsertMedian + ordinalSearchMedian + ordinalRangeSearchMedian + ordinalUpdateMedian) / 
        //                                (cacheInsertMedian + cacheSearchMedian + cacheRangeSearchMedian + cacheUpdateMedian)) << "x faster" << endl;
    }
    
    csvFile.close();
    cout << "\n========================================" << endl;
    cout << "Benchmark complete!" << endl;
    cout << "Results saved to: benchmark_comparison.csv" << endl;
    cout << "========================================" << endl;
    
    return 0;
}
