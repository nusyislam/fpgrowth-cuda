// fpgrowth.h - input loading, initial FP-tree construction and FP-Growth mining.
#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

#include "fptree.h"

// Transactions stored back to back (CSR layout): transaction t occupies
// items[offsets[t] .. offsets[t+1]).  Items within a transaction are sorted
// ascending and de-duplicated by the reader.
struct Dataset {
    std::vector<int>         items;
    std::vector<std::size_t> offsets{0};
    std::size_t numTransactions() const { return offsets.size() - 1; }
};

// Read a FIMI-style .dat file: one transaction per line, space-separated
// non-negative integers.  Blank lines are skipped, duplicate items within a
// line are dropped.  Throws std::runtime_error on I/O or syntax errors.
Dataset readDataset(const std::string& path);

struct BuildStats {
    std::size_t distinct_items = 0;  // items that appear at least once
    std::size_t frequent_items = 0;  // items with support >= min_count
    std::size_t tree_nodes     = 0;  // nodes in the initial FP-tree (root excluded)
};

// Pass 1 (count + order items) and pass 2 (insert transactions).
FPTree buildInitialTree(const Dataset& ds, uint64_t min_count, BuildStats& stats);

// Receives frequent itemsets.  With a stream it writes each itemset as soon as
// it is found; without one it buffers them so writeSorted() can emit them in
// a canonical, diff-able order.
class ItemsetSink {
public:
    explicit ItemsetSink(std::ostream* stream = nullptr) : stream_(stream) {}

    // `items` may be in any order; they are sorted ascending before output.
    void     add(const int* items, std::size_t n, uint64_t support);
    uint64_t count() const { return count_; }

    // Buffered mode only: sort all itemsets lexicographically and write them.
    void writeSorted(std::ostream& os) const;

private:
    std::ostream*         stream_;
    uint64_t              count_ = 0;
    std::vector<int>      scratch_;
    std::vector<int>      buf_;          // all buffered itemsets, concatenated
    std::vector<uint64_t> starts_{0};    // starts_[k] = offset of itemset k in buf_
    std::vector<uint64_t> supports_;
};

struct MiningStats {
    uint64_t cond_trees = 0;  // conditional FP-trees that were built (non-empty)
    uint64_t cond_nodes = 0;  // total nodes across all conditional trees
    uint64_t single_paths = 0;  // trees handled by the single-path shortcut
};

// Recursive FP-Growth over `tree`; every frequent itemset is passed to `sink`.
MiningStats fpgrowth(const FPTree& tree, uint64_t min_count, ItemsetSink& sink);
