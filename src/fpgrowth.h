// fpgrowth.h - input loading, initial FP-tree construction and FP-Growth mining.
#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <utility>
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
    std::size_t filtered_occurrences = 0;  // frequent-item occurrences inserted into the tree
};

// Largest item id accepted (pass 1 counts into a dense array indexed by item).
constexpr int kMaxItemId = 100'000'000;  // 400 MB of uint32 counters

// Pass 1 (count + order items) and pass 2 (insert transactions).
FPTree buildInitialTree(const Dataset& ds, uint64_t min_count, BuildStats& stats);

// The two passes separately, so pass 1 can run elsewhere (the GPU binary).
// countItems returns freq[item] = number of transactions containing the item;
// the vector may extend past the largest item id with zero entries.
std::vector<uint32_t> countItems(const Dataset& ds);
FPTree buildTreeFromCounts(const Dataset& ds, const std::vector<uint32_t>& freq,
                           uint64_t min_count, BuildStats& stats);

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

    // Buffered mode only: indices of the buffered itemsets in sorted order.
    std::vector<uint64_t> sortedOrder() const;

    // Buffered mode only: write the itemsets of several sinks as one sorted
    // list (k-way merge), given each sink's sortedOrder().  Each itemset must
    // be in exactly one sink, which holds for the parallel miners; the output
    // is then the same as writeSorted() on a single sink holding them all.
    static void writeMerged(std::ostream& os, const std::vector<const ItemsetSink*>& sinks,
                            const std::vector<std::vector<uint64_t>>& orders);

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

// Optional accelerator for the counting step of conditional-tree construction
// (implemented on the GPU by the CUDA binary).  countAll() returns, for every
// item of `t` at once, its conditional supports as a lower-triangular matrix:
// row r starts at triRow(r) and holds the summed counts of rank r's nodes for
// each ancestor rank 0 .. r-1.  An empty result means "count on the CPU".
class TreeCounter {
public:
    virtual ~TreeCounter() = default;
    virtual std::vector<uint32_t> countAll(const FPTree& t) = 0;
    static constexpr uint64_t triRow(int32_t r) {
        return static_cast<uint64_t>(r) * static_cast<uint64_t>(r - 1) / 2;
    }
};

// The counting step of conditional-tree construction for every item of `t`
// on one CPU thread: the same node-link and parent walk that fpgrowth() does
// per item, written into a TreeCounter-layout matrix.  Used to compare the CPU
// with the GPU kernel (fpgrowth_cuda --micro-count).
std::vector<uint64_t> conditionalSupportsCpu(const FPTree& t);

// One iteration of FP-Growth's top-level loop, for the parallel binaries:
// emit {item of rank r} and mine its conditional tree into `sink`.  `tree`
// must not be a single path (fpgrowth() handles that case by itself).  `row`
// is rank r's row of a TreeCounter matrix for `tree`, or null to count on the
// CPU; `counter`, if given, is offered every conditional tree further down.
// The union of all ranks' results equals fpgrowth()'s.
MiningStats mineTopLevelItem(const FPTree& tree, int32_t r, const uint32_t* row,
                             uint64_t min_count, ItemsetSink& sink, TreeCounter* counter);
