// fpgrowth.cpp - dataset reader, two-pass tree construction, recursive mining.
#include "fpgrowth.h"

#include <algorithm>
#include <climits>
#include <fstream>
#include <iterator>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <utility>

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

Dataset readDataset(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open input file: " + path);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    Dataset          ds;
    std::vector<int> cur;
    std::size_t      i = 0, n = text.size(), lineno = 1;

    auto fail = [&](const char* msg) {
        throw std::runtime_error(path + ":" + std::to_string(lineno) + ": " + msg);
    };
    auto isSpace = [](char c) { return c == ' ' || c == '\t' || c == '\r'; };

    while (i < n) {
        cur.clear();
        while (i < n && text[i] != '\n') {
            const char c = text[i];
            if (isSpace(c)) { ++i; continue; }
            if (c < '0' || c > '9') fail("expected a non-negative integer item id");
            long long v = 0;
            while (i < n && text[i] >= '0' && text[i] <= '9') {
                v = v * 10 + (text[i] - '0');
                if (v > INT_MAX) fail("item id does not fit in a 32-bit int");
                ++i;
            }
            if (i < n && !isSpace(text[i]) && text[i] != '\n')
                fail("expected a non-negative integer item id");
            cur.push_back(static_cast<int>(v));
        }
        if (i < n) ++i;  // consume '\n'

        if (!cur.empty()) {  // blank (or whitespace-only) lines are skipped
            std::sort(cur.begin(), cur.end());
            cur.erase(std::unique(cur.begin(), cur.end()), cur.end());  // drop duplicates
            ds.items.insert(ds.items.end(), cur.begin(), cur.end());
            ds.offsets.push_back(ds.items.size());
        }
        ++lineno;
    }
    return ds;
}

// ---------------------------------------------------------------------------
// Pass 1 + pass 2: initial FP-tree
// ---------------------------------------------------------------------------

std::vector<uint32_t> countItems(const Dataset& ds) {
    std::vector<uint32_t> freq;
    for (int item : ds.items) {
        if (static_cast<std::size_t>(item) >= freq.size()) {
            if (item > kMaxItemId) throw std::runtime_error("item id exceeds supported maximum");
            freq.resize(std::max(static_cast<std::size_t>(item) + 1, freq.size() * 2), 0);
        }
        ++freq[static_cast<std::size_t>(item)];
    }
    return freq;
}

FPTree buildInitialTree(const Dataset& ds, uint64_t min_count, BuildStats& stats) {
    // Pass 1: support of every item.
    return buildTreeFromCounts(ds, countItems(ds), min_count, stats);
}

FPTree buildTreeFromCounts(const Dataset& ds, const std::vector<uint32_t>& freq,
                           uint64_t min_count, BuildStats& stats) {
    std::vector<int> frequent;
    for (std::size_t it = 0; it < freq.size(); ++it) {
        if (freq[it] == 0) continue;
        ++stats.distinct_items;
        if (freq[it] >= min_count) frequent.push_back(static_cast<int>(it));
    }
    // Descending support; ties broken by ascending item id -> deterministic ranks.
    std::sort(frequent.begin(), frequent.end(), [&](int a, int b) {
        const uint32_t fa = freq[static_cast<std::size_t>(a)], fb = freq[static_cast<std::size_t>(b)];
        return fa != fb ? fa > fb : a < b;
    });
    stats.frequent_items = frequent.size();

    std::vector<HeaderEntry> header;
    header.reserve(frequent.size());
    std::vector<int32_t> rankOf(freq.size(), FPTree::NONE);
    for (std::size_t r = 0; r < frequent.size(); ++r) {
        header.push_back(HeaderEntry{frequent[r], freq[static_cast<std::size_t>(frequent[r])],
                                     FPTree::NONE, FPTree::NONE});
        rankOf[static_cast<std::size_t>(frequent[r])] = static_cast<int32_t>(r);
    }

    // Pass 2: insert every transaction, keeping only frequent items, ordered by rank.
    FPTree               tree(std::move(header));
    std::vector<int32_t> ranks;
    for (std::size_t t = 0; t < ds.numTransactions(); ++t) {
        ranks.clear();
        for (std::size_t k = ds.offsets[t]; k < ds.offsets[t + 1]; ++k) {
            const int32_t r = rankOf[static_cast<std::size_t>(ds.items[k])];
            if (r != FPTree::NONE) ranks.push_back(r);
        }
        if (ranks.empty()) continue;
        std::sort(ranks.begin(), ranks.end());
        tree.insert(ranks.data(), ranks.size());
        stats.filtered_occurrences += ranks.size();
    }
    tree.freeze();
    stats.tree_nodes = tree.nodeCount();
    return tree;
}

// ---------------------------------------------------------------------------
// Result sink
// ---------------------------------------------------------------------------

namespace {
void writeItemset(std::ostream& os, const int* first, const int* last, uint64_t support) {
    for (const int* p = first; p != last; ++p) {
        if (p != first) os << ' ';
        os << *p;
    }
    os << " (" << support << ")\n";
}
}  // namespace

void ItemsetSink::add(const int* items, std::size_t n, uint64_t support) {
    scratch_.assign(items, items + n);
    std::sort(scratch_.begin(), scratch_.end());
    ++count_;
    if (stream_) {
        writeItemset(*stream_, scratch_.data(), scratch_.data() + scratch_.size(), support);
    } else {
        buf_.insert(buf_.end(), scratch_.begin(), scratch_.end());
        starts_.push_back(buf_.size());
        supports_.push_back(support);
    }
}

std::vector<uint64_t> ItemsetSink::sortedOrder() const {
    std::vector<uint64_t> order(supports_.size());
    std::iota(order.begin(), order.end(), 0);
    auto first = [&](uint64_t k) { return buf_.data() + starts_[k]; };
    auto last  = [&](uint64_t k) { return buf_.data() + starts_[k + 1]; };
    std::sort(order.begin(), order.end(), [&](uint64_t a, uint64_t b) {
        return std::lexicographical_compare(first(a), last(a), first(b), last(b));
    });
    return order;
}

void ItemsetSink::writeSorted(std::ostream& os) const {
    for (uint64_t k : sortedOrder())
        writeItemset(os, buf_.data() + starts_[k], buf_.data() + starts_[k + 1], supports_[k]);
}

void ItemsetSink::writeMerged(std::ostream& os, const std::vector<const ItemsetSink*>& sinks,
                              const std::vector<std::vector<uint64_t>>& orders) {
    // Min-heap of (sink, position in that sink's order), keyed by the itemset there.
    struct Cursor { std::size_t s; std::size_t pos; };
    auto first = [&](const Cursor& c) {
        const ItemsetSink& k = *sinks[c.s];
        return k.buf_.data() + k.starts_[orders[c.s][c.pos]];
    };
    auto last = [&](const Cursor& c) {
        const ItemsetSink& k = *sinks[c.s];
        return k.buf_.data() + k.starts_[orders[c.s][c.pos] + 1];
    };
    auto after = [&](const Cursor& a, const Cursor& b) {  // heap order: smallest on top
        return std::lexicographical_compare(first(b), last(b), first(a), last(a));
    };
    std::vector<Cursor> heap;
    for (std::size_t s = 0; s < sinks.size(); ++s)
        if (!orders[s].empty()) heap.push_back(Cursor{s, 0});
    std::make_heap(heap.begin(), heap.end(), after);
    while (!heap.empty()) {
        std::pop_heap(heap.begin(), heap.end(), after);
        Cursor& c = heap.back();
        writeItemset(os, first(c), last(c), sinks[c.s]->supports_[orders[c.s][c.pos]]);
        if (++c.pos < orders[c.s].size())
            std::push_heap(heap.begin(), heap.end(), after);
        else
            heap.pop_back();
    }
}

// ---------------------------------------------------------------------------
// Mining
// ---------------------------------------------------------------------------

namespace {

// Methods are defined below the class but marked inline, as they would be if
// defined inside it: without that GCC inlines less of the recursion and the
// dense datasets mine ~2% slower.
class Miner {
public:
    Miner(uint64_t min_count, ItemsetSink& sink, TreeCounter* counter = nullptr)
        : min_count_(min_count), sink_(sink), counter_(counter) {}

    void run(const FPTree& tree) { mineTree(tree); }

    // One iteration of the loop over a (non-single-path) tree's header: emit
    // {item of rank r} and mine its conditional tree.  `row` is rank r's row
    // of a TreeCounter matrix, or null to count on the CPU.
    void mineItem(const FPTree& t, int32_t r, const uint32_t* row);

    const MiningStats& stats() const { return stats_; }

private:
    using PathNode = std::pair<int, uint32_t>;  // (item, count)

    void   emit(uint64_t support) { sink_.add(suffix_.data(), suffix_.size(), support); }
    void   mineTree(const FPTree& t);
    void   mineSinglePath(const FPTree& t);
    void   enumerateCombinations(const std::vector<PathNode>& path, std::size_t from);
    FPTree conditionalTree(const FPTree& t, int32_t r, const uint32_t* row);

    uint64_t         min_count_;
    ItemsetSink&     sink_;
    TreeCounter*     counter_;
    std::vector<int> suffix_;  // items (original ids) fixed by the enclosing recursion levels
    MiningStats      stats_;
};

inline void Miner::mineTree(const FPTree& t) {
    if (t.singlePath()) {
        ++stats_.single_paths;
        mineSinglePath(t);
        return;
    }
    // Process header items from least frequent (deepest) to most frequent.
    const int32_t top = static_cast<int32_t>(t.header().size()) - 1;
    if (!counter_) {
        for (int32_t r = top; r >= 0; --r) mineItem(t, r, nullptr);
        return;
    }
    // Conditional supports of all items at once, if the accelerator takes this tree.
    const std::vector<uint32_t> counts = counter_->countAll(t);
    for (int32_t r = top; r >= 0; --r)
        mineItem(t, r, counts.empty() ? nullptr : counts.data() + TreeCounter::triRow(r));
}

inline void Miner::mineItem(const FPTree& t, int32_t r, const uint32_t* row) {
    const HeaderEntry& h = t.header()[static_cast<std::size_t>(r)];
    suffix_.push_back(h.item);
    emit(h.support);                             // {item} U suffix is frequent
    FPTree cond = conditionalTree(t, r, row);    // patterns that co-occur with it
    if (!cond.empty()) mineTree(cond);
    suffix_.pop_back();
}

// Single-path optimisation: every non-empty combination of the nodes on
// the path is frequent, with support equal to the count of its deepest
// node (counts never increase going down a path).
inline void Miner::mineSinglePath(const FPTree& t) {
    std::vector<PathNode> path;
    for (int32_t n = t.node(FPTree::ROOT).first_child; n != FPTree::NONE;
         n = t.node(n).first_child)
        path.emplace_back(t.header()[static_cast<std::size_t>(t.node(n).rank)].item,
                          t.node(n).count);
    enumerateCombinations(path, 0);
}

inline void Miner::enumerateCombinations(const std::vector<PathNode>& path, std::size_t from) {
    for (std::size_t i = from; i < path.size(); ++i) {
        suffix_.push_back(path[i].first);
        emit(path[i].second);  // path[i] is the deepest chosen node right now
        enumerateCombinations(path, i + 1);
        suffix_.pop_back();
    }
}

// Build the conditional FP-tree of the item with rank r in tree t.
//   1. Walk the node-links of r; for each node, every ancestor (a prefix
//      path) gets the node's count added to its conditional support.
//      (Skipped when `row` already holds these sums from a TreeCounter.)
//   2. Keep ancestors with conditional support >= min_count, order them
//      (support desc, item asc) to obtain the local ranks of the new tree.
//   3. Walk the node-links again, re-label each prefix path with local
//      ranks, sort and insert it weighted by the node's count.
inline FPTree Miner::conditionalTree(const FPTree& t, int32_t r, const uint32_t* row) {
    const std::vector<HeaderEntry>& H = t.header();

    // Step 1: conditional supports (ancestors always have rank < r).
    std::vector<uint64_t> cnt(static_cast<std::size_t>(r), 0);
    if (row) {
        std::copy(row, row + r, cnt.begin());
    } else {
        for (int32_t n = H[static_cast<std::size_t>(r)].head; n != FPTree::NONE; n = t.node(n).next_link) {
            const uint32_t c = t.node(n).count;
            for (int32_t a = t.node(n).parent; a != FPTree::ROOT; a = t.node(a).parent)
                cnt[static_cast<std::size_t>(t.node(a).rank)] += c;
        }
    }

    // Step 2: the header of the conditional tree.
    std::vector<int32_t> keep;
    for (int32_t i = 0; i < r; ++i)
        if (cnt[static_cast<std::size_t>(i)] >= min_count_) keep.push_back(i);
    if (keep.empty()) return FPTree(std::vector<HeaderEntry>{});

    std::sort(keep.begin(), keep.end(), [&](int32_t a, int32_t b) {
        const uint64_t ca = cnt[static_cast<std::size_t>(a)], cb = cnt[static_cast<std::size_t>(b)];
        if (ca != cb) return ca > cb;
        return H[static_cast<std::size_t>(a)].item < H[static_cast<std::size_t>(b)].item;
    });
    std::vector<int32_t>     local(static_cast<std::size_t>(r), FPTree::NONE);  // parent rank -> local rank
    std::vector<HeaderEntry> header;
    header.reserve(keep.size());
    for (std::size_t j = 0; j < keep.size(); ++j) {
        const std::size_t pr = static_cast<std::size_t>(keep[j]);
        local[pr] = static_cast<int32_t>(j);
        header.push_back(HeaderEntry{H[pr].item, cnt[pr], FPTree::NONE, FPTree::NONE});
    }

    // Step 3: insert the (filtered, re-ranked) prefix paths.
    FPTree cond(std::move(header));
    std::vector<int32_t> path;
    for (int32_t n = H[static_cast<std::size_t>(r)].head; n != FPTree::NONE; n = t.node(n).next_link) {
        path.clear();
        for (int32_t a = t.node(n).parent; a != FPTree::ROOT; a = t.node(a).parent) {
            const int32_t l = local[static_cast<std::size_t>(t.node(a).rank)];
            if (l != FPTree::NONE) path.push_back(l);
        }
        if (path.empty()) continue;
        std::sort(path.begin(), path.end());
        cond.insert(path.data(), path.size(), t.node(n).count);
    }
    cond.freeze();
    ++stats_.cond_trees;
    stats_.cond_nodes += cond.nodeCount();
    return cond;
}

}  // namespace

MiningStats fpgrowth(const FPTree& tree, uint64_t min_count, ItemsetSink& sink) {
    Miner m(min_count, sink);
    m.run(tree);
    return m.stats();
}

MiningStats mineTopLevelItem(const FPTree& tree, int32_t r, const uint32_t* row,
                             uint64_t min_count, ItemsetSink& sink, TreeCounter* counter) {
    Miner m(min_count, sink, counter);
    m.mineItem(tree, r, row);
    return m.stats();
}
