// fptree.h - FP-tree data structure (Han, Pei & Yin 2000).
//
// Memory model
// ------------
// All nodes of one tree live in a single std::vector<FPNode> owned by the
// FPTree (an arena / node pool).  Every link (parent, first_child,
// next_sibling, next_link) is a 32-bit index into that vector, with
// FPTree::NONE (-1) meaning "no node".  This was chosen over per-node
// smart pointers because:
//   * an FP-tree is built once and destroyed all at once, never node by node,
//     so one vector deallocation frees everything and nothing can leak;
//   * 4-byte indices halve the link size compared to 8-byte pointers and keep
//     the nodes contiguous in memory (better cache behaviour while mining);
//   * destroying a unique_ptr-linked tree recurses along every path, which can
//     overflow the stack on long transactions, whereas the pool is freed flat;
//   * index-based nodes map directly onto flat device arrays for the planned
//     CUDA version.
// Child lookup during insertion uses a single hash map keyed by
// (parent index, item rank) so that a parent with many children (typically
// the root) is still O(1) to search instead of a linear sibling scan.
#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <vector>

struct FPNode {
    int32_t  rank;          // position of the item in the header table (-1 for the root)
    uint32_t count;         // number of (weighted) transactions passing through this node
    int32_t  parent;        // index of the parent node (-1 for the root)
    int32_t  first_child;   // head of this node's child list
    int32_t  next_sibling;  // next child of the same parent
    int32_t  next_link;     // node-link: next node in the tree carrying the same item
};

struct HeaderEntry {
    int      item;     // original item id
    uint64_t support;  // support of the item within the transactions of this tree
    int32_t  head;     // first node carrying this item (-1 if none)
    int32_t  tail;     // last node in the node-link chain (lets us append in O(1))
};

class FPTree {
public:
    static constexpr int32_t NONE = -1;
    static constexpr int32_t ROOT = 0;

    // `header` must already be sorted by (support desc, item asc).  The index
    // of an entry in this vector is the item's *rank*; transactions are
    // inserted as ascending sequences of ranks, so the most frequent item is
    // always closest to the root.
    explicit FPTree(std::vector<HeaderEntry> header);

    // Insert one transaction given as strictly ascending ranks, adding `count`
    // to every node on its path (count > 1 is used by conditional trees).
    void insert(const int32_t* ranks, std::size_t n, uint32_t count = 1);

    // Release the child-lookup index once no more insertions will happen.
    void freeze();

    // True if the tree is empty or consists of exactly one path from the root.
    bool singlePath() const;

    bool        empty() const     { return nodes_.size() == 1; }
    std::size_t nodeCount() const { return nodes_.size() - 1; }   // excludes the root

    const FPNode&                   node(int32_t i) const { return nodes_[static_cast<std::size_t>(i)]; }
    const std::vector<FPNode>&      nodes() const         { return nodes_; }
    const std::vector<HeaderEntry>& header() const        { return header_; }

    // Human-readable dump of the header table and the tree (for --print-tree).
    void print(std::ostream& os) const;

private:
    int32_t findOrAddChild(int32_t parent, int32_t rank);
    void    printSubtree(std::ostream& os, int32_t id, const std::string& prefix) const;

    std::vector<FPNode>                   nodes_;        // node pool; nodes_[ROOT] is the root
    std::vector<HeaderEntry>              header_;       // indexed by rank
    std::unordered_map<uint64_t, int32_t> child_index_;  // (parent, rank) -> child node
};
