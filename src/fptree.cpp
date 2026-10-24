// fptree.cpp - FP-tree construction, single-path test and pretty printing.
#include "fptree.h"

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <ostream>
#include <utility>

FPTree::FPTree(std::vector<HeaderEntry> header) : header_(std::move(header)) {
    for (HeaderEntry& h : header_) {
        h.head = NONE;
        h.tail = NONE;
    }
    nodes_.push_back(FPNode{-1, 0, NONE, NONE, NONE, NONE});  // the root
}

int32_t FPTree::findOrAddChild(int32_t parent, int32_t rank) {
    const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(parent)) << 32) |
                         static_cast<uint32_t>(rank);
    auto it = child_index_.find(key);
    if (it != child_index_.end()) return it->second;

    // Create the node and hook it into the parent's child list ...
    const int32_t id      = static_cast<int32_t>(nodes_.size());
    const int32_t sibling = nodes_[static_cast<std::size_t>(parent)].first_child;
    nodes_.push_back(FPNode{rank, 0, parent, NONE, sibling, NONE});
    nodes_[static_cast<std::size_t>(parent)].first_child = id;

    // ... and append it to the node-link chain of its item in the header table.
    HeaderEntry& h = header_[static_cast<std::size_t>(rank)];
    if (h.tail == NONE)
        h.head = id;
    else
        nodes_[static_cast<std::size_t>(h.tail)].next_link = id;
    h.tail = id;

    child_index_.emplace(key, id);
    return id;
}

void FPTree::insert(const int32_t* ranks, std::size_t n, uint32_t count) {
    int32_t cur = ROOT;
    for (std::size_t i = 0; i < n; ++i) {
        assert(i == 0 || ranks[i - 1] < ranks[i]);  // strictly ascending ranks
        assert(ranks[i] >= 0 && static_cast<std::size_t>(ranks[i]) < header_.size());
        cur = findOrAddChild(cur, ranks[i]);
        nodes_[static_cast<std::size_t>(cur)].count += count;
    }
}

void FPTree::freeze() {
    std::unordered_map<uint64_t, int32_t>().swap(child_index_);
}

bool FPTree::singlePath() const {
    for (int32_t n = ROOT; nodes_[static_cast<std::size_t>(n)].first_child != NONE;
         n = nodes_[static_cast<std::size_t>(n)].first_child) {
        const int32_t child = nodes_[static_cast<std::size_t>(n)].first_child;
        if (nodes_[static_cast<std::size_t>(child)].next_sibling != NONE) return false;  // a fork
    }
    return true;
}

void FPTree::print(std::ostream& os) const {
    os << "Header table: " << header_.size()
       << " frequent items, rank = position in (support desc, item asc) order\n";
    os << "  rank  item  support  node-links (item:count, in creation order)\n";
    for (std::size_t r = 0; r < header_.size(); ++r) {
        const HeaderEntry& h = header_[r];
        os << "  " << std::setw(4) << r << "  " << std::setw(4) << h.item << "  "
           << std::setw(7) << h.support << "  ";
        for (int32_t n = h.head; n != NONE; n = nodes_[static_cast<std::size_t>(n)].next_link) {
            os << ' ' << h.item << ':' << nodes_[static_cast<std::size_t>(n)].count;
            if (nodes_[static_cast<std::size_t>(n)].next_link != NONE) os << " ->";
        }
        os << '\n';
    }
    os << "FP-tree: " << nodeCount() << " nodes (excluding root), shown as item:count\n";
    os << "(root)\n";
    printSubtree(os, ROOT, "");
}

void FPTree::printSubtree(std::ostream& os, int32_t id, const std::string& prefix) const {
    // Children are prepended to the sibling list, so reverse to show insertion order.
    std::vector<int32_t> kids;
    for (int32_t c = nodes_[static_cast<std::size_t>(id)].first_child; c != NONE;
         c = nodes_[static_cast<std::size_t>(c)].next_sibling)
        kids.push_back(c);
    std::reverse(kids.begin(), kids.end());

    for (std::size_t i = 0; i < kids.size(); ++i) {
        const bool    last = (i + 1 == kids.size());
        const FPNode& nd   = nodes_[static_cast<std::size_t>(kids[i])];
        os << prefix << (last ? "`-- " : "+-- ")
           << header_[static_cast<std::size_t>(nd.rank)].item << ':' << nd.count << '\n';
        printSubtree(os, kids[i], prefix + (last ? "    " : "|   "));
    }
}
