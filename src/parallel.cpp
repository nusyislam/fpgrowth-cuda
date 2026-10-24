// parallel.cpp - OpenMP top-level mining and merged output (see parallel.h).
#include "parallel.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <optional>
#include <ostream>
#include <streambuf>

#include <omp.h>

// Output buffer of one thread in --unsorted mode.  When it fills up, the
// complete lines in it are written to the shared stream under the mutex and
// the partial last line stays behind, so lines of different threads never
// interleave.
class LineChunkBuf : public std::streambuf {
public:
    LineChunkBuf(std::ostream& out, std::mutex& m, std::size_t cap = std::size_t{1} << 20)
        : out_(out), m_(m), buf_(cap) {
        setp(buf_.data(), buf_.data() + buf_.size());
    }

    // Write everything buffered (called once the thread has finished).
    void flushAll() { handOff(pptr()); }

protected:
    int_type overflow(int_type ch) override {
        char* nl = pptr();
        while (nl != pbase() && nl[-1] != '\n') --nl;
        if (nl != pbase()) {
            handOff(nl);
        } else {  // a single line longer than the buffer: grow it
            const std::size_t used = static_cast<std::size_t>(pptr() - pbase());
            buf_.resize(buf_.size() * 2);
            setp(buf_.data(), buf_.data() + buf_.size());
            pbump(static_cast<int>(used));
        }
        if (traits_type::eq_int_type(ch, traits_type::eof())) return traits_type::not_eof(ch);
        *pptr() = traits_type::to_char_type(ch);
        pbump(1);
        return ch;
    }

private:
    // Write [pbase, end) to the shared stream and keep [end, pptr) buffered.
    void handOff(char* end) {
        if (end != pbase()) {
            std::lock_guard<std::mutex> lock(m_);
            out_.write(pbase(), end - pbase());
        }
        const std::size_t rest = static_cast<std::size_t>(pptr() - end);
        std::memmove(buf_.data(), end, rest);
        setp(buf_.data(), buf_.data() + buf_.size());
        pbump(static_cast<int>(rest));
    }

    std::ostream&     out_;
    std::mutex&       m_;
    std::vector<char> buf_;
};

// Everything one thread writes while it mines, in one object aligned to its
// own cache lines.  The sink's counters and the stream buffer's put pointer
// change on every itemset; as separate small heap allocations, two threads'
// objects could share a cache line, and that false sharing made the 2-thread
// run on Connect-4 at 50% take 17-19 s instead of 10 s, depending on the
// heap layout (which even the length of the command-line paths changed).
struct alignas(128) ThreadOut {
    std::optional<LineChunkBuf> buf;     // unsorted mode only
    std::optional<std::ostream> stream;  // unsorted mode only
    ItemsetSink                 sink;

    ThreadOut(std::ostream* unsorted, std::mutex& m) : sink(open(unsorted, m)) {}

private:
    std::ostream* open(std::ostream* unsorted, std::mutex& m) {
        if (!unsorted) return nullptr;
        buf.emplace(*unsorted, m);
        stream.emplace(&*buf);
        return &*stream;
    }
};

ParallelMiner::ParallelMiner(int threads, std::ostream* unsorted)
    : threads_(threads > 0 ? threads : omp_get_max_threads()), unsorted_(unsorted != nullptr) {
    for (int t = 0; t < threads_; ++t) outs_.push_back(std::make_unique<ThreadOut>(unsorted, out_mutex_));
}

ParallelMiner::~ParallelMiner() = default;

MiningStats ParallelMiner::mine(const FPTree& tree, uint64_t min_count,
                                const std::vector<uint32_t>& topCounts,
                                const std::vector<TreeCounter*>& counters) {
    if (tree.singlePath()) return fpgrowth(tree, min_count, outs_[0]->sink);  // nothing to split

    const int32_t            F = static_cast<int32_t>(tree.header().size());
    std::vector<MiningStats> per(static_cast<std::size_t>(threads_));
    std::exception_ptr       error;
    std::mutex               error_mutex;

#pragma omp parallel num_threads(threads_)
    {
        const std::size_t tid = static_cast<std::size_t>(omp_get_thread_num());
        TreeCounter*      counter = counters.empty() ? nullptr : counters[tid];
        // Same order as the serial loop (least frequent item first): the rare
        // items have the deepest nodes and the largest conditional trees, so
        // the expensive work is handed out early.
#pragma omp for schedule(dynamic, 1)
        for (int32_t i = 0; i < F; ++i) {
            const int32_t   r   = F - 1 - i;
            const uint32_t* row = topCounts.empty() ? nullptr : topCounts.data() + TreeCounter::triRow(r);
            try {
                const MiningStats s = mineTopLevelItem(tree, r, row, min_count, outs_[tid]->sink, counter);
                per[tid].cond_trees += s.cond_trees;
                per[tid].cond_nodes += s.cond_nodes;
                per[tid].single_paths += s.single_paths;
            } catch (...) {  // exceptions must not escape an OpenMP region
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!error) error = std::current_exception();
            }
        }
    }
    if (error) std::rethrow_exception(error);

    MiningStats total;
    for (const MiningStats& s : per) {
        total.cond_trees += s.cond_trees;
        total.cond_nodes += s.cond_nodes;
        total.single_paths += s.single_paths;
    }
    return total;
}

void ParallelMiner::finish(std::ostream& os) {
    if (unsorted_) {
        for (auto& o : outs_) o->buf->flushAll();
        return;
    }
    std::vector<std::vector<uint64_t>> orders(outs_.size());
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads_)
    for (std::size_t s = 0; s < outs_.size(); ++s) orders[s] = outs_[s]->sink.sortedOrder();

    std::vector<const ItemsetSink*> sinks;
    for (const auto& o : outs_) sinks.push_back(&o->sink);
    ItemsetSink::writeMerged(os, sinks, orders);
}

uint64_t ParallelMiner::count() const {
    uint64_t n = 0;
    for (const auto& o : outs_) n += o->sink.count();
    return n;
}
