// parallel.h - multi-threaded FP-Growth (OpenMP), shared by fpgrowth_omp and
// fpgrowth_cuda.
//
// The top-level loop of FP-Growth visits every frequent item of the initial
// tree and mines that item's conditional tree; these subproblems share only
// the read-only initial tree, so they run in parallel with dynamic scheduling
// (work per item is very uneven).  Each thread owns an ItemsetSink:
//   * sorted output: sinks buffer their itemsets, each is sorted in parallel
//     and a k-way merge writes one list identical to the serial binary's;
//   * --unsorted: each sink formats into a private buffer that is handed to
//     the shared output stream in whole lines under a lock.  Same itemsets
//     and lines as the serial binary, but in nondeterministic order.
#pragma once

#include <cstdint>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <vector>

#include "fpgrowth.h"
#include "fptree.h"

class LineChunkBuf;
struct ThreadOut;

class ParallelMiner {
public:
    // threads <= 0 means all available (omp_get_max_threads()).  `unsorted`
    // non-null selects streaming output to that stream.
    ParallelMiner(int threads, std::ostream* unsorted);
    ~ParallelMiner();

    int threads() const { return threads_; }

    // Mine `tree` with one thread per top-level item at a time.  `topCounts` is
    // an optional TreeCounter matrix for `tree` (empty: count on the CPU), and
    // `counters` is empty or holds one TreeCounter per thread for deeper trees.
    MiningStats mine(const FPTree& tree, uint64_t min_count,
                     const std::vector<uint32_t>& topCounts,
                     const std::vector<TreeCounter*>& counters);

    // Sorted mode: write all itemsets in the serial binary's order.
    // Unsorted mode: flush the lines still buffered.
    void finish(std::ostream& os);

    uint64_t count() const;

private:
    int                                     threads_;
    bool                                    unsorted_;
    std::mutex                              out_mutex_;
    std::vector<std::unique_ptr<ThreadOut>> outs_;  // one per thread
};
