// gpu.h - the CUDA part of fpgrowth_cuda behind a plain C++ interface, so
// that everything outside src/cuda/ is compiled by the host compiler alone.
//
// Two kinds of work are offloaded (design notes: GPU_DESIGN.md):
//   * pass 1 of tree construction: the support count of every item;
//   * the counting step of conditional-tree construction, for all items of a
//     tree in one kernel launch (a TreeCounter, see fpgrowth.h).  The
//     initial tree always goes to the GPU; conditional trees go when they
//     have at least `min_nodes` nodes, otherwise the CPU counts them.
// Every transfer and kernel is timed with cudaEvents on its own stream.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "fpgrowth.h"
#include "fptree.h"

struct GpuTimes {
    double   prep_s   = 0;  // CPU side: flatten trees into arrays, unpack results
    double   h2d_s    = 0;  // host -> device copies (cudaEvent)
    double   kernel_s = 0;  // kernels, incl. memsets (cudaEvent)
    double   d2h_s    = 0;  // device -> host copies (cudaEvent)
    uint64_t h2d_bytes = 0, d2h_bytes = 0;
    uint64_t calls = 0;     // trees counted (pass 1: 1)
    uint64_t nodes = 0;     // nodes of the trees counted

    GpuTimes& operator+=(const GpuTimes& o);
};

class Gpu {
public:
    // Selects device 0 and creates the CUDA context (most of the start-up
    // cost).  Throws std::runtime_error if there is no usable GPU.
    Gpu();
    ~Gpu();

    std::string deviceName() const;
    std::string arch() const;  // "sm_89" etc.

    // Pass 1 on the GPU: freq[item] = number of transactions containing it.
    std::vector<uint32_t> countItems(const Dataset& ds);

    // Conditional supports of every item of `t` (TreeCounter matrix layout),
    // whatever its size; used for the initial tree.
    std::vector<uint32_t> countTree(const FPTree& t);

    // One counter per mining thread, each with its own stream and buffers,
    // taking only trees with >= min_nodes nodes.  Owned by this Gpu.
    std::vector<TreeCounter*> threadCounters(int threads, uint64_t min_nodes);

    GpuTimes passOneTimes() const;
    GpuTimes topTreeTimes() const;      // countTree()
    GpuTimes condTreeTimes() const;     // all thread counters together

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
