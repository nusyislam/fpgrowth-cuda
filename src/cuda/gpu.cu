// gpu.cu - CUDA kernels and their host-side driver (interface: src/gpu.h).
#include "../gpu.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

namespace {

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA ") + what + ": " + cudaGetErrorString(err));
}
#define CUDA_CHECK(call) check((call), #call)

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

constexpr int kBlock = 256;
// Static shared memory available to every block on all supported GPUs.
constexpr std::size_t kSharedBytes = 48 * 1024;
// Largest counting matrix offloaded (bigger: the CPU counts that tree).
constexpr uint64_t kMaxMatrixBytes = 512ull << 20;

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Pass 1, one thread per item occurrence (grid-stride).  Each block builds a
// histogram in shared memory and merges it with one global atomic per bin,
// because the dense datasets have <= 129 distinct items and per-occurrence
// global atomics would serialise on them.
__global__ void countItemsShared(const int* __restrict__ items, std::size_t n,
                                 uint32_t* __restrict__ freq, int bins) {
    extern __shared__ uint32_t hist[];
    for (int b = threadIdx.x; b < bins; b += blockDim.x) hist[b] = 0;
    __syncthreads();
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        atomicAdd(&hist[items[i]], 1u);
    __syncthreads();
    for (int b = threadIdx.x; b < bins; b += blockDim.x)
        if (hist[b]) atomicAdd(&freq[b], hist[b]);
}

// Same, for item ids too large for a shared-memory histogram.
__global__ void countItemsGlobal(const int* __restrict__ items, std::size_t n,
                                 uint32_t* __restrict__ freq) {
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        atomicAdd(&freq[items[i]], 1u);
}

// Conditional supports of every item of a flattened FP-tree.
//   up[v]       = (parent node, rank) of node v; node 0 is the root
//   count[v]    = count of node v
//   itemNodes[itemStart[r] .. itemStart[r+1]) = the nodes carrying rank r
// One block per rank r >= 1 (rank 0 has no ancestors), one thread per node of
// that rank: the thread walks from the node to the root and adds the node's
// count to each ancestor's rank.  The block accumulates its row in shared
// memory and writes it to M[r(r-1)/2 .. +r) once, so there are no global
// atomics and M needs no clearing.
__global__ void condCountShared(const int2* __restrict__ up, const uint32_t* __restrict__ count,
                                const int* __restrict__ itemStart,
                                const int* __restrict__ itemNodes, uint32_t* __restrict__ M) {
    extern __shared__ uint32_t row[];
    const int r = blockIdx.x + 1;
    for (int i = threadIdx.x; i < r; i += blockDim.x) row[i] = 0;
    __syncthreads();
    for (int k = itemStart[r] + threadIdx.x; k < itemStart[r + 1]; k += blockDim.x) {
        const int      v = itemNodes[k];
        const uint32_t c = count[v];
        for (int a = up[v].x; a != 0;) {
            const int2 pa = up[a];
            atomicAdd(&row[pa.y], c);
            a = pa.x;
        }
    }
    __syncthreads();
    uint32_t* out = M + static_cast<std::size_t>(r) * (r - 1) / 2;
    for (int i = threadIdx.x; i < r; i += blockDim.x) out[i] = row[i];
}

// Same, for trees with too many items for a shared-memory row: atomics go to
// global memory, which must be zeroed first.
__global__ void condCountGlobal(const int2* __restrict__ up, const uint32_t* __restrict__ count,
                                const int* __restrict__ itemStart,
                                const int* __restrict__ itemNodes, uint32_t* __restrict__ M) {
    const int r   = blockIdx.x + 1;
    uint32_t* out = M + static_cast<std::size_t>(r) * (r - 1) / 2;
    for (int k = itemStart[r] + threadIdx.x; k < itemStart[r + 1]; k += blockDim.x) {
        const int      v = itemNodes[k];
        const uint32_t c = count[v];
        for (int a = up[v].x; a != 0;) {
            const int2 pa = up[a];
            atomicAdd(&out[pa.y], c);
            a = pa.x;
        }
    }
}

// ---------------------------------------------------------------------------
// Buffers that only grow (cudaMalloc/cudaFree are slow and synchronising)
// ---------------------------------------------------------------------------

struct DeviceBuf {
    void*       p   = nullptr;
    std::size_t cap = 0;
    ~DeviceBuf() { if (p) cudaFree(p); }
    void* get(std::size_t bytes) {
        if (bytes > cap) {
            if (p) CUDA_CHECK(cudaFree(p));
            p   = nullptr;
            cap = std::max(bytes, cap * 2);
            CUDA_CHECK(cudaMalloc(&p, cap));
        }
        return p;
    }
};

struct PinnedBuf {
    void*       p   = nullptr;
    std::size_t cap = 0;
    ~PinnedBuf() { if (p) cudaFreeHost(p); }
    void* get(std::size_t bytes) {
        if (bytes > cap) {
            if (p) CUDA_CHECK(cudaFreeHost(p));
            p   = nullptr;
            cap = std::max(bytes, cap * 2);
            CUDA_CHECK(cudaMallocHost(&p, cap));
        }
        return p;
    }
};

// A stream plus the four events that bracket H2D, kernel and D2H.
struct TimedStream {
    cudaStream_t s = nullptr;
    cudaEvent_t  ev[4]{};
    TimedStream() {
        CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        for (cudaEvent_t& e : ev) CUDA_CHECK(cudaEventCreate(&e));
    }
    ~TimedStream() {
        for (cudaEvent_t e : ev) cudaEventDestroy(e);
        cudaStreamDestroy(s);
    }
    void mark(int i) { CUDA_CHECK(cudaEventRecord(ev[i], s)); }
    // Wait for the last mark and add the three intervals to `t`.
    void collect(GpuTimes& t) {
        CUDA_CHECK(cudaEventSynchronize(ev[3]));
        float ms[3];
        for (int i = 0; i < 3; ++i) CUDA_CHECK(cudaEventElapsedTime(&ms[i], ev[i], ev[i + 1]));
        t.h2d_s += ms[0] / 1000.0;
        t.kernel_s += ms[1] / 1000.0;
        t.d2h_s += ms[2] / 1000.0;
    }
};

// ---------------------------------------------------------------------------
// Tree counter: flatten -> H2D -> kernel -> D2H, one tree at a time
// ---------------------------------------------------------------------------

class GpuTreeCounter : public TreeCounter {
public:
    explicit GpuTreeCounter(uint64_t min_nodes) : min_nodes_(min_nodes) {}

    std::vector<uint32_t> countAll(const FPTree& t) override {
        if (t.nodeCount() < min_nodes_) return {};
        return count(t);
    }

    // Empty result if the tree has < 2 items or its matrix would be too large.
    std::vector<uint32_t> count(const FPTree& t) {
        const auto&    H  = t.header();
        const int32_t  F  = static_cast<int32_t>(H.size());
        const uint64_t Mn = TreeCounter::triRow(F);  // entries of the whole matrix
        if (F < 2 || Mn * sizeof(uint32_t) > kMaxMatrixBytes) return {};

        // Flatten into one pinned block so a single copy moves it all:
        //   up[n] (int2) | count[n] | itemStart[F+1] | itemNodes[n-1]
        const auto        t0 = Clock::now();
        const std::size_t n  = t.nodes().size();  // including the root
        const std::size_t words = 2 * n + n + (F + 1) + (n - 1);
        int* host = static_cast<int*>(hin_.get(words * sizeof(int)));
        int2*     up        = reinterpret_cast<int2*>(host);
        uint32_t* cnt       = reinterpret_cast<uint32_t*>(host + 2 * n);
        int*      itemStart = host + 3 * n;
        int*      itemNodes = itemStart + (F + 1);
        for (std::size_t v = 0; v < n; ++v) {
            const FPNode& nd = t.nodes()[v];
            up[v]  = make_int2(nd.parent < 0 ? 0 : nd.parent, nd.rank);
            cnt[v] = nd.count;
        }
        int k = 0;
        for (int32_t r = 0; r < F; ++r) {
            itemStart[r] = k;
            for (int32_t v = H[static_cast<std::size_t>(r)].head; v != FPTree::NONE; v = t.node(v).next_link)
                itemNodes[k++] = v;
        }
        itemStart[F] = k;
        const auto t1 = Clock::now();

        int*      dev  = static_cast<int*>(din_.get(words * sizeof(int)));
        uint32_t* dM   = static_cast<uint32_t*>(dout_.get(Mn * sizeof(uint32_t)));
        uint32_t* hM   = static_cast<uint32_t*>(hout_.get(Mn * sizeof(uint32_t)));
        const bool shared = F * sizeof(uint32_t) <= kSharedBytes;

        ts_.mark(0);
        CUDA_CHECK(cudaMemcpyAsync(dev, host, words * sizeof(int), cudaMemcpyHostToDevice, ts_.s));
        ts_.mark(1);
        const int2*     dUp        = reinterpret_cast<const int2*>(dev);
        const uint32_t* dCnt       = reinterpret_cast<const uint32_t*>(dev + 2 * n);
        const int*      dItemStart = dev + 3 * n;
        const int*      dItemNodes = dItemStart + (F + 1);
        if (shared) {
            condCountShared<<<F - 1, kBlock, F * sizeof(uint32_t), ts_.s>>>(dUp, dCnt, dItemStart,
                                                                           dItemNodes, dM);
        } else {
            CUDA_CHECK(cudaMemsetAsync(dM, 0, Mn * sizeof(uint32_t), ts_.s));
            condCountGlobal<<<F - 1, kBlock, 0, ts_.s>>>(dUp, dCnt, dItemStart, dItemNodes, dM);
        }
        CUDA_CHECK(cudaGetLastError());
        ts_.mark(2);
        CUDA_CHECK(cudaMemcpyAsync(hM, dM, Mn * sizeof(uint32_t), cudaMemcpyDeviceToHost, ts_.s));
        ts_.mark(3);
        ts_.collect(times_);

        const auto            t2 = Clock::now();
        std::vector<uint32_t> M(hM, hM + Mn);
        const auto            t3 = Clock::now();

        times_.prep_s += secs(t0, t1) + secs(t2, t3);
        times_.h2d_bytes += words * sizeof(int);
        times_.d2h_bytes += Mn * sizeof(uint32_t);
        times_.calls += 1;
        times_.nodes += t.nodeCount();
        return M;
    }

    const GpuTimes& times() const { return times_; }

private:
    uint64_t    min_nodes_;
    TimedStream ts_;
    DeviceBuf   din_, dout_;
    PinnedBuf   hin_, hout_;
    GpuTimes    times_;
};

}  // namespace

GpuTimes& GpuTimes::operator+=(const GpuTimes& o) {
    prep_s += o.prep_s;
    h2d_s += o.h2d_s;
    kernel_s += o.kernel_s;
    d2h_s += o.d2h_s;
    h2d_bytes += o.h2d_bytes;
    d2h_bytes += o.d2h_bytes;
    calls += o.calls;
    nodes += o.nodes;
    return *this;
}

struct Gpu::Impl {
    cudaDeviceProp                               prop{};
    GpuTimes                                     pass1;
    std::unique_ptr<GpuTreeCounter>              top;
    std::vector<std::unique_ptr<GpuTreeCounter>> perThread;
};

Gpu::Gpu() : impl_(std::make_unique<Impl>()) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
        throw std::runtime_error("no CUDA device available");
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaGetDeviceProperties(&impl_->prop, 0));
    CUDA_CHECK(cudaFree(nullptr));  // create the context now, not inside a timed phase
    impl_->top = std::make_unique<GpuTreeCounter>(0);
}

Gpu::~Gpu() = default;

std::string Gpu::deviceName() const { return impl_->prop.name; }

std::string Gpu::arch() const {
    return "sm_" + std::to_string(impl_->prop.major) + std::to_string(impl_->prop.minor);
}

std::vector<uint32_t> Gpu::countItems(const Dataset& ds) {
    const std::size_t n = ds.items.size();
    if (n == 0) return {};

    TimedStream ts;
    DeviceBuf   dItems, dFreq;
    int*        items = static_cast<int*>(dItems.get(n * sizeof(int)));

    ts.mark(0);
    CUDA_CHECK(cudaMemcpyAsync(items, ds.items.data(), n * sizeof(int), cudaMemcpyHostToDevice, ts.s));
    ts.mark(1);
    // Largest item id sizes the counter array (thrust reduce on the GPU; it
    // returns the value to the host, so this synchronises the stream).
    const int maxItem = thrust::reduce(thrust::cuda::par.on(ts.s), items, items + n, 0,
                                       thrust::maximum<int>());
    if (maxItem > kMaxItemId) throw std::runtime_error("item id exceeds supported maximum");
    const int bins = maxItem + 1;
    uint32_t* freq = static_cast<uint32_t*>(dFreq.get(bins * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemsetAsync(freq, 0, bins * sizeof(uint32_t), ts.s));
    const int grid = static_cast<int>(std::min<std::size_t>((n + kBlock - 1) / kBlock,
                                                            8 * static_cast<std::size_t>(impl_->prop.multiProcessorCount)));
    if (bins * sizeof(uint32_t) <= kSharedBytes)
        countItemsShared<<<grid, kBlock, bins * sizeof(uint32_t), ts.s>>>(items, n, freq, bins);
    else
        countItemsGlobal<<<grid, kBlock, 0, ts.s>>>(items, n, freq);
    CUDA_CHECK(cudaGetLastError());
    ts.mark(2);
    std::vector<uint32_t> out(static_cast<std::size_t>(bins));
    CUDA_CHECK(cudaMemcpyAsync(out.data(), freq, bins * sizeof(uint32_t), cudaMemcpyDeviceToHost, ts.s));
    ts.mark(3);
    ts.collect(impl_->pass1);
    impl_->pass1.h2d_bytes += n * sizeof(int);
    impl_->pass1.d2h_bytes += bins * sizeof(uint32_t);
    impl_->pass1.calls += 1;
    return out;
}

std::vector<uint32_t> Gpu::countTree(const FPTree& t) { return impl_->top->count(t); }

std::vector<TreeCounter*> Gpu::threadCounters(int threads, uint64_t min_nodes) {
    std::vector<TreeCounter*> out;
    for (int i = 0; i < threads; ++i) {
        impl_->perThread.push_back(std::make_unique<GpuTreeCounter>(min_nodes));
        out.push_back(impl_->perThread.back().get());
    }
    return out;
}

GpuTimes Gpu::passOneTimes() const { return impl_->pass1; }
GpuTimes Gpu::topTreeTimes() const { return impl_->top->times(); }

GpuTimes Gpu::condTreeTimes() const {
    GpuTimes t;
    for (const auto& c : impl_->perThread) t += c->times();
    return t;
}
