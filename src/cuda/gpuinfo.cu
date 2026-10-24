// gpuinfo.cu - CUDA toolchain smoke test and device report.
//
//   gpuinfo
//
// Prints the properties of every visible GPU, then launches a small kernel
// on device 0 and checks its result.  A binary built for a newer ARCH than the
// GPU fails here with "no kernel image is available".  One built for an older
// ARCH still runs (the driver JIT-compiles the embedded PTX), so for timing
// runs make sure ARCH matches the "compute capability" line printed here.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__,       \
                         #call, cudaGetErrorString(err_));                    \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

__global__ void axpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    int runtimeVer = 0, driverVer = 0, count = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVer));
    CUDA_CHECK(cudaDriverGetVersion(&driverVer));
    CUDA_CHECK(cudaGetDeviceCount(&count));
    std::printf("CUDA runtime %d.%d, driver supports %d.%d, %d device(s)\n",
                runtimeVer / 1000, runtimeVer % 1000 / 10,
                driverVer / 1000, driverVer % 1000 / 10, count);

    for (int d = 0; d < count; ++d) {
        cudaDeviceProp p;
        CUDA_CHECK(cudaGetDeviceProperties(&p, d));
        std::printf("Device %d: %s\n", d, p.name);
        std::printf("  compute capability  sm_%d%d\n", p.major, p.minor);
        std::printf("  SMs                 %d\n", p.multiProcessorCount);
        std::printf("  global memory       %.1f GiB%s\n",
                    p.totalGlobalMem / (1024.0 * 1024 * 1024),
                    p.integrated ? " (integrated, shared with CPU)" : "");
        std::printf("  shared mem / block  %zu KiB (opt-in max %zu KiB)\n",
                    p.sharedMemPerBlock / 1024, p.sharedMemPerBlockOptin / 1024);
        std::printf("  L2 cache            %d KiB\n", p.l2CacheSize / 1024);
        std::printf("  max threads / block %d, warp size %d\n",
                    p.maxThreadsPerBlock, p.warpSize);
    }
    if (count == 0) return 1;

    const int n = 1 << 20;
    std::vector<float> x(n, 1.0f), y(n, 2.0f);
    float *dx = nullptr, *dy = nullptr;
    CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, y.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    axpy<<<(n + 255) / 256, 256>>>(n, 3.0f, dx, dy);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(y.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));

    for (int i = 0; i < n; ++i) {
        if (y[i] != 5.0f) {
            std::fprintf(stderr, "kernel check FAILED at %d: %f\n", i, y[i]);
            return 1;
        }
    }
    std::printf("Kernel check on device 0: OK\n");
    return 0;
}
