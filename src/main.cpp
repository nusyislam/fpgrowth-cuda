// main.cpp - command-line front end for the FP-Growth implementation.
//
//   fpgrowth <input_file> <min_support> [output_file] [--print-tree] [--unsorted]
//
// Timing and counters are written to stderr as one line of key=value pairs.
//
// The same file builds the parallel binaries (see the Makefile):
//   -DFPG_OMP   fpgrowth_omp:  top-level mining loop on OpenMP threads (--threads)
//   -DFPG_CUDA  fpgrowth_cuda: as fpgrowth_omp, plus GPU counting (src/gpu.h,
//               --gpu-min-nodes); their stats lines have extra keys at the end.
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "fpgrowth.h"
#include "fptree.h"

#if defined(FPG_OMP) || defined(FPG_CUDA)
#  define FPG_PARALLEL 1
#  include "parallel.h"
#endif
#if defined(FPG_CUDA)
#  include "gpu.h"
#endif

#if defined(_WIN32)
// MinGW/MSVC have no <sys/resource.h>; use the Win32 equivalent of ru_maxrss.
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <psapi.h>
#else
#  include <sys/resource.h>
#endif

namespace {

#if defined(FPG_CUDA)
// Conditional trees below this many nodes are counted on the CPU: a GPU round
// trip (flatten, copy, launch, copy back) costs tens of microseconds, the CPU
// needs about a nanosecond per node and ancestor.  Chosen from the measured
// crossover, see GPU_DESIGN.md.
constexpr uint64_t kDefaultGpuMinNodes = 4096;
#endif

// Peak resident set size in kilobytes (getrusage ru_maxrss on POSIX).
long peakRssKB() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) return -1;
    return static_cast<long>(pmc.PeakWorkingSetSize / 1024);
#else
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return -1;
#  if defined(__APPLE__)
    return static_cast<long>(ru.ru_maxrss / 1024);  // macOS reports bytes
#  else
    return static_cast<long>(ru.ru_maxrss);         // Linux reports kilobytes
#  endif
#endif
}

void usage(const char* prog) {
    std::cerr
        << "usage: " << prog << " <input_file> <min_support> [output_file] [--print-tree] [--unsorted]\n"
        << "  input_file    FIMI .dat: one transaction per line, space-separated integer items\n"
        << "  min_support   absolute count (e.g. 3) or fraction of transactions (e.g. 0.05)\n"
        << "  output_file   frequent itemsets, one per line as \"1 2 5 (2)\"; default stdout\n"
        << "  --print-tree  print the initial FP-tree and header table to stdout\n"
        << "  --unsorted    stream itemsets as found instead of buffering and sorting them\n"
#if defined(FPG_PARALLEL)
        << "                (parallel binaries: same lines, in nondeterministic order)\n"
        << "  --threads N   mining threads (default: all, i.e. omp_get_max_threads())\n"
#endif
#if defined(FPG_CUDA)
        << "  --gpu-min-nodes N  count conditional trees with >= N nodes on the GPU\n"
        << "                (default " << kDefaultGpuMinNodes << "; the initial tree always goes to the GPU)\n"
#endif
        ;
}

#if defined(FPG_PARALLEL)
// Non-negative integer option value (--threads checks its own range).
uint64_t parseCount(const std::string& opt, const char* v) {
    const std::string s = v;
    std::size_t pos = 0;
    unsigned long long n = 0;
    try { n = std::stoull(s, &pos); } catch (const std::exception&) { pos = 0; }
    if (s.empty() || pos != s.size() || s[0] == '-')
        throw std::invalid_argument(opt + " needs a non-negative integer, got: " + s);
    return n;
}
#endif

// "3" -> 3 transactions; "0.05" -> ceil(0.05 * N).  Never less than 1.
//
// Fractions are interpreted at the precision they were typed: if some count k
// satisfies k/N == f when rounded to the same number of decimals as the
// argument, k is used (so "0.266667" with N = 15 gives 4, not 5).  Otherwise
// the threshold is rounded up, since support >= f*N must hold.
uint64_t parseMinSupport(const std::string& arg, std::size_t numTransactions) {
    std::size_t pos = 0;
    if (arg.find_first_of(".eE") != std::string::npos) {
        const double f = std::stod(arg, &pos);
        if (pos != arg.size() || !(f > 0.0) || f > 1.0)
            throw std::invalid_argument("fractional min_support must be in (0, 1]: " + arg);
        if (numTransactions == 0) return 1;

        const double N   = static_cast<double>(numTransactions);
        const double raw = f * N;
        const double k   = std::round(raw);

        double tol = 1e-9;  // scientific notation: treat the value as exact
        if (arg.find_first_of("eE") == std::string::npos) {
            const std::size_t dot = arg.find('.');
            const int decimals = static_cast<int>(arg.size() - dot - 1);
            tol = 0.5 * std::pow(10.0, -decimals) + 1e-12;
        }
        if (k >= 1.0 && std::fabs(k / N - f) <= tol) return static_cast<uint64_t>(k);
        return std::max<uint64_t>(1, static_cast<uint64_t>(std::ceil(raw - 1e-9)));
    }
    const unsigned long long v = std::stoull(arg, &pos);
    if (pos != arg.size()) throw std::invalid_argument("bad min_support: " + arg);
    return std::max<uint64_t>(1, v);
}

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

}  // namespace

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);

    std::vector<std::string> pos;
    bool print_tree = false, unsorted = false;
#if defined(FPG_PARALLEL)
    uint64_t threads = 0;  // 0: all
#endif
#if defined(FPG_CUDA)
    uint64_t gpu_min_nodes = kDefaultGpuMinNodes;
#endif
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
#if defined(FPG_PARALLEL)
        try {
            if (a == "--threads" || a == "--gpu-min-nodes") {
                if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
#  if defined(FPG_CUDA)
                if (a == "--gpu-min-nodes") { gpu_min_nodes = parseCount(a, argv[++i]); continue; }
#  endif
                if (a == "--threads") {
                    threads = parseCount(a, argv[++i]);
                    if (threads == 0 || threads > 4096) throw std::invalid_argument("--threads must be 1..4096");
                    continue;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "error: " << e.what() << '\n';
            return 2;
        }
#endif
        if (a == "--print-tree") print_tree = true;
        else if (a == "--unsorted") unsorted = true;
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else if (a.size() > 1 && a[0] == '-' && !std::isdigit(static_cast<unsigned char>(a[1]))) {
            std::cerr << "unknown option: " << a << "\n";
            usage(argv[0]);
            return 2;
        } else pos.push_back(a);
    }
    if (pos.size() < 2 || pos.size() > 3) { usage(argv[0]); return 2; }
    const std::string& input = pos[0];

    try {
#if defined(FPG_CUDA)
        const auto tg0 = Clock::now();
        Gpu        gpu;  // CUDA context creation: timed apart from the phases below
        const auto tg1 = Clock::now();
#endif
        const auto t0 = Clock::now();
        const Dataset ds = readDataset(input);
        const auto t1 = Clock::now();

        const uint64_t min_count = parseMinSupport(pos[1], ds.numTransactions());
        BuildStats   bs;
#if defined(FPG_CUDA)
        const FPTree tree = buildTreeFromCounts(ds, gpu.countItems(ds), min_count, bs);
#else
        const FPTree tree = buildInitialTree(ds, min_count, bs);
#endif
        const auto   t2 = Clock::now();

        if (print_tree) {
            std::cout << "Initial FP-tree for " << input << " (min_count = " << min_count << ", "
                      << ds.numTransactions() << " transactions)\n";
            tree.print(std::cout);
            std::cout << '\n';
            std::cout.flush();
        }

        std::ofstream file;
        std::ostream* out = &std::cout;
        if (pos.size() == 3) {
            file.open(pos[2], std::ios::binary);
            if (!file) throw std::runtime_error("cannot open output file: " + pos[2]);
            out = &file;
        }

#if defined(FPG_PARALLEL)
        ParallelMiner     sink(static_cast<int>(threads), unsorted ? out : nullptr);
        const auto        t3 = Clock::now();
#  if defined(FPG_CUDA)
        // Initial tree: all conditional supports in one kernel; deeper trees
        // go to the GPU from the mining threads when large enough.
        const std::vector<uint32_t> top = tree.singlePath() ? std::vector<uint32_t>{} : gpu.countTree(tree);
        const MiningStats ms = sink.mine(tree, min_count, top, gpu.threadCounters(sink.threads(), gpu_min_nodes));
#  else
        const MiningStats ms = sink.mine(tree, min_count, {}, {});
#  endif
        const auto        t4 = Clock::now();

        sink.finish(*out);
        out->flush();
        const auto t5 = Clock::now();
#else
        ItemsetSink sink(unsorted ? out : nullptr);
        const auto        t3 = Clock::now();
        const MiningStats ms = fpgrowth(tree, min_count, sink);
        const auto        t4 = Clock::now();

        if (!unsorted) sink.writeSorted(*out);
        out->flush();
        const auto t5 = Clock::now();
#endif

        std::ostringstream st;
        st << std::fixed << std::setprecision(6)
           << "file=" << std::filesystem::path(input).filename().string()  // no directory, no spaces
           << " transactions=" << ds.numTransactions()
           << " distinct_items=" << bs.distinct_items
           << " frequent_items=" << bs.frequent_items
           << " min_count=" << min_count
           << " tree_nodes=" << bs.tree_nodes
           << " filtered_occurrences=" << bs.filtered_occurrences
           << " itemsets=" << sink.count()
           << " cond_trees=" << ms.cond_trees
           << " cond_nodes=" << ms.cond_nodes
           << " single_paths=" << ms.single_paths
           << " read_s=" << secs(t0, t1)
           << " build_s=" << secs(t1, t2)
           << " mine_s=" << secs(t3, t4)
           << " output_s=" << secs(t4, t5)
#if defined(FPG_CUDA)
           << " total_s=" << secs(tg0, t5)  // includes gpu_init_s
#else
           << " total_s=" << secs(t0, t5)
#endif
           << " peak_rss_kb=" << peakRssKB();
#if defined(FPG_PARALLEL)
        st << " threads=" << sink.threads();
#endif
#if defined(FPG_CUDA)
        // Transfers and kernels are cudaEvent times summed over all calls; with
        // several mining threads, calls on different streams can overlap.
        const GpuTimes p1 = gpu.passOneTimes(), tt = gpu.topTreeTimes(), ct = gpu.condTreeTimes();
        GpuTimes       trees = tt;
        trees += ct;
        st << " gpu_min_nodes=" << gpu_min_nodes
           << " gpu_init_s=" << secs(tg0, tg1)
           << " count_h2d_s=" << p1.h2d_s
           << " count_kernel_s=" << p1.kernel_s
           << " count_d2h_s=" << p1.d2h_s
           << " tree_prep_s=" << trees.prep_s
           << " tree_h2d_s=" << trees.h2d_s
           << " tree_kernel_s=" << trees.kernel_s
           << " tree_d2h_s=" << trees.d2h_s
           << " gpu_trees=" << trees.calls
           << " gpu_cond_trees=" << ct.calls
           << " gpu_tree_nodes=" << trees.nodes
           << " h2d_bytes=" << p1.h2d_bytes + trees.h2d_bytes
           << " d2h_bytes=" << p1.d2h_bytes + trees.d2h_bytes;
#endif
        st << '\n';
        std::cerr << st.str();
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
