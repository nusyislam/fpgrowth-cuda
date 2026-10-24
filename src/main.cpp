// main.cpp - command-line front end for the FP-Growth implementation.
//
//   fpgrowth <input_file> <min_support> [output_file] [--print-tree] [--unsorted]
//
// Timing and counters are written to stderr as one line of key=value pairs.
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

#if defined(_WIN32)
// MinGW/MSVC have no <sys/resource.h>; use the Win32 equivalent of ru_maxrss.
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <psapi.h>
#else
#  include <sys/resource.h>
#endif

namespace {

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
        << "  --unsorted    stream itemsets as found instead of buffering and sorting them\n";
}

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
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
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
        const auto t0 = Clock::now();
        const Dataset ds = readDataset(input);
        const auto t1 = Clock::now();

        const uint64_t min_count = parseMinSupport(pos[1], ds.numTransactions());
        BuildStats   bs;
        const FPTree tree = buildInitialTree(ds, min_count, bs);
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

        ItemsetSink sink(unsorted ? out : nullptr);
        const auto        t3 = Clock::now();
        const MiningStats ms = fpgrowth(tree, min_count, sink);
        const auto        t4 = Clock::now();

        if (!unsorted) sink.writeSorted(*out);
        out->flush();
        const auto t5 = Clock::now();

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
           << " total_s=" << secs(t0, t5)
           << " peak_rss_kb=" << peakRssKB() << '\n';
        std::cerr << st.str();
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
