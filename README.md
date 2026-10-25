# FP-Growth frequent itemset mining (ESE 327 project)

A C++17 implementation of the FP-Growth algorithm (Han, Pei & Yin, 2000)
for frequent itemset mining, with:

- a serial reference binary, `fpgrowth`, with built-in phase timers, peak
  memory and counters;
- tests against the Han & Kamber textbook example, a brute-force reference,
  hand-written edge cases (also under AddressSanitizer) and the mlxtend
  library;
- benchmarks on four UCI datasets (Chess, Mushroom, Connect-4, Online
  Retail), with figures and tables;
- an OpenMP version (`fpgrowth_omp`) and a CUDA + OpenMP hybrid
  (`fpgrowth_cuda`); see `GPU_DESIGN.md`.

The project report is `report/report.docx`, built from `report/report.md`.

## Requirements

- `make` and a C++17 compiler (`g++`; set `CXX` to use another)
- `python3` with `venv`; the tests, data and benchmark scripts install
  their packages (`requirements.txt`: mlxtend, pandas, openpyxl, matplotlib)
  into `.venv/` on first use
- `curl` and `gzip` for `make data`
- optional: OpenMP (`-fopenmp`) for `make omp`; the CUDA toolkit (`nvcc`)
  and an NVIDIA GPU for `make cuda`; `pandoc` for `make report`

## Build and run

```bash
make                  # ./fpgrowth (optimised)
make debug            # ./fpgrowth_debug (AddressSanitizer)
make omp              # ./fpgrowth_omp
make cuda             # ./fpgrowth_cuda; override ARCH, NVCC, NVCC_CCBIN, e.g.
                      #   make cuda ARCH=sm_87 NVCC_CCBIN=g++   (Jetson Orin)
make gpuinfo          # ./gpuinfo: CUDA device report and a one-kernel smoke test
```

```
fpgrowth <input_file> <min_support> [output_file] [--print-tree] [--unsorted]
```

- `input_file`: FIMI format, one transaction per line, space-separated
  non-negative integer items.
- `min_support`: an absolute count (`3`) or a fraction of the transactions
  (`0.05`).
- Output: one itemset per line with its support, e.g. `1 2 5 (2)`, sorted
  by default; `--unsorted` streams itemsets as they are found.
- `--print-tree` prints the header table and the initial FP-tree.
- Timings and counters go to stderr as one line of `key=value` pairs.
- The parallel binaries also take `--threads N`; `fpgrowth_cuda` also takes
  `--gpu-min-nodes N` and `--micro-count R`.

Examples:

```bash
make run-han          # the Han & Kamber example with --print-tree
./fpgrowth data/processed/mushroom.dat 0.3 out.txt
./fpgrowth data/processed/mushroom.dat 0.3 | .venv/bin/python scripts/decode.py - data/processed/mushroom.map.tsv --top 20
./fpgrowth_omp data/processed/chess.dat 0.4 /dev/null --unsorted --threads 8
```

## Tests

```bash
make test             # edge cases and brute-force differential tests on ./fpgrowth
                      # and ./fpgrowth_debug (ASan), mlxtend check on the Han example
make test-par         # parallel binaries: edge cases, brute force, and output
                      # identical to ./fpgrowth on every dataset (needs make data)
.venv/bin/python tests/compare_mlxtend.py data/processed/chess.dat 0.6   # one dataset vs mlxtend
```

## Reproducing the results

Every number in the report comes from files in `results/`, made by these
steps. Run them one at a time on an otherwise idle machine.

```bash
make data             # download the UCI datasets into data/raw/ and convert them
                      # to data/processed/<name>.dat and <name>.map.tsv
make bench            # support sweep, scaling, itemset lengths, mlxtend timing
                      # -> results/bench.csv, scaling.csv, lengths.csv, vs_mlxtend.csv
                      # (about 20 min; resumable: delete a CSV to re-measure it)
make figures          # results/figures/*.png|pdf (incl. han_tree) and results/tables.md
make examples         # decoded example patterns -> results/examples.txt
```

Parallel and GPU results (per machine, named by host name):

```bash
make && make omp && make cuda ARCH=<your GPU arch>
python3 scripts/run_parallel_bench.py   # -> results/parallel_<host>.csv,
                                        #    gpu_threshold_<host>.csv, gpu_info_<host>.txt
python3 scripts/kernel_micro.py         # -> results/kernel_micro_<host>.csv
make figures                            # adds omp_speedup_, impl_compare_,
                                        #    gpu_threshold_, kernel_micro_<host>
```

`LAB_RUN.md` has step-by-step instructions for the lab GPU server and the
Jetson Orin. Benchmark caps and the machine used are recorded in
`results/machine.txt` and `results/gpu_info_<host>.txt`.

Finally, `make report` rebuilds `report/report.docx`.

## Repository layout

| Path | Contents |
|:-----|:---------|
| `src/fptree.h`, `src/fptree.cpp` | FP-tree: node pool with 32-bit indices, insertion, node-links, single-path test, `--print-tree` output |
| `src/fpgrowth.h`, `src/fpgrowth.cpp` | Reader, the two passes, recursive mining, output sink |
| `src/main.cpp` | Command line, phase timers, counters, peak memory |
| `src/parallel.h`, `src/parallel.cpp` | OpenMP top-level mining and merged output |
| `src/gpu.h`, `src/cuda/gpu.cu` | CUDA counting kernels |
| `src/cuda/gpuinfo.cu` | CUDA device report and smoke test |
| `tests/` | Edge cases (`edge_cases/`), brute-force differential test, mlxtend cross-check, parallel tests |
| `scripts/` | Download and preprocessing, benchmarks, figures and tables, decoding, examples |
| `data/examples/han.dat` | The Han & Kamber example |
| `data/raw/`, `data/processed/` | Downloaded and converted datasets (not in git; `make data`) |
| `results/` | Benchmark CSVs, `tables.md`, `examples.txt`, machine descriptions |
| `results/figures/` | Figures, PNG (used in the report) and PDF |
| `GPU_DESIGN.md` | Design and results of the OpenMP and CUDA versions |
| `LAB_RUN.md` | Running the parallel experiments on the lab machines |
| `report/` | `report.md` (source), `crossref.lua` (pandoc filter), `make_reference.py` (styles), `report.docx` |
