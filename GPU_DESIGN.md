# Parallel FP-Growth: OpenMP and CUDA design

Three binaries are built from one source tree, with identical output:

| Binary | Build | What runs in parallel |
|:---|:---|:---|
| `fpgrowth` | `make` | nothing (reference) |
| `fpgrowth_omp` | `make omp` | the top-level mining loop, on OpenMP threads |
| `fpgrowth_cuda` | `make cuda` | as `fpgrowth_omp`, plus item counting and conditional-support counting on the GPU |

The host C++ code of all three is compiled by the same compiler with the same
flags. Only `src/cuda/gpu.cu` goes through nvcc, so speed differences come
from the parallelisation, not from different compilers.

## 1. Where the serial time goes

Two measurements on the serial binary decided the split:

1. **Mining is output-bound.** `perf` on Connect-4 at 70% (4.1M itemsets)
   puts about 75% of all samples inside libstdc++: formatting each itemset
   through `std::ostream`. A second measurement agrees: on Connect-4 at
   60%, mining with sorted output (itemsets only buffered; formatting
   happens later) takes 1.7 s, against 5.0 s with `--unsorted` (formatted
   while mining). Building conditional trees is the minority of the work.
   Chess at 25% emits 99M itemsets in 27 s, about 280 ns per itemset.
2. **Conditional trees are tiny.** From `results/bench.csv`, the average
   conditional tree has 4-6 nodes on Chess, Mushroom and Connect-4 and about
   14 on Online Retail (`cond_nodes / cond_trees`). Only the initial tree and
   a few hundred first-level trees on Online Retail exceed a few thousand
   nodes.

A third fact: creating a CUDA context costs about 0.15 s, and a GPU round
trip for one tree (copy in, launch, copy out) costs tens of microseconds.
Most support levels in `bench.csv` finish in under a second in total.

## 2. OpenMP version

FP-Growth's top-level loop visits each frequent item of the initial tree and
mines that item's conditional tree. These subproblems share only the
read-only initial tree, so each one is a task:

- `#pragma omp for schedule(dynamic, 1)` over the items, rarest item first.
  Rare items sit deepest in the tree and have the biggest conditional
  trees, so the expensive tasks are handed out first. Dynamic scheduling
  is required because the work per item varies by orders of magnitude.
- **Per-thread result sinks.** Sorted output: every thread buffers its
  itemsets, the buffers are sorted in parallel, and a k-way merge writes one
  list. Each itemset is found by exactly one task, so the merged list is
  byte-identical to the serial output. `--unsorted`: each thread formats
  into a private 1 MB buffer, which is handed to the output stream in whole
  lines under a lock. The lines are the same as serial, in nondeterministic
  order.
- The itemset formatting, the dominant cost, therefore runs in parallel too.
- **Each thread's output state is padded to its own cache lines.** The
  sink's counters and the stream buffer's put pointer change on every
  itemset. In the first version, the per-thread objects were separate small
  heap allocations, and whether two threads' objects shared a cache line
  depended on the heap layout. Even the length of the file paths on the
  command line changed it. With the bad layout, the 2-thread run on
  Connect-4 at 50% took 17-19 s instead of 10 s (false sharing). That
  benchmark was discarded and rerun after the fix.
- **Caveat when comparing with the serial binary:** in `--unsorted` mode
  the parallel binaries hand output to the stream in 1 MB chunks, while the
  serial binary writes through `std::ofstream`'s much smaller buffer. That
  alone makes `fpgrowth_omp --threads 1` up to 10% faster than `fpgrowth`
  on the dense datasets (Connect-4 60%: 4.55 s vs 5.03 s mining). With
  sorted output, where neither streams during mining, the two are within
  2%. The OpenMP scaling figure therefore uses the 1-thread OpenMP run as
  its baseline.

**Limit of this design:** only the top level is split, so the largest
single item's subtree is a lower bound on the mining time. The dense
datasets scale almost linearly to 4 threads and reach 6-7.5× at 8, then
gain little or lose ground at 28 threads (see section 6). This is most
likely that critical path, plus the 28 logical CPUs being 8 P-cores with
hyper-threading and 12 slower E-cores. Online Retail, with ~2000 top-level
items, keeps scaling to 11× at 28 threads. Splitting deeper recursion levels
into tasks would be the next step.

## 3. CUDA version: what goes where, and why

```
 CPU                                         GPU
 ---                                         ---
 read .dat into CSR arrays
   items[] ------------------ H2D --------->  pass 1: count every item
                                              (thread per occurrence,
   freq[]  <----------------- D2H ----------  shared-memory histogram)
 pass 2: insert transactions (FP-tree)
 flatten tree -> (parent,rank), count,
   nodes grouped by item ---- H2D --------->  conditional-support kernel:
                                              block per item, thread per node,
   M (F x F lower triangle) <- D2H ---------  walk up parent[], add count
 OpenMP: for each item r (dynamic)
   conditional tree from row M[r]  (counting step skipped)
   recurse on CPU; a conditional tree with >= --gpu-min-nodes nodes
   goes through the same kernel on the thread's own CUDA stream
 merge per-thread outputs
```

### Pass 1: item counting (GPU)

The transactions are already in CSR layout (`items[]` plus `offsets[]`).
Counting needs only `items[]`, because the reader removes duplicates within
each transaction. The kernel uses **one thread per item occurrence**
(grid-stride loop) instead of one thread per transaction:

- Online Retail transactions range from 1 to 1106 items. With a thread per
  transaction, one long transaction holds back its whole warp, and the
  accesses are strided rather than coalesced.
- The dense datasets have only 75-129 distinct items, so millions of
  `atomicAdd`s on global memory would collide constantly. Each block
  therefore counts into a **shared-memory histogram** and then adds it to
  global memory with one atomic per item. This applies when the largest item
  id fits in 48 KB of shared memory (true for all four datasets); otherwise
  a global-atomic kernel is used.

The largest item id, which sets the histogram size, is found on the GPU with
`thrust::reduce`.

### Pass 2: tree construction (CPU)

FP-tree insertion is sequential pointer chasing: walk from the root, look up
the child in a hash table, create a node if missing, link it into the
item's node-link chain. Parallel GPU tree building needs lock-free
insertion with heavy contention near the root. On these trees (at most 455k
nodes, built in 30-70 ms) the CPU code is reused unchanged.

### Conditional-support counting (GPU)

Building the conditional tree of item `r` starts by walking every node of
`r` up to the root and adding the node's count to each ancestor's item. This
is the conditional pattern base, which yields the supports of `{a, r}` for
every `a`. For all items at once this is a regular, data-parallel reduction:

- The tree is flattened into arrays: `up[v] = (parent, rank)` as one
  8-byte load per step, `count[v]`, and the node ids grouped by item (CSR).
  Everything is packed into one pinned buffer, so a single copy moves it.
- **One block per item `r`, one thread per node of `r`.** Each thread walks
  up `parent[]` and atomically adds its count to `row[rank(ancestor)]`.
- The row lives in **shared memory**, so the atomics are cheap on-chip
  ones, and each row is written to global memory exactly once. No global
  atomics, and no zeroing of the output matrix. Trees with more than 12,288
  items fall back to global atomics.
- The result is the lower-triangular F×F matrix `M[r][a]`: every frequent
  2-itemset, plus step 1 of every conditional tree of this tree. The CPU
  then builds each conditional tree from its row, skipping its own counting
  walk.

The initial tree always goes to the GPU. Conditional trees go when they have
at least `--gpu-min-nodes` nodes (default 4096). Below that, the round trip
costs more than the CPU walk it saves. The default comes from a sweep with
one mining thread on the RTX 4070 (single runs, made while choosing the
default; `results/gpu_threshold_BLACK-OUT.csv` and figure
`gpu_threshold_BLACK-OUT` have the 3-rep version for Online Retail 0.25% and
Chess 40%, with the same shape):

| `--gpu-min-nodes` | Online Retail 0.2%: mine time (trees on GPU) | Chess 40% | Connect-4 70% |
|---:|---:|---:|---:|
| 0 (every tree) | 48.8 s (594,768) | 27.6 s | 1.52 s |
| 64 | 6.86 s (42,030) | 1.89 s | 0.853 s |
| 256 | 4.68 s (13,449) | 1.68 s | 0.834 s |
| 1024 | 3.88 s (2,608) | 1.65 s | 0.835 s |
| **4096** | **3.76 s (470)** | 1.66 s | 0.833 s |
| 16384 | 3.93 s (7) | 1.66 s | 0.842 s |
| initial tree only | 3.94 s (1) | 1.65 s | 0.837 s |

Offloading every tree is 12× slower than the CPU on Online Retail: about
56 µs per GPU round trip against trees of ~14 nodes. Above ~1000 nodes the
GPU wins slightly, at best ~5% of mining time on Online Retail, and nothing
measurable on the dense datasets, whose conditional trees are all small.
`run_parallel_bench.py` repeats this sweep on every machine
(`gpu_threshold_<host>.csv`), because the crossover depends on the GPU and,
on a Jetson, on the shared CPU/GPU memory.

With OpenMP, each mining thread has its own CUDA stream, events, and
device/pinned buffers that only grow, so threads offload concurrently
without sharing state.

### What stays on the CPU, and why

| Part | Why not on the GPU |
|:---|:---|
| Reading the file | I/O and text parsing |
| Tree insertion | sequential pointer chasing plus hashing; contention at the root |
| Recursion over conditional trees | ~5-14 nodes per tree; a kernel launch costs more than the work |
| Single-path enumeration and output | ~75% of serial mining time is formatting text into a stream; it must end up on the host, and moving GBs of formatted text over PCIe would not be FP-growth |

An alternative that was considered and rejected: GPU itemset
enumeration/formatting for single-path trees. It attacks the dominant cost,
but it is a text-generation kernel rather than FP-growth, and the device-to-host copy
of the output (several GB at low support) would cost about as much as the
formatting saves.

### Timing

`fpgrowth_cuda` adds these keys to the stderr stats line (cudaEvent timers
on the stream that does the work):

| Key | Meaning |
|:---|:---|
| `gpu_init_s` | CUDA context creation (wall clock); included in `total_s` |
| `count_h2d_s`, `count_kernel_s`, `count_d2h_s` | pass 1: copy items in, max + histogram kernels, copy counts out |
| `tree_prep_s` | CPU side of tree counting: flatten, unpack results |
| `tree_h2d_s`, `tree_kernel_s`, `tree_d2h_s` | conditional-support counting, summed over all trees |
| `gpu_trees`, `gpu_cond_trees`, `gpu_tree_nodes` | trees counted on the GPU (total / conditional), their nodes |
| `h2d_bytes`, `d2h_bytes` | bytes moved |

With several mining threads, tree-counting calls on different streams can
overlap, so the summed times are GPU busy time, not wall time.

## 4. Correctness

`make test-par` runs the edge cases and the brute-force differential test
against `fpgrowth_omp` and `fpgrowth_cuda` (at several thread counts, and
with `--gpu-min-nodes 0`, which sends every conditional tree with two or
more items through the kernel). It also checks, on every dataset at 2
support levels, that sorted output is byte-identical to `./fpgrowth`'s,
that `--unsorted` output has the same lines, and that the tree and itemset
counters on stderr match.

## 5. How the experiments are run

`scripts/run_parallel_bench.py` (standard library only; steps for the lab
machines in `LAB_RUN.md`) runs every dataset at 3-4 support levels taken
from `results/bench.csv`, aiming at serial totals near 0.1 s, 1 s, 10 s and
25 s. Configs: serial, OpenMP with 1/2/4/8/all threads, CUDA with 1 and all
mining threads. Each point is the median of 5 runs with `--unsorted` output
to `/dev/null`. Each run is capped at 120 s and 3.8 GiB resident memory.
Figures: `omp_speedup_<host>`, `impl_compare_<host>`, `gpu_threshold_<host>`.

## 6. Results on the development machine (RTX 4070)

Machine `BLACK-OUT`: Intel i7-14700K (8 P + 12 E cores, 28 threads visible
to WSL2), 15.6 GiB RAM, RTX 4070 12 GB (sm_89), CUDA 12.4, Ubuntu 26.04
under WSL2. Details: `results/gpu_info_BLACK-OUT.txt`. Total time in
seconds, median of 5 (`results/parallel_BLACK-OUT.csv`):

| Dataset | Min sup | Itemsets | serial | OMP 1 | OMP 2 | OMP 4 | OMP 8 | OMP 28 | CUDA 1 thr | CUDA 28 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Chess | 60% | 254,944 | 0.078 | 0.071 | 0.038 | 0.022 | 0.019 | 0.038 | 0.216 | 0.217 |
| Chess | 40% | 6,472,980 | 1.824 | 1.650 | 0.831 | 0.429 | 0.280 | **0.275** | 1.809 | 0.478 |
| Chess | 30% | 37,282,962 | 10.466 | 9.485 | 4.716 | 2.409 | 1.335 | **1.271** | 9.673 | 1.417 |
| Chess | 25% | 99,022,533 | 27.771 | 25.786 | 12.558 | 6.411 | **3.445** | 3.724 | 25.226 | 3.862 |
| Mushroom | 10% | 468,003 | 0.098 | 0.088 | 0.047 | 0.033 | **0.028** | 0.044 | 0.297 | 0.194 |
| Mushroom | 5% | 3,082,813 | 0.615 | 0.553 | 0.279 | 0.156 | 0.101 | **0.088** | 0.699 | 0.239 |
| Mushroom | 1% | 77,322,009 | 16.596 | 14.862 | 7.412 | 3.789 | 2.038 | **1.410** | 14.983 | 1.521 |
| Connect-4 | 85% | 142,271 | 0.080 | 0.077 | 0.065 | 0.059 | **0.057** | 0.072 | 0.231 | 0.303 |
| Connect-4 | 70% | 4,129,839 | 0.976 | 0.886 | 0.471 | 0.270 | **0.190** | 0.218 | 1.048 | 0.424 |
| Connect-4 | 60% | 21,252,795 | 5.102 | 4.606 | 2.321 | 1.276 | **0.772** | 0.828 | 4.787 | 0.984 |
| Connect-4 | 50% | 88,316,367 | 22.061 | 19.872 | 9.969 | 5.123 | **3.096** | 3.183 | 20.171 | 3.350 |
| Online Retail | 1% | 1,860 | 0.085 | 0.081 | 0.062 | 0.053 | **0.048** | 0.057 | 0.221 | 0.280 |
| Online Retail | 0.25% | 596,206 | 1.137 | 1.114 | 0.609 | 0.340 | 0.209 | **0.154** | 1.215 | 0.370 |
| Online Retail | 0.2% | 5,944,725 | 4.084 | 3.927 | 2.045 | 1.064 | 0.580 | **0.354** | 4.051 | 0.576 |

What the numbers say:

- **OpenMP is the win:** best speedups over serial of 8.1× (Chess 25%),
  11.8× (Mushroom 1%), 7.1× (Connect-4 50%) and 11.5× (Online Retail 0.2%).
  On every setting above ~0.5 s, scaling is close to linear up to 4
  threads (3.3-4.0× at 4 threads). Below ~0.1 s of total time, thread start-up and the serial read/build phases dominate, and
  28 threads are slower than 8.
- **CUDA never beats OpenMP on this machine.** On the larger settings,
  `cuda_tall` is 0.11-0.22 s slower than `omp_tall`, which is about
  the CUDA context creation (`gpu_init_s` 0.14-0.22 s). On the smallest
  setting of each dataset, that start-up alone makes the CUDA binary
  2-3.8× *slower* than serial.
- **The GPU work itself is tiny:** kernels take 1.5-4 ms and transfers
  under 1.2 ms per run on the dense datasets, against totals of 0.2-4 s. Only
  Online Retail sends conditional trees to the GPU (323-470 trees); there
  the summed transfer time reaches 55-68 ms and kernels 25-38 ms. With 28
  threads competing for the GPU this costs more than it saves: 0.58 s vs
  0.35 s for OpenMP at 0.2%.
- **With one mining thread, the GPU offload has a small but real effect:**
  at the default threshold, Online Retail 0.25% mines in 0.98 s vs 1.04 s
  with only the initial tree on the GPU (`gpu_threshold` sweep). That is the counting
  step of the ~400 largest conditional trees moved off the CPU. The dense
  datasets have no conditional trees large enough to offload.
- The design conclusion for the report: on these datasets FP-growth is
  bounded by irregular, output-heavy CPU work. A GPU can only take over the
  regular counting steps, which are a few percent of the runtime, so a
  multi-core CPU is the better parallel target. The GPU would matter more
  with much larger initial trees (millions of nodes, many transactions) and
  fewer, larger conditional trees, i.e. higher support on bigger data.

### Kernel level: where the GPU does win

`scripts/kernel_micro.py` isolates the one step the GPU replaces: computing
the conditional supports of every item of the *initial* tree. It compares
`conditionalSupportsCpu()` (the miner's own walk, one thread) with the GPU
path, inside one process after the CUDA context exists and after one
warm-up, median of 5, and checks that both matrices are equal
(`results/kernel_micro_BLACK-OUT.csv`, figure `kernel_micro_BLACK-OUT`):

| Dataset | Min sup | Tree nodes | CPU | GPU kernel | + transfers | + flatten/unpack |
|:---|---:|---:|---:|---:|---:|---:|
| Chess | 60% | 10,844 | 0.47 ms | 0.036 ms (13×) | 0.076 ms (6.2×) | 0.16 ms (3.0×) |
| Chess | 25% | 31,813 | 1.69 ms | 0.047 ms (36×) | 0.12 ms (15×) | 0.30 ms (5.6×) |
| Mushroom | 10% | 14,074 | 0.47 ms | 0.025 ms (18×) | 0.067 ms (7.0×) | 0.14 ms (3.2×) |
| Mushroom | 1% | 26,388 | 0.87 ms | 0.033 ms (26×) | 0.096 ms (9.1×) | 0.23 ms (3.8×) |
| Connect-4 | 85% | 411 | 0.008 ms | 0.025 ms (0.33×) | 0.051 ms (0.16×) | 0.072 ms (0.11×) |
| Connect-4 | 50% | 5,715 | 0.27 ms | 0.030 ms (9.2×) | 0.062 ms (4.4×) | 0.10 ms (2.7×) |
| Online Retail | 1% | 297,089 | 30.2 ms | 0.28 ms (110×) | 0.78 ms (39×) | 5.5 ms (5.5×) |
| Online Retail | 0.2% | 454,641 | 73.5 ms | 0.88 ms (83×) | 2.3 ms (32×) | 9.0 ms (8.2×) |

At the kernel level the GPU wins clearly once a tree has more than a few
thousand nodes: 9-110× for the kernel alone and 4-39× with the copies,
mostly growing with tree size (the largest tree, 455k nodes, gains 83×
and 32×, less than the 297k-node one). The work is thousands of independent upward walks
with all atomics in shared memory, which is exactly what a GPU is for. On
a 411-node tree, launch latency (~25 µs) exceeds the CPU's 8 µs of work.
Kernel times below 1 ms vary run to run: the same Online Retail 0.2% point
measured 0.56 ms in an earlier run.

Why none of this shows up end to end:

1. **The step is a tiny part of the run.** Even its largest instance, 73 ms
   on Online Retail 0.2%, is 1.8% of the serial run (4.08 s); on Chess 25% it
   is 1.7 ms of 27.8 s. Mining time is dominated by formatting itemsets and
   by many tiny conditional trees (section 1).
2. **Fixed costs exceed the saving.** Creating the CUDA context costs
   0.14-0.22 s per run, more than the best saving here (73 → 9 ms). For the
   large trees the GPU path is also dominated by its own CPU side:
   flattening 455k nodes into arrays and unpacking the 2.7M-entry matrix
   take ~6.7 ms, three times the copies plus kernel.
3. **Against OpenMP the CPU work is not serial.** In `fpgrowth_omp` each
   item's counting walk runs inside that item's task, spread over all
   threads. The GPU call for the initial tree runs once, before the parallel
   loop, on the critical path: 9 ms of wall time, while 73 ms of CPU work
   spread evenly over 28 threads would cost only a few ms.
4. **Deeper trees are too small.** The rest of the counting happens in
   conditional trees that average 4-14 nodes. At that size the GPU loses
   even on the kernel alone (as the 411-node Connect-4 85% tree shows), so
   the miner keeps them on the CPU (`--gpu-min-nodes`).

So the GPU kernel is fast, but it accelerates a step that is a small
fraction of FP-growth's runtime on these datasets. That fraction would grow
with much larger initial trees, such as more transactions or more distinct
items at a support where conditional trees stay large.
