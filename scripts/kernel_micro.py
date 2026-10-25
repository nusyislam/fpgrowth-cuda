#!/usr/bin/env python3
"""Kernel micro-benchmark: conditional-support counting of the initial FP-tree,
CPU vs GPU, with CUDA context creation excluded.

For every dataset at 2 support levels (the easiest and the hardest level that
run_parallel_bench.py uses), runs `fpgrowth_cuda <data> <support>
--micro-count REPS`.  In one process, after the CUDA context exists and one
untimed warm-up, that times REPS times:
  cpu_s               the counting walk for every item on one CPU thread
                      (the step fpgrowth does per item before building each
                      conditional tree)
  gpu_kernel_s        the counting kernel alone (cudaEvent)
  gpu_kernel_xfer_s   kernel + host->device + device->host copies (cudaEvent)
  gpu_wall_s          the whole GPU path as the miner sees it: also flattening
                      the tree into arrays on the CPU and unpacking the result
and checks that CPU and GPU give the same matrix.  Medians go to
results/kernel_micro_<host>.csv (rewritten on every run; it takes seconds).

Standard library only, like run_parallel_bench.py (same caps per run).

Usage: kernel_micro.py [dataset ...] [--reps N]
"""
import argparse
import csv
import statistics
import sys

from benchlib import DATASETS, RESULTS, dat_path, fmt
from run_parallel_bench import CUDA, HOST, TARGETS_S, check_memory, cuda_usable, pick_levels, run_capped

CSV = RESULTS / f"kernel_micro_{HOST}.csv"
REPS = 5
TIMES = ["cpu_s", "gpu_kernel_s", "gpu_kernel_xfer_s", "gpu_wall_s",
         "gpu_h2d_s", "gpu_d2h_s", "gpu_prep_s"]
FIELDS = ["host", "dataset", "min_support", "status", "reps", "tree_nodes", "frequent_items",
          "matrix_entries", *TIMES, "h2d_bytes", "d2h_bytes",
          "speedup_kernel", "speedup_kernel_xfer", "speedup_wall"]


def parse_reps(stderr):
    reps = []
    for line in stderr.splitlines():
        if not line.startswith("micro "):
            continue
        r = {}
        for kv in line.split()[1:]:
            k, _, v = kv.partition("=")
            try:
                r[k] = float(v)
            except ValueError:
                r[k] = v
        r["gpu_kernel_xfer_s"] = r["gpu_h2d_s"] + r["gpu_kernel_s"] + r["gpu_d2h_s"]
        reps.append(r)
    return reps


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", metavar="dataset")
    ap.add_argument("--reps", type=int, default=REPS)
    args = ap.parse_args()
    if not cuda_usable():
        sys.exit("needs a working ./fpgrowth_cuda (make cuda ARCH=... NVCC_CCBIN=...)")

    rows = []
    for name in args.names or DATASETS:
        if not dat_path(name).exists():
            print(f"{name}: data missing, skipped")
            continue
        levels = pick_levels(name, TARGETS_S)
        for s in sorted({levels[0], levels[-1]}, reverse=True):
            check_memory(f"{name} {s}")
            status, err = run_capped([CUDA, dat_path(name), s, "--micro-count", args.reps], raw=True)
            row = {"host": HOST, "dataset": name, "min_support": s, "status": status}
            reps = parse_reps(err) if status == "ok" else []
            if status == "ok" and len(reps) != args.reps:
                row["status"] = "error"
            if row["status"] == "ok":
                row["reps"] = len(reps)
                for k in ("tree_nodes", "frequent_items", "matrix_entries", "h2d_bytes", "d2h_bytes"):
                    row[k] = int(reps[0][k])
                for k in TIMES:
                    row[k] = statistics.median(r[k] for r in reps)
                row["speedup_kernel"] = row["cpu_s"] / row["gpu_kernel_s"]
                row["speedup_kernel_xfer"] = row["cpu_s"] / row["gpu_kernel_xfer_s"]
                row["speedup_wall"] = row["cpu_s"] / row["gpu_wall_s"]
                print(f"{name} {s}: {row['tree_nodes']:,} nodes, {row['frequent_items']} items  "
                      f"cpu {row['cpu_s'] * 1e3:.3f} ms  kernel {row['gpu_kernel_s'] * 1e3:.3f} ms "
                      f"({row['speedup_kernel']:.1f}x)  +xfer {row['gpu_kernel_xfer_s'] * 1e3:.3f} ms "
                      f"({row['speedup_kernel_xfer']:.1f}x)  wall {row['gpu_wall_s'] * 1e3:.3f} ms "
                      f"({row['speedup_wall']:.1f}x)", flush=True)
            else:
                print(f"{name} {s}: {row['status']}\n{err.strip()[-300:] if isinstance(err, str) else ''}")
            rows.append(row)

    with CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            # times are microseconds-scale: keep ns resolution
            w.writerow({k: (f"{v:.9f}" if k in TIMES else fmt(v)) for k, v in r.items()})
    print(f"wrote {CSV.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
