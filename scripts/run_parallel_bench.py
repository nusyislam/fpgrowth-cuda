#!/usr/bin/env python3
"""Parallel benchmark: serial vs OpenMP vs CUDA on this machine.

For every dataset at 3-4 support levels taken from results/bench.csv, times
  serial       ./fpgrowth
  omp_tN       ./fpgrowth_omp --threads N    for N in 1, 2, 4, 8 (those <= cores)
  omp_tall     ./fpgrowth_omp                (all logical CPUs)
  cuda_t1      ./fpgrowth_cuda --threads 1   (GPU counting, one mining thread)
  cuda_tall    ./fpgrowth_cuda               (GPU counting, all mining threads)
with REPS runs each (median), --unsorted output to /dev/null as in
run_bench.py.  Also sweeps fpgrowth_cuda's --gpu-min-nodes on two settings to
show where offloading a conditional tree starts to pay on this GPU.

Writes (the hostname keeps machines apart):
  results/parallel_<host>.csv        one row per (dataset, support, config)
  results/gpu_threshold_<host>.csv   the --gpu-min-nodes sweep
  results/gpu_info_<host>.txt        CPU, cores, memory, OS, compilers, CUDA, gpuinfo

Levels: per dataset, the completed levels of results/bench.csv whose serial
total time is closest (in log scale) to 0.1 s, 1 s, 10 s and 25 s on the
machine that made bench.csv (duplicates dropped, so 3-4 levels).  They are
visited easiest first; a config that times out or hits the memory cap is not
run at harder levels of that dataset.

Caps, one job at a time: every run has the same wall-clock timeout as
benchlib (120 s) and a 4 GB cap (less on machines with little RAM).  The cap
is on resident memory, polled every 20 ms, not on address space (prlimit
--as) as in benchlib: CUDA reserves large virtual ranges at start-up and
OpenMP threads each get a malloc arena, so an address-space cap would stop
those binaries for memory they never use.  Available memory is checked before
every level.

Standard library only (no venv, no internet needed); Linux /proc required.
Resumable: rows already in the CSV are skipped (delete it to re-measure).

Usage: run_parallel_bench.py [dataset ...] [--reps N] [--quick] [--no-sweep]
  --quick     levels up to the ~10 s target only, 3 reps (a first look)
"""
import argparse
import math
import os
import platform
import signal
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

from benchlib import (DATASETS, RESULTS, ROOT, TIME_KEYS, TIMEOUT_S, CsvAppender, dat_path,
                      fmt, parse_stats, read_rows)

HOST = socket.gethostname().split(".")[0]
CSV = RESULTS / f"parallel_{HOST}.csv"
SWEEP_CSV = RESULTS / f"gpu_threshold_{HOST}.csv"
INFO = RESULTS / f"gpu_info_{HOST}.txt"

SERIAL, OMP, CUDA = ROOT / "fpgrowth", ROOT / "fpgrowth_omp", ROOT / "fpgrowth_cuda"
TARGETS_S = [0.1, 1.0, 10.0, 25.0]
REPS = 5
POLL_S = 0.02

COUNT_KEYS = ["peak_rss_kb", "itemsets", "frequent_items", "tree_nodes", "cond_trees",
              "cond_nodes", "single_paths"]
CUDA_KEYS = ["gpu_init_s", "count_h2d_s", "count_kernel_s", "count_d2h_s", "tree_prep_s",
             "tree_h2d_s", "tree_kernel_s", "tree_d2h_s", "gpu_trees", "gpu_cond_trees",
             "gpu_tree_nodes", "h2d_bytes", "d2h_bytes"]
FIELDS = ["host", "dataset", "min_support", "config", "threads", "gpu_min_nodes", "status",
          "reps", "transactions", "min_count", *TIME_KEYS, *COUNT_KEYS, *CUDA_KEYS]
SWEEP_FIELDS = ["host", "dataset", "min_support", "gpu_min_nodes", "status", "reps",
                "mine_s", "total_s", "gpu_trees", "gpu_tree_nodes", "tree_prep_s",
                "tree_h2d_s", "tree_kernel_s", "tree_d2h_s"]
SWEEP_NODES = [64, 256, 1024, 4096, 16384, 10**9]   # 10**9: initial tree only


# ---------------------------------------------------------------------------
# Machine and memory
# ---------------------------------------------------------------------------

def meminfo_kb(key):
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1])
    return None


TOTAL_KB = meminfo_kb("MemTotal") or 16_000_000
MEM_CAP_KB = min(4_000_000, TOTAL_KB * 2 // 5)        # 4 GB, or 40% of RAM if smaller
MIN_AVAILABLE_KB = min(6_000_000, TOTAL_KB // 2)       # refuse to start below this


def check_memory(what):
    avail = meminfo_kb("MemAvailable")
    if avail is None:
        return
    print(f"  [free: {avail / 1024**2:.1f} of {TOTAL_KB / 1024**2:.1f} GiB available "
          f"before {what}]", flush=True)
    if avail < MIN_AVAILABLE_KB:
        sys.exit(f"only {avail / 1024**2:.1f} GiB available; not starting {what}. "
                 "Close other programs (check `free -h`) and rerun; finished rows are kept.")


def sh(cmd):
    """Output of a shell command, or '' if it fails or is missing."""
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def write_machine_info(cuda_ok, gpuinfo_text):
    cpu = (sh("lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1")
           or sh("sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -1")
           or platform.processor() or "unknown")
    board = sh("tr -d '\\0' < /proc/device-tree/model 2>/dev/null")
    lines = [
        f"Machine: {HOST}   (parallel benchmarks: results/parallel_{HOST}.csv)",
        f"Collected: {time.strftime('%Y-%m-%d %H:%M %Z')}",
        "",
        f"CPU:        {cpu}" + (f"   (board: {board})" if board else ""),
        f"Logical CPUs: {os.cpu_count()}   (omp_tall uses all of them)",
        f"lscpu:      {sh('lscpu | grep -E \"^(Core|Socket|Thread|CPU max MHz)\" | tr -s \" \" | paste -sd \";\" -')}",
        f"RAM:        {TOTAL_KB / 1024**2:.1f} GiB   (per-run memory cap {MEM_CAP_KB / 1024**2:.1f} GiB RSS)",
        f"OS:         {sh('. /etc/os-release 2>/dev/null && echo $PRETTY_NAME') or platform.platform()}",
        f"Kernel:     {platform.release()}"
        + ("   (WSL2 virtual machine)" if "microsoft" in platform.release().lower() else ""),
        f"C++:        {sh('${CXX:-g++} --version | head -1')}",
        f"nvcc:       {sh('nvcc --version | tail -2 | paste -sd \" \" -') or 'not found'}",
        f"Python:     {platform.python_version()}",
    ]
    power = sh("nvpmodel -q 2>/dev/null | paste -sd ' ' -")
    if power:
        lines.append(f"Jetson power mode (nvpmodel -q): {power}")
    lines += ["", "gpuinfo:" if cuda_ok else "gpuinfo: (no usable GPU / fpgrowth_cuda not run)",
              gpuinfo_text, "",
              "Notes:",
              "- Times are medians; --unsorted output to /dev/null (as results/bench.csv).",
              "- One run at a time; each capped at "
              f"{TIMEOUT_S} s wall clock and {MEM_CAP_KB / 1024**2:.1f} GiB resident memory.",
              "- CUDA phase times are cudaEvent sums over all calls; with several mining",
              "  threads, calls on different streams can overlap in time."]
    INFO.write_text("\n".join(lines) + "\n")
    print(f"wrote {INFO.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Capped runs
# ---------------------------------------------------------------------------

def descendants(pid):
    """pid and all its descendant pids (via /proc/<pid>/task/*/children)."""
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        out.append(p)
        try:
            for task in Path(f"/proc/{p}/task").iterdir():
                todo += [int(c) for c in (task / "children").read_text().split()]
        except OSError:
            pass
    return out


def rss_kb(pids):
    total = 0
    for p in pids:
        try:
            for line in Path(f"/proc/{p}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1])
                    break
        except OSError:
            pass
    return total


def run_capped(cmd, timeout=TIMEOUT_S, raw=False):
    """One run, through `timeout` (so the binary is a fresh process and its own
    peak_rss_kb is not inflated by Python's), with the resident-memory cap.
    Returns (status, stats): status "ok", "timeout", "memory_limit" or "error";
    stats is the parsed last stderr line, or with raw=True the whole stderr."""
    proc = subprocess.Popen(["timeout", str(timeout), *map(str, cmd)], stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, text=True, start_new_session=True)
    deadline = time.monotonic() + timeout + 30
    killed = None
    while proc.poll() is None:
        if rss_kb(descendants(proc.pid)) > MEM_CAP_KB:
            killed = "memory_limit"
        elif time.monotonic() > deadline:
            killed = "timeout"
        if killed:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            break
        time.sleep(POLL_S)
    err = proc.stderr.read()
    proc.stderr.close()
    if killed:
        return killed, {}
    if proc.returncode == 124:
        return "timeout", {}
    if proc.returncode != 0:
        if "bad_alloc" in err or proc.returncode > 128:
            return "memory_limit", {}
        print(f"    failed (exit {proc.returncode}): {err.strip()[-300:]}", file=sys.stderr)
        return "error", {}
    return "ok", err if raw else parse_stats(err)


def median_row(runs):
    row = {}
    for k in TIME_KEYS + COUNT_KEYS + CUDA_KEYS:
        vals = [r[k] for r in runs if k in r]
        if vals:
            row[k] = statistics.median(vals)
    return row


# ---------------------------------------------------------------------------
# Configurations and levels
# ---------------------------------------------------------------------------

def configs(cuda_ok):
    cores = os.cpu_count() or 1
    out = [("serial", SERIAL, [], 1)]
    out += [(f"omp_t{n}", OMP, ["--threads", n], n) for n in (1, 2, 4, 8) if n < cores]
    out.append(("omp_tall", OMP, [], cores))
    if cuda_ok:
        out += [("cuda_t1", CUDA, ["--threads", 1], 1), ("cuda_tall", CUDA, [], cores)]
    return out


def pick_levels(name, targets):
    rows = [r for r in read_rows(RESULTS / "bench.csv")
            if r["dataset"] == name and r["status"] == "ok"]
    if not rows:
        return []
    chosen = []
    for t in targets:
        best = min(rows, key=lambda r: abs(math.log(float(r["total_s"]) / t)))
        s = float(best["min_support"])
        if s not in chosen:
            chosen.append(s)
    return sorted(chosen, reverse=True)       # easiest (highest support) first


def cuda_usable():
    if not CUDA.exists():
        print("fpgrowth_cuda not built: CUDA configs skipped (make cuda)")
        return False
    status, _ = run_capped([CUDA, ROOT / "data/examples/han.dat", 2], timeout=60)
    if status != "ok":
        print("fpgrowth_cuda does not run here (no GPU?): CUDA configs skipped")
    return status == "ok"


def bench(names, reps, targets, cuda_ok):
    done = {(r["dataset"], float(r["min_support"]), r["config"]) for r in read_rows(CSV)}
    failed = {(r["dataset"], r["config"]) for r in read_rows(CSV) if r["status"] != "ok"}
    out = CsvAppender(CSV, FIELDS)
    try:
        for name in names:
            dat = dat_path(name)
            if not dat.exists():
                print(f"{name}: {dat.relative_to(ROOT)} missing, skipped")
                continue
            for s in pick_levels(name, targets):
                check_memory(f"{name} {s}")
                for config, binary, args, threads in configs(cuda_ok):
                    if (name, s, config) in done:
                        print(f"{name} {s} {config}: in CSV, skipping")
                        continue
                    if (name, config) in failed:
                        print(f"{name} {s} {config}: stopped at an easier level, skipping")
                        continue
                    cmd = [binary, dat, s, "/dev/null", "--unsorted", *args]
                    runs, status = [], "ok"
                    while len(runs) < reps:
                        status, st = run_capped(cmd)
                        if status != "ok":
                            runs = []
                            break
                        runs.append(st)
                    row = {"host": HOST, "dataset": name, "min_support": s, "config": config,
                           "threads": threads, "status": status, "reps": len(runs)}
                    if runs:
                        row.update(median_row(runs))
                        row["threads"] = runs[0].get("threads", 1)
                        row["gpu_min_nodes"] = runs[0].get("gpu_min_nodes", "")
                        row["transactions"] = runs[0]["transactions"]
                        row["min_count"] = runs[0]["min_count"]
                        print(f"{name} {s} {config}: total={row['total_s']:.3f}s "
                              f"mine={row['mine_s']:.3f}s itemsets={int(row['itemsets'])} "
                              f"rss={row['peak_rss_kb'] / 1024:.0f}MB ({len(runs)} reps)", flush=True)
                    else:
                        failed.add((name, config))
                        print(f"{name} {s} {config}: {status}", flush=True)
                    out.write({k: fmt(v) for k, v in row.items()})
    finally:
        out.close()


def threshold_sweep(reps):
    """--gpu-min-nodes sweep with one mining thread (so the GPU effect is not
    hidden by parallelism) at the 2nd-easiest level of two datasets."""
    done = {(r["dataset"], float(r["min_support"]), int(r["gpu_min_nodes"]))
            for r in read_rows(SWEEP_CSV)}
    out = CsvAppender(SWEEP_CSV, SWEEP_FIELDS)
    try:
        for name in ("online_retail", "chess"):
            levels = pick_levels(name, TARGETS_S)
            if len(levels) < 2 or not dat_path(name).exists():
                continue
            s = levels[1]
            check_memory(f"threshold sweep {name} {s}")
            for nodes in SWEEP_NODES:
                if (name, s, nodes) in done:
                    continue
                cmd = [CUDA, dat_path(name), s, "/dev/null", "--unsorted", "--threads", 1,
                       "--gpu-min-nodes", nodes]
                runs, status = [], "ok"
                while len(runs) < reps:
                    status, st = run_capped(cmd)
                    if status != "ok":
                        runs = []
                        break
                    runs.append(st)
                row = {"host": HOST, "dataset": name, "min_support": s, "gpu_min_nodes": nodes,
                       "status": status, "reps": len(runs)}
                if runs:
                    for k in SWEEP_FIELDS[6:]:
                        row[k] = statistics.median(r[k] for r in runs)
                    print(f"sweep {name} {s} --gpu-min-nodes {nodes}: mine={row['mine_s']:.3f}s "
                          f"gpu_trees={int(row['gpu_trees'])}", flush=True)
                out.write({k: fmt(v) for k, v in row.items()})
    finally:
        out.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", metavar="dataset")
    ap.add_argument("--reps", type=int, default=None, help=f"runs per point (default {REPS})")
    ap.add_argument("--quick", action="store_true", help="levels up to ~10 s, 3 reps")
    ap.add_argument("--no-sweep", action="store_true", help="skip the --gpu-min-nodes sweep")
    args = ap.parse_args()

    for b in (SERIAL, OMP):
        if not b.exists():
            sys.exit(f"{b.name} not built: run `make && make omp` (and `make cuda ARCH=...`)")
    reps = args.reps or (3 if args.quick else REPS)
    targets = TARGETS_S[:3] if args.quick else TARGETS_S

    print(f"host {HOST}: {os.cpu_count()} logical CPUs, {TOTAL_KB / 1024**2:.1f} GiB RAM, "
          f"cap {MEM_CAP_KB / 1024**2:.1f} GiB RSS / {TIMEOUT_S} s per run, {reps} reps")
    cuda_ok = cuda_usable()
    gpuinfo = ROOT / "gpuinfo"
    info_text = sh(str(gpuinfo)) if cuda_ok and gpuinfo.exists() else ""
    if cuda_ok and not info_text:
        info_text = "(./gpuinfo not built: make gpuinfo)"
    write_machine_info(cuda_ok, info_text)

    bench(args.names or DATASETS, reps, targets, cuda_ok)
    if cuda_ok and not args.no_sweep:
        threshold_sweep(min(reps, 3))
    print(f"done: {CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
