#!/usr/bin/env python3
"""Support sweep: run the binary on every dataset over a range of min_support
levels and write per-level medians to results/bench.csv.

Levels are visited from high to low support.  Each level is run REPS times
with --unsorted and output to /dev/null (so result buffering and formatting
to a real file do not distort time or memory), and the median of every phase
time and counter is recorded.  A dataset stops going lower once a run
exceeds the time budget (~60 s) or memory budget (~4 GB), or is killed by
the hard timeout / address-space limit; that level is recorded with a status
other than "ok" and a single repetition.

Resumable: levels already in the CSV are skipped, and a dataset whose CSV
rows already contain a stopping point is not extended.

Usage: run_bench.py [dataset ...] [--lowest S]   (default: all datasets)
  --lowest S  only run levels >= S in this invocation (to split long sweeps)
"""
import argparse

from benchlib import (DATASETS, RESULTS, TIME_KEYS, COUNT_KEYS, CsvAppender, check_memory,
                      dat_path, fmt, median_stats, read_rows, run_binary)

# Chosen from itemset counts in single-run probes: from tens of itemsets down
# to where a run costs about a minute (the last level is expected to stop).
LEVELS = {
    "chess":         [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2],
    "mushroom":      [0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005],
    "connect4":      [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4],
    "online_retail": [0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.0025, 0.002, 0.0015],
}
REPS = 5
TIME_BUDGET_S = 60
MEM_BUDGET_KB = 4_000_000

CSV = RESULTS / "bench.csv"
FIELDS = ["dataset", "min_support", "status", "reps", "transactions", "min_count",
          *TIME_KEYS, *COUNT_KEYS]


def sweep(name, done, out, lowest=0.0):
    rows = [r for r in done if r["dataset"] == name]
    if any(r["status"] != "ok" for r in rows):
        print(f"{name}: already stopped at {[r['min_support'] for r in rows if r['status'] != 'ok'][0]}")
        return
    have = {float(r["min_support"]) for r in rows}
    for s in LEVELS[name]:
        if s < lowest:
            print(f"{name}: levels below {lowest} left for a later run")
            return
        if s in have:
            print(f"{name} {s}: in CSV, skipping")
            continue
        check_memory(f"{name} {s}")
        status, first = run_binary(dat_path(name), s)
        runs = [first] if status == "ok" else []
        if status == "ok" and first["total_s"] > TIME_BUDGET_S:
            status = "over_time_budget"
        elif status == "ok" and first["peak_rss_kb"] > MEM_BUDGET_KB:
            status = "over_memory_budget"
        while status == "ok" and len(runs) < REPS:
            st, r = run_binary(dat_path(name), s)
            if st != "ok":
                status, runs = st, []
                break
            runs.append(r)

        row = {"dataset": name, "min_support": s, "status": status, "reps": len(runs)}
        if runs:
            row.update(median_stats(runs))
            row["transactions"], row["min_count"] = runs[0]["transactions"], runs[0]["min_count"]
        out.write({k: fmt(v) for k, v in row.items()})
        if runs:
            print(f"{name} {s}: {status} itemsets={int(row['itemsets'])} "
                  f"total={row['total_s']:.3f}s rss={row['peak_rss_kb'] / 1024:.0f}MB "
                  f"({len(runs)} reps)", flush=True)
        else:
            print(f"{name} {s}: {status}", flush=True)
        if status != "ok":
            print(f"{name}: stopping sweep at {s} ({status})")
            return


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", metavar="dataset")
    ap.add_argument("--lowest", type=float, default=0.0,
                    help="only run levels >= this support in this invocation")
    args = ap.parse_args()
    done = read_rows(CSV)
    out = CsvAppender(CSV, FIELDS)
    try:
        for name in args.names or DATASETS:
            sweep(name, done, out, args.lowest)
    finally:
        out.close()


if __name__ == "__main__":
    main()
