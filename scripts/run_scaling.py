#!/usr/bin/env python3
"""Scalability: runtime vs number of transactions.

For Connect-4 and Online Retail, shuffle the transactions once with a fixed
seed and take the first 10/25/50/75/100% as subsets (so each subset contains
the smaller ones), write them to data/processed/subsets/, and run the binary
at a fixed *relative* min_support on each, REPS times, recording medians to
results/scaling.csv.  Same --unsorted / /dev/null / timeout / memory limits
as run_bench.py.  Resumable: (dataset, fraction) pairs already in the CSV
are skipped.

Usage: run_scaling.py [dataset ...]
"""
import argparse
import random

from benchlib import (PROCESSED, RESULTS, TIME_KEYS, COUNT_KEYS, CsvAppender, check_memory,
                      dat_path, fmt, median_stats, read_rows, run_binary)

# Relative support per dataset: a moderate itemset count at 100% that stays
# moderate on the 10% subset as well.
SUPPORT = {"connect4": 0.7, "online_retail": 0.005}
FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]
SEED = 327
REPS = 5

CSV = RESULTS / "scaling.csv"
SUBSETS = PROCESSED / "subsets"
FIELDS = ["dataset", "fraction", "min_support", "status", "reps", "transactions",
          "min_count", *TIME_KEYS, *COUNT_KEYS]


def make_subsets(name):
    """Write <name>_<pct>.dat for every fraction; returns {fraction: path}."""
    lines = [l for l in dat_path(name).read_text().splitlines() if l.strip()]
    order = list(range(len(lines)))
    random.Random(SEED).shuffle(order)
    SUBSETS.mkdir(parents=True, exist_ok=True)
    paths = {}
    for frac in FRACTIONS:
        path = SUBSETS / f"{name}_{round(frac * 100):03d}.dat"
        k = round(frac * len(lines))
        # Keep the original file order inside the subset; only membership is random.
        keep = sorted(order[:k])
        path.write_text("".join(lines[i] + "\n" for i in keep))
        paths[frac] = path
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", metavar="dataset")
    args = ap.parse_args()
    done = {(r["dataset"], float(r["fraction"])) for r in read_rows(CSV)}
    out = CsvAppender(CSV, FIELDS)
    try:
        for name in args.names or SUPPORT:
            paths = make_subsets(name)
            s = SUPPORT[name]
            for frac in FRACTIONS:
                if (name, frac) in done:
                    print(f"{name} {frac:.0%}: in CSV, skipping")
                    continue
                check_memory(f"{name} {frac:.0%}")
                runs, status = [], "ok"
                while len(runs) < REPS:
                    status, r = run_binary(paths[frac], s)
                    if status != "ok":
                        runs = []
                        break
                    runs.append(r)
                row = {"dataset": name, "fraction": frac, "min_support": s,
                       "status": status, "reps": len(runs)}
                if runs:
                    row.update(median_stats(runs))
                    row["transactions"] = runs[0]["transactions"]
                    row["min_count"] = runs[0]["min_count"]
                out.write({k: fmt(v) for k, v in row.items()})
                print(f"{name} {frac:.0%}: {status}"
                      + (f" n={row['transactions']} itemsets={int(row['itemsets'])} "
                         f"total={row['total_s']:.3f}s" if runs else ""), flush=True)
    finally:
        out.close()


if __name__ == "__main__":
    main()
