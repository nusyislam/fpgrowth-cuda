#!/usr/bin/env python3
"""Distribution of frequent-itemset lengths at a few support levels per dataset.

Runs the binary with its normal (sorted, buffered) output to stdout, streams
that output through a pipe -- nothing is written to disk -- and counts the
itemsets of each length.  The total is checked against the binary's own
itemsets= counter.  Writes results/lengths.csv with one row per
(dataset, min_support, length).  Resumable per (dataset, min_support).

Usage: itemset_lengths.py [dataset ...]
"""
import argparse
import subprocess
import sys
from collections import Counter

from benchlib import (RESULTS, CsvAppender, binary_cmd, check_memory, dat_path, parse_stats,
                      read_rows)

LEVELS = {
    "chess":         [0.8, 0.6, 0.4],
    "mushroom":      [0.3, 0.1, 0.03],
    "connect4":      [0.9, 0.8, 0.7],
    "online_retail": [0.02, 0.005, 0.002],
}
TIMEOUT_S = 300

CSV = RESULTS / "lengths.csv"
FIELDS = ["dataset", "min_support", "length", "count", "fraction"]


def lengths(dat, support):
    """Counter {length: itemsets}, or None if the run was killed / failed."""
    proc = subprocess.Popen(binary_cmd(dat, support, output=None, unsorted=False,
                                       timeout=TIMEOUT_S),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    hist = Counter()
    for line in proc.stdout:          # "1 2 5 (2)": number of spaces == length
        hist[line.count(" ")] += 1
    stderr = proc.stderr.read()
    if proc.wait() != 0:
        print(f"  run failed or was killed (exit {proc.returncode}): {stderr.strip()[:200]}")
        return None
    expected = parse_stats(stderr)["itemsets"]
    if sum(hist.values()) != expected:
        sys.exit(f"parsed {sum(hist.values())} itemsets but the binary reported {expected}")
    return hist


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", metavar="dataset")
    args = ap.parse_args()
    done = {(r["dataset"], float(r["min_support"])) for r in read_rows(CSV)}
    out = CsvAppender(CSV, FIELDS)
    try:
        for name in args.names or LEVELS:
            for s in LEVELS[name]:
                if (name, s) in done:
                    print(f"{name} {s}: in CSV, skipping")
                    continue
                check_memory(f"{name} {s}")
                hist = lengths(dat_path(name), s)
                if hist is None:
                    print(f"{name} {s}: skipped")
                    continue
                total = sum(hist.values())
                for k in sorted(hist):
                    out.write({"dataset": name, "min_support": s, "length": k,
                               "count": hist[k], "fraction": f"{hist[k] / total:.6f}"})
                mode = max(hist, key=hist.get)
                print(f"{name} {s}: {total} itemsets, lengths {min(hist)}..{max(hist)}, "
                      f"mode {mode}", flush=True)
    finally:
        out.close()


if __name__ == "__main__":
    main()
