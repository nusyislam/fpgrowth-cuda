#!/usr/bin/env python3
"""Cross-check the fpgrowth binary against mlxtend's fpgrowth on one dataset.

Runs both on the same .dat file and min_support (absolute count or fraction,
same CLI semantics as the binary), verifies that itemsets and supports match
exactly, and reports both runtimes.

Usage: compare_mlxtend.py <file.dat> <min_support> [--binary PATH]
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from minsupport import parse_min_support


def read_dat(path):
    """Same semantics as the C++ reader: skip blank lines, dedupe items."""
    transactions = []
    for line in Path(path).read_text().splitlines():
        items = line.split()
        if items:
            transactions.append(sorted(set(int(x) for x in items)))
    return transactions


def parse_binary_output(stdout):
    result = {}
    for line in stdout.splitlines():
        if not line.strip():
            continue
        head, _, sup = line.rpartition(" (")
        result[frozenset(int(x) for x in head.split())] = int(sup.rstrip(")"))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dat")
    ap.add_argument("min_support", help="absolute count (e.g. 3) or fraction (e.g. 0.05)")
    ap.add_argument("--binary", default="./fpgrowth")
    args = ap.parse_args()

    transactions = read_dat(args.dat)
    n = len(transactions)
    if n == 0:
        print("dataset has no transactions; nothing to compare")
        return 0
    threshold = parse_min_support(args.min_support, n)

    # --- my binary ---------------------------------------------------------
    t0 = time.perf_counter()
    proc = subprocess.run([str(Path(args.binary).resolve()), args.dat, args.min_support],
                          capture_output=True, text=True)
    t_binary = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"binary failed (exit {proc.returncode}):\n{proc.stderr}")
        return 1
    mine = parse_binary_output(proc.stdout)
    stats = dict(kv.split("=", 1) for kv in proc.stderr.split())
    mine_s = float(stats.get("mine_s", "nan"))

    # --- mlxtend -----------------------------------------------------------
    import pandas as pd
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import fpgrowth

    te = TransactionEncoder()
    df = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)
    # (threshold - 0.5) / n, not threshold / n: mlxtend compares float
    # fractional supports, and count >= threshold - 0.5 selects exactly the
    # integer counts >= threshold without floating-point boundary effects.
    t0 = time.perf_counter()
    res = fpgrowth(df, min_support=(threshold - 0.5) / n, use_colnames=True)
    t_mlxtend = time.perf_counter() - t0
    theirs = {frozenset(s): int(round(sup * n))
              for s, sup in zip(res["itemsets"], res["support"])}

    # --- compare -----------------------------------------------------------
    print(f"{args.dat}: {n} transactions, min_support {args.min_support} "
          f"-> threshold {threshold}")
    print(f"  fpgrowth binary : {len(mine):6d} itemsets  "
          f"{t_binary:.4f} s wall ({mine_s:.6f} s mining)")
    print(f"  mlxtend fpgrowth: {len(theirs):6d} itemsets  "
          f"{t_mlxtend:.4f} s mining call")

    if mine == theirs:
        print("MATCH: itemsets and supports identical")
        return 0
    print("MISMATCH:")
    for k in sorted(set(mine) | set(theirs), key=lambda s: sorted(s)):
        if mine.get(k) != theirs.get(k):
            items = " ".join(map(str, sorted(k)))
            print(f"  {items}: binary={mine.get(k)} mlxtend={theirs.get(k)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
