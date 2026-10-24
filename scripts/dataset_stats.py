#!/usr/bin/env python3
"""Summarise the processed .dat datasets and write results/dataset_stats.csv.

Per dataset: transactions, distinct items, avg/min/max transaction length,
density (avg length / distinct items) and file size.  Transactions are read
with the same semantics as the C++ reader (blank lines skipped, duplicate
items within a line counted once).

Usage: dataset_stats.py [file.dat ...]   (default: data/processed/*.dat)
"""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ["dataset", "transactions", "distinct_items", "avg_len", "min_len",
          "max_len", "density", "file_size_bytes"]


def stats(path):
    items, n, total, lo, hi = set(), 0, 0, None, 0
    with open(path) as f:
        for line in f:
            t = set(line.split())
            if not t:
                continue
            n += 1
            total += len(t)
            lo = len(t) if lo is None else min(lo, len(t))
            hi = max(hi, len(t))
            items |= t
    avg = total / n if n else 0.0
    return {
        "dataset": path.stem,
        "transactions": n,
        "distinct_items": len(items),
        "avg_len": round(avg, 2),
        "min_len": lo or 0,
        "max_len": hi,
        "density": round(avg / len(items), 4) if items else 0.0,
        "file_size_bytes": path.stat().st_size,
    }


def human(size):
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--csv", type=Path, default=ROOT / "results" / "dataset_stats.csv")
    args = ap.parse_args()
    files = args.files or sorted((ROOT / "data" / "processed").glob("*.dat"))

    rows = [stats(p) for p in files]
    header = ["dataset", "transactions", "items", "avg len", "min", "max", "density", "size"]
    table = [[r["dataset"], f"{r['transactions']:,}", f"{r['distinct_items']:,}",
              f"{r['avg_len']:.2f}", str(r["min_len"]), str(r["max_len"]),
              f"{r['density']:.4f}", human(r["file_size_bytes"])] for r in rows]
    widths = [max(len(x) for x in col) for col in zip(header, *table)]
    fmt = "  ".join(f"{{:<{w}}}" if i == 0 else f"{{:>{w}}}" for i, w in enumerate(widths))
    print(fmt.format(*header))
    print("  ".join("-" * w for w in widths))
    for t in table:
        print(fmt.format(*t))

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.csv.relative_to(ROOT) if args.csv.is_relative_to(ROOT) else args.csv}")


if __name__ == "__main__":
    main()
