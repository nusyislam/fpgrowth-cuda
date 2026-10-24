#!/usr/bin/env python3
"""Regenerate tests/edge_cases/*.expected by independent brute-force
enumeration (same reference algorithm as brute_force.py, not the binary).
Run from anywhere: python3 tests/gen_expected.py
"""
import itertools
from pathlib import Path

EDGE = Path(__file__).resolve().parent / "edge_cases"

CASES = {  # name -> min_support threshold (absolute)
    "empty": 1,
    "single_transaction": 1,
    "all_identical": 5,
    "none_frequent": 2,
    "all_frequent": 2,
    "long_transaction": 1,
}


def brute_force(transactions, min_count):
    all_items = sorted(set().union(*transactions)) if transactions else []
    out = []
    for size in range(1, len(all_items) + 1):
        for combo in itertools.combinations(all_items, size):
            s = frozenset(combo)
            c = sum(1 for t in transactions if s <= t)
            if c >= min_count:
                out.append((combo, c))
    out.sort()
    return out


def main():
    # long_transaction.dat: one transaction of 12 items -> 4095 itemsets at
    # min_support 1, exercising the single-path shortcut on a deep path.
    (EDGE / "long_transaction.dat").write_text(
        " ".join(str(i) for i in range(1, 13)) + "\n")
    (EDGE / "empty.dat").write_text("")

    for name, min_count in CASES.items():
        transactions = [
            frozenset(int(x) for x in line.split())
            for line in (EDGE / f"{name}.dat").read_text().splitlines()
            if line.split()
        ]
        lines = [f"{' '.join(map(str, combo))} ({c})\n"
                 for combo, c in brute_force(transactions, min_count)]
        (EDGE / f"{name}.expected").write_text("".join(lines))
        print(f"{name}: {len(lines)} itemsets")


if __name__ == "__main__":
    main()
