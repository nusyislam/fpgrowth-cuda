#!/usr/bin/env python3
"""Print fpgrowth output itemsets with human-readable item names.

Reads the binary's output ("1 2 5 (2)" per line) from a file or stdin, and a
<name>.map.tsv written by preprocess.py.

Usage:
  decode.py <itemsets.txt|-> <name.map.tsv> [--top N] [--min-len K] [--max-len K]
            [--percent N_TRANSACTIONS]

Examples:
  ./fpgrowth data/processed/mushroom.dat 0.3 | scripts/decode.py - data/processed/mushroom.map.tsv --top 20 --min-len 2
"""
import argparse
import csv
import sys


def read_map(path):
    with open(path, newline="") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)  # header
        return {int(i): name for i, name in r}


def read_itemsets(f):
    for line in f:
        if not line.strip():
            continue
        head, _, sup = line.rpartition(" (")
        yield [int(x) for x in head.split()], int(sup.rstrip(")\n"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("itemsets", help="fpgrowth output file, or - for stdin")
    ap.add_argument("map", help="<name>.map.tsv from preprocess.py")
    ap.add_argument("--top", type=int, metavar="N", help="only the N itemsets with highest support")
    ap.add_argument("--min-len", type=int, default=1, metavar="K", help="only itemsets with >= K items")
    ap.add_argument("--max-len", type=int, metavar="K", help="only itemsets with <= K items")
    ap.add_argument("--percent", type=int, metavar="N_TRANSACTIONS",
                    help="also show support as a percentage of N_TRANSACTIONS")
    args = ap.parse_args()

    names = read_map(args.map)
    f = sys.stdin if args.itemsets == "-" else open(args.itemsets)
    sets = [(items, sup) for items, sup in read_itemsets(f)
            if len(items) >= args.min_len and (args.max_len is None or len(items) <= args.max_len)]
    # Highest support first; ties: shorter, then by item ids, for stable output.
    sets.sort(key=lambda s: (-s[1], len(s[0]), s[0]))
    if args.top is not None:
        sets = sets[:args.top]

    width = len(str(sets[0][1])) if sets else 1
    for items, sup in sets:
        pct = f" {100 * sup / args.percent:5.1f}%" if args.percent else ""
        readable = ", ".join(names.get(i, f"<unknown item {i}>") for i in items)
        print(f"{sup:>{width}}{pct}  {{{readable}}}")


if __name__ == "__main__":
    main()
