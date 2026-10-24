#!/usr/bin/env python3
"""Randomized differential test for the fpgrowth binary.

Each trial generates a small random FIMI-style dataset (5-30 transactions,
4-10 distinct items, 1-8 items per line, with duplicate items inside a line,
blank/whitespace-only lines and varying whitespace), computes every frequent
itemset with its exact support by brute-force subset enumeration, then runs
the binary at several min_support levels (absolute and fractional, always
including 1) and compares itemsets and supports exactly.

On a mismatch the dataset and an expected-vs-actual diff are saved under
tests/failures/ so the case can be replayed.

Usage: brute_force.py [--trials N] [--seed S] [--binary PATH]
"""
import argparse
import itertools
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from minsupport import parse_min_support

TESTS_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Dataset generation and brute-force reference
# ---------------------------------------------------------------------------

def gen_dataset(rng):
    """Return (file text, list of transactions as frozensets)."""
    n_tx = rng.randint(5, 30)
    pool = rng.sample(range(0, 80), rng.randint(4, 10))
    lines, transactions = [], []
    for _ in range(n_tx):
        if rng.random() < 0.15:
            lines.append(rng.choice(["", "   ", "\t"]))  # must be skipped by the reader
        items = [rng.choice(pool) for _ in range(rng.randint(1, 8))]  # duplicates allowed
        sep = "  " if rng.random() < 0.1 else " "
        lines.append(sep.join(str(x) for x in items))
        transactions.append(frozenset(items))
    text = "\n".join(lines)
    if rng.random() < 0.9:  # occasionally no final newline
        text += "\n"
    return text, transactions


def brute_force(transactions):
    """Exact support of every itemset that occurs in the data."""
    all_items = sorted(set().union(*transactions))
    counts = {}
    for size in range(1, len(all_items) + 1):
        for combo in itertools.combinations(all_items, size):
            s = frozenset(combo)
            c = sum(1 for t in transactions if s <= t)
            if c > 0:
                counts[combo] = c
    return counts


def support_args(rng, n_tx, counts):
    """min_support arguments to test, absolute and fractional, incl. 1."""
    max_sup = max(counts.values())
    levels = ["1", "2", str(rng.randint(1, n_tx)), str(max_sup), str(max_sup + 1)]
    # A fraction that encodes an exact count k/N, and a short "as typed" one.
    levels.append(f"{rng.randint(1, n_tx) / n_tx:.9f}")
    levels.append(f"{rng.uniform(0.05, 1.0):.{rng.randint(1, 3)}f}")
    out, seen = [], set()
    for s in levels:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Running and comparing
# ---------------------------------------------------------------------------

def parse_output(stdout):
    """'1 2 5 (2)' lines -> {(1, 2, 5): 2}."""
    result = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        head, _, sup = line.rpartition(" (")
        if not head or not sup.endswith(")"):
            raise ValueError(f"unparseable output line: {line!r}")
        items = tuple(int(x) for x in head.split())
        if items in result:
            raise ValueError(f"duplicate itemset in output: {line!r}")
        result[items] = int(sup[:-1])
    return result


def fmt_itemsets(m):
    return "".join(f"{' '.join(map(str, k))} ({v})\n" for k, v in sorted(m.items()))


def save_failure(trial, sup_arg, text, expected, actual, proc, note):
    d = TESTS_DIR / "failures" / f"trial{trial:04d}_sup{sup_arg.replace('.', 'p')}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "data.dat").write_text(text)
    (d / "expected.txt").write_text(fmt_itemsets(expected))
    (d / "actual.txt").write_text(fmt_itemsets(actual) if actual is not None else "")
    diff = []
    for k in sorted(set(expected) | set(actual or {})):
        e, a = expected.get(k), (actual or {}).get(k)
        if e != a:
            diff.append(f"  {' '.join(map(str, k))}: expected {e}, got {a}\n")
    (d / "info.txt").write_text(
        f"note: {note}\nmin_support arg: {sup_arg}\nexit code: {proc.returncode}\n"
        f"replay: <binary> {d / 'data.dat'} {sup_arg}\n"
        f"stderr:\n{proc.stderr}\ndiff (expected vs actual):\n{''.join(diff)}"
    )
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--binary", default="./fpgrowth")
    args = ap.parse_args()

    binary = str(Path(args.binary).resolve())
    rng = random.Random(args.seed)
    checks = passed = failed = sanitizer = 0

    with tempfile.TemporaryDirectory(prefix="fpgrowth_bf_") as tmp:
        dat = Path(tmp) / "data.dat"
        for trial in range(1, args.trials + 1):
            text, transactions = gen_dataset(rng)
            dat.write_text(text)
            counts = brute_force(transactions)

            for sup_arg in support_args(rng, len(transactions), counts):
                checks += 1
                threshold = parse_min_support(sup_arg, len(transactions))
                expected = {k: c for k, c in counts.items() if c >= threshold}

                proc = subprocess.run([binary, str(dat), sup_arg],
                                      capture_output=True, text=True, timeout=120)
                note, actual = None, None
                if "Sanitizer" in proc.stderr or "runtime error" in proc.stderr:
                    note = "sanitizer report in stderr"
                    sanitizer += 1
                elif proc.returncode != 0:
                    note = f"non-zero exit code {proc.returncode}"
                else:
                    try:
                        actual = parse_output(proc.stdout)
                    except ValueError as e:
                        note = str(e)
                if note is None and actual != expected:
                    note = "itemsets/supports differ from brute force"

                if note is None:
                    passed += 1
                else:
                    failed += 1
                    d = save_failure(trial, sup_arg, text, expected, actual, proc, note)
                    print(f"FAIL trial {trial} min_support={sup_arg}: {note} -> {d}")

            if trial % 100 == 0:
                print(f"  ... {trial}/{args.trials} trials, {checks} runs, {failed} failures")

    print(f"brute force vs {args.binary}: {args.trials} datasets, {checks} runs, "
          f"{passed} passed, {failed} failed, {sanitizer} sanitizer reports (seed {args.seed})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
