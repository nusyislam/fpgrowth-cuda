#!/usr/bin/env python3
"""Runtime of the fpgrowth binary vs mlxtend's fpgrowth on the 13 settings that
were cross-validated in tests/compare_mlxtend.py.  Writes results/vs_mlxtend.csv.

Binary: REPS capped runs (--unsorted, output to /dev/null), median total_s,
which includes reading the file, building the tree, mining and formatting
every itemset.  mlxtend: a separate worker process (same ~4 GB address-space
limit, 300 s timeout) one-hot encodes the transactions once (timed
separately) and then times REPS calls of mlxtend.frequent_patterns.fpgrowth
on the encoded DataFrame; the median call time is recorded.  So the speedup
column (mlxtend_s / binary_total_s) does not charge mlxtend for input
parsing or encoding.  A worker that is killed is recorded as
"skipped_mlxtend_limit".  Itemset counts of both are checked to agree.

Usage: vs_mlxtend.py
"""
import argparse
import json
import resource
import statistics
import subprocess
import sys
import time

from benchlib import (RESULTS, ROOT, CsvAppender, check_memory, dat_path, limit_memory,
                      read_rows, run_binary)

SETTINGS = [
    ("chess", 0.9), ("chess", 0.8), ("chess", 0.7), ("chess", 0.6),
    ("mushroom", 0.3), ("mushroom", 0.2), ("mushroom", 0.1),
    ("connect4", 0.95), ("connect4", 0.9), ("connect4", 0.85),
    ("online_retail", 0.02), ("online_retail", 0.01), ("online_retail", 0.005),
]
REPS = 3
MLXTEND_TIMEOUT_S = 300

CSV = RESULTS / "vs_mlxtend.csv"
FIELDS = ["dataset", "min_support", "status", "itemsets", "binary_total_s", "binary_mine_s",
          "binary_peak_rss_kb", "mlxtend_s", "mlxtend_encode_s", "mlxtend_peak_rss_kb",
          "speedup"]


def worker(dat, support, reps):
    """Runs inside the capped child: prints one JSON line of timings."""
    sys.path.insert(0, str(ROOT / "tests"))
    from minsupport import parse_min_support
    import pandas as pd
    from mlxtend.frequent_patterns import fpgrowth
    from mlxtend.preprocessing import TransactionEncoder

    transactions = [sorted(set(map(int, l.split()))) for l in open(dat) if l.strip()]
    n = len(transactions)
    threshold = parse_min_support(support, n)
    t0 = time.perf_counter()
    te = TransactionEncoder()
    df = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)
    encode_s = time.perf_counter() - t0
    times, count = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        # Same threshold convention as tests/compare_mlxtend.py.
        res = fpgrowth(df, min_support=(threshold - 0.5) / n)
        times.append(time.perf_counter() - t0)
        count = len(res)
        del res
    print(json.dumps({"times": times, "encode_s": encode_s, "itemsets": count,
                      "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))


def run_mlxtend(dat, support):
    cmd = [sys.executable, __file__, "--worker", str(dat), str(support)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=MLXTEND_TIMEOUT_S,
                              preexec_fn=limit_memory())
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode < 0 or "MemoryError" in proc.stderr:   # killed / hit the memory cap
        return None
    if proc.returncode != 0:
        sys.exit(f"mlxtend worker failed (exit {proc.returncode}):\n{proc.stderr.strip()[-1000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worker", nargs=2, metavar=("DAT", "SUPPORT"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        worker(args.worker[0], args.worker[1], REPS)
        return

    done = {(r["dataset"], float(r["min_support"])) for r in read_rows(CSV)}
    out = CsvAppender(CSV, FIELDS)
    try:
        for name, s in SETTINGS:
            if (name, s) in done:
                print(f"{name} {s}: in CSV, skipping")
                continue
            check_memory(f"{name} {s}")
            runs = []
            for _ in range(REPS):
                status, r = run_binary(dat_path(name), s)
                if status != "ok":
                    sys.exit(f"{name} {s}: binary run failed ({status})")
                runs.append(r)
            row = {"dataset": name, "min_support": s, "itemsets": runs[0]["itemsets"],
                   "binary_total_s": f"{statistics.median(r['total_s'] for r in runs):.6f}",
                   "binary_mine_s": f"{statistics.median(r['mine_s'] for r in runs):.6f}",
                   "binary_peak_rss_kb": statistics.median(r["peak_rss_kb"] for r in runs)}
            m = run_mlxtend(dat_path(name), s)
            if m is None:
                row["status"] = "skipped_mlxtend_limit"
                print(f"{name} {s}: mlxtend killed / over limits, skipped")
            else:
                if m["itemsets"] != runs[0]["itemsets"]:
                    sys.exit(f"{name} {s}: itemset counts differ "
                             f"(binary {runs[0]['itemsets']}, mlxtend {m['itemsets']})")
                ml = statistics.median(m["times"])
                row.update(status="ok", mlxtend_s=f"{ml:.6f}",
                           mlxtend_encode_s=f"{m['encode_s']:.6f}",
                           mlxtend_peak_rss_kb=m["peak_rss_kb"],
                           speedup=f"{ml / float(row['binary_total_s']):.2f}")
                print(f"{name} {s}: {row['itemsets']} itemsets, binary {row['binary_total_s']} s, "
                      f"mlxtend {ml:.4f} s -> {row['speedup']}x", flush=True)
            out.write(row)
    finally:
        out.close()


if __name__ == "__main__":
    main()
