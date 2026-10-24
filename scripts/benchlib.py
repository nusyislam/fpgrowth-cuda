"""Shared helpers for the benchmark scripts: capped binary runs, CSV I/O.

Every run of the binary goes through run_binary(), which applies a wall-clock
timeout and an address-space limit (RLIMIT_AS, the same as `ulimit -v`) so a
blow-up is recorded as a stopping point instead of taking the machine down.

The binary is launched as `prlimit --as=N -- timeout T ./fpgrowth ...` rather
than directly from Python: Linux carries ru_maxrss across exec, so a process
exec'd straight from the (much larger) Python interpreter would report
Python's RSS as its peak.  `timeout` fork()s the binary as a fresh process,
so the binary's peak_rss_kb reflects only itself.
"""
import csv
import resource
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "fpgrowth"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

MEM_LIMIT_KB = 4_000_000      # address-space cap per run (~4 GB)
TIMEOUT_S = 120               # hard kill; the sweeps stop going lower at ~60 s
MIN_AVAILABLE_KB = 6_000_000  # refuse to start a heavy run below this

DATASETS = ["chess", "mushroom", "connect4", "online_retail"]

# Phase times, then counters, as printed on the binary's stderr line.
TIME_KEYS = ["read_s", "build_s", "mine_s", "output_s", "total_s"]
COUNT_KEYS = ["peak_rss_kb", "tree_nodes", "filtered_occurrences", "frequent_items",
              "itemsets", "cond_trees", "cond_nodes", "single_paths"]


def dat_path(name):
    return PROCESSED / f"{name}.dat"


def mem_available_kb():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    return None


def check_memory(what=""):
    """Print available memory; exit if it is too low to start a heavy run."""
    avail = mem_available_kb()
    if avail is None:
        return
    print(f"  [free: {avail / 1024**2:.1f} GiB available{' before ' + what if what else ''}]",
          flush=True)
    if avail < MIN_AVAILABLE_KB:
        sys.exit(f"only {avail / 1024**2:.1f} GiB available; not starting {what}")


def limit_memory(limit_kb=MEM_LIMIT_KB):
    """preexec_fn for subprocess: cap the child's address space."""
    def apply():
        resource.setrlimit(resource.RLIMIT_AS, (limit_kb * 1024, limit_kb * 1024))
    return apply


def parse_stats(stderr):
    """Parse the binary's key=value stderr line into numbers where possible."""
    line = stderr.strip().splitlines()[-1]
    out = {}
    for kv in line.split():
        k, _, v = kv.partition("=")
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def binary_cmd(dat, support, output="/dev/null", unsorted=True, timeout=TIMEOUT_S):
    """Command line for one capped run (see the module docstring)."""
    return (["prlimit", f"--as={MEM_LIMIT_KB * 1024}", "--", "timeout", str(timeout),
             str(BINARY), str(dat), str(support)]
            + ([output] if output else []) + (["--unsorted"] if unsorted else []))


def run_binary(dat, support, output="/dev/null", unsorted=True, timeout=TIMEOUT_S):
    """One capped run.  Returns (status, stats); status is "ok", "timeout",
    "memory_limit" or "error", and stats is {} unless status is "ok"."""
    cmd = binary_cmd(dat, support, output, unsorted, timeout)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 30)
    except subprocess.TimeoutExpired:   # only if `timeout` itself failed to kill it
        return "timeout", {}
    if proc.returncode == 124:
        return "timeout", {}
    if proc.returncode != 0:
        # bad_alloc under RLIMIT_AS is caught in main() and reported as an error;
        # a kill by signal (128 + n from `timeout`) is also treated as out of memory.
        if "bad_alloc" in proc.stderr or proc.returncode > 128:
            return "memory_limit", {}
        print(f"binary failed (exit {proc.returncode}): {proc.stderr.strip()}", file=sys.stderr)
        return "error", {}
    return "ok", parse_stats(proc.stderr)


def median_stats(runs):
    """Median of every timing and counter over repeated runs."""
    return {k: statistics.median(r[k] for r in runs) for k in TIME_KEYS + COUNT_KEYS}


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


class CsvAppender:
    """Append rows to a CSV (header written on creation), flushing each row so
    an interrupted sweep keeps everything finished so far."""

    def __init__(self, path, fields):
        self.path, self.fields = path, fields
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists() or path.stat().st_size == 0
        self.f = path.open("a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fields, extrasaction="ignore")
        if new:
            self.w.writeheader()
            self.f.flush()

    def write(self, row):
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()


def fmt(x):
    """Compact CSV formatting for floats."""
    return f"{x:.6f}" if isinstance(x, float) else x
