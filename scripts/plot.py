#!/usr/bin/env python3
"""Make the report figures from results/*.csv into results/figures/ (PDF + PNG).

  runtime_vs_support      total runtime vs min_support, log y, one panel per dataset
  itemsets_vs_support     number of frequent itemsets vs min_support, log y
  memory_vs_support       peak RSS vs min_support
  compression_vs_support  FP-tree nodes / frequent-item occurrences vs min_support
  phase_breakdown         share of read / build / mine time at one support per dataset
  itemset_lengths         distribution of itemset lengths, 3 supports per dataset
  scaling                 runtime vs % of transactions (Connect-4, Online Retail)
  vs_mlxtend              binary vs mlxtend runtime, grouped bars with speedups

and for every machine with results/parallel_<host>.csv (run_parallel_bench.py):
  omp_speedup_<host>      OpenMP end-to-end speedup over its own 1-thread run vs threads
  impl_compare_<host>     serial vs OpenMP vs CUDA runtime per support level, CUDA
                          split into GPU start-up / transfers / kernels / CPU work
  gpu_threshold_<host>    mining time vs fpgrowth_cuda --gpu-min-nodes (1 thread)
  kernel_micro_<host>     initial-tree conditional-support counting, CPU vs GPU
                          kernel (kernel_micro.py; machines with that CSV only)

Print-friendly: each dataset has one fixed color *and* marker, phases and
implementations use gray tones plus hatching, so every figure reads in
grayscale.  Datasets never share an axis, so color is never the only cue.

Usage: plot.py
"""
import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

from benchlib import DATASETS, RESULTS  # noqa: E402

FIG = RESULTS / "figures"

LABEL = {"chess": "Chess", "mushroom": "Mushroom", "connect4": "Connect-4",
         "online_retail": "Online Retail"}
COLOR = {"chess": "#2a78d6", "mushroom": "#eb6834", "connect4": "#1baf7a",
         "online_retail": "#c98500"}
MARKER = {"chess": "o", "mushroom": "s", "connect4": "^", "online_retail": "D"}
# Supports below ~2% read better on a log x-axis.
LOG_X = {"mushroom", "online_retail"}

INK, MUTED, GRID = "#1f1f1e", "#62615c", "#e4e3df"
# Phases / implementations: gray tones + hatching (no clash with dataset hues).
PHASES = [("read_s", "read input", "#d9d8d3", ""),
          ("build_s", "build FP-tree", "#9a9993", "////"),
          ("mine_s", "mine (+ write itemsets)", "#4a4945", "")]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9,
    "axes.titleweight": "bold", "axes.labelsize": 8.5, "axes.labelcolor": INK,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "legend.frameon": False, "lines.linewidth": 1.6, "lines.markersize": 4.5,
    "hatch.linewidth": 0.6, "hatch.color": "#ffffff",
    "pdf.fonttype": 42, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def read(name):
    with (RESULTS / name).open(newline="") as f:
        return list(csv.DictReader(f))


def num(row, key):
    return float(row[key])


def pct(x, _pos=None):
    """0.005 -> '0.5%', 0.25 -> '25%'."""
    v = 100 * x
    return f"{v:.2g}%" if v < 1 else f"{v:.3g}%"


def save(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{stem}.{ext}")
    plt.close(fig)
    print(f"wrote results/figures/{stem}.pdf / .png")


def bench():
    by = defaultdict(list)
    for r in read("bench.csv"):
        by[r["dataset"]].append(r)
    for rows in by.values():
        rows.sort(key=lambda r: -num(r, "min_support"))
    return by


def support_axis(ax, name, supports, keep=6):
    """x = min_support, decreasing to the right (harder problems to the right).
    Every measured level gets a tick; about `keep` of them (always the first
    and last) get a label."""
    if name in LOG_X:
        ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    step = max(1, -(-(len(supports) - 1) // (keep - 1)))
    shown = set(supports[::step])
    if supports[-1] not in shown:
        shown.discard(supports[-2])   # no two adjacent labels at the crowded end
        shown.add(supports[-1])
    ax.set_xticks(supports, [pct(x) if x in shown else "" for x in supports])
    pad = 1.12 if name in LOG_X else None
    if pad:
        ax.set_xlim(max(supports) * pad, min(supports) / pad)
    else:
        span = max(supports) - min(supports)
        ax.set_xlim(max(supports) + 0.04 * span, min(supports) - 0.04 * span)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


def per_dataset(stem, ylabel, value, logy=False, ylim0=False, note_stop=True):
    """2x2 panels, one per dataset, of value(row) vs min_support (ok rows only)."""
    by = bench()
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.8))
    for ax, name in zip(axes.flat, DATASETS):
        ok = [r for r in by[name] if r["status"] == "ok"]
        xs = [num(r, "min_support") for r in ok]
        ys = [value(r) for r in ok]
        ax.plot(xs, ys, color=COLOR[name], marker=MARKER[name],
                markeredgecolor="white", markeredgewidth=0.6)
        if logy:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
        if ylim0:
            ax.set_ylim(0, max(ys) * 1.18)
        elif logy:
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi * 4)   # room for the stop note
        support_axis(ax, name, xs)
        stop = [r for r in by[name] if r["status"] != "ok"]
        if note_stop:
            if stop:
                st = stop[0]["status"]
                why = (f"{num(stop[0], 'total_s'):.0f} s, over 60 s budget"
                       if st == "over_time_budget" else
                       {"timeout": "killed at 120 s timeout",
                        "memory_limit": "hit 4 GB limit"}.get(st, st))
                text = f"stopped at {pct(num(stop[0], 'min_support'))}\n({why})"
            else:
                text = "not stopped"
            ax.text(0.03, 0.96, text, transform=ax.transAxes, va="top", ha="left",
                    fontsize=7, color=MUTED)
        ax.set_title(LABEL[name], color=INK, loc="left")
        ax.set_xlabel("minimum support")
        ax.set_ylabel(ylabel)
    fig.tight_layout(h_pad=1.6, w_pad=1.6)
    save(fig, stem)


def fig_phase_breakdown():
    # Representative supports: mid-range, total time ~0.1-1 s.
    rep = {"chess": 0.5, "mushroom": 0.1, "connect4": 0.7, "online_retail": 0.005}
    by = bench()
    rows = [next(r for r in by[n] if abs(num(r, "min_support") - rep[n]) < 1e-12)
            for n in DATASETS]
    fig, ax = plt.subplots(figsize=(6.5, 2.3))
    ys = list(range(len(rows)))[::-1]
    for y, r in zip(ys, rows):
        parts = [num(r, k) for k, *_ in PHASES]
        total = sum(parts)
        left = 0.0
        for (k, label, color, hatch), v in zip(PHASES, parts):
            share = 100 * v / total
            ax.barh(y, share, left=left, height=0.62, color=color, hatch=hatch,
                    edgecolor="white", linewidth=1.2, label=label if y == ys[0] else None)
            if share >= 7:
                ax.text(left + share / 2, y, f"{share:.0f}%", ha="center", va="center",
                        fontsize=7, color="white" if color == PHASES[2][2] else INK,
                        bbox=None if color != PHASES[1][2] else
                        dict(boxstyle="round,pad=0.15", fc=PHASES[1][2], ec="none"))
            left += share
        ax.text(101.5, y, f"{total * 1000:.0f} ms total" if total < 1 else f"{total:.2f} s total",
                va="center", ha="left", fontsize=7, color=MUTED)
    ax.set_yticks(ys, [f"{LABEL[r['dataset']]} @ {pct(num(r, 'min_support'))}" for r in rows])
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of total runtime (%)")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1.4)
    fig.tight_layout()
    save(fig, "phase_breakdown")


def fig_itemset_lengths():
    by = defaultdict(lambda: defaultdict(list))
    for r in read("lengths.csv"):
        by[r["dataset"]][float(r["min_support"])].append((int(r["length"]), float(r["fraction"])))
    styles = [("-", 1.0), ("--", 0.8), (":", 0.65)]
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.6))
    for ax, name in zip(axes.flat, DATASETS):
        for (s, pts), (ls, alpha) in zip(sorted(by[name].items(), reverse=True), styles):
            pts.sort()
            ax.plot([p[0] for p in pts], [100 * p[1] for p in pts], ls=ls, color=COLOR[name],
                    alpha=alpha, marker=MARKER[name], markersize=3.5, markeredgecolor="white",
                    markeredgewidth=0.5, label=f"min_sup {pct(s)}")
        ax.set_title(LABEL[name], color=INK, loc="left")
        ax.set_xlabel("itemset length (items)")
        ax.set_ylabel("share of frequent itemsets (%)")
        ax.set_ylim(bottom=0)
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.legend(loc="upper right", handlelength=2.6)
    fig.tight_layout(h_pad=1.6, w_pad=1.6)
    save(fig, "itemset_lengths")


def fig_scaling():
    by = defaultdict(list)
    for r in read("scaling.csv"):
        if r["status"] == "ok":
            by[r["dataset"]].append(r)
    names = [n for n in DATASETS if n in by]
    fig, axes = plt.subplots(1, len(names), figsize=(6.5, 2.6))
    for ax, name in zip(axes, names):
        rows = sorted(by[name], key=lambda r: num(r, "fraction"))
        xs = [100 * num(r, "fraction") for r in rows]
        ax.plot(xs, [num(r, "total_s") for r in rows], color=COLOR[name], marker=MARKER[name],
                markeredgecolor="white", markeredgewidth=0.6, label="total")
        ax.plot(xs, [num(r, "mine_s") for r in rows], color=PHASES[2][2], ls="--", marker="x",
                markersize=4, lw=1.2, label="mine (+ write)")
        ax.plot(xs, [num(r, "read_s") + num(r, "build_s") for r in rows], color=PHASES[1][2],
                ls=":", marker="+", markersize=5, lw=1.4, label="read + build")
        ax.set_ylim(bottom=0)
        ax.set_xticks(xs)
        ax.set_xlabel("transactions used (% of dataset)")
        ax.set_ylabel("runtime (s)")
        n = int(num(rows[-1], "transactions"))
        ax.set_title(f"{LABEL[name]}  (min_sup {pct(num(rows[0], 'min_support'))}, "
                     f"N = {n:,})", color=INK, loc="left")
        ax.set_ylim(0, max(num(r, "total_s") for r in rows) * 1.6)
        ax.legend(loc="upper left", handlelength=2.4)
    fig.tight_layout(w_pad=2.0)
    save(fig, "scaling")


def fig_vs_mlxtend():
    rows = [r for r in read("vs_mlxtend.csv")]
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    x, ticks, labels, prev, groups = 0.0, [], [], None, {}
    w = 0.38
    for r in rows:
        if prev is not None and r["dataset"] != prev:
            x += 0.6                    # gap between datasets
        prev = r["dataset"]
        mine = num(r, "binary_total_s")
        ax.bar(x - w / 2, mine, w, color=PHASES[2][2], edgecolor="white", linewidth=0.8,
               label="this FP-Growth (C++)" if not ticks else None)
        if r["status"] == "ok":
            theirs = num(r, "mlxtend_s")
            ax.bar(x + w / 2, theirs, w, color=PHASES[0][2], hatch="////", edgecolor=MUTED,
                   linewidth=0.5, label="mlxtend fpgrowth" if not ticks else None)
            ax.text(x, theirs * 1.12, f"{num(r, 'speedup'):.0f}×", ha="center", va="bottom",
                    fontsize=6.5, color=INK)
        else:
            ax.text(x + w / 2, mine, "skipped", rotation=90, fontsize=6, color=MUTED)
        ticks.append(x)
        labels.append(pct(num(r, "min_support")))
        groups.setdefault(r["dataset"], []).append(x)
        x += 1.0
    ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * 2.5)
    ax.set_xticks(ticks, labels, fontsize=7)
    for name, xs in groups.items():
        ax.annotate(LABEL[name], xy=(sum(xs) / len(xs), 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -24), textcoords="offset points", ha="center", va="top",
                    fontsize=8, color=INK, fontweight="bold")
    ax.set_xlabel("minimum support (grouped by dataset)", labelpad=30)
    ax.set_ylabel("runtime (s, log scale)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("Runtime vs mlxtend (labels: speedup = mlxtend / this implementation)",
                 color=INK, loc="left")
    fig.tight_layout()
    save(fig, "vs_mlxtend")


# ---------------------------------------------------------------------------
# Parallel versions (one set of figures per machine)
# ---------------------------------------------------------------------------

# Parts of a run, bottom to top.  Serial and OpenMP bars are all "CPU".
RUN_PARTS = [("cpu", "CPU: read, build, mine, write", "#4a4945", ""),
             ("init", "GPU start-up (CUDA context)", "#d9d8d3", ""),
             ("xfer", "host<->device transfers", "#9a9993", "////"),
             ("kernel", "GPU kernels", "#c4c3bd", "xxxx")]
IMPLS = [("serial", "S"), ("omp_tall", "O"), ("cuda_t1", "C1"), ("cuda_tall", "C")]
LEVEL_STYLES = ["-", "--", "-.", ":"]


def run_parts(r):
    """Seconds per RUN_PARTS entry for one row (GPU parts are 0 for CPU builds)."""
    def g(k):
        return float(r[k]) if r.get(k) not in (None, "") else 0.0
    init = g("gpu_init_s")
    xfer = g("count_h2d_s") + g("count_d2h_s") + g("tree_h2d_s") + g("tree_d2h_s")
    kernel = g("count_kernel_s") + g("tree_kernel_s")
    cpu = max(0.0, g("total_s") - init - xfer - kernel)
    return {"cpu": cpu, "init": init, "xfer": xfer, "kernel": kernel}


def fmt_s(x):
    return f"{x * 1000:.0f} ms" if x < 1 else f"{x:.1f} s"


def parallel_hosts():
    return sorted(p.stem[len("parallel_"):] for p in RESULTS.glob("parallel_*.csv"))


def parallel_rows(host):
    by = defaultdict(dict)        # (dataset, support) -> config -> row
    for r in read(f"parallel_{host}.csv"):
        by[(r["dataset"], float(r["min_support"]))][r["config"]] = r
    return by


def panels(names, figsize):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    return fig, axes


def fig_omp_speedup(host):
    by = parallel_rows(host)
    names = [n for n in DATASETS if any(k[0] == n for k in by)]
    fig, axes = panels(names, (6.5, 4.9))
    for ax, name in zip(axes.flat, names):
        levels = sorted((s for (n, s) in by if n == name), reverse=True)
        ticks = set()
        for s, ls in zip(levels, LEVEL_STYLES):
            # Baseline: the same binary on 1 thread.  (Its --unsorted output goes
            # through a larger buffer than the serial binary's, which alone makes
            # it up to ~10% faster on the dense datasets; see GPU_DESIGN.md.)
            cfg = by[(name, s)]
            if cfg.get("omp_t1", {}).get("status") != "ok":
                continue
            base = num(cfg["omp_t1"], "total_s")
            pts = sorted((int(r["threads"]), base / num(r, "total_s")) for c, r in cfg.items()
                         if c.startswith("omp_") and r["status"] == "ok")
            ticks.update(p[0] for p in pts)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], ls=ls, color=COLOR[name],
                    marker=MARKER[name], markeredgecolor="white", markeredgewidth=0.6,
                    label=f"{pct(s)}  (1 thread {fmt_s(base)})")
        ticks = sorted(ticks)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        top = max(ln.get_ydata().max() for ln in ax.get_lines()) * 2.4   # room for the legend
        ax.plot([1, ticks[-1]], [1, ticks[-1]], color=MUTED, lw=0.8, ls=(0, (1, 2)),
                label="ideal (linear)")
        ax.set_ylim(0, top)            # the ideal line leaves the panel at the top
        ax.set_title(LABEL[name], color=INK, loc="left")
        ax.set_xlabel("OpenMP threads")
        ax.set_ylabel("speedup over 1 thread (total time)")
        ax.legend(loc="upper left", title="min support", title_fontsize=7, handlelength=2.6,
                  frameon=True, facecolor="white", edgecolor="none", framealpha=0.92)
    fig.suptitle(f"OpenMP speedup on {host}", x=0.01, ha="left", fontsize=9,
                 fontweight="bold", color=INK)
    fig.tight_layout(h_pad=1.6, w_pad=1.6)
    save(fig, f"omp_speedup_{host}")


def fig_impl_compare(host):
    by = parallel_rows(host)
    names = [n for n in DATASETS if any(k[0] == n for k in by)]
    # One full-width row per dataset: 4 bars x up to 4 levels need the width.
    fig, axes = plt.subplots(len(names), 1, figsize=(6.5, 1.9 * len(names) + 0.9), squeeze=False)
    w = 0.8
    for ax, name in zip(axes.flat, names):
        levels = sorted((s for (n, s) in by if n == name), reverse=True)
        x, ticks, labels, ymax = 0.0, [], [], 1.0
        for s in levels:
            cfg = by[(name, s)]
            if cfg.get("serial", {}).get("status") != "ok":
                continue
            base = num(cfg["serial"], "total_s")
            xs = []
            for key, label in IMPLS:
                r = cfg.get(key)
                if r is None:
                    continue
                if r["status"] != "ok":
                    ax.text(x, 0.05, r["status"], rotation=90, fontsize=6, color=MUTED,
                            ha="center", va="bottom")
                else:
                    parts = run_parts(r)
                    bottom = 0.0
                    for k, plabel, color, hatch in RUN_PARTS:
                        h = parts[k] / base
                        if h > 0:
                            ax.bar(x, h, w, bottom=bottom, color=color, hatch=hatch,
                                   edgecolor="white", linewidth=0.8, label=plabel)
                        bottom += h
                    ymax = max(ymax, bottom)
                    if key != "serial":
                        ax.text(x, bottom + 0.02 * ymax, f"{base / num(r, 'total_s'):.1f}×",
                                ha="center", va="bottom", fontsize=6, color=INK)
                ticks.append(x)
                labels.append(label)
                xs.append(x)
                x += 1.0
            ax.annotate(f"min sup {pct(s)}\nserial {fmt_s(base)}",
                        xy=(sum(xs) / len(xs), 0), xycoords=("data", "axes fraction"),
                        xytext=(0, -14), textcoords="offset points", ha="center", va="top",
                        fontsize=7, color=INK)
            x += 0.9
        ax.set_xticks(ticks, labels, fontsize=6.5)
        ax.tick_params(axis="x", length=0, pad=2)
        ax.axhline(1.0, color=MUTED, lw=0.6, ls=(0, (1, 2)))
        ax.set_ylim(0, ymax * 1.22)
        ax.grid(axis="x", visible=False)
        ax.set_title(LABEL[name], color=INK, loc="left")
        ax.set_ylabel("time / serial")
    handles = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    order = [p[1] for p in RUN_PARTS if p[1] in handles]
    fig.suptitle(f"Serial vs OpenMP vs CUDA on {host}: runtime relative to serial\n"
                 "S serial, O OpenMP, C1 CUDA with 1 mining thread, C CUDA "
                 "(O and C: all threads); labels: speedup over serial",
                 x=0.01, y=0.998, ha="left", fontsize=8, fontweight="bold", color=INK)
    fig.legend([handles[o] for o in order], order, loc="upper left", ncol=4,
               bbox_to_anchor=(0.01, 1 - 0.42 / fig.get_figheight()), fontsize=7)
    fig.tight_layout(h_pad=2.6, rect=(0, 0, 1, 1 - 0.62 / fig.get_figheight()))
    save(fig, f"impl_compare_{host}")


def fig_gpu_threshold(host):
    path = RESULTS / f"gpu_threshold_{host}.csv"
    if not path.exists():
        return
    by = defaultdict(list)
    for r in read(path.name):
        if r["status"] == "ok":
            by[(r["dataset"], float(r["min_support"]))].append(r)
    if not by:
        return
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    xs_all = set()
    for i, ((name, s), rows) in enumerate(sorted(by.items())):
        rows.sort(key=lambda r: int(r["gpu_min_nodes"]))
        cpu_only = num(rows[-1], "mine_s")       # largest threshold: initial tree only
        pts = [(int(r["gpu_min_nodes"]), num(r, "mine_s") / cpu_only, int(num(r, "gpu_trees")))
               for r in rows[:-1]]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLOR[name],
                marker=MARKER[name], markeredgecolor="white", markeredgewidth=0.6,
                label=f"{LABEL[name]} @ {pct(s)}")
        xs_all.update(p[0] for p in pts)
        above = i % 2 == 0          # alternate label sides so the two lines' labels never meet
        for xv, yv, n in pts:
            ax.annotate(f"{n:,}", (xv, yv), xytext=(0, 5 if above else -6),
                        textcoords="offset points", ha="center",
                        va="bottom" if above else "top", fontsize=6, color=MUTED)
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (1, 2)), label="conditional trees all on CPU")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(xs_all), [f"{x:,}" for x in sorted(xs_all)])
    ax.margins(y=0.15)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("--gpu-min-nodes (conditional trees at least this large go to the GPU)")
    ax.set_ylabel("mining time, relative")
    ax.set_title(f"GPU offload threshold on {host}, 1 mining thread "
                 "(point labels: trees counted on the GPU)", color=INK, loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, f"gpu_threshold_{host}")


def node_count(n):
    return f"{n / 1000:.3g}k" if n >= 1000 else f"{n:.0f}"


def fig_kernel_micro(host):
    path = RESULTS / f"kernel_micro_{host}.csv"
    if not path.exists():
        return
    rows = [r for r in read(path.name) if r["status"] == "ok"]
    if not rows:
        return
    # Implementations in gray tones + hatching, as elsewhere (hues mean datasets).
    bars = [("cpu_s", "CPU, 1 thread", PHASES[2][2], ""),
            ("gpu_kernel_s", "GPU kernel", "#9a9993", ""),
            ("gpu_kernel_xfer_s", "GPU kernel + transfers", "#c4c3bd", "////"),
            ("gpu_wall_s", "GPU incl. flatten tree + unpack (CPU side)", "#e4e3df", "xxxx")]
    fig, ax = plt.subplots(figsize=(6.5, 3.1))
    w = 0.2
    for i, r in enumerate(rows):
        cpu = num(r, "cpu_s")
        for j, (k, label, color, hatch) in enumerate(bars):
            x = i + (j - 1.5) * w
            v = num(r, k) * 1e3
            ax.bar(x, v, w * 0.92, color=color, hatch=hatch,
                   edgecolor="white" if j < 2 else MUTED, linewidth=0.6 if j < 2 else 0.4,
                   label=label if i == 0 else None)
            if j > 0:
                ax.text(x, v * 1.12, f"{cpu / num(r, k):.3g}×", ha="center", va="bottom",
                        fontsize=5.5, color=INK, rotation=90)
    ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * 6)
    ax.set_xticks(range(len(rows)),
                  [f"{LABEL[r['dataset']]}\n{pct(num(r, 'min_support'))}\n"
                   f"{node_count(num(r, 'tree_nodes'))} nodes" for r in rows], fontsize=6.5)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("time (ms, log scale)")
    ax.set_title(f"Conditional supports of the initial FP-tree on {host}, CUDA context excluded "
                 "(labels: speedup over CPU)", color=INK, loc="left", fontsize=8)
    ax.legend(loc="upper left", ncol=2, fontsize=7)
    fig.tight_layout()
    save(fig, f"kernel_micro_{host}")


def main():
    per_dataset("runtime_vs_support", "total runtime (s, log scale)",
                lambda r: num(r, "total_s"), logy=True)
    per_dataset("itemsets_vs_support", "frequent itemsets (log scale)",
                lambda r: num(r, "itemsets"), logy=True)
    per_dataset("memory_vs_support", "peak RSS (MB)",
                lambda r: num(r, "peak_rss_kb") / 1024, ylim0=True, note_stop=False)
    per_dataset("compression_vs_support", "tree nodes / item occurrences",
                lambda r: num(r, "tree_nodes") / num(r, "filtered_occurrences"),
                ylim0=True, note_stop=False)
    fig_phase_breakdown()
    fig_itemset_lengths()
    fig_scaling()
    fig_vs_mlxtend()
    for host in parallel_hosts():
        fig_omp_speedup(host)
        fig_impl_compare(host)
        fig_gpu_threshold(host)
        fig_kernel_micro(host)


if __name__ == "__main__":
    main()
