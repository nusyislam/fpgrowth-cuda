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


if __name__ == "__main__":
    main()
