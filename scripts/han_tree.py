#!/usr/bin/env python3
"""Draw the initial FP-tree of the Han & Kamber example from
`./fpgrowth data/examples/han.dat 2 --print-tree` -> results/figures/han_tree.{png,pdf}.

The tree shape and counts come from the binary's printout.  Node-links are
drawn in the order the binary created the nodes, which is recovered by
replaying the transactions in file order over the printed tree and checked
against the counts the binary prints for every node-link chain.

Usage: han_tree.py
"""
import re
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

from benchlib import RESULTS, ROOT  # noqa: E402

DAT = ROOT / "data" / "examples" / "han.dat"
FIG = RESULTS / "figures"
INK, MUTED = "#1f1f1e", "#62615c"
# One color per item's node-link chain (dashes keep them apart in grayscale too).
LINKS = ["#2a78d6", "#eb6834", "#1baf7a", "#8e44ad", "#c98500"]
DASHES = [(0, (4, 2)), (0, (1, 1.5)), (0, (6, 2, 1, 2)), (0, (3, 1.5)), (0, (2, 2))]


def run():
    return subprocess.run([str(ROOT / "fpgrowth"), str(DAT), "2", "--print-tree"],
                          capture_output=True, text=True, check=True).stdout


def parse(out):
    """-> header [(rank, item, support, [chain counts])], nodes [{item, count, parent, kids}]"""
    lines = out.splitlines()
    header = []
    for l in lines:
        m = re.match(r"^\s+(\d+)\s+(\d+)\s+(\d+)\s+(.*)$", l)
        if m and ":" in m.group(4):
            chain = [int(c.split(":")[1]) for c in m.group(4).split("->")]
            header.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), chain))
    nodes = [{"item": None, "count": None, "parent": None, "kids": [], "depth": -1}]
    stack = [0]
    start = lines.index("(root)") + 1
    for l in lines[start:]:
        m = re.match(r"^([| ]*)[+`]-- (\d+):(\d+)$", l)
        if not m:
            break
        depth = len(m.group(1)) // 4
        del stack[depth + 1:]
        nid = len(nodes)
        nodes.append({"item": int(m.group(2)), "count": int(m.group(3)), "parent": stack[-1],
                      "kids": [], "depth": depth})
        nodes[stack[-1]]["kids"].append(nid)
        stack.append(nid)
    return header, nodes


def creation_order(header, nodes):
    """Replay han.dat over the printed tree: the order in which each node is first reached."""
    rank = {item: r for r, item, _, _ in header}
    order = []
    for line in DAT.read_text().splitlines():
        cur = 0
        for item in sorted({int(x) for x in line.split()} & rank.keys(), key=rank.get):
            cur = next(k for k in nodes[cur]["kids"] if nodes[k]["item"] == item)
            if cur not in order:
                order.append(cur)
    chains = {item: [n for n in order if nodes[n]["item"] == item] for _, item, _, _ in header}
    for _, item, _, counts in header:  # must agree with the binary's node-link printout
        assert [nodes[n]["count"] for n in chains[item]] == counts, (item, counts)
    return chains


def layout(nodes):
    """x by leaf order (children in printed order), y by depth."""
    pos, nxt = {}, [0]

    def place(n):
        if not nodes[n]["kids"]:
            x = nxt[0]
            nxt[0] += 1
        else:
            xs = [place(k) for k in nodes[n]["kids"]]
            x = (xs[0] + xs[-1]) / 2
        pos[n] = (1.5 * x, -1.25 * nodes[n]["depth"])
        return x

    place(0)
    return pos


def draw(header, nodes, chains, pos):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.set_axis_off()
    x0 = min(x for x, _ in pos.values())
    shift = 3.6  # room for the header table on the left
    P = {n: (x + shift - x0, y) for n, (x, y) in pos.items()}

    for n, nd in enumerate(nodes):
        if nd["parent"] is not None:
            (x1, y1), (x2, y2) = P[nd["parent"]], P[n]
            ax.plot([x1, x2], [y1, y2], color=INK, lw=1.0, zorder=1)
    for n, nd in enumerate(nodes):
        label = "null" if n == 0 else f"I{nd['item']}:{nd['count']}"
        ax.text(*P[n], label, ha="center", va="center", fontsize=8.5, color=INK, zorder=3,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=INK, lw=0.9))

    # Header table
    top = 0.6
    ax.text(0.15, top, "item", fontsize=8, weight="bold", ha="left", va="center")
    ax.text(1.15, top, "support", fontsize=8, weight="bold", ha="left", va="center")
    rows = {}
    for i, (_, item, sup, _) in enumerate(header):
        y = top - 0.7 * (i + 1)
        rows[item] = (2.15, y)
        ax.text(0.25, y, f"I{item}", fontsize=8.5, ha="left", va="center")
        ax.text(1.35, y, str(sup), fontsize=8.5, ha="left", va="center")

    # Node-links: header entry -> first node -> next node ... (creation order)
    for i, (item, chain) in enumerate(chains.items()):
        pts = [rows[item]] + [P[n] for n in chain]
        for k, (a, b) in enumerate(zip(pts, pts[1:])):
            rad = 0.12 if k == 0 else (-0.35 if b[0] >= a[0] else 0.35)
            ax.add_patch(FancyArrowPatch(a, b, connectionstyle=f"arc3,rad={rad}",
                                         arrowstyle="-|>", mutation_scale=8, color=LINKS[i],
                                         lw=1.0, ls=DASHES[i], shrinkA=4, shrinkB=13,
                                         zorder=2))
    xs = [x for x, _ in P.values()]
    ys = [y for _, y in P.values()]
    ax.set_xlim(-0.1, max(xs) + 0.6)
    ax.set_ylim(min(min(ys), top - 0.7 * len(header)) - 0.5, top + 0.4)
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"han_tree.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    header, nodes = parse(run())
    chains = creation_order(header, nodes)
    draw(header, nodes, chains, layout(nodes))
    print(f"wrote {FIG / 'han_tree.png'}: {len(nodes) - 1} nodes, {len(header)} header items")


if __name__ == "__main__":
    main()
