#!/usr/bin/env python3
"""Convert the raw UCI datasets in data/raw/ to FIMI .dat files.

For each dataset writes
  data/processed/<name>.dat      one transaction per line, space-separated item ids
  data/processed/<name>.map.tsv  item id <TAB> human-readable name

Categorical datasets (mushroom, chess, connect4): every attribute=value pair
is one item, the class label included as an attribute; missing values ("?")
are skipped.  Item ids are assigned 1.. in column order, values sorted within
each column, so the numbering is deterministic.

Online Retail: one transaction per invoice, items are StockCodes.

Usage: preprocess.py [name ...]     (default: all datasets)
"""
import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

MISSING = "?"

# --- attribute names and value legends (from the .names files) -------------

MUSHROOM_ATTRS = [
    ("class", "edible=e,poisonous=p"),
    ("cap-shape", "bell=b,conical=c,convex=x,flat=f,knobbed=k,sunken=s"),
    ("cap-surface", "fibrous=f,grooves=g,scaly=y,smooth=s"),
    ("cap-color", "brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y"),
    ("bruises", "bruises=t,no=f"),
    ("odor", "almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s"),
    ("gill-attachment", "attached=a,descending=d,free=f,notched=n"),
    ("gill-spacing", "close=c,crowded=w,distant=d"),
    ("gill-size", "broad=b,narrow=n"),
    ("gill-color", "black=k,brown=n,buff=b,chocolate=h,gray=g,green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y"),
    ("stalk-shape", "enlarging=e,tapering=t"),
    ("stalk-root", "bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r"),
    ("stalk-surface-above-ring", "fibrous=f,scaly=y,silky=k,smooth=s"),
    ("stalk-surface-below-ring", "fibrous=f,scaly=y,silky=k,smooth=s"),
    ("stalk-color-above-ring", "brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y"),
    ("stalk-color-below-ring", "brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y"),
    ("veil-type", "partial=p,universal=u"),
    ("veil-color", "brown=n,orange=o,white=w,yellow=y"),
    ("ring-number", "none=n,one=o,two=t"),
    ("ring-type", "cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z"),
    ("spore-print-color", "black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y"),
    ("population", "abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y"),
    ("habitat", "grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d"),
]

CHESS_ATTRS = (
    "bkblk bknwy bkon8 bkona bkspr bkxbq bkxcr bkxwp blxwp bxqsq cntxt dsopp "
    "dwipd hdchk katri mulch qxmsq r2ar8 reskd reskr rimmx rkxwp rxmsq simpl "
    "skach skewr skrxp spcop stlmt thrsk wkcti wkna8 wknck wkovl wkpos wtoeg"
).split() + ["class"]

CONNECT4_ATTRS = [f"{col}{row}" for col in "abcdefg" for row in range(1, 7)] + ["class"]
CONNECT4_VALUES = {"x": "x", "o": "o", "b": "blank"}

# Online Retail StockCodes that are not products: postage, carriage, fees,
# manual adjustments, discounts, samples, gift vouchers.
RETAIL_NON_PRODUCT = {"POST", "DOT", "M", "C2", "D", "S", "BANK CHARGES",
                      "AMAZONFEE", "CRUK", "PADS", "B"}
RETAIL_NON_PRODUCT_RE = re.compile(r"^GIFT_\d+_\d+$")


def legend(spec):
    """'edible=e,poisonous=p' -> {'e': 'edible', 'p': 'poisonous'}"""
    return {code: name for name, code in (kv.split("=") for kv in spec.split(","))}


# --- writers ------------------------------------------------------------------

def write_dataset(name, transactions, names):
    """transactions: lists of item ids; names: {id: readable name}."""
    OUT.mkdir(parents=True, exist_ok=True)
    dat = OUT / f"{name}.dat"
    with dat.open("w") as f:
        for t in transactions:
            f.write(" ".join(map(str, sorted(t))) + "\n")
    with (OUT / f"{name}.map.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["id", "name"])
        for i in sorted(names):
            w.writerow([i, names[i]])
    print(f"{name}: {len(transactions)} transactions, {len(names)} items -> {dat.relative_to(ROOT)}")


def categorical(name, path, attrs, decode, class_first=False):
    """Generic attribute=value encoder for comma-separated categorical data.

    attrs:  attribute names in *output* column order
    decode: (attr_index, raw_value) -> readable value
    class_first: the class label is the first CSV column (mushroom) rather
                 than the last; it is moved to the front of `attrs` either way.
    """
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append([v.strip() for v in line.split(",")])
    width = len(attrs)
    bad = [i for i, r in enumerate(rows) if len(r) != width]
    if bad:
        sys.exit(f"{path}: {len(bad)} rows without {width} fields (first: line {bad[0] + 1})")
    if not class_first:  # put the class column at index 0 like `attrs`
        rows = [[r[-1]] + r[:-1] for r in rows]
        attrs = [attrs[-1]] + attrs[:-1]

    ids, names = {}, {}
    for col in range(width):
        for v in sorted({r[col] for r in rows} - {MISSING}):
            ids[col, v] = len(ids) + 1
            names[ids[col, v]] = f"{attrs[col]}={decode(col, v)}"

    transactions = [[ids[c, v] for c, v in enumerate(r) if v != MISSING] for r in rows]
    skipped = sum(r.count(MISSING) for r in rows)
    if skipped:
        print(f"{name}: skipped {skipped} missing values")
    write_dataset(name, transactions, names)


# --- one function per dataset -----------------------------------------------------

def mushroom():
    legends = [legend(spec) for _, spec in MUSHROOM_ATTRS]
    categorical("mushroom", RAW / "agaricus-lepiota.data",
                [a for a, _ in MUSHROOM_ATTRS],
                lambda col, v: legends[col][v], class_first=True)


def chess():
    # Board features are mostly boolean (t/f), a few are l/g/n/b/w; values are
    # kept as in the file.  Class: won / nowin.
    categorical("chess", RAW / "kr-vs-kp.data", CHESS_ATTRS, lambda col, v: v)


def connect4():
    def decode(col, v):
        return v if col == 0 else CONNECT4_VALUES[v]   # col 0 = class after reorder
    categorical("connect4", RAW / "connect-4.data", CONNECT4_ATTRS, decode)


def online_retail():
    import pandas as pd

    df = pd.read_excel(RAW / "Online Retail.xlsx",
                       dtype={"InvoiceNo": str, "StockCode": str})
    n0 = len(df)
    df["InvoiceNo"] = df["InvoiceNo"].str.strip()
    df["StockCode"] = df["StockCode"].str.strip().str.upper()

    cancelled = df["InvoiceNo"].str.startswith("C")
    nonpos = df["Quantity"] <= 0
    nonprod = (df["StockCode"].isin(RETAIL_NON_PRODUCT)
               | df["StockCode"].str.match(RETAIL_NON_PRODUCT_RE))
    keep = ~cancelled & ~nonpos & ~nonprod & df["StockCode"].notna()
    print(f"online_retail: {n0} rows; dropping {cancelled.sum()} cancelled, "
          f"{(nonpos & ~cancelled).sum()} more with quantity <= 0, "
          f"{(nonprod & ~cancelled & ~nonpos).sum()} more non-product -> {keep.sum()} rows")
    df = df[keep]

    codes = sorted(df["StockCode"].unique())
    ids = {c: i + 1 for i, c in enumerate(codes)}
    # Descriptions vary slightly between rows of one code; use the most common.
    desc = {}
    for code, d in df.dropna(subset=["Description"]).groupby("StockCode")["Description"]:
        desc[code] = Counter(s.strip() for s in d).most_common(1)[0][0]
    names = {ids[c]: f"{c} {desc.get(c, '(no description)')}" for c in codes}

    transactions = [sorted({ids[c] for c in g})
                    for _, g in df.groupby("InvoiceNo", sort=True)["StockCode"]]
    write_dataset("online_retail", transactions, names)


DATASETS = {
    "mushroom": mushroom,
    "chess": chess,
    "connect4": connect4,
    "online_retail": online_retail,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", metavar="name",
                    help=f"datasets to convert: {', '.join(DATASETS)} (default all)")
    args = ap.parse_args()
    unknown = set(args.names) - set(DATASETS)
    if unknown:
        ap.error(f"unknown dataset(s): {', '.join(sorted(unknown))}")
    for name in args.names or DATASETS:
        DATASETS[name]()


if __name__ == "__main__":
    main()
