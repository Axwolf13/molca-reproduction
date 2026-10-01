#!/usr/bin/env python3
"""The SMILES recency curve at three distances, all on one machine.

distance_test.py measured two points on the cluster: corrupting the SMILES costs
23.9% of the clean baseline when the SMILES is adjacent, 21.3% when filler pushes
it 37 Galactica tokens away. Two points show that recency exists. This script
adds a middle point (13 tokens) and runs the whole curve on the laptop, so no
point in it carries a cross-machine difference.

Costs are a share of each distance's own clean baseline, as in section 12, and
every contrast is paired: all six conditions saw the same 3300 molecules in the
same order.

Usage:  python distance_curve_local.py [--resamples 2000] [--seed 0] [--out results]
Writes <out>/distance_curve_local.{txt,json}.
"""
import argparse
import json
import os
import random
import sys

from bootstrap import BERT, bleu2
from distance_test import ci, load

HERE = os.path.dirname(os.path.abspath(__file__))

# filler units between the SMILES span and the soft prompts -> Galactica tokens
DISTANCES = [(0, 0), (2, 13), (6, 37)]
FILES = {
    (0, "clean"): "results/predictions/baseline_full.jsonl",
    (0, "shuf"): "results/predictions/shuffle_smiles_full.jsonl",
    (2, "clean"): "results/predictions/filler_mid2_full.jsonl",
    (2, "shuf"): "results/predictions/filler_mid2_shufsmiles_full.jsonl",
    (6, "clean"): "results/predictions/filler_mid6_full.jsonl",
    (6, "shuf"): "results/predictions/filler_mid6_shufsmiles_full.jsonl",
}
CLUSTER = {0: 23.9, 6: 21.3}   # section 12, same measure on the cluster


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    missing = [p for p in FILES.values() if not os.path.exists(os.path.join(HERE, p))]
    if missing:
        sys.exit("missing:\n  " + "\n  ".join(missing))

    from transformers import BertTokenizer
    tok = BertTokenizer.from_pretrained(BERT)
    cache, data, n = {}, {}, None
    for key, path in FILES.items():
        data[key] = load(path, tok, cache)
        size = len(data[key][2])
        if n is None:
            n = size
        elif size != n:
            sys.exit("%s has %d rows, expected %d" % (path, size, n))

    def costs(idx):
        out = {}
        for k, _ in DISTANCES:
            clean = bleu2(idx, *data[(k, "clean")])
            shuf = bleu2(idx, *data[(k, "shuf")])
            out[k] = (clean, 100.0 * (clean - shuf) / clean if clean else 0.0)
        return out

    point = costs(list(range(n)))
    rng = random.Random(args.seed)
    draws = []
    for _ in range(args.resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        draws.append(costs(idx))

    lines, report = [], {"n": n, "resamples": args.resamples, "seed": args.seed, "points": [], "changes": []}

    def emit(text=""):
        print(text)
        lines.append(text)

    emit("SMILES cost by distance, laptop only, n = %d molecules, %d resamples\n" % (n, args.resamples))
    emit("  %-10s %-8s %8s %10s   %-18s %s" % ("units", "tokens", "clean", "SMILES cost", "95% CI", "cluster"))
    for k, t in DISTANCES:
        lo, hi = ci([d[k][1] for d in draws])
        cl = "%.1f%%" % CLUSTER[k] if k in CLUSTER else "-"
        emit("  %-10d %-8d %8.2f %9.1f%%   [%5.1f, %5.1f]    %s" % (k, t, point[k][0], point[k][1], lo, hi, cl))
        report["points"].append({"filler_units": k, "tokens": t, "clean_bleu2": round(point[k][0], 2),
                                 "smiles_cost_pct": round(point[k][1], 2), "ci95": [round(lo, 2), round(hi, 2)],
                                 "cluster_cost_pct": CLUSTER.get(k)})

    emit("\nChange in SMILES cost, paired (percentage points)\n")
    for (a, ta), (b, tb) in [(DISTANCES[0], DISTANCES[1]), (DISTANCES[1], DISTANCES[2]), (DISTANCES[0], DISTANCES[2])]:
        diffs = [d[b][1] - d[a][1] for d in draws]
        lo, hi = ci(diffs)
        verdict = "excludes 0" if lo > 0 or hi < 0 else "includes 0"
        emit("  %2d -> %2d tokens   %+5.1f pp   [%+5.1f, %+5.1f]   %s"
             % (ta, tb, point[b][1] - point[a][1], lo, hi, verdict))
        report["changes"].append({"from_tokens": ta, "to_tokens": tb, "pp": round(point[b][1] - point[a][1], 2),
                                  "ci95": [round(lo, 2), round(hi, 2)], "excludes_zero": lo > 0 or hi < 0})

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "distance_curve_local.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "distance_curve_local.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nwrote %s/distance_curve_local.{txt,json}" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
