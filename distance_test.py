#!/usr/bin/env python3
"""Does the graph dominate because it is the graph, or because it is nearest?

Section 3 shows the model follows the graph over an intact SMILES string. The
soft prompts also sit nearest the generation point, so "the graph wins" and "the
nearest channel wins" predict the same outcome. Section 6 tried to separate them
by reordering the prompt and destroyed generation instead.

Jobs 186523, 186610 and 186611 separate them a different way. Neutral filler
between the SMILES span and the soft prompts pushes the SMILES 37 Galactica
tokens further from generation while leaving the graph exactly where it was, and
leaving the template ending the checkpoint was fine-tuned on intact.

The recency account predicts the SMILES contributes less once it is further away,
so corrupting it should cost less. The modality account predicts no such change.

Costs are reported as a share of each condition's own clean baseline, since the
filler lowers that baseline from 62.32 to 53.60 and absolute points are therefore
not comparable across the two distances. All four conditions ran on the same 3300
molecules in the same order, so every contrast is paired.

Usage:  python distance_test.py [--resamples 2000] [--seed 0]
"""
import argparse
import json
import os
import random
import sys

from bootstrap import BERT, SPECIAL, bleu2, stats

HERE = os.path.dirname(os.path.abspath(__file__))

CONDITIONS = {
    "near_clean":       "cluster/predictions/chebi_evaluation.jsonl",
    "near_shufsmiles":  "cluster/predictions/chebi_shufsmiles.jsonl",
    "near_shufgraph":   "cluster/predictions/chebi_shufgraph.jsonl",
    "far_clean":        "cluster/predictions/chebi_filler_mid6.jsonl",
    "far_shufsmiles":   "cluster/predictions/chebi_filler_mid6_shufsmiles.jsonl",
    "far_shufgraph":    "cluster/predictions/chebi_filler_mid6_shufgraph.jsonl",
}


def load(path, tok, cache):
    rows = []
    with open(os.path.join(HERE, path), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    def toks(text):
        if text not in cache:
            cache[text] = [t for t in tok.tokenize(text) if t not in SPECIAL]
        return cache[text]

    num, den = ([], []), ([], [])
    hlen, rlen = [], []
    for row in rows:
        st, h, r = stats(toks(row["prediction"]), toks(row["target"]))
        for n in (0, 1):
            num[n].append(st[n][0])
            den[n].append(st[n][1])
        hlen.append(h)
        rlen.append(r)
    return num, den, hlen, rlen


def ci(vals, lo=2.5, hi=97.5):
    s = sorted(vals)
    return s[int(len(s) * lo / 100)], s[min(len(s) - 1, int(len(s) * hi / 100))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import BertTokenizer
    tok = BertTokenizer.from_pretrained(BERT)
    cache = {}

    data, n = {}, None
    for label, path in CONDITIONS.items():
        if not os.path.exists(os.path.join(HERE, path)):
            sys.exit("missing %s" % path)
        data[label] = load(path, tok, cache)
        size = len(data[label][2])
        if n is None:
            n = size
        elif size != n:
            sys.exit("%s has %d rows, expected %d" % (label, size, n))
        print("scored %-18s %4d molecules" % (label, size), file=sys.stderr)

    def shares(idx):
        """Corruption cost as a share of the matching clean baseline."""
        out = {}
        for dist in ("near", "far"):
            clean = bleu2(idx, *data[dist + "_clean"])
            for ch in ("shufsmiles", "shufgraph"):
                bad = bleu2(idx, *data["%s_%s" % (dist, ch)])
                out[(dist, ch)] = 100.0 * (clean - bad) / clean if clean else 0.0
            out[(dist, "clean")] = clean
        return out

    base = list(range(n))
    point = shares(base)

    print("\nBLEU-2, and what corrupting each channel costs, at two distances\n")
    print("  %-34s %8s %8s %8s" % ("", "clean", "SMILES", "graph"))
    for dist, name in (("near", "SMILES adjacent (as shipped)"),
                       ("far", "SMILES pushed 37 tokens away")):
        print("  %-34s %8.2f %7.1f%% %7.1f%%"
              % (name, point[(dist, "clean")],
                 point[(dist, "shufsmiles")], point[(dist, "shufgraph")]))

    rng = random.Random(args.seed)
    draws = {"smiles": [], "graph": []}
    for _ in range(args.resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        s = shares(idx)
        for ch, key in (("shufsmiles", "smiles"), ("shufgraph", "graph")):
            draws[key].append(s[("far", ch)] - s[("near", ch)])

    print("\nChange in relative cost when the SMILES moves away, 95% paired "
          "bootstrap\n")
    for key, ch in (("smiles", "shufsmiles"), ("graph", "shufgraph")):
        obs = point[("far", ch)] - point[("near", ch)]
        lo, hi = ci(draws[key])
        verdict = "excludes 0" if lo > 0 or hi < 0 else "INCLUDES 0"
        print("  %-8s %+6.1f pp   [%+5.1f, %+5.1f]  %s" % (key, obs, lo, hi, verdict))

    near = point[("near", "shufgraph")] / point[("near", "shufsmiles")]
    far = point[("far", "shufgraph")] / point[("far", "shufsmiles")]
    print("\n  graph cost over SMILES cost: %.2fx adjacent, %.2fx at 37 tokens"
          % (near, far))

    print("""
Reading. Moving the SMILES 37 tokens from the generation point reduces what
corrupting it costs, by 2.6 points of a 23.9 percent effect, and the interval
excludes zero. Recency is real and measurable rather than absent.

It is also far too small to be the mechanism. Were the graph winning because it
sits nearest, displacing its rival that far should have shifted the balance
substantially. The graph's own cost barely moves, and the ratio between the two
channels goes from 2.46x to 2.74x. A recency account would have to explain a
2.5x dominance out of an effect worth a tenth of the smaller channel.

The confound section 3 flagged is real, now quantified, and does not carry the
result.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
