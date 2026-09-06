#!/usr/bin/env python3
"""How far does agreement between the two machines go past the aggregate score?

Matching BLEU-2 to two decimals is weak evidence on its own: two runs could
reach the same corpus score through different captions. This compares the
predictions themselves, molecule by molecule, between the RTX 4060 runs in
results/predictions/ and the Tesla P100 runs in cluster/predictions/.

It also decomposes the `shuffle_graph` disagreement. That intervention rotates
graphs WITHIN a batch, so the laptop's batch of 4 and the cluster's batch of 8
hand a quarter of the molecules a different wrong graph. Those molecules should
disagree completely; the rest should behave like the baseline.

Usage:  python cross_stack_agreement.py
"""
import difflib
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

PAIRS = [
    ("baseline", "results/predictions/baseline_full.jsonl",
     "cluster/predictions/chebi_evaluation.jsonl"),
    ("shuffle_graph", "results/predictions/shuffle_graph_full.jsonl",
     "cluster/predictions/chebi_shufgraph.jsonl"),
]

LOCAL_BS, CLUSTER_BS = 4, 8


def load(rel):
    path = os.path.join(HERE, rel)
    if not os.path.exists(path):
        sys.exit("missing %s" % rel)
    with io.open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def partner(i, bs, n):
    """Index whose graph molecule i receives under within-batch rotation by +1."""
    start = (i // bs) * bs
    return start + ((i - start + 1) % min(bs, n - start))


def main():
    for name, a, b in PAIRS:
        A, B = load(a), load(b)
        if len(A) != len(B):
            sys.exit("%s: %d vs %d rows" % (name, len(A), len(B)))
        n = len(A)

        aligned = sum(1 for x, y in zip(A, B)
                      if x["target"].strip() == y["target"].strip())
        ident = [i for i in range(n)
                 if A[i]["prediction"].strip() == B[i]["prediction"].strip()]
        sims = sorted(difflib.SequenceMatcher(None, x["prediction"],
                                              y["prediction"]).ratio()
                      for x, y in zip(A, B))

        print("\n%s  (n = %d)" % (name, n))
        print("  same molecules in the same order : %d / %d" % (aligned, n))
        print("  character-identical captions     : %d  (%.1f%%)"
              % (len(ident), 100.0 * len(ident) / n))
        print("  median similarity                : %.3f" % sims[n // 2])
        print("  captions above 0.95 similarity   : %d  (%.1f%%)"
              % (sum(1 for s in sims if s > 0.95),
                 100.0 * sum(1 for s in sims if s > 0.95) / n))

        if name != "shuffle_graph":
            continue

        same = [i for i in range(n)
                if partner(i, LOCAL_BS, n) == partner(i, CLUSTER_BS, n)]
        diff = [i for i in range(n) if i not in set(same)]
        print("  decomposed by whether both runs substituted the same graph:")
        for label, idx in (("same wrong graph", same), ("different wrong graph", diff)):
            hits = sum(1 for i in idx
                       if A[i]["prediction"].strip() == B[i]["prediction"].strip())
            print("    %-22s %4d molecules, %4d identical (%.1f%%)"
                  % (label, len(idx), hits,
                     100.0 * hits / len(idx) if idx else 0.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
