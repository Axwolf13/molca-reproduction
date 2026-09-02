#!/usr/bin/env python3
"""Paired bootstrap confidence intervals over the CheBI-20 test set.

The paper reports every generative table as a bare point estimate, and so did
the first draft of RESULTS.md. Retraining to get error bars was out of budget.
Resampling the test set is not: generation is deterministic here, so the only
sampling variation left to quantify is *which molecules the test set happens to
contain*, and a bootstrap over those 3300 molecules measures exactly that.

Every condition is resampled with the SAME index draw, which makes each contrast
paired. That is the right design, since all conditions saw the identical split
in the identical order.

BLEU-2 is recomputed the way `model/help_funcs.py:caption_evaluate` does it:
SciBERT tokenisation, `[PAD]`/`[CLS]`/`[SEP]` stripped, then nltk `corpus_bleu`
with weights (.5, .5) and no smoothing. Sufficient statistics are cached per
molecule, so a resample is a sum rather than a rescoring.

Usage:  python bootstrap.py [--resamples 2000] [--seed 0]
"""
import argparse
import collections
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BERT = "allenai/scibert_scivocab_uncased"      # what init_tokenizer() returns

# label -> prediction file, all on the full 3300-molecule split
CONDITIONS = [
    ("baseline",       "cluster/predictions/chebi_evaluation.jsonl"),
    ("shuffle_smiles", "cluster/predictions/chebi_shufsmiles.jsonl"),
    ("null_graph",     "cluster/predictions/chebi_nullgraph.jsonl"),
    ("shuffle_graph",  "cluster/predictions/chebi_shufgraph.jsonl"),
    ("rewire_graph",   "cluster/predictions/chebi_rewire.jsonl"),
]

# (label, a, b) reported as a - b, with the claim each one tests
CONTRASTS = [
    ("cost of substituting the SMILES", "baseline", "shuffle_smiles",
     "corrupting the text channel hurts"),
    ("cost of substituting the graph", "baseline", "shuffle_graph",
     "corrupting the graph channel hurts"),
    ("graph cost minus SMILES cost", "shuffle_smiles", "shuffle_graph",
     "the graph channel matters MORE than the text channel"),
    ("no graph minus wrong graph", "null_graph", "shuffle_graph",
     "a wrong graph is worse than an uninformative one"),
    ("wrong graph minus incoherent graph", "shuffle_graph", "rewire_graph",
     "an incoherent graph is worse than a merely wrong one"),
]

SPECIAL = ("[PAD]", "[CLS]", "[SEP]")


def ngrams(seq, n):
    return collections.Counter(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))


def stats(hyp, ref):
    """nltk corpus_bleu sufficient statistics for one sentence pair, n = 1, 2."""
    out = []
    for n in (1, 2):
        h, r = ngrams(hyp, n), ngrams(ref, n)
        out.append((sum(min(c, r[g]) for g, c in h.items()), max(len(hyp) - n + 1, 0)))
    return out, len(hyp), len(ref)


def bleu2(idx, num, den, hlen, rlen):
    """Corpus BLEU-2 over the molecules named by `idx`."""
    n1 = d1 = n2 = d2 = h = r = 0
    for i in idx:
        n1 += num[0][i]; d1 += den[0][i]
        n2 += num[1][i]; d2 += den[1][i]
        h += hlen[i];    r += rlen[i]
    if d1 == 0 or d2 == 0 or n1 == 0 or n2 == 0 or h == 0:
        return 0.0
    bp = 1.0 if h > r else math.exp(1 - r / h)
    return 100.0 * bp * math.exp(0.5 * math.log(n1 / d1) + 0.5 * math.log(n2 / d2))


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

    num = ([], [])
    den = ([], [])
    hlen, rlen = [], []
    for row in rows:
        st, h, r = stats(toks(row["prediction"]), toks(row["target"]))
        for n in (0, 1):
            num[n].append(st[n][0])
            den[n].append(st[n][1])
        hlen.append(h)
        rlen.append(r)
    return num, den, hlen, rlen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import random
    from transformers import BertTokenizer

    tok = BertTokenizer.from_pretrained(BERT)
    cache = {}

    data, n = {}, None
    for label, path in CONDITIONS:
        if not os.path.exists(os.path.join(HERE, path)):
            sys.exit("missing %s" % path)
        data[label] = load(path, tok, cache)
        size = len(data[label][2])
        if n is None:
            n = size
        elif size != n:
            sys.exit("%s has %d rows, expected %d" % (label, size, n))
        print("scored %-16s %4d molecules" % (label, size), file=sys.stderr)

    base = list(range(n))
    point = {k: bleu2(base, *v) for k, v in data.items()}

    rng = random.Random(args.seed)
    draws = {k: [] for k in data}
    for _ in range(args.resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        for k, v in data.items():
            draws[k].append(bleu2(idx, *v))

    def ci(vals, lo=2.5, hi=97.5):
        s = sorted(vals)
        return (s[int(len(s) * lo / 100)], s[min(len(s) - 1, int(len(s) * hi / 100))])

    print("\nBLEU-2 with a 95%% percentile bootstrap, n = %d molecules, "
          "%d resamples\n" % (n, args.resamples))
    print("  %-16s %8s   %-18s %s" % ("condition", "BLEU-2", "95% CI", "width"))
    for label, _ in CONDITIONS:
        lo, hi = ci(draws[label])
        print("  %-16s %8.2f   [%6.2f, %6.2f]  %5.2f"
              % (label, point[label], lo, hi, hi - lo))

    print("\nPaired contrasts. A CI excluding zero means the ordering survives "
          "resampling\nof the test set.\n")
    for name, a, b in [(c[0], c[1], c[2]) for c in CONTRASTS]:
        diffs = [x - y for x, y in zip(draws[a], draws[b])]
        lo, hi = ci(diffs)
        obs = point[a] - point[b]
        verdict = "excludes 0" if lo > 0 or hi < 0 else "INCLUDES 0"
        print("  %-36s %+7.2f   [%+6.2f, %+6.2f]  %s"
              % (name, obs, lo, hi, verdict))

    print()
    for name, a, b, claim in CONTRASTS:
        diffs = [x - y for x, y in zip(draws[a], draws[b])]
        lo, hi = ci(diffs)
        mark = "supported" if lo > 0 else ("REFUTED" if hi < 0 else "NOT SUPPORTED")
        print("  %-52s %s" % (claim, mark))
    return 0


if __name__ == "__main__":
    sys.exit(main())
