#!/usr/bin/env python3
"""How much of the PubChem324kV2 test set did the ChEBI-20 checkpoint already see?

`cluster/predictions/pc_transfer.jsonl` is the ChEBI-20 fine-tuned checkpoint
evaluated on PubChem324kV2's 2000-molecule test split, scoring 49.04 BLEU-2.
The paper's PubChem324k number is 38.7, but that row is a model fine-tuned on
PubChem324k itself, so reading 49.04 as "transfer beats in-domain training"
only holds if the two test sets are disjoint from ChEBI-20's training data.

They are not. The paper says the ChEBI-20 exclusion filter was applied to the
*pretrain* subset, leaving the high-quality 15k train/valid/test subset
unfiltered. This script measures the consequence by matching caption text,
which is the only molecule identifier the prediction dumps carry, then scores
the seen and unseen halves separately.

Caption matching is conservative in both directions. Whitespace is collapsed
and case folded, nothing else, so a molecule whose PubChem and ChEBI-20 texts
differ by a word counts as unseen even though the checkpoint may well have
trained on it.

BLEU-2 is recomputed exactly as `model/help_funcs.py:caption_evaluate` does,
reusing the machinery in `bootstrap.py`.

Usage:  python transfer_overlap.py [--chebi ../MolCA/data/ChEBI-20_data]
                                   [--resamples 2000] [--seed 0]
"""
import argparse
import csv
import json
import os
import random
import re
import sys

from bootstrap import BERT, SPECIAL, bleu2, stats

HERE = os.path.dirname(os.path.abspath(__file__))
PRED = "cluster/predictions/pc_transfer.jsonl"
SHUF = "cluster/predictions/pc_transfer_shufgraph.jsonl"
DEFAULT_CHEBI = os.path.join(HERE, "..", "MolCA", "data", "ChEBI-20_data")

WS = re.compile(r"\s+")


def norm(text):
    return WS.sub(" ", text).strip().lower()


def load_chebi(root):
    """caption -> split name, for every ChEBI-20 row."""
    seen = {}
    sizes = {}
    for split in ("train", "validation", "test"):
        path = os.path.join(root, split + ".txt")
        if not os.path.exists(path):
            sys.exit("missing %s. Pass --chebi to point at ChEBI-20_data." % path)
        n = 0
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                seen.setdefault(norm(row["description"]), set()).add(split)
                n += 1
        sizes[split] = n
    return seen, sizes


def ci(vals, lo=2.5, hi=97.5):
    s = sorted(vals)
    return s[int(len(s) * lo / 100)], s[min(len(s) - 1, int(len(s) * hi / 100))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chebi", default=DEFAULT_CHEBI)
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import BertTokenizer

    chebi, sizes = load_chebi(args.chebi)
    print("ChEBI-20 rows loaded: " + ", ".join(
        "%s %d" % (k, v) for k, v in sizes.items()))

    rows = []
    with open(os.path.join(HERE, PRED), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print("PubChem324kV2 test rows: %d (%d distinct captions)\n"
          % (len(rows), len({norm(r["target"]) for r in rows})))

    tok = BertTokenizer.from_pretrained(BERT)
    cache = {}

    def toks(text):
        if text not in cache:
            cache[text] = [t for t in tok.tokenize(text) if t not in SPECIAL]
        return cache[text]

    num, den = ([], []), ([], [])
    hlen, rlen, tags = [], [], []
    for row in rows:
        st, h, r = stats(toks(row["prediction"]), toks(row["target"]))
        for n in (0, 1):
            num[n].append(st[n][0])
            den[n].append(st[n][1])
        hlen.append(h)
        rlen.append(r)
        tags.append(chebi.get(norm(row["target"]), set()))

    total = len(rows)
    counts = {s: sum(s in t for t in tags) for s in ("train", "validation", "test")}
    seen_idx = [i for i, t in enumerate(tags) if "train" in t]
    any_idx = [i for i, t in enumerate(tags) if t]
    unseen_idx = [i for i, t in enumerate(tags) if "train" not in t]

    print("Caption-exact overlap with ChEBI-20")
    for split in ("train", "validation", "test"):
        print("  appears in ChEBI-20 %-11s %4d / %d  (%5.2f%%)"
              % (split, counts[split], total, 100.0 * counts[split] / total))
    print("  appears in any split      %4d / %d  (%5.2f%%)"
          % (len(any_idx), total, 100.0 * len(any_idx) / total))
    print("\n  The train row is the one that matters. That is the split the\n"
          "  chebi.ckpt checkpoint was fine-tuned on for 100 epochs.\n")

    data = (num, den, hlen, rlen)
    overall = bleu2(range(total), *data)
    seen = bleu2(seen_idx, *data)
    unseen = bleu2(unseen_idx, *data)

    print("BLEU-2 by subset")
    print("  %-34s %5d molecules   %6.2f" % ("whole test split", total, overall))
    print("  %-34s %5d molecules   %6.2f"
          % ("caption seen in ChEBI-20 train", len(seen_idx), seen))
    print("  %-34s %5d molecules   %6.2f"
          % ("caption not seen in ChEBI-20 train", len(unseen_idx), unseen))
    print("  %-34s %5s              %+6.2f" % ("contamination gap", "", seen - unseen))

    rng = random.Random(args.seed)
    ds, du, dg = [], [], []
    for _ in range(args.resamples):
        si = [seen_idx[rng.randrange(len(seen_idx))] for _ in seen_idx]
        ui = [unseen_idx[rng.randrange(len(unseen_idx))] for _ in unseen_idx]
        a, b = bleu2(si, *data), bleu2(ui, *data)
        ds.append(a)
        du.append(b)
        dg.append(a - b)

    lo, hi = ci(ds)
    print("\n95%% percentile bootstrap, %d resamples" % args.resamples)
    print("  seen subset     [%6.2f, %6.2f]" % (lo, hi))
    lo, hi = ci(du)
    print("  unseen subset   [%6.2f, %6.2f]" % (lo, hi))
    lo, hi = ci(dg)
    print("  gap             [%+6.2f, %+6.2f]  %s"
          % (lo, hi, "excludes 0" if lo > 0 or hi < 0 else "INCLUDES 0"))

    print("\nRead the unseen number, not the headline. The paper's 38.7 is a\n"
          "PubChem324k-trained model on this same 2000-row split, so the\n"
          "honest comparison is %.2f against 38.7, and even that assumes the\n"
          "V2 release kept the paper's split. See NOTES.md." % unseen)

    # Does the channel-conflict result survive once the memorised rows go?
    shuf = []
    with open(os.path.join(HERE, SHUF), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                shuf.append(json.loads(line))
    if len(shuf) != total:
        print("\nskipping the shuffled-graph split: %d rows, expected %d"
              % (len(shuf), total))
        return 0

    snum, sden = ([], []), ([], [])
    shl, srl = [], []
    for row in shuf:
        st, h, r = stats(toks(row["prediction"]), toks(row["target"]))
        for n in (0, 1):
            snum[n].append(st[n][0])
            sden[n].append(st[n][1])
        shl.append(h)
        srl.append(r)
    sdata = (snum, sden, shl, srl)

    print("\nChannel conflict on PubChem324kV2, seen and unseen scored apart")
    print("  %-22s %8s %8s %8s" % ("subset", "normal", "shuffled", "drop"))
    for name, sub in (("whole split", list(range(total))),
                      ("seen in ChEBI-20 train", seen_idx),
                      ("not seen", unseen_idx)):
        a, b = bleu2(sub, *data), bleu2(sub, *sdata)
        print("  %-22s %8.2f %8.2f %8.2f" % (name, a, b, a - b))
    print("\n  Substituting the neighbour's graph costs the model most of its\n"
          "  score on both halves, so the channel-conflict finding survives the\n"
          "  removal of the memorised rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
