#!/usr/bin/env python3
"""Does the paper's own contamination filter actually hold?

Section 4.1 states that the PubChem324k pretrain subset was filtered to exclude
molecules from the valid/test splits of ChEBI-20, PCDes and MoMu. Every headline
number in this study rests on that claim, because `chebi.ckpt` was pretrained on
that subset and is then evaluated on ChEBI-20's test split. If the filter leaked,
the 62.32 reproduction in section 1 is contaminated by the authors' own pipeline
rather than by anything we did.

Section 11 found 49.25% contamination running the comparison in the *other*
direction, so the question is no longer hypothetical. This script settles it, on
the released V2 artefact, using the same canonical-SMILES matching.

No GPU. Roughly 300k molecules to canonicalise, so a few minutes.

    cd /home/mllp26_team007/MolCA
    source /home/mllp26_team007/molca_env/bin/activate
    python verify_pretrain_filter.py > results_pretrain_filter.txt 2>&1

Read the result as follows.

  Pretrain against ChEBI-20 valid/test
      Should be ZERO. This is exactly what section 4.1 says was removed. A
      non-zero count means the released V2 pretrain subset does not carry the
      filter the paper describes, and section 1's reproduction needs a caveat.

  Pretrain against ChEBI-20 train
      Expected to be large and entirely benign. Nothing claims this was filtered
      and it leaks nothing into any evaluation.

  PubChem train/valid/test against ChEBI-20 anything
      Expected to be large. Section 11 already established that the high-quality
      15k subset was never filtered. Reported here for completeness, since it
      lets the two directions be read off one table.
"""
import argparse
import csv
import os
import sys

PUBCHEM = "data/PubChem324kV2/PubChem324kV2"
CHEBI = "data/ChEBI-20_data"
SPLITS = ("pretrain", "train", "valid", "test")


def canon_all(smiles_iter):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    out, bad = set(), 0
    for s in smiles_iter:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            bad += 1
        else:
            out.add(Chem.MolToSmiles(mol))
    return out, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pubchem", default=PUBCHEM)
    ap.add_argument("--chebi", default=CHEBI)
    args = ap.parse_args()

    sys.path.insert(0, os.getcwd())
    from data_provider.molecule_caption_dataset import MoleculeCaptionV2

    pub = {}
    for split in SPLITS:
        path = os.path.join(args.pubchem, split + ".pt")
        if not os.path.exists(path):
            print("skipping %s, not present" % path)
            continue
        ds = MoleculeCaptionV2(path, 128)
        s, bad = canon_all(ds.get(i).smiles for i in range(len(ds)))
        pub[split] = s
        print("PubChem324kV2 %-9s %7d rows, %7d distinct structures, %d unparseable"
              % (split, len(ds), len(s), bad))

    che = {}
    for split in ("train", "validation", "test"):
        path = os.path.join(args.chebi, split + ".txt")
        with open(path, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        s, bad = canon_all(r["SMILES"] for r in rows)
        che[split] = s
        print("ChEBI-20      %-9s %7d rows, %7d distinct structures, %d unparseable"
              % (split, len(rows), len(s), bad))

    print("\nOverlap, counted as distinct structures present in both\n")
    print("  %-14s %-12s %8s   %s" % ("PubChem split", "ChEBI split", "shared", "verdict"))
    for p in SPLITS:
        if p not in pub:
            continue
        for c in ("train", "validation", "test"):
            n = len(pub[p] & che[c])
            if p == "pretrain" and c in ("validation", "test"):
                verdict = "FILTER HOLDS" if n == 0 else "FILTER LEAKED"
            else:
                verdict = "not filtered, expected"
            print("  %-14s %-12s %8d   %s" % (p, c, n, verdict))

    if "pretrain" in pub:
        leak = len(pub["pretrain"] & (che["validation"] | che["test"]))
        print("\nHeadline: %d ChEBI-20 valid/test structures appear in the "
              "PubChem324kV2\npretrain subset. Section 4.1 says this should be zero."
              % leak)
        if leak:
            print("\nIf this is non-zero, the 62.32 reproduction in section 1 is\n"
                  "evaluated partly on molecules the checkpoint saw during\n"
                  "pretraining, and RESULTS.md needs a limitation saying so.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
