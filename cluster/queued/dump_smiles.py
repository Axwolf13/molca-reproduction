#!/usr/bin/env python3
"""Dump canonical SMILES so the contamination overlap can be redone on structures.

`transfer_overlap.py` currently matches molecules by caption text, because the
prediction dumps carry nothing else. That makes 23.35% a floor rather than a
measurement: any molecule whose PubChem wording differs from its ChEBI-20 wording
by a single word is scored as unseen.

This script closes that gap. It needs no GPU and takes under a minute. Run it on
the cluster, where both datasets and rdkit are present, then commit the two
output files and rerun `transfer_overlap.py --smiles`.

    cd /home/mllp26_team007/MolCA
    source /home/mllp26_team007/molca_env/bin/activate
    python dump_smiles.py --out /home/mllp26_team007/smiles_dump

Writes:
    pubchem_test_smiles.jsonl   one row per test molecule, in dataset order
    chebi_smiles.jsonl          one row per ChEBI-20 molecule, all three splits

Both carry the raw SMILES and the rdkit canonical form. Canonicalisation is what
makes the comparison meaningful, since the two datasets need not agree on
kekulisation, atom ordering, or stereo notation for the same structure.
"""
import argparse
import csv
import json
import os
import sys

PUBCHEM = "data/PubChem324kV2/PubChem324kV2/test.pt"
CHEBI = "data/ChEBI-20_data"


def canon(smiles):
    """rdkit canonical SMILES, or None when the string will not parse."""
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="smiles_dump")
    ap.add_argument("--pubchem", default=PUBCHEM)
    ap.add_argument("--chebi", default=CHEBI)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # PubChem324kV2 test split, through the repository's own loader so the row
    # order matches the prediction dumps exactly.
    sys.path.insert(0, os.getcwd())
    from data_provider.molecule_caption_dataset import MoleculeCaptionV2

    ds = MoleculeCaptionV2(args.pubchem, 128)
    path = os.path.join(args.out, "pubchem_test_smiles.jsonl")
    bad = 0
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(len(ds)):
            row = ds.get(i)
            c = canon(row.smiles)
            bad += c is None
            # text_head lets the consumer assert row alignment against the
            # prediction dump rather than trusting that the order held.
            fh.write(json.dumps({
                "i": i,
                "smiles": row.smiles,
                "canonical": c,
                "text_head": row.text[:80],
            }) + "\n")
    print("wrote %s: %d rows, %d unparseable" % (path, len(ds), bad))

    # ChEBI-20, all three splits.
    path = os.path.join(args.out, "chebi_smiles.jsonl")
    total = bad = 0
    with open(path, "w", encoding="utf-8") as out:
        for split in ("train", "validation", "test"):
            src = os.path.join(args.chebi, split + ".txt")
            with open(src, encoding="utf-8") as fh:
                for row in csv.DictReader(fh, delimiter="\t"):
                    c = canon(row["SMILES"])
                    bad += c is None
                    total += 1
                    out.write(json.dumps({
                        "split": split,
                        "cid": row["CID"],
                        "smiles": row["SMILES"],
                        "canonical": c,
                    }) + "\n")
    print("wrote %s: %d rows, %d unparseable" % (path, total, bad))
    print("\nCommit both files under cluster/results/ and rerun "
          "transfer_overlap.py --smiles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
