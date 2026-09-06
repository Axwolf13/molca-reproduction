#!/usr/bin/env python3
"""Check every retrieval metric in the raw Condor logs against the paper.

Reference values are Tables 7b (PCDes) and 7c (MoMu) of Liu et al., EMNLP 2023
(arXiv:2310.12798v4). The paper's "MolCA w/o MTM" row is contrastive scoring
alone; its "MolCA" row adds matching-head re-ranking of the top-128 candidates.
Lightning names those `test_*` and `rerank_test_*` respectively.

This reads runlogs/pcdes.183393.out and runlogs/momu.183394.out, the unedited
stdout of the two jobs. results/results_retrieval.txt is a hand-made excerpt of
the same run carrying only the full-test-set rows, so the in-batch columns can
be checked against the logs alone.

Usage:  python cluster/verify_retrieval.py
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = {
    "PCDes": os.path.join(HERE, "runlogs", "pcdes.183393.out"),
    "MoMu": os.path.join(HERE, "runlogs", "momu.183394.out"),
}

PAPER = {
    "PCDes": {                              # Table 7b
        "test_inbatch_g2t_acc":          80.9,
        "test_inbatch_g2t_rec20":        98.1,
        "test_inbatch_t2g_acc":          77.9,
        "test_inbatch_t2g_rec20":        97.5,
        "test_fullset_g2t_acc":          37.7,
        "test_fullset_g2t_rec20":        80.6,
        "test_fullset_t2g_acc":          35.3,
        "test_fullset_t2g_rec20":        76.5,
        "rerank_test_inbatch_g2t_acc":   86.4,
        "rerank_test_inbatch_g2t_rec20": 99.8,
        "rerank_test_inbatch_t2g_acc":   84.8,
        "rerank_test_inbatch_t2g_rec20": 98.5,
        "rerank_test_fullset_g2t_acc":   48.1,
        "rerank_test_fullset_g2t_rec20": 85.6,
        "rerank_test_fullset_t2g_acc":   46.0,
        "rerank_test_fullset_t2g_rec20": 82.3,
    },
    "MoMu": {                               # Table 7c
        "test_inbatch_g2t_acc":          65.0,
        "test_inbatch_g2t_rec20":        95.9,
        "test_inbatch_t2g_acc":          63.3,
        "test_inbatch_t2g_rec20":        95.9,
        "test_fullset_g2t_acc":          22.5,
        "test_fullset_g2t_rec20":        68.5,
        "test_fullset_t2g_acc":          21.1,
        "test_fullset_t2g_rec20":        64.8,
        "rerank_test_inbatch_g2t_acc":   73.4,
        "rerank_test_inbatch_g2t_rec20": 98.5,
        "rerank_test_inbatch_t2g_acc":   72.8,
        "rerank_test_inbatch_t2g_rec20": 97.5,
        "rerank_test_fullset_g2t_acc":   30.6,
        "rerank_test_fullset_g2t_rec20": 76.8,
        "rerank_test_fullset_t2g_acc":   29.8,
        "rerank_test_fullset_t2g_rec20": 73.3,
    },
}

BAR = "│"
ROW = re.compile(BAR + r"\s*([a-z0-9_]+)\s*" + BAR + r"\s*([0-9.]+)\s*" + BAR)


def parse(path):
    """Pull every `| metric | value |` row out of one Lightning summary table."""
    out = {}
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh.read().replace("\r", "\n").splitlines():
            m = ROW.search(line)
            if m:
                out[m.group(1)] = float(m.group(2))
    return out


def main():
    got = {}
    for split, path in LOGS.items():
        if not os.path.exists(path):
            sys.exit("missing %s" % path)
        got[split] = parse(path)

    worst, worst_key, checked, failed = 0.0, None, 0, 0
    for split, refs in PAPER.items():
        print("\n%s  (paper Table 7%s, log %s)"
              % (split, "b" if split == "PCDes" else "c",
                 os.path.basename(LOGS[split])))
        print("  %-32s %8s %8s %8s" % ("metric", "ours", "paper", "delta"))
        for key, ref in refs.items():
            if key not in got[split]:
                print("  %-32s %8s %8.1f   MISSING" % (key, "-", ref))
                failed += 1
                continue
            mine = got[split][key]
            delta = mine - ref
            if abs(delta) > worst:
                worst, worst_key = abs(delta), "%s %s" % (split, key)
            checked += 1
            print("  %-32s %8.2f %8.1f %+8.2f" % (key, mine, ref, delta))

    print("\n%d metrics checked, %d missing, max |deviation| = %.2f (%s)"
          % (checked, failed, worst, worst_key))

    # The full-test-set columns are the ones the paper leads with. In-batch
    # accuracy depends on how the evaluation batches happen to be drawn, which
    # makes it the looser of the two comparisons.
    fullset = max(abs(got[sp][k] - v)
                  for sp, refs in PAPER.items()
                  for k, v in refs.items()
                  if "fullset" in k and k in got[sp])
    print("full-test-set columns only (16 metrics): max |deviation| = %.2f" % fullset)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
