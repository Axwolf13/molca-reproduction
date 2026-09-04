#!/usr/bin/env python3
"""Re-derive the documented numbers that no saved tool output already records.

Most figures in `README.md` and `results/RESULTS.md` can be traced by grep to a
file under `cluster/results/`, `cluster/runlogs/` or `results/logs/`. A handful
cannot, because they came from interactive analysis whose output was never
written to disk, or because the local run's log was captured with progress bars
stripped and lost its final metric block.

This script recomputes exactly those, from the prediction dumps, so that every
number in the documentation has a reproducible source. It is the last check
before the coding work is called finished.

Scoring reproduces `model/help_funcs.py:caption_evaluate`: strip `[PAD]`,
`[CLS]` and `[SEP]`, then nltk `corpus_bleu` with weights (.5, .5) and no
smoothing. Two tokenisers appear below and the distinction matters. MolCA's own
scorer calls `init_tokenizer()`, which returns SciBERT. Section 6 of RESULTS.md
predates that discovery and reports `bert-base-uncased` figures, which is why
its table reads 63.16 where the pipeline reports 62.65. Both are recomputed here
under the tokeniser each was produced with.

Usage:  python verify_numbers.py
Exit status is non-zero if any check fails.
"""
import json
import os
import sys

from bootstrap import BERT, SPECIAL, bleu2, stats

HERE = os.path.dirname(os.path.abspath(__file__))
PLAIN = "bert-base-uncased"

# (description, prediction file, tokeniser, field, expected, where it is quoted)
CHECKS = [
    ("local baseline, 1000 molecules", "results/predictions/baseline.jsonl",
     BERT, "bleu2", 62.65, "RESULTS.md sec 1, sec 2"),
    ("local baseline, full 3300 split", "results/predictions/baseline_full.jsonl",
     BERT, "bleu2", 62.77, "README.md, RESULTS.md sec 1"),
    ("sec 6 default order", "results/predictions/baseline.jsonl",
     PLAIN, "bleu2", 63.16, "README.md, RESULTS.md sec 6"),
    ("sec 6 default order", "results/predictions/baseline.jsonl",
     PLAIN, "words", 42.9, "RESULTS.md sec 6"),
    ("sec 6 default order", "results/predictions/baseline.jsonl",
     PLAIN, "empty", 0, "RESULTS.md sec 6"),
    ("sec 6 graph prompts first", "results/predictions/baseline_graphfirst.jsonl",
     PLAIN, "bleu2", 0.01, "README.md, RESULTS.md sec 6"),
    ("sec 6 graph prompts first", "results/predictions/baseline_graphfirst.jsonl",
     PLAIN, "words", 123.0, "RESULTS.md sec 6"),
    ("sec 6 graph prompts first", "results/predictions/baseline_graphfirst.jsonl",
     PLAIN, "empty", 55, "RESULTS.md sec 6"),
    ("sec 12 baseline generation", "cluster/predictions/chebi_evaluation.jsonl",
     BERT, "words", 42.4, "RESULTS.md sec 12"),
    ("sec 12 filler between", "cluster/predictions/chebi_filler_mid6.jsonl",
     BERT, "words", 37.3, "RESULTS.md sec 12"),
    ("sec 12 filler between", "cluster/predictions/chebi_filler_mid6.jsonl",
     BERT, "empty", 0, "RESULTS.md sec 12"),
    ("sec 12 filler at end", "cluster/predictions/chebi_filler_end6.jsonl",
     BERT, "words", 26.0, "RESULTS.md sec 12"),
    ("sec 12 filler at end", "cluster/predictions/chebi_filler_end6.jsonl",
     BERT, "empty", 54, "RESULTS.md sec 12"),
    ("sec 12 void control", "cluster/predictions/pc_stage2_control.jsonl",
     BERT, "empty", 1605, "RESULTS.md sec 12, NOTES.md"),
]

TOL = {"bleu2": 0.005, "words": 0.051, "empty": 0}


def measure(path, tok, cache):
    rows = []
    with open(os.path.join(HERE, path), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    def toks(text):
        key = (id(tok), text)
        if key not in cache:
            cache[key] = [t for t in tok.tokenize(text) if t not in SPECIAL]
        return cache[key]

    num, den = ([], []), ([], [])
    hlen, rlen = [], []
    for row in rows:
        st, h, r = stats(toks(row["prediction"]), toks(row["target"]))
        for n in (0, 1):
            num[n].append(st[n][0])
            den[n].append(st[n][1])
        hlen.append(h)
        rlen.append(r)

    words = [len(row["prediction"].split()) for row in rows]
    return {
        "bleu2": bleu2(range(len(rows)), num, den, hlen, rlen),
        "words": sum(words) / len(words),
        "empty": sum(1 for row in rows if not row["prediction"].strip()),
        "rows": len(rows),
    }


def main():
    from transformers import BertTokenizer

    toks, cache, measured = {}, {}, {}
    for name in (BERT, PLAIN):
        try:
            toks[name] = BertTokenizer.from_pretrained(name)
        except Exception as exc:
            sys.exit("cannot load %s: %s" % (name, exc))

    fails = 0
    print("%-32s %-10s %9s %9s  %s" % ("what", "field", "expected", "got", "quoted in"))
    for desc, path, tokname, field, expected, where in CHECKS:
        full = os.path.join(HERE, path)
        if not os.path.exists(full):
            print("%-32s %-10s %9s %9s  MISSING FILE %s" % (desc, field, expected, "-", path))
            fails += 1
            continue
        key = (path, tokname)
        if key not in measured:
            measured[key] = measure(path, toks[tokname], cache)
        got = measured[key][field]
        ok = abs(got - expected) <= TOL[field]
        fails += not ok
        fmt = "%9d" if field == "empty" else "%9.2f"
        print(("%-32s %-10s " + fmt + " " + fmt + "  %s%s")
              % (desc, field, expected, got, where, "" if ok else "   <-- MISMATCH"))

    print()
    if fails:
        print("%d check(s) failed. The documentation does not match the artefacts."
              % fails)
        return 1
    print("All %d checks reproduce. Every documented number without other saved\n"
          "provenance is derivable from the prediction dumps in this repository."
          % len(CHECKS))
    print("\nTwo figures in RESULTS.md section 6 are not covered here: the 63.10\n"
          "ROUGE-L and 66.04 METEOR in the same table. Both need `rouge_score`,\n"
          "which the local environment lacks. They come from the same run as the\n"
          "63.16 and 123.0 verified above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
