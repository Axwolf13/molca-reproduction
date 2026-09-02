#!/usr/bin/env python3
"""
analysis.py - scores all completed MolCA eval runs and writes results.

Two parts:
  1. Pipeline-reported BLEU-2 for every run, scraped from Condor logs.
  2. Channel-conflict analysis: when the graph and the SMILES describe
     different molecules, which one does the model follow?

Part 2 recomputes BLEU with nltk, which does NOT match the pipeline's own
tokenisation/smoothing. Those numbers are internally comparable only.

Usage:  ~/hfenv/bin/python analysis.py > results.txt
"""
import json, random, glob, re, os
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

SM = SmoothingFunction().method1
BATCH = 8          # --inference_batch_size
PAPER_BLEU2 = 62.0 # Table 2b, MolCA Galac1.3B, CheBI-20

def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def bleu(refs, hyps, n=2):
    w = tuple([1.0/n]*n)
    return corpus_bleu(refs, hyps, weights=w, smoothing_function=SM) * 100

# ---------- part 1: pipeline-reported scores ----------
print("=" * 68)
print("PIPELINE-REPORTED SCORES (as computed by MolCA's own caption_evaluate)")
print("=" * 68)

rows = []
for f in sorted(glob.glob(os.path.expanduser('~/MolCA/runlogs/*.out'))):
    txt = open(f, errors='ignore').read()
    m = {k: re.search(rf"{p} score: ([\d.e+-]+)", txt)
         for k, p in (('BLEU-2', 'BLEU-2'), ('BLEU-4', 'BLEU-4'),
                      ('METEOR', 'Average Meteor'))}
    if not m['BLEU-2']:
        continue
    name = os.path.basename(f).rsplit('.', 2)[0]
    jobid = os.path.basename(f).rsplit('.', 2)[1]
    _all = re.findall(r"\[(\d+:\d+:\d+)<00:00", txt)
    t = _all[-1] if _all else None
    rows.append((name, jobid,
                 float(m['BLEU-2'].group(1)),
                 float(m['BLEU-4'].group(1)) if m['BLEU-4'] else float('nan'),
                 float(m['METEOR'].group(1)) if m['METEOR'] else float('nan'),
                 t if t else '?'))

print(f"{'run':24s} {'job':8s} {'BLEU-2':>10s} {'BLEU-4':>10s} {'METEOR':>8s}  {'wall':>9s}")
print("-" * 68)
for name, jid, b2, b4, mt, wall in sorted(rows, key=lambda r: -r[2]):
    b2s = f"{b2:10.2f}" if b2 > 1e-6 else f"{b2:10.2e}"
    print(f"{name:24s} {jid:8s} {b2s} {b4:10.2f} {mt:8.2f}  {wall:>9s}")

base = next((r[2] for r in rows if 'chebi_eval' in r[0]), None)
if base:
    print(f"\nPaper (Table 2b): {PAPER_BLEU2}   reproduction: {base:.2f}"
          f"   deviation: {base - PAPER_BLEU2:+.2f}")

# ---------- part 2: channel conflict ----------
CK = os.path.expanduser('~/MolCA/all_checkpoints')
def find(run):
    g = sorted(glob.glob(f'{CK}/{run}/lightning_logs/version_*/predictions.txt'))
    return g[-1] if g else None

p_norm, p_shuf = find('chebi_evaluation'), find('chebi_shufgraph')
if not (p_norm and p_shuf):
    print("\n[channel-conflict analysis skipped: predictions not found]")
    raise SystemExit

norm, shuf = load(p_norm), load(p_shuf)
n = len(norm)
assert all(a['target'] == b['target'] for a, b in zip(norm[:50], shuf[:50])), \
    "run order differs; rotation index would be invalid"

def rot(i):
    k, off = divmod(i, BATCH)
    return k*BATCH + (off + 1) % min(BATCH, n - k*BATCH)

tgt   = [d['target'].split()     for d in norm]
h_nrm = [d['prediction'].split() for d in norm]
h_shf = [d['prediction'].split() for d in shuf]

own  = [[tgt[i]]      for i in range(n)]
grph = [[tgt[rot(i)]] for i in range(n)]
random.seed(0)
perm = list(range(n)); random.shuffle(perm)
rnd  = [[tgt[perm[i]]] for i in range(n)]

print("\n" + "=" * 68)
print("CHANNEL CONFLICT: which modality does the model follow?")
print("=" * 68)
print("In the shuffled condition each molecule keeps its own SMILES but")
print("receives the NEXT molecule's graph. If the model follows the graph,")
print("its output should match the rotated target, not its own.\n")
print("BLEU-2 (nltk, not comparable to pipeline numbers above)\n")
print(f"{'':26s} {'own target':>12s} {'graph target':>14s} {'random':>10s}")
print("-" * 68)
print(f"{'normal predictions':26s} {bleu(own,h_nrm):12.2f} "
      f"{bleu(grph,h_nrm):14.2f} {bleu(rnd,h_nrm):10.2f}")
print(f"{'shuffled-graph predictions':26s} {bleu(own,h_shf):12.2f} "
      f"{bleu(grph,h_shf):14.2f} {bleu(rnd,h_shf):10.2f}")

floor = bleu(grph, tgt)
print(f"\nBoilerplate floor (target vs rotated target): {floor:.2f}")
print(f"  -> shuffled preds exceed floor against the GRAPH's target by "
      f"{bleu(grph,h_shf)-floor:+.2f}")
print(f"  -> and against their OWN target by only "
      f"{bleu(own,h_shf)-floor:+.2f}")

# qualitative example
print("\n" + "-" * 68)
print("EXAMPLE (index 0)")
print("-" * 68)
for label, s in (("own target      ", norm[0]['target']),
                 ("graph's target  ", norm[rot(0)]['target']),
                 ("normal pred     ", norm[0]['prediction']),
                 ("shuffled pred   ", shuf[0]['prediction'])):
    print(f"{label}: {s.strip()[:150]}")
