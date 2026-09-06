#!/usr/bin/env python3
"""
analysis2.py - multi-metric channel-conflict analysis for MolCA.

Extends analysis.py in two ways:

  1. Scores the channel-conflict table under BLEU-2, BLEU-4, ROUGE-L and
     METEOR instead of BLEU-2 alone, so the finding does not rest on one
     n-gram metric.

  2. Adds a chemistry-aware measure. ChEBI-20 captions follow the ontology
     template "The molecule is a <CLASS> ...", and also assert membership
     via "It is a <CLASS>". Extracting those class terms and measuring
     overlap tests whether the model names the right KIND of molecule,
     which BLEU cannot distinguish from using the right vocabulary in the
     wrong arrangement.

All metrics here are recomputed locally and are NOT comparable to the
pipeline's own reported numbers (different tokenisation and smoothing).
They are internally comparable across the cells of each table.

Usage:  ~/hfenv/bin/python analysis2.py | tee results_multimetric.txt
"""
import json, random, glob, os, re
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

SM = SmoothingFunction().method1
BATCH = 8
CK = os.path.expanduser('~/MolCA/all_checkpoints')

# ---------------------------------------------------------------- loading
def find(run):
    g = sorted(glob.glob(f'{CK}/{run}/lightning_logs/version_*/predictions.txt'))
    return g[-1] if g else None

def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

# ---------------------------------------------------------------- metrics
_rs = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def m_bleu(refs, hyps, n):
    return corpus_bleu(refs, hyps, weights=tuple([1.0/n]*n),
                       smoothing_function=SM) * 100

def m_rougeL(refs, hyps):
    tot = 0.0
    for r, h in zip(refs, hyps):
        tot += _rs.score(' '.join(r[0]), ' '.join(h))['rougeL'].fmeasure
    return tot / len(hyps) * 100

def m_meteor(refs, hyps):
    tot = 0.0
    for r, h in zip(refs, hyps):
        try:
            tot += meteor_score([r[0]], h)
        except Exception:
            pass
    return tot / len(hyps) * 100

# ------------------------------------------------ chemical class extraction
# ChEBI descriptions assert class membership in two stereotyped ways:
#   "The molecule is a <class> that ..."   /   "It is a <class>, a <class> and ..."
LEAD = re.compile(r"^The molecule is (?:an?|the)\s+([^.,;]+?)(?:\s+that\b|\s+which\b|[.,;])",
                  re.I)
MEMB = re.compile(r"\bIt is (?:an?|the)\s+(.+?)\.", re.I)
DERI = re.compile(r"\bIt derives from (?:an?|the)\s+([^.,;]+)", re.I)

STOP = {'molecule', 'compound', 'conjugate', 'acid', 'base', 'member', 'role',
        'hydride', 'it', 'a', 'an', 'the', 'of', 'and'}

def classes(text):
    """Return the set of chemical class terms asserted by a ChEBI caption."""
    out = set()
    m = LEAD.search(text)
    if m:
        out.add(m.group(1).strip().lower())
    m = MEMB.search(text)
    if m:
        for part in re.split(r",| and ", m.group(1)):
            part = re.sub(r"^\s*(an?|the)\s+", "", part.strip(), flags=re.I)
            if part and part.lower() not in STOP:
                out.add(part.strip().lower())
    m = DERI.search(text)
    if m:
        out.add("derives:" + m.group(1).strip().lower())
    return {c for c in out if c and c not in STOP}

def class_prf(pred_txt, ref_txt):
    """Micro-averaged precision/recall/F1 over asserted class terms."""
    tp = fp = fn = 0
    for p, r in zip(pred_txt, ref_txt):
        P, R = classes(p), classes(r)
        tp += len(P & R); fp += len(P - R); fn += len(R - P)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2*prec*rec / (prec + rec) if prec + rec else 0.0
    return prec*100, rec*100, f1*100

# ---------------------------------------------------------------- setup
p_norm, p_shuf = find('chebi_evaluation'), find('chebi_shufgraph')
if not (p_norm and p_shuf):
    raise SystemExit("predictions not found for chebi_evaluation / chebi_shufgraph")

norm, shuf = load(p_norm), load(p_shuf)
n = len(norm)
assert all(a['target'] == b['target'] for a, b in zip(norm[:50], shuf[:50])), \
    "run order differs between the two runs; rotation index invalid"

def rot(i):
    k, off = divmod(i, BATCH)
    return k*BATCH + (off + 1) % min(BATCH, n - k*BATCH)

tgt_txt = [d['target']     for d in norm]
nrm_txt = [d['prediction'] for d in norm]
shf_txt = [d['prediction'] for d in shuf]

tgt   = [t.split() for t in tgt_txt]
h_nrm = [t.split() for t in nrm_txt]
h_shf = [t.split() for t in shf_txt]

own  = [[tgt[i]]      for i in range(n)]
grph = [[tgt[rot(i)]] for i in range(n)]
random.seed(0)
perm = list(range(n)); random.shuffle(perm)
rnd  = [[tgt[perm[i]]] for i in range(n)]

own_txt  = tgt_txt
grph_txt = [tgt_txt[rot(i)]  for i in range(n)]
rnd_txt  = [tgt_txt[perm[i]] for i in range(n)]

print(f"n = {n} test molecules\n")

# ---------------------------------------------------------------- table 1
print("=" * 74)
print("CHANNEL CONFLICT ACROSS FOUR METRICS")
print("=" * 74)
print("Shuffled condition: each molecule keeps its own SMILES but receives")
print("the NEXT molecule's graph. 'graph target' is the caption of the")
print("molecule whose graph was supplied.\n")

for label, fn in (("BLEU-2",  lambda r, h: m_bleu(r, h, 2)),
                  ("BLEU-4",  lambda r, h: m_bleu(r, h, 4)),
                  ("ROUGE-L", m_rougeL),
                  ("METEOR",  m_meteor)):
    print(f"--- {label} " + "-" * (68 - len(label)))
    print(f"{'':28s} {'own tgt':>10s} {'graph tgt':>11s} {'random':>9s}")
    print(f"{'normal predictions':28s} {fn(own,h_nrm):10.2f} "
          f"{fn(grph,h_nrm):11.2f} {fn(rnd,h_nrm):9.2f}")
    print(f"{'shuffled-graph predictions':28s} {fn(own,h_shf):10.2f} "
          f"{fn(grph,h_shf):11.2f} {fn(rnd,h_shf):9.2f}")
    floor = fn(grph, tgt)
    print(f"{'[floor: tgt vs rotated tgt]':28s} {'':10s} {floor:11.2f}\n")

# ---------------------------------------------------------------- table 2
print("=" * 74)
print("CHEMICAL CLASS AGREEMENT (ontology terms, not n-grams)")
print("=" * 74)
print("Class terms extracted from 'The molecule is a X', 'It is a X, a Y',")
print("and 'It derives from a Z'. Micro-averaged over the test set.\n")
print(f"{'predictions':28s} {'scored against':16s} {'P':>7s} {'R':>7s} {'F1':>7s}")
print("-" * 74)
for pname, ptxt in (("normal", nrm_txt), ("shuffled-graph", shf_txt)):
    for rname, rtxt in (("own target", own_txt),
                        ("graph's target", grph_txt),
                        ("random target", rnd_txt)):
        p, r, f = class_prf(ptxt, rtxt)
        print(f"{pname:28s} {rname:16s} {p:7.2f} {r:7.2f} {f:7.2f}")
print("-" * 74)
p, r, f = class_prf(own_txt, grph_txt)
print(f"{'[floor: tgt vs rotated tgt]':28s} {'':16s} {p:7.2f} {r:7.2f} {f:7.2f}")

# ---------------------------------------------------------------- examples
print("\n" + "=" * 74)
print("QUALITATIVE EXAMPLES: classes asserted")
print("=" * 74)
shown = 0
for i in range(n):
    if shown >= 3:
        break
    Cs, Cg, Co = classes(shf_txt[i]), classes(grph_txt[i]), classes(own_txt[i])
    if not (Cs and Cg and Co) or Cg == Co:
        continue
    shown += 1
    print(f"\n[{i}]")
    print(f"  own SMILES describes : {sorted(Co)}")
    print(f"  graph supplied       : {sorted(Cg)}")
    print(f"  model said           : {sorted(Cs)}")
    print(f"  -> overlap with graph: {len(Cs & Cg)}   with own: {len(Cs & Co)}")
