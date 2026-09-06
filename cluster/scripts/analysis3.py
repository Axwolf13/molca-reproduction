#!/usr/bin/env python3
"""
analysis3.py - chemical class agreement with fuzzy matching.

analysis2.py used exact string equality on ChEBI class terms, which
undercounts badly. Two failure modes were visible in its own examples:

  * acid/base pairs      "indolylmethylglucosinolate"
                     vs  "indolylmethylglucosinolic acid"
  * long vs short forms  "branched amino tetrasaccharide"
                     vs  "branched amino tetrasaccharide consisting of
                          N-acetyl-beta-D-glucosamine having two ..."

Both are the same chemical class and both scored zero. This script keeps
exact matching as a conservative baseline and adds two looser criteria,
so the reported effect can be bracketed rather than stated as one number.

Matching criteria, from strict to loose:
  exact     - normalised strings identical
  contain   - one normalised string contains the other
  jaccard   - token-set Jaccard >= THRESHOLD after dropping stopwords

Usage:  ~/hfenv/bin/python analysis3.py | tee results_class_fuzzy.txt
"""
import json, random, glob, os, re

BATCH = 8
JACCARD_MIN = 0.5
CK = os.path.expanduser('~/MolCA/all_checkpoints')

# ------------------------------------------------------------ extraction
LEAD = re.compile(r"^The molecule is (?:an?|the)\s+([^.,;]+?)(?:\s+that\b|\s+which\b|[.,;])", re.I)
MEMB = re.compile(r"\bIt is (?:an?|the)\s+(.+?)\.", re.I)
DERI = re.compile(r"\bIt derives from (?:an?|the)\s+([^.,;]+)", re.I)

STOP = {'molecule', 'compound', 'conjugate', 'member', 'role', 'hydride',
        'it', 'a', 'an', 'the', 'of', 'and', 'is', 'that', 'which', 'from'}

# chemical suffix pairs that denote the same class in different ionisation
# states; ChEBI uses both forms freely
IONIC = [('ate', 'ic acid'), ('ate', 'ic'), ('ide', 'ic acid')]

def norm(term):
    t = term.strip().lower()
    t = re.sub(r"^\s*(an?|the)\s+", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip(" .,;")

def classes(text):
    out = set()
    m = LEAD.search(text)
    if m:
        out.add(norm(m.group(1)))
    m = MEMB.search(text)
    if m:
        for part in re.split(r",| and ", m.group(1)):
            p = norm(part)
            if p and p not in STOP:
                out.add(p)
    m = DERI.search(text)
    if m:
        out.add("derives:" + norm(m.group(1)))
    return {c for c in out if c and c not in STOP}

# ------------------------------------------------------------- matching
def toks(s):
    return {w for w in re.split(r"[^a-z0-9()\[\]>-]+", s) if w and w not in STOP}

def ionic_equiv(a, b):
    """True if a and b differ only by an acid/base suffix swap."""
    for x, y in IONIC:
        for p, q in ((a, b), (b, a)):
            if p.endswith(x) and q.endswith(y):
                if p[:-len(x)].rstrip() == q[:-len(y)].rstrip():
                    return True
    return False

def match(a, b, mode):
    if a == b:
        return True
    if mode == 'exact':
        return False
    if ionic_equiv(a, b):
        return True
    if mode in ('contain', 'jaccard'):
        # a long ChEBI phrase that starts with the short class name
        if a.startswith(b) or b.startswith(a):
            return True
        if len(a) > 12 and a in b:
            return True
        if len(b) > 12 and b in a:
            return True
    if mode == 'jaccard':
        A, B = toks(a), toks(b)
        if A and B:
            j = len(A & B) / len(A | B)
            if j >= JACCARD_MIN:
                return True
    return False

def prf(pred_txt, ref_txt, mode):
    tp = fp = fn = 0
    for p_text, r_text in zip(pred_txt, ref_txt):
        P, R = classes(p_text), classes(r_text)
        used = set()
        hit = 0
        for a in P:
            for b in R:
                if b in used:
                    continue
                if match(a, b, mode):
                    used.add(b); hit += 1
                    break
        tp += hit
        fp += len(P) - hit
        fn += len(R) - hit
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    return prec*100, rec*100, f1*100

# ---------------------------------------------------------------- setup
def find(run):
    g = sorted(glob.glob(f'{CK}/{run}/lightning_logs/version_*/predictions.txt'))
    return g[-1] if g else None

def load(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]

norm_runs = load(find('chebi_evaluation'))
shuf_runs = load(find('chebi_shufgraph'))
n = len(norm_runs)
assert all(a['target'] == b['target'] for a, b in zip(norm_runs[:50], shuf_runs[:50]))

def rot(i):
    k, off = divmod(i, BATCH)
    return k*BATCH + (off + 1) % min(BATCH, n - k*BATCH)

tgt_txt = [d['target']     for d in norm_runs]
nrm_txt = [d['prediction'] for d in norm_runs]
shf_txt = [d['prediction'] for d in shuf_runs]
grph_txt = [tgt_txt[rot(i)] for i in range(n)]
random.seed(0); perm = list(range(n)); random.shuffle(perm)
rnd_txt = [tgt_txt[perm[i]] for i in range(n)]

# ---------------------------------------------------------------- report
print(f"n = {n} test molecules")
print(f"Jaccard threshold = {JACCARD_MIN}\n")

nc = sum(len(classes(t)) for t in tgt_txt)
print(f"Class terms extracted from targets: {nc} "
      f"({nc/n:.2f} per molecule)\n")

for mode in ('exact', 'contain', 'jaccard'):
    print("=" * 72)
    print(f"MATCHING: {mode}")
    print("=" * 72)
    print(f"{'predictions':26s} {'scored against':16s} {'P':>7s} {'R':>7s} {'F1':>7s}")
    print("-" * 72)
    cells = {}
    for pname, ptxt in (("normal", nrm_txt), ("shuffled-graph", shf_txt)):
        for rname, rtxt in (("own target", tgt_txt),
                            ("graph's target", grph_txt),
                            ("random target", rnd_txt)):
            p, r, f = prf(ptxt, rtxt, mode)
            cells[(pname, rname)] = f
            print(f"{pname:26s} {rname:16s} {p:7.2f} {r:7.2f} {f:7.2f}")
    fl = prf(tgt_txt, grph_txt, mode)[2]
    print("-" * 72)
    print(f"{'[floor: tgt vs rot tgt]':26s} {'':16s} {'':7s} {'':7s} {fl:7.2f}")
    sg = cells[("shuffled-graph", "graph's target")]
    so = cells[("shuffled-graph", "own target")]
    ratio = sg / so if so else float('inf')
    print(f"\nshuffled preds: F1 {sg:.2f} vs the GRAPH's classes, "
          f"{so:.2f} vs their OWN -> {ratio:.1f}x\n")

# --------------------------------------------------------------- example
print("=" * 72)
print("EXAMPLES WHERE FUZZY MATCHING RECOVERS A HIT MISSED BY EXACT")
print("=" * 72)
shown = 0
for i in range(n):
    if shown >= 5:
        break
    P, G = classes(shf_txt[i]), classes(grph_txt[i])
    for a in P:
        for b in G:
            if not match(a, b, 'exact') and match(a, b, 'jaccard'):
                print(f"\n[{i}] model: {a!r}")
                print(f"      graph: {b!r}")
                shown += 1
                break
        else:
            continue
        break
