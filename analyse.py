#!/usr/bin/env python3
"""analyse.py - channel-conflict analysis for the local MolCA runs.

Akshay Ashok (7071170), MLLP seminar 2026.

Question. MolCA feeds the language model two views of the same molecule: the
SMILES string as text, and the 2D graph as eight soft prompts from the
Q-Former. Table 5a of the paper measures what each view contributes during
TRAINING (34.6 SMILES-only, 34.5 graph-only, 38.7 both, on PubChem324k).
It never measures what the trained model does with them at INFERENCE.

Method. Take the released both-modalities checkpoint and put the two channels
into conflict: give molecule i its own SMILES but molecule i+1's graph. If the
model describes molecule i, it is reading the text. If it describes i+1, it is
reading the graph. Rotation is within-batch, so the substituted graph is always
a real molecule from the same split - this separates "wrong molecule" from
"malformed input".

Scoring. Corpus BLEU here is NOT the pipeline's BLEU (different tokenisation
and smoothing), so these numbers are comparable only within this file. Every
comparison carries a floor: ChEBI captions share heavy boilerplate ("The
molecule is a...", "It has a role as..."), so unrelated captions already score
~25 BLEU-2 against each other. Chemical-class agreement is reported alongside
because its floor is near zero and it is a claim about chemistry rather than
about n-gram overlap.

Usage:  molca_venv/Scripts/python.exe analyse.py [--out DIR]
"""
import argparse
import glob
import json
import os
import random
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
# MUST match --inference_batch_size used for the runs: the rotation that
# defines "the graph molecule i was given" is within-batch, so a wrong value
# here silently misaligns every comparison.
BATCH = int(os.environ.get("MOLCA_BATCH", "4"))
JACCARD_MIN = 0.5
SEED = 0

# --------------------------------------------------------------- discovery
SEARCH = [
    os.path.join(HERE, "MolCA", "all_checkpoints", "local_{}", "lightning_logs",
                 "version_*", "predictions.txt"),
    os.path.join(HERE, "MolCA", "all_checkpoints", "{}", "lightning_logs",
                 "version_*", "predictions.txt"),
]


def _version_no(path):
    m = re.search(r"version_(\d+)", path)
    return int(m.group(1)) if m else -1


def find(run):
    """Newest version_N with predictions. Sorts numerically: plain string
    sort would rank version_10 before version_2."""
    for pat in SEARCH:
        hits = glob.glob(pat.format(run))
        if hits:
            return max(hits, key=_version_no)
    return None


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


# ------------------------------------------------------- class extraction
LEAD = re.compile(r"^The molecule is (?:an?|the)\s+([^.,;]+?)(?:\s+that\b|\s+which\b|[.,;])", re.I)
MEMB = re.compile(r"\bIt is (?:an?|the)\s+(.+?)\.", re.I)
DERI = re.compile(r"\bIt derives from (?:an?|the)\s+([^.,;]+)", re.I)
STOP = {"molecule", "compound", "conjugate", "member", "role", "hydride",
        "it", "a", "an", "the", "of", "and", "is", "that", "which", "from"}
IONIC = [("ate", "ic acid"), ("ate", "ic"), ("ide", "ic acid")]


def norm(term):
    t = re.sub(r"^\s*(an?|the)\s+", "", term.strip().lower())
    return re.sub(r"\s+", " ", t).strip(" .,;")


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


def toks(s):
    return {w for w in re.split(r"[^a-z0-9()\[\]>-]+", s) if w and w not in STOP}


def ionic_equiv(a, b):
    for x, y in IONIC:
        for p, q in ((a, b), (b, a)):
            if p.endswith(x) and q.endswith(y) and p[:-len(x)].rstrip() == q[:-len(y)].rstrip():
                return True
    return False


def match(a, b, mode):
    if a == b:
        return True
    if mode == "exact":
        return False
    if ionic_equiv(a, b):
        return True
    if a.startswith(b) or b.startswith(a):
        return True
    if len(a) > 12 and a in b:
        return True
    if len(b) > 12 and b in a:
        return True
    if mode == "jaccard":
        A, B = toks(a), toks(b)
        if A and B and len(A & B) / len(A | B) >= JACCARD_MIN:
            return True
    return False


def n_matches(pred_text, ref_text, mode):
    """How many of the reference's class terms the prediction also asserts."""
    P, R = classes(pred_text), classes(ref_text)
    used, hit = set(), 0
    for a in P:
        for b in R:
            if b not in used and match(a, b, mode):
                used.add(b)
                hit += 1
                break
    return hit, len(P), len(R)


def prf(preds, refs, mode):
    tp = fp = fn = 0
    for p, r in zip(preds, refs):
        hit, np_, nr = n_matches(p, r, mode)
        tp += hit
        fp += np_ - hit
        fn += nr - hit
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec * 100, rec * 100, f1 * 100


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "deliverables", "results"))
    # Analyse a differently-sized sweep, e.g. --suffix _full for the 3300
    # molecule runs. Mixing sizes in one table is not meaningful, so the two
    # sweeps are analysed separately rather than merged.
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    sfx = args.suffix
    os.makedirs(args.out, exist_ok=True)

    base_p = find("baseline" + sfx) or find("chebi_evaluation")
    if not base_p:
        sys.exit("no baseline predictions found yet - is the run still going?")
    base = load(base_p)
    n = len(base)
    print("baseline predictions: %s (n=%d)" % (base_p, n))

    tgt = [d["target"] for d in base]
    nrm = [d["prediction"] for d in base]

    def rot(i):
        """Index whose graph molecule i receives under graphs[1:]+graphs[:1]."""
        k, off = divmod(i, BATCH)
        return k * BATCH + (off + 1) % min(BATCH, n - k * BATCH)

    def rot_rev(i):
        k, off = divmod(i, BATCH)
        size = min(BATCH, n - k * BATCH)
        return k * BATCH + (off - 1) % size

    random.seed(SEED)
    perm = list(range(n))
    random.shuffle(perm)

    ident = list(range(n))
    rot_i = [rot(i) for i in range(n)]
    rot_r = [rot_rev(i) for i in range(n)]

    # Which molecule supplied each channel, per condition. This MUST be stated
    # per condition: shuffle_graph rotates the graph and leaves the SMILES
    # alone, shuffle_smiles does the reverse. Treating "own target" as
    # "the SMILES molecule" in both cases inverts the mirror control.
    #                       graph came from | SMILES came from
    CHANNEL_SOURCE = {
        "normal":            (ident, ident),
        # prompt-order test: soft prompts moved BEFORE the SMILES span
        "graphfirst-baseline": (ident, ident),
        "graphfirst-shuffled-graph": (rot_i, ident),
        "shuffled-graph":    (rot_i, ident),
        "shuffled-graph-rev": (rot_r, ident),
        "shuffled-SMILES":   (ident, rot_i),
        "shuffled-graph+no-SMILES": (rot_i, None),
        "graph-only":        (ident, None),    # SMILES withheld
        "rewired-graph":     (None, ident),    # graph corrupted, no referent
        "null-graph":        (None, ident),
    }

    refs = {
        "own target": tgt,
        "graph's target": [tgt[j] for j in rot_i],
        "random target (floor)": [tgt[perm[i]] for i in range(n)],
    }

    # which other conditions finished
    runs = {"normal": nrm}
    for name, key in (("shuffled-graph", "shuffle_graph"),
                      ("graphfirst-baseline", "baseline_graphfirst"),
                      ("graphfirst-shuffled-graph", "shuffle_graph_graphfirst"),
                      ("shuffled-graph-rev", "shuffle_graph_rev"),
                      ("shuffled-SMILES", "shuffle_smiles"),
                      ("shuffled-graph+no-SMILES", "shuffle_graph_only"),
                      ("rewired-graph", "rewire_graph"),
                      ("graph-only", "graph_only"),
                      ("null-graph", "null_graph")):
        p = find(key + sfx)
        if not p:
            continue
        d = load(p)
        if len(d) != n:
            print("  skipping %s: %d rows, expected %d" % (key, len(d), n))
            continue
        if [x["target"] for x in d[:200]] != tgt[:200]:
            print("  skipping %s: example order differs from baseline" % key)
            continue
        runs[name] = [x["prediction"] for x in d]
        print("  loaded %-16s %s" % (name, p))


    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit()
    emit("=" * 78)
    emit("A. N-GRAM AGREEMENT (own recomputation; not comparable to the paper)")
    emit("=" * 78)

    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        from nltk.translate.meteor_score import meteor_score
        from rouge_score import rouge_scorer
    except ImportError as e:
        sys.exit("missing metric package: %s" % e)

    sm = SmoothingFunction().method1
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def bleu(refs_txt, hyps_txt, w):
        return corpus_bleu([[r.split()] for r in refs_txt],
                           [h.split() for h in hyps_txt],
                           weights=w, smoothing_function=sm) * 100

    def rougeL(refs_txt, hyps_txt):
        return 100 * sum(rs.score(r, h)["rougeL"].fmeasure
                         for r, h in zip(refs_txt, hyps_txt)) / len(hyps_txt)

    def meteor(refs_txt, hyps_txt):
        return 100 * sum(meteor_score([r.split()], h.split())
                         for r, h in zip(refs_txt, hyps_txt)) / len(hyps_txt)

    metrics = [("BLEU-2", lambda r, h: bleu(r, h, (.5, .5))),
               ("BLEU-4", lambda r, h: bleu(r, h, (.25,) * 4)),
               ("ROUGE-L", rougeL),
               ("METEOR", meteor)]

    floor_ref = [tgt[perm[i]] for i in range(n)]

    def condition_refs(rname):
        """Per condition: the molecule that supplied the graph, and the one
        that supplied the SMILES. These are NOT the same mapping for every
        condition - shuffle_graph rotates +1, shuffle_graph_rev rotates -1,
        shuffle_smiles rotates the text instead. A single fixed 'graph's
        target' column silently scores some conditions against the wrong
        molecule."""
        g_src, s_src = CHANNEL_SOURCE.get(rname, (None, None))
        g = [tgt[j] for j in g_src] if g_src is not None else None
        s = [tgt[j] for j in s_src] if s_src is not None else None
        return g, s

    ngram = {}
    hdr = "%22s  %22s  %22s" % ("graph's molecule", "SMILES' molecule", "random (floor)")
    for mname, fn in metrics:
        emit()
        emit("%-26s %s" % (mname, hdr))
        emit("-" * 96)
        for rname, hyps in runs.items():
            g, s = condition_refs(rname)
            vals = []
            for ref in (g, s, floor_ref):
                vals.append(fn(ref, hyps) if ref is not None else None)
            ngram[(mname, rname)] = {"graph": vals[0], "smiles": vals[1], "floor": vals[2]}
            cells = "  ".join("%22.2f" % v if v is not None else "%22s" % "n/a"
                              for v in vals)
            emit("%-26s %s" % (rname, cells))
    emit()
    emit("n/a = that channel had no referent molecule: the graph was corrupted")
    emit("      rather than substituted (rewire, null), or the SMILES was withheld.")

    emit()
    emit("=" * 78)
    emit("B. CHEMICAL-CLASS AGREEMENT (F1; floor is near zero, unlike BLEU)")
    emit("=" * 78)
    klass = {}
    for mode in ("exact", "contain", "jaccard"):
        emit()
        emit("matching = %-15s %s" % (mode, hdr))
        emit("-" * 96)
        for rname, hyps in runs.items():
            g, s = condition_refs(rname)
            vals = [prf(hyps, ref, mode)[2] if ref is not None else None
                    for ref in (g, s, floor_ref)]
            klass[(mode, rname)] = {"graph": vals[0], "smiles": vals[1], "floor": vals[2]}
            cells = "  ".join("%22.2f" % v if v is not None else "%22s" % "n/a"
                              for v in vals)
            emit("%-26s %s" % (rname, cells))

    # ------------------------------------------------- per-example verdicts
    emit()
    emit("=" * 78)
    emit("C. PER-EXAMPLE VERDICTS (what the micro-averaged F1 above hides)")
    emit("=" * 78)
    emit("For each molecule: does the caption assert the chemical class of the")
    emit("molecule that supplied its GRAPH, or of the one that supplied its")
    emit("SMILES? Only conditions where those two differ can be scored, and")
    emit("which one was rotated flips between conditions - shuffle_graph moves")
    emit("the graph, shuffle_smiles moves the SMILES.")

    per_example = {}
    for rname, hyps in runs.items():
        src = CHANNEL_SOURCE.get(rname)
        if not src:
            continue
        g_src, s_src = src
        if g_src is None or s_src is None or g_src == s_src:
            emit()
            emit("%s: not a conflict condition (the two channels do not name"
                 " two different molecules); skipped" % rname)
            continue
        verdicts = Counter()
        for i in range(n):
            g_hit = n_matches(hyps[i], tgt[g_src[i]], "jaccard")[0]
            s_hit = n_matches(hyps[i], tgt[s_src[i]], "jaccard")[0]
            if g_hit > s_hit:
                v = "follows GRAPH"
            elif s_hit > g_hit:
                v = "follows SMILES"
            elif g_hit == 0:
                v = "neither"
            else:
                v = "both"
            verdicts[v] += 1
        per_example[rname] = verdicts
        emit()
        emit("%s (n=%d)   [graph rotated: %s | SMILES rotated: %s]"
             % (rname, n, g_src != ident, s_src != ident))
        for v in ("follows GRAPH", "follows SMILES", "both", "neither"):
            c = verdicts[v]
            emit("    %-16s %5d  %5.1f%%" % (v, c, 100.0 * c / n))
        decisive = verdicts["follows GRAPH"] + verdicts["follows SMILES"]
        if decisive:
            emit("    among the %d decisive cases: %.1f%% follow the GRAPH"
                 % (decisive, 100.0 * verdicts["follows GRAPH"] / decisive))

    # -------------------------------------------------------- worked example
    if "shuffled-graph" in runs:
        emit()
        emit("=" * 78)
        emit("D. WORKED EXAMPLE")
        emit("=" * 78)
        for i in range(n):
            if n_matches(runs["shuffled-graph"][i], refs["graph's target"][i], "jaccard")[0] > 0 \
               and n_matches(runs["shuffled-graph"][i], refs["own target"][i], "jaccard")[0] == 0:
                emit("index %d (graph came from index %d)" % (i, rot(i)))
                emit("  its own SMILES describes : %s" % refs["own target"][i][:150])
                emit("  the graph it was given   : %s" % refs["graph's target"][i][:150])
                emit("  the model wrote          : %s" % runs["shuffled-graph"][i][:150])
                break

    with open(os.path.join(args.out, "channel_conflict%s.txt" % sfx), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "channel_conflict%s.json" % sfx), "w", encoding="utf-8") as fh:
        json.dump({"n": n,
                   "runs": sorted(runs),
                   "ngram": {"%s|%s" % k: v for k, v in ngram.items()},
                   "class_f1": {"%s|%s" % k: v for k, v in klass.items()},
                   "per_example": {k: dict(v) for k, v in per_example.items()}},
                  fh, indent=2)
    print()
    print("wrote %s/channel_conflict%s.{txt,json}" % (args.out, sfx))


if __name__ == "__main__":
    main()
