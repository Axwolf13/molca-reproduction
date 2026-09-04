# Results

Akshay Ashok (7071170) · seminar: Machine Learning for Language Processing · 31 Aug 2026

All numbers come from MolCA's **released** CheBI-20 checkpoint. Inference only:
no training, no gradient updates. Twelve runs across seven distinct
manipulations, plus a full-test-set reproduction.

Local configuration: RTX 4060 Laptop 8 GB, Python 3.13.2, torch 2.13.0+cu126,
transformers 4.46.3, fp16, 5-beam search, batch 4. A parallel set of runs on a
Tesla P100 (Python 3.10, torch 2.3.1, transformers 4.44.2, fp16, batch 8) is
referenced where available; those artifacts live under `../cluster/`. Both
machines ran fp16, the P100 having rejected bfloat16 outright.

---

## 1. Reproduction

| | n | BLEU-2 |
|---|---|---|
| Paper, Table 2b | 3300 | 62.0 |
| Tesla P100 (cluster) | 3300 | 62.32 |
| **RTX 4060 (local)** | **3300** | **62.77** |
| RTX 4060 (local) | 1000 | 62.65 |

Three independent measurements of the same checkpoint on the identical test
split span **0.77 BLEU-2**, across two GPU generations (Pascal → Ada), two
Python versions, two torch majors, two transformers versions, and a LAVIS that
is pip-installed on one machine and hand-vendored on the other.

The 1000-molecule subset used for the condition matrix lands within **0.12** of
the full split, so the subset is not biased with respect to the full test set.

### Agreement goes well past the aggregate score

Two runs could reach the same corpus BLEU through different captions, which
would make a matching number weak evidence. Comparing the predictions themselves
molecule by molecule, via
[`../cross_stack_agreement.py`](../cross_stack_agreement.py):

| | Character-identical | Above 0.95 similarity |
|---|---|---|
| baseline, 3300 molecules | **93.8%** | 96.1% |

Between a Pascal card and an Ada card, on different Python majors, different
torch majors, and a LAVIS installed one way and vendored the other, fourteen
captions in fifteen come out byte for byte the same.

`shuffle_graph` agrees on only 68.5% of captions. That reason is mechanical
rather than worrying. Because rotation happens **within a batch**, the laptop's
batch of 4 and the cluster's batch of 8 hand a quarter of the molecules a
different wrong graph:

| Molecules | Character-identical |
|---|---|
| Given the same wrong graph by both runs (2476) | 91.3% |
| Given a different wrong graph (824) | 0.0% |

The split is exactly clean, carrying a robustness result worth more than the
agreement figure itself. A quarter of the test set was paired with an entirely
different wrong molecule, yet the aggregate moved by 0.07 BLEU-2 (25.62 against
25.55). **The channel effect does not depend on which particular wrong graph is
substituted.**

## 2. Condition ladder (pipeline BLEU-2, MolCA's own scorer)

The subset conditions all see the **same first 1000 molecules** in the same
order, so they are exactly comparable to one another.

| condition | n | BLEU-2 | Δ | manipulation |
|---|---|---|---|---|
| baseline | 3300 | 62.77 | n/a | unmodified |
| baseline | 1000 | 62.65 | n/a | unmodified |
| shuffle_smiles | 1000 | 48.05 | −14.60 | own graph, next molecule's SMILES |
| null_graph | 1000 | 28.19 | −34.46 | graph present, atom features zeroed |
| shuffle_graph_rev | 1000 | 26.10 | −36.55 | own SMILES, previous molecule's graph |
| shuffle_graph | 1000 | 25.71 | −36.94 | own SMILES, next molecule's graph |
| **shuffle_graph** | **3300** | **25.55** | **−37.22** | same, full split |
| rewire_graph | 1000 | 20.46 | −42.19 | atoms kept, edge_index resampled |
| graph_only | 1000 | 2.42 | −60.23 | SMILES withheld |
| shuffle_graph_only | 1000 | 2.23 | −60.42 | wrong graph AND no SMILES |

The channel effect replicates to **0.07 BLEU-2** between the two machines on the
full split (25.62 cluster / 25.55 local), and to 0.16 against the subset.

## 3. Central result: which channel does the model follow?

When the graph and the SMILES name **different molecules**, score the caption
against both candidates. This is an own recomputation with different
tokenisation and smoothing from the pipeline scorer, so it is comparable only
within this section. The floor is a random pairing: ChEBI captions share heavy
boilerplate, which puts unrelated captions near 26 BLEU-2 of each other.

| condition | vs GRAPH's molecule | vs SMILES' molecule | floor |
|---|---|---|---|
| shuffle_graph | **45.91** | 28.80 | 26.16 |
| shuffle_graph_rev | **47.14** | 29.32 | 25.91 |
| shuffle_smiles | **47.15** | 29.34 | 25.82 |

Chemical-class F1 on ChEBI ontology terms. Its floor is ≈0.3, so unlike BLEU
there is no boilerplate to hide behind:

| condition | vs GRAPH | vs SMILES | ratio |
|---|---|---|---|
| shuffle_graph | **17.92** | 2.18 | 8.2× |
| shuffle_graph_rev | **17.86** | 1.89 | 9.5× |
| shuffle_smiles | **17.95** | 1.89 | 9.5× |

Stable under looser matching (contain ≈23.4 vs ≈2.9; jaccard ≈27.3 vs ≈3.9), so
the effect is not an artifact of the matching threshold.

Per example, counting only cases that decisively matched one candidate:

| condition | decisive | follow the GRAPH |
|---|---|---|
| shuffle_graph | 528 | **88.6%** |
| shuffle_graph_rev | 510 | **90.8%** |
| shuffle_smiles | 511 | **90.8%** |

**The model describes the molecule whose graph it holds ~90% of the time, even
with a contradicting SMILES string present in its prompt.**

### Worked example (test index 0)

```
its own SMILES describes : a steroid ester, methyl (17E)-pregna-4,17-dien-21-oate
the graph it was given   : a branched amino tetrasaccharide
the model wrote          : "The molecule is a branched amino tetrasaccharide ..."
```

## 4. Corrupting the graph costs 2.5× what corrupting the text costs

Matched manipulation, same magnitude applied to each channel:

- substituting the **graph** costs **36.9** BLEU-2
- substituting the **SMILES** costs **14.6**

## 5. The model trusts the graph rather than hedging

| graph condition | BLEU-2 |
|---|---|
| uninformative (features zeroed) | 28.19 |
| wrong (a different real molecule) | 25.71 |
| incoherent (random edges) | 20.46 |

A **wrong** graph hurts more than **no** graph, and an incoherent one hurts
most. That first gap is the narrow one, at 1.78 BLEU-2 on the full split. It
survives resampling: 95% CI [+1.37, +2.30], §10. A model treating the graph as a weak side-channel would be no worse off
with garbage than with nothing. Under `rewire_graph` the model also does not
fall back on the SMILES: 23.27 against its own molecule, barely above the 21.10
floor.

## 6. MolCA is extremely brittle to prompt layout

The template appends the graph soft prompts *after* the SMILES span
(`\1\3\4%s`). Moving them in front, with identical tokens and identical
content verified as the same character multiset, collapses the model:

| | BLEU-2 | ROUGE-L | METEOR | mean length | empty |
|---|---|---|---|---|---|
| default order | 63.16 | 63.10 | 66.04 | 42.9 words | 0 |
| graph prompts first | **0.01** | 0.05 | 0.09 | 123.0 words | 55 |

(Both rows come from the same recomputation, which tokenises with
`bert-base-uncased`, hence 63.16 against the pipeline's 62.65. MolCA's scorer
actually calls `init_tokenizer()`, returning `allenai/scibert_scivocab_uncased`.
Matching that vocabulary closes the gap exactly, as §10 shows. The comparison
inside this table is unaffected, since both rows use the same tokeniser.)
Output degenerates into repeated tokens:
`"trans trans trans trans de trans ..."`, `"p p p p p p ..."`.

**This experiment failed as an instrument.** It was designed to separate
modality from position. Because the graph prompts sit nearest the generation
point, "the graph dominates" stays confounded with "the nearest channel
dominates".
Because reordering alone destroys generation, it cannot separate them.

The confound stayed open until §12 attacked it from the other side, holding the
order fixed and varying the distance instead. Reordering was the wrong lever, not
the wrong question.

## 7. Relation to the paper's own ablation

Table 5a reports what each view contributes during **training**: 34.6
SMILES-only, 34.5 graph-only, 38.7 both (PubChem324k). Read alone that suggests
two interchangeable views combining for a modest gain.

At **inference** the trained model is not even-handed: it follows the graph ~90%
of the time and overrides an explicit contradicting SMILES string. The two
measurements are compatible but tell different stories, and only the first
appears in the paper.

## 8. Retrieval (cluster only)

MolCA's second task evaluates molecule-text retrieval from `stage1.ckpt`.
Reaching it at all requires fixing two bugs, one of which
([issue #13](https://github.com/acharkq/MolCA/issues/13)) has been open upstream
since July 2024 with its reporter unable to recover the paper's accuracy. See
the README for how the two fixes differ. It never ran locally: two re-ranking passes per split at 13.4 s
per iteration works out to roughly 11 GPU-hours. On the cluster both splits
completed. Raw Lightning output, box drawing intact, sits in
[`../cluster/results/results_retrieval.txt`](../cluster/results/results_retrieval.txt).

Full test set, contrastive scoring alone. This is the paper's `MolCA w/o MTM`
row, Tables 7b and 7c:

| Split | Direction | Acc | Acc (paper) | R@20 | R@20 (paper) |
|---|---|---|---|---|---|
| PCDes | graph → text | 37.69 | 37.7 | 80.59 | 80.6 |
| PCDes | text → graph | 35.36 | 35.3 | 76.55 | 76.5 |
| MoMu | graph → text | 22.47 | 22.5 | 68.45 | 68.5 |
| MoMu | text → graph | 21.14 | 21.1 | 64.76 | 64.8 |

After re-ranking the top-128 candidates with the molecule-text matching head,
which is the paper's plain `MolCA` row:

| Split | Direction | Acc | Acc (paper) | R@20 | R@20 (paper) |
|---|---|---|---|---|---|
| PCDes | graph → text | 48.20 | 48.1 | 85.56 | 85.6 |
| PCDes | text → graph | 45.96 | 46.0 | 82.22 | 82.3 |
| MoMu | graph → text | 30.55 | 30.6 | 76.77 | 76.8 |
| MoMu | text → graph | 29.68 | 29.8 | 73.32 | 73.3 |

All sixteen full-test-set metrics land within **0.12**. Adding the sixteen
in-batch metrics brings the comparison to **32 published values with no
exceptions**, at a maximum deviation of 0.49.
[`../cluster/verify_retrieval.py`](../cluster/verify_retrieval.py) recomputes
the whole diff from the raw Condor logs.

The in-batch columns are the looser of the two, predictably so. In-batch
retrieval ranks each query only against the molecules sharing its batch, and
`Stage1KVPLMDM` builds those batches from a `.shuffle()` seeded by
`pl.seed_everything(42)`. Since a seeded permutation is not stable across torch
versions, the two machines group different molecules together even at the same
`--match_batch_size 64`. Full-test-set retrieval ranks against every candidate
and carries no such dependence, which is why I treat those columns as the
reproduction and the in-batch ones as corroboration.

Re-ranking lifts PCDes graph-to-text accuracy by **10.51** points (37.69 →
48.20) and text-to-graph by **10.60**. Reading the same two rows out of Table 7b
gives 10.4 and 10.7, so the mechanism's contribution reproduces as closely as
the endpoints do.

One sanity check the logs happen to supply: every `val_*` row is identical
between the PCDes job and the MoMu job. Since `--use_phy_eval` swaps only the
test split, that identity is what correct runs produce. Divergence there would
have meant state leaking between jobs.

## 9. What the cluster adds beyond the local runs

Every cluster condition ran the **full 3300-molecule split**, which extends two
axes the laptop could only sample:

| Sweep | Condition | BLEU-2 |
|---|---|---|
| Beam width | 1 beam | 61.53 |
| | 2 beams | 62.16 |
| | 3 beams | 62.29 |
| | 4 beams | 62.28 |
| | 5 beams (default) | 62.32 |
| Rewiring fraction | 25% of edges resampled | 23.41 |
| | 50% | 20.19 |
| | 100% | 20.18 |

Buying 0.79 BLEU-2 in total and saturating by three beams, beam width is not
what carries the reproduction.

The rewiring sweep is the more interesting one. Damage saturates at half the
edges: resampling every edge costs no more than resampling half of them. The
graph channel therefore degrades sharply rather than gracefully, which fits the
trust ladder in §5.

A further cluster condition has no local counterpart: `null_shufsmiles` (21.01)
zeroes the graph features **and** rotates the SMILES, stripping both channels of
correct information at once. Landing below `null_graph` (27.40) while staying
above `rewire_graph` (20.18) places it where the ladder predicts.

## 10. Error bars

Generation here is deterministic, which leaves one source of variation worth
measuring: the test set is a sample of molecules, and a different 3300 would
give a different number. A paired bootstrap over those molecules quantifies
exactly that. [`../bootstrap.py`](../bootstrap.py) resamples the split 2000
times, applying the **same index draw to every condition** so each contrast
stays paired.

The script reimplements `caption_evaluate` rather than approximating it: SciBERT
tokenisation, special tokens stripped, nltk `corpus_bleu` at weights (.5, .5)
with no smoothing. Its point estimates reproduce all five pipeline scores to the
last reported decimal (62.32, 47.41, 27.40, 25.62, 20.18), so the intervals below
are on MolCA's own metric rather than on a proxy for it.

| Condition | BLEU-2 | 95% CI |
|---|---|---|
| baseline | 62.32 | [61.45, 63.18] |
| shuffle_smiles | 47.41 | [46.54, 47.98] |
| null_graph | 27.40 | [27.01, 27.82] |
| shuffle_graph | 25.62 | [25.22, 25.94] |
| rewire_graph | 20.18 | [19.71, 20.65] |

The contrasts matter more than the levels, because a paired design cancels the
shared difficulty of whichever molecules a resample happens to draw:

| Contrast | Observed | 95% CI | Claim it tests |
|---|---|---|---|
| Substituting the SMILES | +14.91 | [+14.32, +15.91] | the text channel carries signal |
| Substituting the graph | +36.70 | [+35.91, +37.60] | the graph channel carries signal |
| Graph cost over SMILES cost | +21.79 | [+20.87, +22.50] | **the graph matters more** |
| No graph over wrong graph | +1.78 | [+1.37, +2.30] | **a wrong graph beats no graph** |
| Wrong graph over incoherent | +5.45 | [+4.86, +5.89] | incoherence is worse still |

Every interval excludes zero. The one I expected to be fragile is the fourth:
1.78 BLEU-2 is a small gap, and the whole trust-ladder argument in §5 rests on
it. Its interval clears zero by a comfortable margin, so the ordering holds.

Raw output is in [`bootstrap.txt`](bootstrap.txt). Two caveats on scope. The
bootstrap runs on the cluster's full-split predictions, since the laptop reached
3300 molecules for only two conditions. And it measures test-set sampling alone,
not training variance; see Limitation 6.

---

## 11. Cross-dataset transfer, and the contamination underneath it

Three cluster jobs (`185367` to `185369`) evaluate the ChEBI-20 checkpoint on
the PubChem324kV2 test split, a pairing the authors never ran. Every other
result in this document lives on ChEBI-20, so this is the only place the channel
conflict meets a second caption distribution.

The headline is misleading and worth dismantling before anything else. The run
scores **49.04 BLEU-2** where the paper's Table 2a reports **38.7** for a model
fine-tuned on PubChem324k itself. Transfer appearing to beat in-domain training
is exactly the kind of result that deserves suspicion.

### Half the test set is contaminated

Job `186489` dumped rdkit-canonical SMILES for both datasets, letting
[`../transfer_overlap.py`](../transfer_overlap.py) match on structure rather than
on caption text. **985 of the 2000 test molecules (49.25%)** are structurally
present in ChEBI-20's training split, the split `chebi.ckpt` was fine-tuned on
for 100 epochs.

An earlier pass matched captions instead and found 467. The two sets nest
exactly: every caption match is also a structure match, and 518 molecules are the
same compound described in different words across the two datasets. Caption
matching was therefore conservative in precisely the way claimed, and it
understated the problem by a factor of two.

No filter the authors built could have caught this, and the reason is directional
rather than an oversight. Section 4.1 excludes molecules from the **valid/test
sets** of ChEBI-20, PCDes and MoMu, which protects their ChEBI-20 evaluation
against leakage out of PubChem324k pretraining. Our run points the other way, and
the overlap lands on ChEBI-20's **train** split:

| Overlap falls in | Count | Could the paper's filter have caught it? |
|---|---:|---|
| ChEBI-20 train only | 985 | No. The filter never targets train |
| ChEBI-20 valid or test only | 239 | Only if applied to the downstream subset too |
| Both | 0 | ChEBI-20's own splits are structurally disjoint |

Even a version of the filter extended to the high-quality 15k subset would have
removed 239 of the 1224 matches, leaving all 985 train overlaps intact. This is
contamination in a direction nobody had reason to guard against, because until
now nobody had pointed a ChEBI-20 checkpoint at PubChem324k.

| Subset | Molecules | BLEU-2 | 95% CI |
|---|---:|---:|---|
| Whole test split | 2000 | 49.04 | |
| Structure seen in ChEBI-20 train | 985 | **63.75** | [59.09, 68.77] |
| Structure not seen | 1015 | **28.45** | [26.45, 30.46] |
| Gap | | +35.31 | [+30.43, +40.66] |

The suspicion was right and the correction is larger than the first pass
suggested. Removing the contaminated half leaves **28.45**, which sits **10.25
BLEU-2 below** the paper's 38.7 rather than level with it.

Splitting the contaminated half by how it was matched gives a clean gradient, and
it is the most direct evidence in this study of what memorisation buys:

| What the checkpoint saw in training | Molecules | BLEU-2 |
|---|---:|---:|
| This structure **and** this caption | 467 | **94.33** |
| This structure, worded differently | 518 | **45.63** |
| Neither | 1015 | **28.45** |

Seeing the exact caption is worth 66 BLEU-2 over seeing nothing. Seeing only the
molecule, with the description rewritten, is worth 17. The middle row is the
interesting one, because it is neither recall nor clean generalisation: the model
has learned this compound's chemistry and is being asked to describe it in
another dataset's idiom.

An earlier draft of this section, working from caption matching, reported 38.86
and concluded that transfer performed on par with in-domain training. That
conclusion was an artefact of an undercounted overlap and does not survive
structural matching. Transfer from ChEBI-20 to PubChem324k is **materially worse**
than training on PubChem324k, which is the unsurprising result. What is worth
keeping is how thoroughly contamination disguised it: the raw 49.04 read as a
27% improvement over the paper, when the clean subset is a 27% shortfall.

That comparison still carries an unresolved assumption, since PubChem324kV2 was
released in January 2024, after both the preprint and the conference, and the v1 dataset the
paper used is no longer downloadable. [`../NOTES.md`](../NOTES.md) documents what
I could and could not establish there.

### The conflict result does not depend on any of this

| Subset | Normal | Neighbour's graph | Drop |
|---|---:|---:|---:|
| Whole split | 49.04 | 18.63 | 30.41 |
| Seen in ChEBI-20 train | 63.75 | 20.72 | 43.04 |
| Not seen | 28.45 | 15.61 | 12.84 |

Substituting the neighbouring molecule's graph costs 12.84 BLEU-2 on the 1015
molecules the checkpoint provably never trained on, roughly halving the score.
Withholding the SMILES entirely (`185369`) collapses the run to 2.41e-154,
reproducing on PubChem the same degenerate floor §3 found on ChEBI-20. The drop
is smaller here than the 22.83 the caption-matched split suggested, which follows
from the cleaner subset starting 10 points lower. Since the finding replicates
across datasets, across caption styles, and on uncontaminated data, neither the
overlap nor the split question touches it.

---

## 12. Distance: the recency confound, measured

§6 tried to separate modality from position by reordering the prompt, and
reordering alone collapsed generation. Jobs `186488` and `186523` tried the other
lever, holding the order fixed and inserting neutral filler
(`It is a chemical entity. `, 37 Galactica tokens at k=6) to change how far each
channel sits from the generation point.

| Job | Filler position | SMILES distance | Graph distance | BLEU-2 |
|---|---|---|---|---:|
| `183288` | none | near | nearest | 62.32 |
| `186523` | between SMILES and soft prompts | **+37 tokens** | nearest | **53.60** |
| `186488` | after the whole prompt | +37 tokens | **+37 tokens** | **37.26** |

Read as a decomposition, moving the SMILES away costs **8.72** and moving the
graph away as well costs a further **16.34**. Taken at face value that says the
graph is roughly twice as sensitive to distance as the SMILES, which would fit
§3's picture of the graph as the channel the model actually leans on.

**Do not take it at face value.** The two rows differ in a second respect. In
`186523` the tokens immediately preceding generation are the soft prompts and the
period, exactly the pattern the checkpoint was fine-tuned on. In `186488` they are
filler text. Given how brittle §6 showed this checkpoint to be about its template,
the 16.34 confounds "the graph moved" with "the template ending changed", and
generation quality confirms something broke:

| Condition | Mean words | Empty outputs |
|---|---:|---:|
| Baseline | 42.4 | 0 |
| `186523` filler between | 37.3 | 0 |
| `186488` filler at end | 26.0 | 54 |

So the 8.72 is a clean measurement and the 16.34 is not.

### What the gate actually established

`186488` was designed as the gate, on the reasoning that filler moving both
channels equally leaves the relative geometry untouched. That reasoning was
wrong: appending filler necessarily displaces the soft prompts from the
generation point, so the "neutral" control was itself a manipulation.

The useful gate turned out to be the other job. `186523` preserves the template
ending, keeps every output non-empty, and retains 86% of baseline BLEU-2 while
pushing the SMILES 37 tokens away. That is a working instrument, and it was
never used, because the four conditions that would have used it were cancelled
when the gate appeared to fail.

### Using it: the confound is real, and it is small

Jobs `186610` and `186611` corrupt each channel at the increased distance, scored
against `186523`. Because the filler lowers the clean baseline from 62.32 to
53.60, the costs are reported as a share of each condition's own baseline rather
than in absolute points. All four conditions ran on the same 3300 molecules in
the same order, so [`../distance_test.py`](../distance_test.py) bootstraps the
contrasts paired.

| SMILES position | Clean | Cost of corrupting the SMILES | Cost of corrupting the graph |
|---|---:|---:|---:|
| Adjacent, as shipped | 62.32 | 23.9% | 58.9% |
| Pushed 37 tokens away | 53.60 | **21.3%** | **58.3%** |
| Change | | −2.6 pp [−3.9, −1.9] | −0.5 pp [−1.1, −0.1] |

Both intervals exclude zero, so recency is real rather than absent. Moving the
SMILES 37 tokens from the generation point does reduce what corrupting it costs.

It is nowhere near large enough to be the mechanism. The reduction is 2.6 points
of a 23.9% effect, roughly a tenth of the smaller channel, while the graph's own
cost barely moves. The ratio between the two channels goes from **2.46× to
2.74×**: displacing the SMILES that far widened the gap slightly, in the
direction recency predicts, and left the graph just as dominant. A recency
account would have to produce a 2.5× dominance out of an effect worth a tenth of
the weaker channel.

**§3's confound is therefore closed rather than merely acknowledged.** It exists,
it is now measured, and it does not carry the result. That is a stronger position
than the reordering test in §6 could ever have reached, since that test destroyed
the model instead of moving one variable.

### The contamination control returned nothing

Job `186483` evaluated `stage2.ckpt`, the one PubChem-native checkpoint in the
release, on the PubChem324kV2 test split. Since it has never seen ChEBI-20's
train split, its score on the 985 contaminated molecules should have matched its
score on the other 1015, turning §11's contamination diagnosis into a direct
measurement.

It scored **0.18 BLEU-2, with 1605 of 2000 outputs empty.** `stage2.ckpt` is the
pre-fine-tune model, and without the fine-tune it does not produce captions at
all. The control is void rather than negative: it measures nothing about
contamination, only that stage-2 pretraining alone does not yield a captioner.
That is consistent with the paper, where every row of Table 8 ablates the
pretrain stages **after** fine-tuning and no un-fine-tuned score is reported.

---

## 13. The paper's own contamination filter holds exactly

§11 found half the PubChem324kV2 test split sitting in ChEBI-20's training data,
which raises an obvious question about the direction that matters more. Every
headline number in this document comes from `chebi.ckpt` evaluated on ChEBI-20's
test split, and that checkpoint was pretrained on PubChem324k's pretrain subset.
Section 4.1 says that subset was filtered to exclude ChEBI-20's valid and test
molecules. If it leaked, the 62.32 in §1 is partly recall rather than captioning.

Job `186609` checked, on the released V2 artefact, matching canonical SMILES:

| PubChem324kV2 split | ChEBI-20 split | Shared structures | |
|---|---|---:|---|
| pretrain (298,010 distinct) | validation | **0** | filter holds |
| pretrain | test | **0** | filter holds |
| pretrain | train | 16,777 | never claimed to be filtered |
| train (12,000) | test | 778 | never claimed to be filtered |
| test (2,000) | train | 985 | §11's contamination |

**Zero out of 6601.** Not a small number, an exact one, across 298,010 pretrain
structures. The filter the paper describes is present in the released dataset and
works, so the reproduction in §1 rests on clean data and the study's own
foundation is sound.

Two things follow. The contamination in §11 is genuinely a gap in the filter's
*scope* rather than a failure of its implementation: the authors filtered exactly
what they said they filtered, and the direction they did not cover is the one
nobody had reason to run. And the 985 in the last row is reached here through a
different code path from §11's, loading `test.pt` directly rather than the dumped
identifiers, which makes it an independent confirmation of the 49.25% figure.

---

## Limitations

**1. `shuffle_smiles` is not an independent control.** Rotating the graph −1 and
rotating the SMILES +1 produce the same (graph of A, SMILES of B) pairing.
Verified: `shuffle_smiles[i] == shuffle_graph_rev[rot(i)]` for **983/1000**
predictions. It is a consistency check across two code paths, not additional
evidence. The eight conditions are seven distinct manipulations.

**2. Position is confounded with modality, and the confound is small.** Closed
rather than open, though not for free. Reordering the prompt (§6) destroyed the
model and settled nothing. Displacing the SMILES 37 tokens instead (§12) reduces
its contribution by 2.6 points of a 23.9% effect while leaving the graph's
unchanged, taking the ratio between the channels from 2.46× to 2.74×. Recency
exists and cannot account for the dominance. The residual caveat is that one
displacement at one distance is a two-point curve; `chebi_filler_mid2` would add
a third.

**3. The no-SMILES conditions are distribution shift, not information removal.**
`graph_only` (2.42) and `shuffle_graph_only` (2.23) degenerate into
non-terminating repeated tokens, and both ran ~2× longer because generation
never emits a stop token. They show brittleness off-distribution; they do not
show the SMILES channel carries more information than the graph.

**4. The chemical-class extractor has a ceiling.** It is regex-based over ChEBI's
stereotyped phrasing. On correct baseline captions it registers a match only
**76.3%** of the time, so part of the ~45% "neither" bucket is extractor failure
rather than model failure. Ratios between conditions are unaffected; absolute
rates are underestimates.

**5. §3 and §6 do not use MolCA's scorer.** Both tokenise with
`bert-base-uncased` where the pipeline uses SciBERT, and both apply smoothing
the pipeline omits, which is why the pipeline reports 25.71 for `shuffle_graph`
while that recomputation gives 28.80 on the identical file. Compare only within
a table. §10 is the exception: it reimplements `caption_evaluate` faithfully and
reproduces all five pipeline scores to the last decimal.

**6. Error bars cover test-set sampling only.** Generation is deterministic
here (`do_sample=False`, 5 beams), so re-running gives identical output. The
remaining source of variation is which molecules the test set contains, and §10
quantifies it with a paired bootstrap. What that does **not** cover is training
variance: a second checkpoint trained from a different seed could land elsewhere,
and measuring that would require retraining. The paper reports no interval of
either kind.

**7. Retrieval was measured on the cluster only.** Two real bugs were found and
patched, and the data path was verified end to end on both machines. Locally the
eval needs ~11 GPU-hours at the paper's settings (two re-ranking passes per
split, 13.4 s per iteration × 750 iterations × 2 splits) and did not complete.
The cluster ran it to completion; see §8.

**8. The transfer comparison rests on an unverified split.** Structural matching
has closed the measurement half of this: §11 now matches rdkit-canonical SMILES,
finds 49.25% contamination, and reports 28.45 on the clean subset. An earlier
draft matched captions, found 23.35%, and concluded the opposite about transfer.
The two overlap sets nest exactly, so the correction was one of coverage rather
than of method, but it is a reminder of how far a conservative proxy can be from
the quantity it stands in for.

What remains open is whether PubChem324kV2 preserves the split the paper's 38.7
was measured on. V2 postdates publication, the v1 dataset is gone from Hugging
Face, and the release documents no changelog, so the 28.45 against 38.7
comparison is approximate in a way no local work can fix.
[`../NOTES.md`](../NOTES.md) sets out what I could and could not establish.

**9. Two of the eight cluster experiments returned nothing usable.** §12 records
both. The `stage2.ckpt` control was void, since the pre-fine-tune model does not
caption at all. The first distance gate was confounded, because appending filler
displaces the soft prompts rather than leaving the geometry alone. Both are kept
in the repository rather than dropped, since a study that reports only the
experiments that worked is not reporting its method.
