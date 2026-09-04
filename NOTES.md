# Open Questions

Everything else in this repository is settled: a number was measured, checked against
the paper, and written down. This file is the opposite. It holds what I could not
close, kept here so that the next person to open the repository does not quietly
assume it was answered.

It began as two questions about one number. The second closed with job `186489`, and
closing it moved the number a long way. The first is still open and will stay open,
because the evidence needed to settle it no longer exists in public.

## The number in question

Job 185367 evaluates the **ChEBI-20 fine-tuned checkpoint** on the **PubChem324kV2
test split**, a pairing the authors never ran. It scores **49.04 BLEU-2**. Table 2a of
the paper reports **38.7** for MolCA Galac1.3B on PubChem324k.

The tempting reading is that a model trained on one dataset transfers to another and
beats the model trained in-domain. That reading needs two things to hold. One is now
settled and kills the reading outright; the other cannot be settled at all.

| | Question | Status |
|---|---|---|
| 1 | Is PubChem324kV2 the split the paper's 38.7 was measured on? | **Open.** The evidence leans against assuming it |
| 2 | How far do the ChEBI-20 training set and this test set overlap? | **Closed, and the answer is bad.** 49.25% structural overlap |

Question 2 turned out to be answerable, first from files already in the repository and
then properly, once job `186489` dumped canonical structures. It is recorded below as a
finding rather than a question. Question 1 remains genuinely open.

## Question 1: is PubChem324kV2 the paper's split?

Unresolved. What I could establish points at "probably not identical, and definitely not
verifiable from public artefacts."

### What checks out

Table 1 of the paper gives the split sizes as pretrain 298,083, train 12,000, valid
1,000 and test 2,000. Our run emitted exactly **2000 prediction rows**, so the V2 test
split is the same size as the paper's. Size agreement is consistent with the splits
being the same. It does not identify them, since any 2000-row resplit would agree too.

### What does not check out

Tracing the upstream repository's own history puts V2 firmly after publication:

| Commit | Date | What it did |
|---|---|---|
| `25625e5` | 2024-01-16 | Introduced `MoleculeCaptionV2` and the packed `.pt` loader |
| `24966d1` | 2024-01-17 | Repointed the README from `acharkq/PubChem324k` to `acharkq/PubChem324kV2` |

MolCA appeared at EMNLP in December 2023, with the arXiv preprint in October. Whatever
V2 is, it was assembled **three months after the preprint and one after the
conference**, which makes it a re-release rather than the artefact the numbers came
from.

Three further details make the original unrecoverable:

- The v1 dataset `acharkq/PubChem324k` is **gone from Hugging Face**. Querying the
  datasets API for that author returns only `PubChem324kV2` and `RNADataset`.
- The V2 card documents nothing. Its entire description is that this is "the second
  version of the PubChem324k Dataset used in the paper," with no split sizes, no
  changelog, and no statement about what changed.
- The v1 code path still exists in `data_provider/stage2_dm.py:162` but sits behind a
  hardcoded `if False:`. Loading the paper's directory-of-files layout now requires
  editing the source. The `--filtered_cid_path` argument that v1 used to exclude
  downstream test molecules is unreachable on the V2 path, so whatever filtering V2
  applies is baked into the `.pt` files and cannot be inspected from the loader.

### What would settle it

Nothing available locally. The `.pt` files carry a `smiles` field per row, so dumping
the 2000 test SMILES is easy, but there is no v1 release left to compare them against.
Closing this needs either an archived copy of `acharkq/PubChem324k` or a direct answer
from the authors.

**Until then, treat any comparison against 38.7 as approximate rather than exact.**

## Question 2: the overlap, which turned out to be the real problem

Run `python transfer_overlap.py` to reproduce this. It reads the prediction dumps
from `cluster/predictions/` and needs the ChEBI-20 split files, which the
repository does not ship. Point `--chebi` at wherever `train.txt`,
`validation.txt` and `test.txt` live; the default assumes the upstream clone sits
beside this directory at `../MolCA/data/ChEBI-20_data`. Saved output is in
`cluster/results/transfer_overlap_output.txt`.

Job `186489` dumped rdkit-canonical SMILES for both datasets, so the match is on
structure rather than on wording:

| PubChem324kV2 test molecule also appears in | Count | Share of 2000 |
|---|---:|---:|
| ChEBI-20 **train** | 985 | **49.25%** |
| ChEBI-20 validation | 120 | 6.00% |
| ChEBI-20 test | 119 | 5.95% |
| any ChEBI-20 split | 1224 | 61.20% |

The train row is the one that matters, because `chebi.ckpt` was fine-tuned on that
split for 100 epochs. **Half** the "transfer" test set is material the checkpoint
was trained on.

An earlier pass matched caption text and found 467 rather than 985. The two sets
nest exactly, with 518 molecules being the same compound worded differently across
the datasets, so caption matching was conservative exactly as claimed and
understated the contamination by a factor of two.

### No filter the authors built could have caught this

Section 4.1 describes the filtering:

> Additionally, we filter our pretrain subset to exclude molecules from the valid/test
> sets of other downstream datasets, including CheBI-20, PCDes, and MoMu datasets.

Two things are scoped there. The filter applies to the **pretrain subset**, leaving the
high-quality 15k subset that becomes train/valid/test untouched. More decisively, it
targets the **valid/test** splits of the downstream datasets, and our overlap does not
live there:

| Overlap falls in | Count | Reachable by the paper's filter? |
|---|---:|---|
| ChEBI-20 train only | 985 | No. The filter never targets train |
| ChEBI-20 valid or test only | 239 | Only if extended to the downstream subset |
| Both | 0 | ChEBI-20's own splits are structurally disjoint |

Extending the filter to cover the 15k subset would still have removed only 239 of
the 1224 matches, leaving all 985 train overlaps intact. The filter exists to
protect the authors' **ChEBI-20 evaluation** from leakage out of PubChem324k
pretraining. Our run is the reverse direction, ChEBI-20 training leaking into a
PubChem324k evaluation, which nobody had reason to guard against because nobody had
pointed a ChEBI-20 checkpoint at PubChem324k.

### Scoring the halves separately

| Subset | Molecules | BLEU-2 | 95% CI |
|---|---:|---:|---|
| Whole test split | 2000 | 49.04 | |
| Structure seen in ChEBI-20 train | 985 | **63.75** | [59.09, 68.77] |
| Structure not seen | 1015 | **28.45** | [26.45, 30.46] |
| Gap | | +35.31 | [+30.43, +40.66] |

Strip the contaminated half and the honest transfer number is **28.45**, which sits
**10.25 BLEU-2 below** the paper's 38.7.

This reverses what the caption-matched pass concluded. On 467 overlaps the clean
subset scored 38.86 and transfer looked level with in-domain training; on the true
985 it scores 28.45 and transfer is clearly worse, which is the unsurprising
outcome. What is worth keeping is the size of the disguise. The raw 49.04 reads as
27% above the paper, while the uncontaminated subset is 27% below it.

### Where this measurement is still weak

Structure matching is exact-match on canonical SMILES, so it counts a molecule as
unseen when the two datasets record different stereochemistry, salt forms, or
tautomers for what a chemist would call the same compound. All 2000 test SMILES
parsed, so there is no silent drop-out, but 49.25% remains a floor rather than a
ceiling. An InChIKey-prefix or scaffold-level match would put an upper bound on it.

The larger caveat is question 1. Comparing 28.45 to 38.7 assumes V2 preserved the
paper's split, and nothing local can establish that.

## What the contamination does not touch

The channel-conflict result, which is the reason the transfer run exists, holds on both
halves of the split:

| Subset | Normal | Neighbour's graph | Drop |
|---|---:|---:|---:|
| Whole split | 49.04 | 18.63 | 30.41 |
| Seen in ChEBI-20 train | 63.75 | 20.72 | 43.04 |
| Not seen | 28.45 | 15.61 | 12.84 |

Substituting the neighbouring molecule's graph roughly halves the score on the 1015
molecules the checkpoint never trained on. Since the finding replicates on a second
dataset, under a second caption distribution, on data the model has provably not seen,
contamination cannot explain it.

## Why there is no PubChem-native comparison run

The obvious control is a PubChem324k model on this same split, since it would be clean
on all 985 contaminated rows. It does not exist. Enumerating the release gives seven
files:

| File | What it is |
|---|---|
| `stage1.ckpt`, `archived/stage1.ckpt` | Stage-1 retrieval, 1679 bytes apart |
| `stage2.ckpt` | Stage-2 **pretrained**, LM frozen, PubChem324k pretrain subset |
| `archived/chebi.ckpt` + `chebi_lora/` | ChEBI-20 fine-tuned, what every run here uses |

Table 2a's 38.7 comes from a LoRA fine-tune on PubChem324k's train subset for 100
epochs, and those weights never shipped. `stage2.ckpt` is PubChem-native but sits before
the fine-tune, and since every row of Table 8 ablates the pretrain stages **after**
fine-tuning, the paper gives no score for it in that state.

This is the same wall that keeps MoleculeNet and IUPAC out of the study. Neither is a
gap I chose to leave: property prediction needs a trained classifier head and IUPAC
needs an IUPAC-tuned checkpoint, and the release contains neither. Transfer was the only
way to touch PubChem324k at all.

That control was tried anyway, as job `186483`, on the reasoning that `stage2.ckpt`
has never seen ChEBI-20's train split and so should score alike on both halves.

**It returned nothing usable: 0.18 BLEU-2, with 1605 of 2000 outputs empty.**
Without the fine-tune the model does not produce captions at all, so the result is
void rather than negative. It says nothing about contamination and only confirms
what the release inventory already implied, that stage-2 pretraining alone is not a
captioner. The contamination diagnosis in question 2 therefore rests on the
seen-versus-unseen split within `chebi.ckpt`'s own predictions, which is weaker than
an independent control would have been.

## For whoever picks this up

1. **Do not quote 49.04 as a transfer result.** Quote 28.45, and say what was removed.
2. ~~Redo the overlap on structures.~~ **Done**, job `186489`. 49.25%, and it moved the
   conclusion rather than confirming it.
3. Look for an archived copy of `acharkq/PubChem324k`, or ask the authors directly
   whether V2 preserved the split. Either closes question 1, and nothing else will.
4. ~~Run `stage2.ckpt` as the contamination control.~~ **Done and void**, job `186483`,
   0.18 BLEU-2 with 1605 of 2000 outputs empty.
5. Put an upper bound on the contamination with InChIKey-prefix or scaffold matching.
   Exact canonical SMILES misses stereochemistry and salt-form variants, so 49.25% is
   a floor. This needs no cluster time, only rdkit locally.
6. If question 1 closes favourably, the 28.45 against 38.7 comparison becomes worth
   writing up properly. If it does not, the transfer run still carries the channel
   conflict replication, which never depended on the comparison.

## One thing this file no longer needs to warn about

An earlier draft flagged a worry that ran the other way: if PubChem324k's *pretrain*
subset contained ChEBI-20 test molecules, the whole study's headline 62.32 would be
partly recall. Job `186609` checked it against the released V2 artefact and found
**zero of 6601 ChEBI-20 valid/test structures among the 298,010 pretrain structures**.

Section 4.1's filter is present in the shipped dataset and it is exact. So the
contamination in question 2 is a gap in that filter's *scope* rather than in its
implementation, and the reproduction this repository is built on rests on clean data.
`results/RESULTS.md` section 13 has the full table.
