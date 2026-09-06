# Queued, Not Yet Run

Everything in `../scripts/` ran and produced the logs in `../runlogs/`. Nothing in
this directory has. It is kept separate so the evidentiary record stays clean:
a reader can tell at a glance which scripts produced results and which are
proposals.

**The experimental programme is finished.** What remains here is one optional
refinement, not a gap. Every question this study can answer with the released
artefacts has been answered, and everything still open needs either checkpoints
the authors never published or a training run outside the scope of a
reproduction.

## What went through, and what it settled

| Job | Question | Answer |
|---|---|---|
| `186489` | How much of PubChem324kV2's test split did the ChEBI-20 checkpoint already see? | 49.25% by structure, against 23.35% by caption. Reversed section 11's conclusion |
| `186609` | Does the paper's own pretrain filter hold? | Zero of 6601, exactly. Section 13 |
| `186523`, `186610`, `186611` | Is the graph winning, or is the nearest channel winning? | The graph. Recency is real and worth a tenth of the weaker channel. Section 12 |
| `186483` | Can `stage2.ckpt` serve as a contamination-free control? | No. 0.18 BLEU-2, 1605 of 2000 outputs empty |
| `186488` | Does inserting filler cost anything by itself? | Confounded question, see below |

## The one thing left

`chebi_filler_mid2` and `chebi_filler_mid2_shufsmiles` repeat the distance
measurement at 13 Galactica tokens rather than 37.

Section 12 rests on two points: the SMILES costs 23.9% of baseline when adjacent
and 21.3% when displaced. Two points establish that recency exists and is small.
A third would establish whether the relationship is smooth, which is the
difference between "the number moved" and "the number moves with distance". It
would strengthen a claim already made rather than test a new one.

```bash
cd /home/mllp26_team007/MolCA
python <this directory>/apply_filler_patch.py     # already applied there
cp <this directory>/*.sub <this directory>/run_chebi_filler_mid2*.sh <job dir>/
condor_submit chebi_filler_mid2.sub
```

Expect the SMILES cost to land between 21.3% and 23.9% if distance acts smoothly,
and outside that range if the 37-token result was a threshold effect rather than
a gradient.

## Why `186488` is filed as a failure rather than a result

It was designed as the gate for the whole distance experiment, on the reasoning
that filler appended after the entire prompt moves both channels equally and so
leaves the relative geometry untouched. That reasoning was wrong. Appending
filler necessarily displaces the soft prompts from the generation point, which
makes it a manipulation rather than a control, and the checkpoint is brittle
about exactly that: 37.26 BLEU-2 with 54 empty outputs.

The useful gate was `186523`, which preserves the template ending, keeps every
output non-empty, and retains 86% of baseline. Reading `186488` as a failed gate
rather than as a null result is what allowed the experiment to continue.

## What is deliberately not here

Fine-tuning `stage2.ckpt` on PubChem324k's train subset would reproduce Table 2a's
38.7 directly and remove the need for the transfer comparison entirely. It is 100
epochs, costed at 6 GPU-hours on the authors' hardware and considerably more on a
P100. That is a training run rather than an evaluation.

MoleculeNet property prediction and IUPAC name prediction are absent for the same
reason: the release ships no classifier head and no IUPAC-tuned checkpoint.
`../../ENVIRONMENT.md` enumerates all seven released files.
