# Queued, Not Yet Run

Everything in `../scripts/` ran and produced the logs in `../runlogs/`. Nothing in
this directory has. It is kept separate so the evidentiary record stays clean:
a reader can tell at a glance which scripts produced results and which are
proposals.

Two pieces of work sit here. Neither is necessary. The study in `../../README.md`
is complete and internally consistent without them.

## 1. Structure-level overlap (no GPU, about a minute)

`transfer_overlap.py` matches molecules by caption text, because the prediction
dumps carry nothing else. That makes the 23.35% contamination figure a floor
rather than a measurement, and it is the whole content of Limitation 8.

`dump_smiles.py` fixes it. Run it on the cluster, where both datasets and rdkit
2026.3.5 are present:

```bash
cd /home/mllp26_team007/MolCA
source /home/mllp26_team007/molca_env/bin/activate
cp <this directory>/dump_smiles.py .
python dump_smiles.py --out smiles_dump
```

It writes `pubchem_test_smiles.jsonl` (2000 rows, in dataset order, carrying a
`text_head` field so row alignment against the prediction dump can be asserted
rather than assumed) and `chebi_smiles.jsonl` (all three ChEBI-20 splits). Both
carry the rdkit canonical form alongside the raw string, which is what makes the
comparison meaningful when two datasets disagree about kekulisation or atom
ordering for the same structure.

Bring both files back, drop them in `../results/`, and the overlap can be redone
on structures. Expect the number to rise, since caption matching can only miss
contamination and never invent it.

## 2. The distance experiment (six jobs, roughly 1.8 GPU-hours each)

This is the only open methodological weakness in the study. Section 3 shows the
model follows the graph rather than the SMILES, but the graph soft prompts sit
nearest the generation point, so "the graph wins" stays confounded with "the
nearest channel wins". Section 6 attacked this by reordering, which collapsed
generation to 0.01 BLEU-2 and settled nothing.

Reordering is not the only lever. Leaving the order alone and changing the
*distance* asks the same question without breaking the prompt:

| Variable | Effect |
|---|---|
| `MOLCA_FILLER_MID=k` | Filler between the SMILES span and the soft prompts. The SMILES moves away from generation; the graph does not |
| `MOLCA_FILLER_END=k` | Filler after the whole prompt. Both channels move equally, so the relative geometry is untouched |

The unit is `It is a chemical entity. `, true of every molecule, so repeating it
adds distance rather than information. Measured against the Galactica tokeniser,
k=2 is 13 tokens and k=6 is 37, against the graph's footprint of 8.

### Running it

```bash
cd /home/mllp26_team007/MolCA
python <this directory>/apply_filler_patch.py     # idempotent, --revert undoes it
cp <this directory>/*.sub <this directory>/run_chebi_filler_*.sh <job dir>/
condor_submit chebi_filler_end6.sub               # the gate, read it first
```

The patch is line-anchored rather than block-anchored, because the two machines
patched `smiles_handler` differently. It refuses to run rather than guessing if
the file has drifted further, and it saves a `.prefiller` backup.

### The gate

`chebi_filler_end6` runs baseline inputs with filler appended after the whole
prompt. Since both channels move equally, nothing about the comparison in section
3 should change, and the score should land near the 62.32 baseline.

**If it collapses the way the reordering test did, stop.** The filler itself is
then the confound and the remaining five jobs measure nothing. That failure is
worth recording either way, since it would establish that this checkpoint tolerates
no deviation from its fine-tuning template at all, which is a sharper version of
section 6's finding rather than a wasted night.

### What the remaining five jobs decide

The reference costs on the unmodified prompt, both from `../../bootstrap.py`:

| Contrast | Cost at normal distance |
|---|---|
| baseline − shuffle_smiles | **+14.91** |
| baseline − shuffle_graph | **+36.70** |

| Job pair | Measures |
|---|---|
| `mid2` − `mid2_shufsmiles` | SMILES cost with the SMILES 13 tokens further away |
| `mid6` − `mid6_shufsmiles` | The same at 37 tokens |
| `mid6` − `mid6_shufgraph` | Graph cost. The graph never moved, so this should stay near +36.70 |

Read the three numbers as a curve:

- **Cost falls as k grows.** Distance is doing real work, and section 3's result
  is at least partly positional. That would qualify the central finding, which is
  exactly why the experiment is worth running.
- **Cost stays near +14.91 at both levels.** Distance is not the mechanism, and
  the modality reading survives a test designed to break it.
- **The graph control moves.** The filler is doing something other than adding
  distance, and neither reading is safe. Treat the whole experiment as void.

I put the odds of a clean result at roughly two in five, the gate being the
likely failure point. That is a reasonable bet against idle GPUs overnight and a
poor one against anything with a deadline.

## What is deliberately not here

Fine-tuning `stage2.ckpt` on PubChem324k's train subset would reproduce Table 2a's
38.7 directly and remove the need for the transfer comparison entirely. It is 100
epochs, costed at 6 GPU-hours on the authors' hardware and considerably more on a
P100. That is a training run rather than an evaluation, which puts it outside what
this study set out to do.
