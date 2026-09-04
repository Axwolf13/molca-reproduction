# Queued, Not Yet Run

Everything in `../scripts/` ran and produced the logs in `../runlogs/`. Nothing in
this directory has. It is kept separate so the evidentiary record stays clean:
a reader can tell at a glance which scripts produced results and which are
proposals.

Four jobs remain here, and between them they would close the last open question
this study can close on its own hardware.

## What already ran, and what it changed

Two items that used to sit in this directory have since gone through the cluster.
Both are worth reading before submitting anything else.

- **`dump_smiles.py`** (job `186489`) worked exactly as intended and moved a
  headline number. Structural matching found 49.25% contamination in the
  PubChem324kV2 test split against caption matching's 23.35%, which took the clean
  transfer score from 38.86 to 28.45 and reversed section 11's conclusion. The
  script now lives in `../scripts/`.
- **`chebi_filler_end6`** (job `186488`) was the designated gate and it was
  confounded. Appending filler after the whole prompt does not leave the geometry
  untouched, because it displaces the soft prompts from the generation point. It
  scored 37.26 with 54 empty outputs, and the four jobs below were cancelled on
  the strength of that.

That cancellation was the wrong call, though a reasonable one at the time.

## The gate that matters passed

`chebi_filler_mid6` (job `186523`) puts the filler **between** the SMILES span and
the soft prompts, so the template ending the checkpoint was fine-tuned on stays
intact. It scored **53.60 BLEU-2 with zero empty outputs**, retaining 86% of
baseline while pushing the SMILES 37 Galactica tokens further from generation.

That is a working instrument, and it is the reference the remaining jobs score
against.

## Run this one first

`verify_pretrain_filter.py` checks the paper's own hygiene claim, and it is the
highest-value job left. No GPU, a few minutes.

Section 4.1 says the PubChem324k pretrain subset was filtered to exclude
molecules from ChEBI-20's valid/test splits. Every headline number in this study
depends on that holding, because `chebi.ckpt` was pretrained on that subset and
is then evaluated on ChEBI-20's test split. Section 11 found 49.25% contamination
running the comparison the other way, so the filter is no longer something to
take on trust.

```bash
cd /home/mllp26_team007/MolCA
source /home/mllp26_team007/molca_env/bin/activate
cp <this directory>/verify_pretrain_filter.py .
python verify_pretrain_filter.py > results_pretrain_filter.txt 2>&1
```

A zero in the pretrain-against-valid/test row confirms the filter and closes the
question. A non-zero one means section 1's 62.32 is evaluated partly on molecules
the checkpoint saw in pretraining, which would be the most consequential finding
in the study and would need saying before anyone else notices.

## The four jobs

| Job | Measures | Reference |
|---|---|---|
| `chebi_filler_mid6_shufsmiles` | SMILES cost at +37 tokens | 53.60, against +14.91 at normal distance |
| `chebi_filler_mid6_shufgraph` | Graph cost, the graph having not moved | 53.60, against +36.70 at normal distance |
| `chebi_filler_mid2` | Baseline at +13 tokens | 62.32 at zero |
| `chebi_filler_mid2_shufsmiles` | SMILES cost at +13 tokens | the row above |

The first two are the measurement. The `mid2` pair turns a single contrast into a
dose-response curve, which is the difference between "the number moved" and "the
number moves with distance", so run all four if the slots exist and the first two
if they do not.

Reading the result:

- **SMILES cost falls as distance grows.** Position is doing real work and
  section 3's central finding needs qualifying.
- **SMILES cost holds near +14.91 at both distances.** Distance is not the
  mechanism, and the modality reading survives a test built to break it.
- **Graph cost drifts off +36.70.** The graph never moved, so a change there means
  the filler is doing something other than adding distance. Treat the whole
  experiment as void.

## Running them

```bash
cd /home/mllp26_team007/MolCA
python <this directory>/apply_filler_patch.py     # idempotent, --revert undoes it
cp <this directory>/*.sub <this directory>/run_chebi_filler_*.sh <job dir>/
condor_submit chebi_filler_mid6_shufsmiles.sub
```

The patch is line-anchored rather than block-anchored, because the two machines
patched `smiles_handler` differently. It refuses to run rather than guessing if
the file has drifted further, and it saves a `.prefiller` backup. It is already
applied on the cluster, where `../patches_cluster.diff` records the result.

## What is deliberately not here

Fine-tuning `stage2.ckpt` on PubChem324k's train subset would reproduce Table 2a's
38.7 directly and remove the need for the transfer comparison entirely. It is 100
epochs, costed at 6 GPU-hours on the authors' hardware and considerably more on a
P100. That is a training run rather than an evaluation, which puts it outside what
this study set out to do.

Job `186483` tested whether the released `stage2.ckpt` could stand in without that
fine-tune. It could not: 0.18 BLEU-2, 1605 of 2000 outputs empty.
