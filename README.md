# Does the Graph Channel Matter?

### A Reproducibility and Modality-Conflict Study of MolCA

**Akshay Ashok** (7071170)
Seminar: Machine Learning for Language Processing, Saarland University
Supervisor: Prof. Dietrich Klakow

---

## What I Tested

[MolCA](https://github.com/acharkq/MolCA) (Liu et al., EMNLP 2023) wires a 2D
molecular graph encoder into Galactica through a Q-Former cross-modal projector.
Its Table 5a measures what each input view contributes during *training*,
reporting 34.6 BLEU-2 for SMILES alone, 34.5 for the graph alone, and 38.7 for
both together. Read on its own, that table suggests two interchangeable views
combining for a modest gain.

The paper never measures what the trained model does with those two views at
*inference*. I closed that gap by putting the channels into direct conflict on
the released checkpoint, feeding each molecule its own SMILES string alongside a
different molecule's graph.

### Coverage, and what the release puts out of reach

| Dataset | What I ran | Why |
|---|---|---|
| ChEBI-20 | 17 cluster jobs, plus the local matrix | `archived/chebi.ckpt` is released |
| PCDes + MoMu | 2 retrieval evals, 32 metrics | `stage1.ckpt` is released |
| PubChem324kV2 | 3 transfer runs | No fine-tuned checkpoint exists, so transfer only |
| MoleculeNet | none | Needs a trained classifier head |
| IUPAC naming | none | Needs an IUPAC-tuned checkpoint |

The last two rows are not omissions. `acharkq/MolCA` ships seven files, covering
ChEBI-20 captioning and stage-1 retrieval and nothing else; `ENVIRONMENT.md`
enumerates them. Property prediction and IUPAC naming would each require training
weights the authors never released, as would reproducing Table 2a's 38.7 on
PubChem324k directly. The experimental program is bounded by the artefact rather
than by the budget.

## Headline Results

### Reproduction

Running the released checkpoint over the full 3300-molecule CheBI-20 test split:

| Source | BLEU-2 |
|---|---|
| Paper, Table 2b | 62.0 |
| Tesla P100 (cluster) | 62.32 |
| RTX 4060 Laptop (local) | **62.77** |

Three measurements of one checkpoint span 0.77 BLEU-2 across two GPU
generations, two Python versions, two torch majors, and two transformers
versions. Agreement runs deeper than the score: **93.8% of the 3300 captions are
character-identical** between the two machines. Because I also hand-vendored LAVIS on the laptop rather than
installing it, the software stack differs substantially between the two
machines. Both ran fp16, the P100 having rejected bfloat16 outright.

### The Central Finding

Handing a molecule its own SMILES together with a different molecule's graph, I
found the model describes the *other* molecule:

```
its own SMILES describes : a steroid ester, methyl (17E)-pregna-4,17-dien-21-oate
the graph it was given   : a branched amino tetrasaccharide
the model wrote          : "The molecule is a branched amino tetrasaccharide ..."
```

Across 1000 molecules, roughly 90% of decisive cases assert the chemical class
of the molecule that supplied the **graph**. Measured as class-agreement F1
against a near-zero floor, the ratio runs about 9:1.

| Manipulation | Cost in BLEU-2 |
|---|---|
| Substituting the graph | **36.9** |
| Substituting the SMILES | 14.6 |

Corrupting the graph therefore costs roughly 2.5 times what corrupting the text
costs, under a matched manipulation applied to each channel.

### Retrieval Reproduces Too

The cluster additionally reproduced MolCA's retrieval task from `stage1.ckpt`,
once two release bugs described below were fixed. Diffing both Condor logs
against Tables 7b and 7c of the paper covers **all 32 published metrics**:

| Scope | Metrics | Max deviation |
|---|---|---|
| Full test set, both splits | 16 | **0.12** |
| In batch, both splits | 16 | 0.49 |

`cluster/verify_retrieval.py` recomputes that comparison from the raw logs. The
in-batch figures deviate more because in-batch retrieval ranks each query only
against its own batch, making the score depend on an evaluation batch size the
paper never states. Full-test-set retrieval ranks against every candidate and
carries no such dependence.

Re-ranking the top-128 contrastive candidates with the matching head lifts PCDes
graph-to-text accuracy from 37.69 to 48.20, reproducing the direction and
magnitude of the gain the paper credits to that step.

### The Model Trusts the Graph Rather Than Hedging

| Graph condition | BLEU-2 |
|---|---|
| Uninformative (atom features zeroed) | 28.19 |
| Wrong (a different real molecule) | 25.71 |
| Incoherent (edges resampled at random) | 20.46 |

A wrong graph hurts more than no graph at all. Were the graph a weak side
channel, feeding it garbage would cost no more than feeding it nothing.
Since it costs considerably more, the language model treats those eight soft
prompts as authoritative evidence rather than as advice.

### The Conflict Replicates on a Second Dataset

Pointing the ChEBI-20 checkpoint at PubChem324kV2's test split gives the finding
a second caption distribution to survive.

| Subset of the 2000 test molecules | Normal | Neighbour's graph | Drop |
|---|---:|---:|---:|
| Whole split | 49.04 | 18.63 | 30.41 |
| Never seen during ChEBI-20 training | 38.86 | 16.03 | 22.83 |

The split matters because 23.35% of that test set carries a caption the
checkpoint was fine-tuned on verbatim, scoring 94.33 BLEU-2 on recall alone.
Reading the raw 49.04 as transfer beating the paper's in-domain 38.7 would be
wrong twice over, so **[NOTES.md](NOTES.md)** works through what the number can
and cannot support. The conflict result needs neither reading: it holds at 22.83
BLEU-2 on molecules the model provably never trained on.

Full numbers, controls, and limitations live in
**[results/RESULTS.md](results/RESULTS.md)**.

## Repository Layout

| Path | Contents |
|---|---|
| `apply_patches.py` | 14 idempotent source patches across four groups |
| `analyse.py` | n-gram agreement, chemical-class agreement, per-example verdicts |
| `bootstrap.py` | Paired bootstrap confidence intervals over the test set |
| `cross_stack_agreement.py` | Caption-level agreement between the two machines |
| `transfer_overlap.py` | ChEBI-20 contamination inside the PubChem324kV2 test split |
| `NOTES.md` | The two questions about the transfer run I could not close |
| `run_eval.sh` | A single condition |
| `run_queue.sh` | Several conditions back to back |
| `supervise.sh` | Resumable supervisor that skips finished conditions and retries failures |
| `run_fullset.sh` | The 3300-molecule runs |
| `run_promptorder.sh` | The position-versus-modality test |
| `run_retrieval.sh` | Stage-1 retrieval |
| `ENVIRONMENT.md` | Environment recipe, plus why the paper's pins no longer build |
| `patches/molca.diff` | The resulting diff against `acharkq/MolCA` at `f728a47` |
| `results/predictions/` | 11 prediction files, one per condition, as JSONL |
| `results/logs/` | Run logs with progress bars stripped and repeats collapsed |
| `vendor/lavis/` | Minimal vendored LAVIS, BSD-3-Clause, licence retained |
| `cluster/` | The parallel P100 runs: predictions, logs, submit files, retrieval |

Checkpoints, model weights, datasets, and the virtualenv stay out of the
repository. `ENVIRONMENT.md` explains how to obtain each one.

## Reproducing

```bash
git clone https://github.com/acharkq/MolCA          # upstream, at f728a47
python apply_patches.py                             # idempotent
./run_eval.sh baseline                              # or any condition below
python analyse.py
```

Available conditions:

- `baseline`, `baseline_full`
- `shuffle_graph`, `shuffle_graph_rev`, `shuffle_graph_full`
- `shuffle_smiles`
- `rewire_graph`, `null_graph`
- `graph_only`, `shuffle_graph_only`
- `baseline_graphfirst`, `shuffle_graph_graphfirst`

Every intervention lives inside `InferenceCollater` and stays inert unless its
environment variable is set, which leaves training collation untouched.

## What I Found in the Released Code

### Portability

MolCA assumes the authors' rig: two A100s, bf16, an initialised
`torch.distributed` process group. Each assumption fails somewhere different on
a single consumer GPU. Two of them discard work *after* it has already finished.

| ID | Problem | Consequence |
|---|---|---|
| A1 | Checkpoints carry `cuda:4` device tags | Unpickling fails on a single-GPU machine |
| A2 | `blip2_opt.py` hardcodes bf16 in two of three branches | Dtype mismatch on cards without bf16 support |
| A3 | `all_gather_object` runs unguarded in the eval hook | A finished two-hour run gets thrown away at the last step |
| A4 | `dist.get_rank()` runs unguarded in the stage-1 loss | Same root cause |
| A5 | `persistent_workers=True` hardcoded across four datamodules | Illegal whenever `num_workers=0` |

Three further issues fall outside that table:

- `val_dataloader` uses `batch_size` rather than `inference_batch_size`. On an
  8 GB card under Windows WDDM, the oversized batch spills silently into system
  RAM instead of raising OOM. Throughput collapsed to 0.03 it/s before the
  machine froze outright.
- The README points at `all_checkpoints/share/`, a path absent from the release.
  The files actually sit under `archived/`.
- `stage2_chebi_dm.py` lacks the `graph_only` option that its `stage2_dm.py`
  sibling exposes. Three `if False:` branches in `stage1_dm.py` misdirect
  debugging further.

### Retrieval: One Known Bug and One New One

`stage1.py` routes on the root path. Because `--root data/kv_data` contains the
substring `kv`, every documented retrieval command takes the `Stage1KVPLMDM`
branch.

1. **`Stage1KVPLMDM` never receives a tokenizer.** Grepping the file for
   `tokenizer` returns zero hits, yet it builds `GINPretrainDataset`, whose
   `tokenizer_text()` calls `self.tokenizer(...)`. Since `trainer.validate()`
   iterates precisely that dataset, every documented retrieval command dies at
   `pretrain_dataset.py:78` with `TypeError: 'NoneType' object is not callable`.
   A suitable tokenizer sits in scope one line above the call site in
   `stage1.py`. **This one is not mine.** It is
   [issue #13](https://github.com/acharkq/MolCA/issues/13), reported in July
   2024 and still open. I rediscovered it independently before reading the
   tracker, which is worth stating plainly rather than dressing up.
2. **torch 2.6 flipped the `weights_only` default in `torch.load` to `True`**,
   which rejects the pickled PyG graph objects the dataset stores. This stayed
   invisible on the authors' torch 2.0. It also stayed invisible throughout
   captioning, because ChEBI-20 ships `.txt` files while `kv_data` ships
   pre-pickled `.pt` graphs.

I patched both, then verified the data path end to end. Locally the evaluation
never completed: two re-ranking passes per split, at 13.4 s per iteration over
750 iterations, works out to roughly 11 GPU-hours. On the cluster, where the
same two bugs surfaced at the same two lines, the patched pipeline ran to
completion. Numbers live in
[`cluster/results/results_retrieval.txt`](cluster/results/results_retrieval.txt).

Finishing the run settles the question issue #13 left open. Its reporter patched
the tokenizer by constructing one inside `pretrain_dataset.py`, then found the
resulting accuracy far below the paper and asked whether the tokenizer was the
culprit. The maintainer replied that the tokenizer was probably right and the
cause lay elsewhere, without demonstrating it. Threading the model's own
tokenizer through the datamodule constructor instead reproduces **all 32
published retrieval metrics**. The maintainer's reading was therefore correct.
A correctly wired tokenizer is sufficient, which leaves whatever degraded that
reporter's numbers as a separate and still-undiagnosed problem.

### Relation to the Issue Tracker

Fourteen issues exist upstream, nine closed. Reading them separates what I
rediscovered from what appears to be new. I searched every issue body and every
comment for each finding:

| Finding | Status upstream |
|---|---|
| Retrieval tokenizer is `None` | [#13](https://github.com/acharkq/MolCA/issues/13), open since Jul 2024 |
| `all_checkpoints/share/` does not exist | Users copy that path from the README in #13 and #14 without anyone flagging it |
| `environment.yml` will not build | [#8](https://github.com/acharkq/MolCA/issues/8), closed with a hand-written recipe that still omits `ogb`, `peft`, and `rdkit` |
| `torch.load` `weights_only` under torch ≥2.6 | Unreported |
| `persistent_workers=True` with `num_workers=0` | Unreported |
| Unguarded `all_gather_object` and `dist.get_rank()` | Unreported |
| `cuda:4` device tags in the checkpoints | Unreported |
| `graph_only` missing from `stage2_chebi_dm.py` | Unreported |
| `caption_evaluate` discarding its truncation arguments | Unreported |

Two closed issues are worth knowing about before trusting any reproduction of
this repository. [#7](https://github.com/acharkq/MolCA/issues/7) records that
`script/chebi.sh` never loads a pretrained checkpoint. Anyone running the
shipped script therefore measures a different model from the one the paper
describes.
[#3](https://github.com/acharkq/MolCA/issues/3) records a naming bug that had
stage-1 pretraining reading the wrong subset of PubChem324k, fixed in
`2d46afe`. Since the released `stage1.ckpt` postdates that commit, the
checkpoint I evaluate is unaffected.

## Honest Limitations

I am keeping these visible because they change how the results should be read.

- **`shuffle_smiles` is not the independent control I designed it to be.**
  Rotating the graph by minus one and rotating the SMILES by plus one produce
  the same pairing. After re-indexing, 983 of 1000 predictions match exactly.
  It functions as a two-code-path consistency check rather than as extra
  evidence, which leaves eight conditions covering seven distinct manipulations.
- **The position-versus-modality test failed as an instrument.** Since the graph
  prompts sit nearest the generation point, "the graph dominates" stays
  confounded with "the nearest channel dominates". Because moving the prompts
  ahead of the SMILES collapses the model from 63.16 to 0.01 BLEU-2, the
  reordering cannot separate the two explanations. That confound remains open.
- **The chemical-class extractor has a ceiling.** Built on regular expressions
  over ChEBI's stereotyped phrasing, it registers a match on only 76.3% of
  *correct* baseline captions. Absolute per-example rates are therefore
  underestimates, although ratios between conditions hold.
- **`results/logs/baseline_graphfirst.log` is nearly empty.** That run's Python
  process finished and wrote all 1000 predictions. Its shell wrapper broke
  mid-run when `run_eval.sh` was edited while bash was still reading it. I
  rescored the predictions afterwards using MolCA's own scorer.
- **MolCA's `caption_evaluate` silently ignores its truncation settings.** It
  passes `truncation`, `max_length`, and `padding` into `tokenizer.tokenize()`,
  which accepts none of them. Transformers warns once per batch before
  discarding them, leaving the intended 512-token truncation inoperative.

## Licence

This work carries the MIT licence, recorded in `LICENSE`. The `vendor/lavis/`
directory holds a minimal subset of
[Salesforce LAVIS](https://github.com/salesforce/LAVIS) under BSD-3-Clause, with
its licence file and SPDX headers retained. `ENVIRONMENT.md` documents every
modification I made to it. MolCA itself is not redistributed here, only a patch
against it.
