# The Cluster Runs

Everything in this directory comes from the Saarland CS HTCondor pool rather
than from the laptop that produced `../results/`. Two machines, two software
stacks, one checkpoint. Holding the model fixed while changing almost everything
around it is what turns a single number into a reproduction.

## Configuration

| | Cluster (this directory) | Local (`../results/`) |
|---|---|---|
| GPU | Tesla P100 for captioning, see below for retrieval | RTX 4060 Laptop, 8 GB |
| Scheduler | HTCondor, `universe = docker` | none, a bash supervisor |
| Base image | `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel` | n/a |
| Python | 3.10 | 3.13.2 |
| torch | 2.3.1 / CUDA 12.1 | 2.13.0 / CUDA 12.6 |
| transformers | 4.44.2 | 4.46.3 |
| LAVIS | `salesforce-lavis==1.0.2`, installed | hand-vendored under `../vendor/lavis/` |
| Precision | `16-mixed` | `16-mixed` |
| Inference batch | 8 | 4 |
| Test split | full 3300 molecules, every ChEBI-20 condition | 3300 for two conditions, first 1000 for the rest |
| Second dataset | PubChem324kV2, full 2000-molecule test split | not attempted |

Every condition here ran the **complete 3300-molecule CheBI-20 test split**. On
the laptop only `baseline` and `shuffle_graph` reached the full split, because a
single 8 GB card puts each run near two hours. The condition matrix there uses
the first 1000 molecules, landing within 0.12 BLEU-2 of the full split.

Because the two stacks differ in Python version, torch major, transformers
version, LAVIS provenance, and batch size, small numeric differences between
this directory and `../results/` are expected rather than alarming. The full
split gives 62.32 here against 62.77 locally, a spread of 0.45 on a checkpoint
the paper reports at 62.0.

## Checkpoints

The upstream README points readers at `all_checkpoints/share/`. No such
directory exists in the release. Every job below loads instead from:

```
all_checkpoints/archived/chebi.ckpt
all_checkpoints/archived/chebi_lora/
all_checkpoints/stage1.ckpt
```

All three come from the [`acharkq/MolCA`](https://github.com/acharkq/MolCA)
release. None of them is redistributed here.

## What Is in This Directory

| Path | Contents |
|---|---|
| `results/results.txt` | Pipeline BLEU-2, BLEU-4, METEOR and wall time for 26 jobs |
| `results/results_retrieval.txt` | Stage-1 retrieval, raw Lightning output |
| `results/verify_retrieval_output.txt` | Output of `verify_retrieval.py`, all 32 metrics |
| `results/transfer_overlap_output.txt` | Output of `../transfer_overlap.py`, the contamination split |
| `results/results_overlap_smiles.txt` | The same, as produced on the cluster, unedited |
| `results/results_pretrain_filter.txt` | Verification that the paper's own contamination filter holds |
| `results/distance_test_output.txt` | Output of `../distance_test.py`, modality against recency |
| `results/pubchem_test_smiles.jsonl` | Canonical SMILES for the 2000 PubChem324kV2 test rows |
| `results/chebi_smiles.jsonl` | Canonical SMILES for all 33,008 ChEBI-20 rows |
| `verify_retrieval.py` | Diffs both retrieval logs against the paper's Tables 7b and 7c |
| `results/results_multimetric.txt` | Channel conflict across BLEU-2, BLEU-4, ROUGE-L, METEOR |
| `results/results_class_fuzzy.txt` | Chemical-class agreement under three matching rules |
| `predictions/` | 25 prediction files, one per job, as JSONL |
| `runlogs/` | Condor `.out` and `.err` for every job, including the failures |
| `scripts/` | The `.sub` submit files, their `.sh` payloads, and three analysis scripts |
| `patches_cluster.diff` | The source patches applied on the cluster |
| `environment_cluster_linux.txt` | `pip freeze` from the cluster virtualenv |

The result files keep their original formatting, box-drawing characters
included. Unedited tool output is stronger evidence than a table I retyped.

## Retrieval

Retrieval is the one task the cluster finished and the laptop did not. Local
runs need roughly 11 GPU-hours for two re-ranking passes across both splits,
which never fit into a night.

`verify_retrieval.py` parses the two Condor logs and diffs every metric against
Tables 7b and 7c of the paper. Its output is kept in
`results/verify_retrieval_output.txt`. Running it:

```
32 metrics checked, 0 missing, max |deviation| = 0.49 (MoMu rerank_test_inbatch_t2g_acc)
full-test-set columns only (16 metrics): max |deviation| = 0.12
```

Lightning's `test_*` keys correspond to the paper's `MolCA w/o MTM` row, which
scores by contrastive similarity alone. Its `rerank_test_*` keys correspond to
the plain `MolCA` row, re-ranking the top-128 candidates with the matching head.

Full test set, the column the paper leads with:

| Split | Direction | Acc | Acc (paper) | R@20 | R@20 (paper) |
|---|---|---|---|---|---|
| PCDes | graph → text | 37.69 | 37.7 | 80.59 | 80.6 |
| PCDes | text → graph | 35.36 | 35.3 | 76.55 | 76.5 |
| MoMu | graph → text | 22.47 | 22.5 | 68.45 | 68.5 |
| MoMu | text → graph | 21.14 | 21.1 | 64.76 | 64.8 |

The same four rows after re-ranking:

| Split | Direction | Acc | Acc (paper) | R@20 | R@20 (paper) |
|---|---|---|---|---|---|
| PCDes | graph → text | 48.20 | 48.1 | 85.56 | 85.6 |
| PCDes | text → graph | 45.96 | 46.0 | 82.22 | 82.3 |
| MoMu | graph → text | 30.55 | 30.6 | 76.77 | 76.8 |
| MoMu | text → graph | 29.68 | 29.8 | 73.32 | 73.3 |

Re-ranking lifts PCDes graph-to-text accuracy by 10.51 points, from 37.69 to
48.20. That mechanism is the paper's own claim about the matching head,
reproducing here in both direction and magnitude.

### Where the deviations concentrate

The sixteen full-test-set metrics stay within 0.12. The sixteen in-batch metrics
are looser, with two outliers: MoMu text-to-graph in-batch accuracy misses by
0.21 without re-ranking and by 0.49 with it. Every other in-batch metric sits
inside 0.07.

That asymmetry is structural rather than mysterious. Full-test-set retrieval
ranks each query against all candidates, giving a quantity independent of how
the loader groups examples. In-batch retrieval ranks each query only against its
own batch, which makes the score a function of `--match_batch_size` and of the
shuffle seed. Since the paper does not state the batch size it evaluated with,
the in-batch columns are the weaker comparison of the two. I therefore treat
the full-test-set figures as the reproduction.

### Which GPU ran retrieval

Every captioning job used a Tesla P100, and job `183286` proves it by dying on a
bfloat16 dtype mismatch that Pascal hardware produces and Ampere does not.

Retrieval is less certain, and the honest answer is that the artefacts do not
say. `scripts/` holds two submit files per split: `pcdes.sub` and `momu.sub`
place no constraint on the device, while `pcdes_a100.sub` and `momu_a100.sub`
differ from them by exactly one line, `require_gpus = (Capability >= 8.0)`, which
excludes the P100 at capability 6.0. Both variants write their output to the same
`runlogs/pcdes.$(ClusterId).out` path, so the surviving logs cannot say which one
was submitted, and Lightning records only `GPU available: True (cuda)` with a
device UUID rather than a model name.

Nothing in the retrieval result depends on this. Full-test-set retrieval is
deterministic given the checkpoint and the candidate pool, and all 32 metrics
were diffed against the paper regardless of device. It is recorded here because
the configuration table above would otherwise claim more than the logs support.

One incidental check the logs happen to supply: every `val_*` row is identical
between the PCDes job and the MoMu job. Since `--use_phy_eval` swaps only the
test split, that identity is what correct runs produce. Divergence there would
have signalled state leaking between jobs.

## Cross-Dataset Transfer

Three later jobs point the ChEBI-20 checkpoint at a dataset it was never
fine-tuned on, giving the channel-conflict finding a second caption
distribution to survive:

| Job | Condition | Rows | BLEU-2 |
|---|---|---:|---:|
| `185367` | `pc_transfer`, normal inputs | 2000 | 49.04 |
| `185368` | `pc_transfer_shufgraph`, neighbour's graph | 2000 | 18.63 |
| `185369` | `pc_transfer_graphonly`, SMILES withheld | 2000 | 2.41e-154 |

All three load `archived/chebi.ckpt` with `--root data/PubChem324kV2/`, so the
weights are the ChEBI-20 ones while the split is PubChem's. The 2000 rows match
Table 1 of the paper exactly.

Do not read 49.04 against the paper's 38.7 without reading
[`../NOTES.md`](../NOTES.md) first. **49.25%** of the PubChem324kV2 test split is
structurally present in ChEBI-20's training set, which is what the checkpoint
spent 100 epochs on. Scoring the halves apart puts the contaminated rows at 63.75
and the rest at 28.45, so the clean transfer number is 10.25 BLEU-2 *below* the
paper rather than above it.

## The Distance Experiment

Six jobs separate "the graph wins" from "the channel nearest generation wins",
which the shipped prompt layout confounds.

| Job | Condition | BLEU-2 |
|---|---|---:|
| `186523` | filler between SMILES and soft prompts, clean inputs | 53.60 |
| `186610` | the same, SMILES rotated | 42.17 |
| `186611` | the same, graph rotated | 22.33 |

Against the unmodified baselines of 62.32 / 47.41 / 25.62, corrupting the SMILES
costs 23.9% of baseline when it sits adjacent and 21.3% when it is 37 tokens
further away, while the graph stays at 58.9% and 58.3%. Recency is measurable and
too small to explain a 2.5x dominance. `../results/RESULTS.md` section 12 has the
intervals.

## The Paper's Filter, Verified

Job `186609` needed no GPU and produced the cleanest result of the whole set.
Matching canonical structures, **zero** of ChEBI-20's 6601 valid and test
molecules appear among the 298,010 distinct structures in PubChem324kV2's
pretrain subset. Section 4.1 claims exactly that exclusion, and the released
dataset delivers it.

Since every number in `../results/` comes from a checkpoint pretrained on that
subset and evaluated on ChEBI-20's test split, this is the check that keeps the
reproduction honest. Raw output is in `results/results_pretrain_filter.txt`.

## Two Jobs That Did Not Deliver

Kept here because a null result that cost GPU-hours is still part of the record.

| Job | Intent | Outcome |
|---|---|---|
| `186483` | `stage2.ckpt` as a contamination-free control | **Void.** 0.18 BLEU-2, 1605 of 2000 outputs empty. The pre-fine-tune model does not caption |
| `186488` | Gate for the distance experiment, filler after the whole prompt | **Confounded.** 37.26 BLEU-2, 54 empty. Appending filler displaces the soft prompts from the generation point, so it is a manipulation rather than a control |

Both are kept rather than dropped. A study that reports only the experiments
that worked is not reporting its method.

## Two Findings the Logs Preserve

**bf16 does not survive on this hardware.** Job `183283` ran under
`precision = bf16-mixed`. Job `183286` then died with `Index put requires the
source and destination dtypes match, got BFloat16 for the destination and Half
for the source`, which traces back to `blip2_opt.py` hardcoding bfloat16 in two
of its three model-loading branches. Both jobs are in `runlogs/`. The fix,
visible in `patches_cluster.diff`, forces float16 in all three branches.
Consequently **both machines ran fp16**, contrary to what an earlier draft of
`../results/RESULTS.md` claimed.

**The retrieval tokenizer bug reproduces identically.** `stage1.py` routes on
the root path. Because `--root data/kv_data` contains the substring `kv`, every
documented retrieval command goes down the `Stage1KVPLMDM` branch. That class
never receives a tokenizer, while the dataset it builds calls one. The cluster
patch and the local patch fix it the same way, by threading `tokenizer` through
the constructor. Two independent stacks failing at the same line is the
strongest evidence available that the bug is in the release rather than in
either environment.

## Reproducing a Job

```bash
cd cluster/scripts
condor_submit chebi_eval.sub          # or any other .sub in this directory
```

Each `.sub` names its `.sh` payload, requests one GPU with 8 CPUs and 32 GB, and
writes into `runlogs/`. The payloads activate `molca_env`, change into the
`MolCA` checkout, and call `stage2.py` or `stage1.py` directly.
