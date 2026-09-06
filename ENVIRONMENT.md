# Reproducing This Environment

`pip freeze` alone will not rebuild this setup. Because I vendored one
dependency (`lavis`) rather than installing it, a fresh virtualenv built purely
from `results/environment_local_win.txt` fails immediately at
`ModuleNotFoundError: No module named 'lavis'`. The full recipe follows.

## Why the Published Pins No Longer Build

MolCA's `environment.yml` targets 2023: torch 2.0.0 with CUDA 11.7, Python 3.8,
transformers 4.28.1, pytorch-lightning 1.9.0, alongside `salesforce-lavis` and
`flash-attn 1.0.5`. On Python 3.13 that chain is unbuildable.

| Pin | Failure |
|---|---|
| `numpy<2` | No cp313 wheel exists. The source build invokes Meson, which crashed on the non-ASCII path (`Universität`) |
| `tokenizers<0.20` | No cp313 wheel, which makes `transformers==4.44.2` uninstallable |
| `transformers>=5` | Dropped `transformers.file_utils`, imported by LAVIS's `Qformer.py` |
| `salesforce-lavis` | `lavis/__init__.py` imports the entire model zoo, pulling in spacy, decord, timm, fairscale, and opencv |

Squeezed between those constraints, `transformers==4.46.3` is the only version
carrying cp313 wheels while still shipping `file_utils`.

## Installation

```bash
python3.13 -m venv molca_venv
molca_venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
molca_venv/Scripts/python.exe -m pip install --only-binary=:all: \
    numpy "transformers==4.46.3" "peft==0.13.2" "pytorch-lightning==2.3.3" \
    torch-geometric rdkit nltk scikit-learn networkx pandas pyyaml tqdm
molca_venv/Scripts/python.exe -m pip install ogb rouge_score iopath "omegaconf>=2.3"
molca_venv/Scripts/python.exe -c "import nltk; [nltk.download(r, quiet=True) for r in ('wordnet','punkt','punkt_tab','omw-1.4')]"
```

Although LAVIS pins `omegaconf` at 2.0, you must install 2.3 or later.
Lightning's CSV logger calls `OmegaConf.save`, which on 2.0 raises inside the
*exception handler*, masking whatever the real error was.

## The Vendored LAVIS

MolCA touches six symbols from LAVIS. Importing any one of them executes
`lavis/__init__.py`, which in turn imports every model LAVIS ships. Rather than
install that dependency tree, I vendored the package from
`salesforce_lavis-1.0.2-py3-none-any.whl` under `vendor/lavis/` and applied four
edits.

| File | Edit | Reason |
|---|---|---|
| `lavis/__init__.py` | Emptied | Imported the datasets, models, processors, and tasks zoo |
| `lavis/models/__init__.py` | Emptied | Same |
| `lavis/models/blip2_models/blip2.py` | Made `create_eva_vit_g` and `create_clip_vit_L` lazy | Pulled in timm and fairscale, although MolCA never calls `init_vision_encoder` |
| `lavis/common/utils.py` | Made the torchvision import optional | Used only by dataset-download helpers |
| `lavis/common/dist_utils.py` | Made the timm import optional | Used only to download ViT weights |

I altered nothing on MolCA's execution path. Every edit targets a module-level
import that would otherwise force an unrelated dependency. The Q-Former
implementation in `Qformer.py`, all 1216 lines of it, stays byte-identical to
the release, which matters because the checkpoint's weights are keyed to it.

**Every run must point `PYTHONPATH` at the `vendor/` directory.** While
`run_eval.sh` handles this, a bare `python stage2.py` will not find `lavis`.

## Assets

| Asset | Source | Size |
|---|---|---|
| `chebi.ckpt` | `acharkq/MolCA` at `archived/chebi.ckpt` | 1.28 GB |
| `chebi_lora/` | `acharkq/MolCA` at `archived/chebi_lora/` | 50 MB |
| `stage1.ckpt` | `acharkq/MolCA` at `stage1.ckpt` | 2.15 GB |
| `galactica-1.3b` | `facebook/galactica-1.3b` | 2.63 GB |
| ChEBI-20 | `data/dataset.zip`, shipped in the repo | 3300 test rows |
| PubChem324kV2 | `acharkq/PubChem324kV2`, cluster only | 2000 test rows |

The README documents these under `all_checkpoints/share/`, a directory absent
from the released repository. The files actually sit under `archived/` and at
the repository root.

### What the release contains, in full

`acharkq/MolCA` holds seven files and no more:

| File | Size | What it is |
|---|---|---|
| `stage1.ckpt` | 2.15 GB | Stage-1 retrieval model |
| `archived/stage1.ckpt` | 2.15 GB | Near-identical, 1679 bytes apart |
| `stage2.ckpt` | 1.18 GB | Stage-2 **pretrained**, LM frozen, PubChem324k pretrain subset |
| `archived/chebi.ckpt` | 1.28 GB | ChEBI-20 fine-tuned |
| `archived/chebi_lora/adapter_model.bin` | 50 MB | Its LoRA adapter |
| `archived/chebi_lora/adapter_config.json` | 382 B | Adapter config |
| `.gitattributes` | 1.5 kB | |

This inventory determines which tables in the paper are reachable without
training. Only ChEBI-20 captioning and stage-1 retrieval ship as usable weights.
Table 2a's 38.7 comes from a LoRA fine-tune on PubChem324k's train subset, 100
epochs per Table 11, and those weights are not released. Neither are the
MoleculeNet classifier heads nor an IUPAC-tuned checkpoint.

`stage2.ckpt` is the one PubChem-native artefact, but it is the pre-fine-tune
model. Table 8 ablates pretrain stages **after** fine-tuning in every row, so
the paper reports no captioning score for it in this state and it cannot stand
in for 38.7.

## The Second Machine

A parallel set of runs went through the Saarland CS HTCondor pool, where the
environment is deliberately unlike this one. Documenting both is the point:
a reproduction that only holds on the machine that produced it is not a
reproduction.

| | Laptop (this file) | Cluster (`cluster/environment_cluster_linux.txt`) |
|---|---|---|
| Python | 3.13.2 | 3.10 |
| torch | 2.13.0 / CUDA 12.6 | 2.3.1 / CUDA 12.1 |
| transformers | 4.46.3 | 4.44.2 |
| numpy | 2.x | 1.26.4 |
| peft | 0.13.2 | 0.12.0 |
| LAVIS | vendored, four edits | `salesforce-lavis==1.0.2`, pip-installed |
| Base image | none | `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel` |

Two details are worth carrying forward. The cluster could install
`salesforce-lavis` outright, since Python 3.10 still has wheels for its
dependency tree; vendoring was forced by 3.13 rather than chosen. And the P100
rejected bfloat16 with a dtype mismatch, which means patch A2 was load-bearing
on both machines. `cluster/README.md` records the failing job.

## Hardware Notes

I ran everything on an RTX 4060 Laptop with 8 GB of VRAM, which constrains the
configuration in two ways worth recording.

**Generation batch size caps at 4.** Batches of 6, 8, 16, and 24 all exhaust
memory during five-beam search. Counterintuitively, batch 4 runs *faster* per
molecule than batch 8 appeared to, because batch 8 only ever seemed to fit by
oversubscribing into system RAM.

**Set a hard memory cap.** Windows WDDM permits a process to spill VRAM into
system RAM instead of raising OOM. An oversized batch therefore does not fail;
it thrashes. I measured throughput collapsing from 0.37 it/s to 0.03 it/s before
system RAM ran out and the machine froze. Exporting `MOLCA_MEM_FRAC=0.90`
converts that silent death spiral into an immediate, diagnosable OOM. After
applying the cap, the same loop ran at 6.30 it/s.

## Source Modifications

`apply_patches.py` applies every change and remains idempotent across repeated
runs. Running `git diff` inside `MolCA/` reproduces the exact change set.

| Group | Count | Purpose |
|---|---|---|
| A | 5 | Portability |
| B | 4 | Ablation harness |
| C | 2 | Harness convenience |
| D | 3 | Retrieval path |
