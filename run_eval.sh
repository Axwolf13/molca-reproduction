#!/usr/bin/env bash
# run_eval.sh - one CheBI-20 captioning eval from MolCA's released checkpoint.
#
#   ./run_eval.sh <condition> [extra stage2.py args...]
#
# <condition> is a key of CONDITIONS in apply_patches.py, or "baseline".
# Everything is inference only: no training, no gradient, released weights.
#
#   ./run_eval.sh baseline --limit_val_batches 3      # smoke test, 24 molecules
#   ./run_eval.sh baseline                            # full 3300-molecule pass
#   ./run_eval.sh shuffle_graph
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/molca_venv/Scripts/python.exe"
export PYTHONPATH="$HERE/vendor"          # minimal vendored LAVIS
export TOKENIZERS_PARALLELISM=false
# 8 GB card: beam search KV cache fragments badly without this
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Windows WDDM will silently spill VRAM into system RAM rather than raising
# OOM. That does not fail, it thrashes: observed 0.37 -> 0.03 it/s, then the
# machine ran out of system RAM and froze. Cap it so an overrun raises instead.
export MOLCA_MEM_FRAC="${MOLCA_MEM_FRAC:-0.90}"
# Batch 4 is the ceiling for 5-beam generation on 8 GB once spilling into
# system RAM is disallowed. It is also faster per molecule than batch 8 was,
# because batch 8 only appeared to fit by oversubscribing.
BS="${MOLCA_BATCH:-4}"
# Molecules per condition. 250 x 4 = the first 1000 of the test split, so every
# condition sees exactly the same molecules in the same order.
LIMIT="${MOLCA_LIMIT_BATCHES:-250}"

COND="${1:-baseline}"; shift || true
case "$COND" in
  baseline)          ;;
  graph_only)        export MOLCA_GRAPH_ONLY=1 ;;
  shuffle_graph)     export MOLCA_SHUFFLE_GRAPH=1 ;;
  shuffle_graph_rev) export MOLCA_SHUFFLE_GRAPH_REV=1 ;;
  shuffle_smiles)    export MOLCA_SHUFFLE_SMILES=1 ;;
  rewire_graph)      export MOLCA_REWIRE_GRAPH=1 ;;
  null_graph)        export MOLCA_NULL_GRAPH=1 ;;
  # the 2x2 corner: wrong graph AND no SMILES to fall back on
  shuffle_graph_only) export MOLCA_SHUFFLE_GRAPH=1 MOLCA_GRAPH_ONLY=1 ;;
  # position-vs-modality: the template normally puts the soft prompts last,
  # nearest the generation point. These two move them first. Both are needed:
  # the baseline measures what reordering alone costs, which is the only way
  # to interpret what reordering does to graph-following.
  baseline_graphfirst)      export MOLCA_GRAPH_FIRST=1 ;;
  shuffle_graph_graphfirst) export MOLCA_GRAPH_FIRST=1 MOLCA_SHUFFLE_GRAPH=1 ;;
  # full 3300-molecule versions. Same interventions as baseline/shuffle_graph,
  # but kept under separate names: writing 3300 rows into local_baseline would
  # make it the newest version there, and analyse.py would then reject every
  # 1000-row condition as a length mismatch.
  baseline_full)            ;;
  shuffle_graph_full)       export MOLCA_SHUFFLE_GRAPH=1 ;;
  *) echo "unknown condition: $COND" >&2; exit 2 ;;
esac

cd "$HERE/MolCA"
mkdir -p ../deliverables/results/logs

echo "=== condition: $COND ==="
env | grep '^MOLCA_' || echo "(no interventions - baseline)"

"$PY" stage2.py \
  --devices "[0]" \
  --filename "local_$COND" \
  --stage2_path      "all_checkpoints/archived/chebi.ckpt" \
  --init_checkpoint  "all_checkpoints/archived/chebi.ckpt" \
  --peft_dir         "all_checkpoints/archived/chebi_lora" \
  --opt_model        "../models/galactica-1.3b" \
  --mode eval \
  --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' \
  --tune_gnn \
  --llm_tune lora \
  --inference_batch_size "$BS" \
  --batch_size "$BS" \
  --limit_val_batches "$LIMIT" \
  --precision '16-mixed' \
  --num_workers 0 \
  --root "data/ChEBI-20_data" \
  "$@" 2>&1 | tee "../deliverables/results/logs/$COND.log"
