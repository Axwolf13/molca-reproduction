#!/usr/bin/env bash
# run_retrieval.sh - molecule-text retrieval (stage 1), the paper's Table 4.
#
# A second task, and a second falsifiable claim: the paper attributes +8.3
# accuracy (58.3 -> 66.6) to re-ranking the top-128 contrastive candidates with
# the molecule-text matching head. Both numbers come out of one run.
#
# Two evals:
#   pcdes  PCDes test split
#   momu   the MoMu "phy_data" split (--use_phy_eval)
#
# Inference only, from the released stage1.ckpt. No beam search, so this is
# minutes rather than hours.
#
# Waits for the full-set captioning queue to finish before touching the GPU.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/molca_venv/Scripts/python.exe"
export PYTHONPATH="$HERE/vendor"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MOLCA_MEM_FRAC="${MOLCA_MEM_FRAC:-0.90}"

LOGS="$HERE/deliverables/results/logs"
STAMP="$LOGS/RETRIEVAL_STATUS.txt"
mkdir -p "$LOGS"
echo "retrieval queued: $(date '+%Y-%m-%d %H:%M:%S')" > "$STAMP"

# wait for the captioning queue to declare itself done, then for the GPU
for _ in $(seq 1 480); do
    grep -q "full-set runs finished" "$LOGS/FULLSET_STATUS.txt" 2>/dev/null && break
    sleep 30
done
for _ in $(seq 1 240); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -z "${used:-}" ] && break
    [ "$used" -lt 2000 ] && break
    sleep 30
done
sleep 30
echo "[$(date '+%H:%M:%S')] GPU free, starting retrieval" >> "$STAMP"

cd "$HERE/MolCA"
# 64 is the match_batch_size that reproduced the paper on the cluster, and
# the value the upstream README's own retrieval command uses.
for eval_name in pcdes momu; do
    extra=""
    [ "$eval_name" = "momu" ] && extra="--use_phy_eval"
    t0=$(date +%s)
    echo "[$(date '+%H:%M:%S')] START $eval_name" >> "$STAMP"
    "$PY" stage1.py \
      --root 'data/kv_data' \
      --gtm --lm \
      --devices "[0]" \
      --filename "local_retrieval_$eval_name" \
      --init_checkpoint "all_checkpoints/stage1.ckpt" \
      --rerank_cand_num 128 \
      --num_query_token 8 \
      --match_batch_size 64 \
      --batch_size 64 \
      --num_workers 0 \
      --precision '16-mixed' \
      --mode eval $extra \
      > "$LOGS/retrieval_$eval_name.log" 2>&1
    printf '[%s] DONE  %-6s (%d min)\n' "$(date '+%H:%M:%S')" "$eval_name" \
           "$((($(date +%s)-t0)/60))" >> "$STAMP"
    tr '\r' '\n' < "$LOGS/retrieval_$eval_name.log" \
      | grep -iE "acc|recall|R@" | tail -12 >> "$STAMP"
done
echo "" >> "$STAMP"
echo "retrieval finished: $(date '+%Y-%m-%d %H:%M:%S')" >> "$STAMP"
