#!/usr/bin/env bash
# run_fullset.sh - the two headline conditions on the COMPLETE 3300-molecule
# CheBI-20 test split, rather than the 1000-molecule subset the condition
# matrix used.
#
# Why: the subset runs are exactly comparable to EACH OTHER (identical 1000
# molecules), which is what the ablation argument needs. They are not
# comparable to the paper's Table 2b, which is on all 3300. These two runs
# close that gap so the reproduction claim is like-for-like.
#
# Waits for the GPU first, so it can be queued behind a running job.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$HERE/deliverables/results/logs/FULLSET_STATUS.txt"
mkdir -p "$(dirname "$STAMP")"

export MOLCA_LIMIT_BATCHES=825      # 825 x 4 = 3300, the whole split
export MOLCA_BATCH=4

{
  echo "full-set runs"
  echo "molecules: 825 batches x 4 = 3300 (complete test split)"
  echo "queued   : $(date '+%Y-%m-%d %H:%M:%S')"
} > "$STAMP"

# do not collide with whatever is already on the GPU
for _ in $(seq 1 360); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -z "${used:-}" ] && break
    [ "$used" -lt 2000 ] && break
    sleep 30
done
sleep 45
echo "[$(date '+%H:%M:%S')] GPU free, starting" >> "$STAMP"

for cond in baseline_full shuffle_graph_full; do
    t0=$(date +%s)
    echo "[$(date '+%H:%M:%S')] START $cond" >> "$STAMP"
    bash "$HERE/run_eval.sh" "$cond" > /dev/null 2>&1
    rows=0
    for f in "$HERE/MolCA/all_checkpoints/local_$cond"/lightning_logs/version_*/predictions.txt; do
        [ -f "$f" ] || continue
        n=$(grep -c . "$f"); [ "$n" -gt "$rows" ] && rows=$n
    done
    s=$(tr '\r' '\n' < "$HERE/deliverables/results/logs/$cond.log" 2>/dev/null \
        | grep -oE "BLEU-2 score: [0-9.]+" | tail -1)
    printf '[%s] DONE  %-20s %s  rows=%s  (%d min)\n' "$(date '+%H:%M:%S')" \
           "$cond" "${s:-<no score>}" "$rows" "$((($(date +%s)-t0)/60))" >> "$STAMP"
done

echo "" >> "$STAMP"
echo "full-set runs finished $(date '+%Y-%m-%d %H:%M:%S')" >> "$STAMP"
