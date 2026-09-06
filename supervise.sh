#!/usr/bin/env bash
# supervise.sh - keep the condition queue alive without a human watching it.
#
# Failure modes this handles:
#   * the queue process dies (OOM kill, driver reset, stray Ctrl-C)  -> restart
#   * one condition fails but others could still run                 -> skip it
#   * a condition OOMs                                               -> retry
#     once at half batch, then give up on it rather than loop forever
#
# It does NOT survive a machine reboot. If the PC restarts, run
#   bash supervise.sh &
# again; everything already finished is on disk and will be skipped.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CK="$HERE/MolCA/all_checkpoints"
LOGS="$HERE/deliverables/results/logs"
SUP="$LOGS/SUPERVISOR.txt"
mkdir -p "$LOGS"

CONDS="baseline shuffle_graph shuffle_smiles rewire_graph graph_only null_graph shuffle_graph_rev shuffle_graph_only"
WANT_ROWS=$(( ${MOLCA_LIMIT_BATCHES:-250} * ${MOLCA_BATCH:-4} ))
MAX_TRIES=2

say() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" >> "$SUP"; }

# a condition counts as done when some version_N holds a full predictions file
done_rows() {
    local best=0 f n
    for f in "$CK/local_$1"/lightning_logs/version_*/predictions.txt; do
        [ -f "$f" ] || continue
        n=$(grep -c . "$f" 2>/dev/null || echo 0)
        [ "$n" -gt "$best" ] && best=$n
    done
    echo "$best"
}
is_done() { [ "$(done_rows "$1")" -ge "$WANT_ROWS" ]; }

{
  echo "supervisor started $(date '+%Y-%m-%d %H:%M:%S')"
  echo "expecting $WANT_ROWS rows per condition"
  echo ""
} > "$SUP"

# An earlier queue may still be mid-condition. Two jobs on one 8 GB card would
# OOM both, so wait for the GPU to go idle before taking over. Idle desktop is
# ~750 MiB; a live run sits around 4700 MiB.
for _ in $(seq 1 240); do          # up to 2h
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -z "${used:-}" ] && break
    [ "$used" -lt 2000 ] && break
    say "waiting for GPU to free (${used} MiB in use)"
    sleep 30
done
# VRAM is released when the process exits, which is just after it writes
# predictions.txt. Settle briefly so the is_done check sees the finished file
# rather than re-running a condition that had in fact completed.
sleep 45
say "GPU free, taking over"

for cond in $CONDS; do
    if is_done "$cond"; then
        say "SKIP  $cond (already has $(done_rows "$cond") rows)"
        continue
    fi
    # Rotation-based conditions substitute WITHIN a batch, so their meaning
    # depends on the batch size. Retrying those at a smaller batch would
    # silently change which molecule's graph each example received and
    # misalign them against the batch-4 baseline. Only per-molecule
    # conditions may be retried smaller.
    case "$cond" in
        shuffle_graph|shuffle_graph_rev|shuffle_smiles|shuffle_graph_only)
            SHRINKABLE=0 ;;
        *) SHRINKABLE=1 ;;
    esac
    for try in $(seq 1 $MAX_TRIES); do
        bs=${MOLCA_BATCH:-4}
        if [ "$try" -gt 1 ] && [ "$SHRINKABLE" -eq 1 ]; then
            bs=2
            say "retrying $cond at batch $bs (per-molecule condition, safe to shrink)"
        elif [ "$try" -gt 1 ]; then
            say "retrying $cond at batch $bs (rotation condition, batch size is fixed)"
        fi
        say "START $cond (attempt $try/$MAX_TRIES, batch $bs)"
        t0=$(date +%s)
        MOLCA_BATCH=$bs bash "$HERE/run_eval.sh" "$cond" > /dev/null 2>&1
        mins=$(( ($(date +%s) - t0) / 60 ))
        if is_done "$cond"; then
            score=$(tr '\r' '\n' < "$LOGS/$cond.log" 2>/dev/null \
                    | grep -oE "BLEU-2 score: [0-9.]+" | tail -1)
            say "DONE  $cond  ${score:-<no score>}  (${mins} min)"
            break
        fi
        oom=$(grep -c "out of memory" "$LOGS/$cond.log" 2>/dev/null || echo 0)
        say "FAIL  $cond (attempt $try, ${mins} min, oom_lines=$oom)"
        if [ "$try" -eq "$MAX_TRIES" ]; then
            say "GIVING UP on $cond - moving to the next condition"
        fi
    done
done

say ""
say "all conditions attempted; running analysis"
"$HERE/molca_venv/Scripts/python.exe" "$HERE/analyse.py" >> "$SUP" 2>&1 \
    && say "analysis written to deliverables/results/channel_conflict.txt" \
    || say "analysis FAILED - see above"

say ""
say "SUMMARY"
for cond in $CONDS; do
    printf '  %-20s %5s rows\n' "$cond" "$(done_rows "$cond")" >> "$SUP"
done
say "supervisor finished $(date '+%Y-%m-%d %H:%M:%S')"
