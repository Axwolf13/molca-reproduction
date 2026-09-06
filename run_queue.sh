#!/usr/bin/env bash
# run_queue.sh - run conditions back to back on one GPU.
#
# An 8 GB card fits exactly one 1.3B five-beam job, so conditions are
# serialised. Ordered by value, because a crash loses only what has not run
# yet: everything already finished is on disk.
#
#   baseline            anchor - every other condition is read against it
#   shuffle_graph       the effect: own SMILES, next molecule's graph
#   shuffle_smiles      MIRROR CONTROL - does the model follow the graph, or
#                       merely whichever channel was left uncorrupted?
#   rewire_graph        coherent-but-wrong molecule vs incoherent topology
#   graph_only          SMILES withheld entirely
#   null_graph          graph present but uninformative (zeroed atom features)
#   shuffle_graph_rev   same effect with rotation -1, a replication check
#   shuffle_graph_only  2x2 corner: wrong graph AND no SMILES to fall back on
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUEUE="${*:-baseline shuffle_graph shuffle_smiles rewire_graph graph_only null_graph shuffle_graph_rev shuffle_graph_only}"
STAMP="$HERE/deliverables/results/logs/QUEUE_STATUS.txt"
mkdir -p "$(dirname "$STAMP")"

{
  echo "queue    : $QUEUE"
  echo "molecules: ${MOLCA_LIMIT_BATCHES:-250} batches x ${MOLCA_BATCH:-4}"
  echo "started  : $(date '+%Y-%m-%d %H:%M:%S')"
} > "$STAMP"

for cond in $QUEUE; do
    t0=$(date +%s)
    echo "" >> "$STAMP"
    echo "[$(date '+%H:%M:%S')] START $cond" >> "$STAMP"
    if bash "$HERE/run_eval.sh" "$cond" > /dev/null 2>&1; then
        s=$(tr '\r' '\n' < "$HERE/deliverables/results/logs/$cond.log" \
            | grep -oE "BLEU-2 score: [0-9.]+" | tail -1)
        printf '[%s] DONE  %-20s %s  (%d min)\n' "$(date '+%H:%M:%S')" "$cond" \
               "${s:-<no score parsed>}" "$((($(date +%s)-t0)/60))" >> "$STAMP"
    else
        printf '[%s] FAIL  %-20s see logs/%s.log  (%d min)\n' "$(date '+%H:%M:%S')" \
               "$cond" "$cond" "$((($(date +%s)-t0)/60))" >> "$STAMP"
    fi
done
echo "" >> "$STAMP"
echo "queue finished: $(date '+%Y-%m-%d %H:%M:%S')" >> "$STAMP"
