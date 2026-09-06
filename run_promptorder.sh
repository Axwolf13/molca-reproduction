#!/usr/bin/env bash
# Position vs modality. The prompt template normally appends the graph soft
# prompts AFTER the SMILES span, so they sit nearest the generation point.
# Both runs move them in front:
#   baseline_graphfirst       what does reordering alone cost?
#   shuffle_graph_graphfirst  does graph-following survive reordering?
# The first is required to interpret the second.
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_queue.sh" baseline_graphfirst shuffle_graph_graphfirst
