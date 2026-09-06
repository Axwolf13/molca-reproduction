#!/bin/bash
source /home/mllp26_team007/molca_env/bin/activate
cd /home/mllp26_team007/MolCA
python stage1.py \
  --root 'data/kv_data' \
  --gtm --lm \
  --devices '[0]' \
  --filename pcdes_evaluation \
  --init_checkpoint "all_checkpoints/stage1.ckpt" \
  --rerank_cand_num 128 \
  --num_query_token 8 \
  --match_batch_size 64 \
  --precision '16-mixed' \
  --mode eval
