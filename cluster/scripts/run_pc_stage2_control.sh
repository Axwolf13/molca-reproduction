#!/bin/bash
source /home/mllp26_team007/molca_env/bin/activate
cd /home/mllp26_team007/MolCA
python stage2.py \
  --devices '[0]' \
  --filename pc_stage2_control \
  --stage2_path "all_checkpoints/stage2.ckpt" \
  --init_checkpoint "all_checkpoints/stage2.ckpt" \
  --opt_model 'facebook/galactica-1.3b' \
  --mode eval \
  --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' \
  --tune_gnn --llm_tune freeze \
  --inference_batch_size 8 \
  --precision '16-mixed' \
  --root "data/PubChem324kV2/PubChem324kV2/"
