#!/bin/bash
source /home/mllp26_team007/molca_env/bin/activate
cd /home/mllp26_team007/MolCA
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt')"
python stage2.py \
  --devices '[0]' \
  --filename chebi_beam1 \
  --stage2_path "all_checkpoints/archived/chebi.ckpt" \
  --init_checkpoint "all_checkpoints/archived/chebi.ckpt" \
  --peft_dir "all_checkpoints/archived/chebi_lora" \
  --opt_model 'facebook/galactica-1.3b' \
  --mode eval \
  --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' \
  --tune_gnn \
  --llm_tune lora \
  --inference_batch_size 8 \
  --num_beams 1 \
  --precision '16-mixed' \
  --root "data/ChEBI-20_data"
