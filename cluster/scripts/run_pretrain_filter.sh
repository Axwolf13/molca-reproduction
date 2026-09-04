#!/bin/bash
source /home/mllp26_team007/molca_env/bin/activate
cd /home/mllp26_team007/MolCA
python verify_pretrain_filter.py > results_pretrain_filter.txt 2>&1
