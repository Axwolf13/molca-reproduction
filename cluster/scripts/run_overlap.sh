#!/bin/bash
source /home/mllp26_team007/molca_env/bin/activate
cd /home/mllp26_team007/MolCA
python transfer_overlap.py --smiles smiles_dump > results_overlap_smiles.txt 2>&1
