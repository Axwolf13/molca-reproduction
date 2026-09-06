#!/bin/bash
source /home/mllp26_team007/molca_env/bin/activate
cd /home/mllp26_team007/MolCA
pip install -q rdkit 2>/dev/null
python dump_smiles.py --out smiles_dump
