#!/usr/bin/env python3
"""Write the four submit files and payloads for the distance experiment.

Run once, here, then submit the four `.sub` files from the cluster's job
directory. Kept as a generator rather than four hand-written pairs so the
Condor stanza and the stage2 invocation stay identical across conditions,
leaving the environment variables as the only difference between jobs.
"""
import os
import stat

HERE = os.path.dirname(os.path.abspath(__file__))

# name -> environment that defines the condition
JOBS = [
    # Gate. Filler after the whole prompt moves both channels away equally, so
    # the relative geometry is untouched. If this collapses the way MOL_FIRST
    # did, the filler itself is the problem and jobs 2 to 4 are uninterpretable.
    ("chebi_filler_end6", {"MOLCA_FILLER_END": "6"}),

    # Filler between the SMILES span and the soft prompts. The SMILES is now
    # 37 Galactica tokens further from generation; the graph has not moved.
    ("chebi_filler_mid6", {"MOLCA_FILLER_MID": "6"}),

    # The same manipulation at 13 tokens rather than 37. Two levels turn a
    # single contrast into a dose-response curve, which is the difference
    # between "the number moved" and "the number moves with distance".
    ("chebi_filler_mid2", {"MOLCA_FILLER_MID": "2"}),
    ("chebi_filler_mid2_shufsmiles",
     {"MOLCA_FILLER_MID": "2", "MOLCA_SHUFFLE_SMILES": "1"}),

    # The measurement. Against job 2, this gives the cost of corrupting the
    # SMILES at increased distance. The known cost at normal distance is
    # 62.32 - 47.41 = +14.91.
    ("chebi_filler_mid6_shufsmiles",
     {"MOLCA_FILLER_MID": "6", "MOLCA_SHUFFLE_SMILES": "1"}),

    # Internal control. The graph's distance is unchanged, so its cost should
    # also be unchanged at 62.32 - 25.62 = +36.70. A moved number here would
    # mean the filler is doing something other than adding distance.
    ("chebi_filler_mid6_shufgraph",
     {"MOLCA_FILLER_MID": "6", "MOLCA_SHUFFLE_GRAPH": "1"}),
]

PAYLOAD = """#!/bin/bash
{exports}source /home/mllp26_team007/molca_env/bin/activate
cd /home/mllp26_team007/MolCA
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt')"
python stage2.py \\
  --devices '[0]' \\
  --filename {name} \\
  --stage2_path "all_checkpoints/archived/chebi.ckpt" \\
  --init_checkpoint "all_checkpoints/archived/chebi.ckpt" \\
  --peft_dir "all_checkpoints/archived/chebi_lora" \\
  --opt_model 'facebook/galactica-1.3b' \\
  --mode eval \\
  --prompt '[START_I_SMILES]{{}}[END_I_SMILES]. ' \\
  --tune_gnn \\
  --llm_tune lora \\
  --inference_batch_size 8 \\
  --precision '16-mixed' \\
  --root "data/ChEBI-20_data"
"""

SUB = """universe                = docker
docker_image            = pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
executable              = run_{name}.sh
output                  = runlogs/{name}.$(ClusterId).out
error                   = runlogs/{name}.$(ClusterId).err
log                     = runlogs/{name}.$(ClusterId).log
should_transfer_files   = YES
request_GPUs            = 1
request_CPUs            = 8
request_memory          = 32G
requirements            = UidDomain == "cs.uni-saarland.de"
+WantGPUHomeMounted     = true
+WantScratchMounted     = true
stream_output = True
stream_error  = True
queue
"""


def main():
    for name, env in JOBS:
        exports = "".join("export %s=%s\n" % kv for kv in sorted(env.items()))
        sh = os.path.join(HERE, "run_%s.sh" % name)
        with open(sh, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(PAYLOAD.format(name=name, exports=exports))
        os.chmod(sh, os.stat(sh).st_mode | stat.S_IXUSR | stat.S_IXGRP)
        sub = os.path.join(HERE, "%s.sub" % name)
        with open(sub, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(SUB.format(name=name))
        print("wrote run_%s.sh and %s.sub" % (name, name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
