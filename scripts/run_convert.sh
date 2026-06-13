#!/bin/bash
set -euo pipefail
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate /work/mech-ai-scratch/alloy/.conda/envs/sam3
cd /work/mech-ai-scratch/alloy/sam3
export HF_HUB_DISABLE_PROGRESS_BARS=0
python scripts_refcoco/convert_refcoco.py --splits val testA --out-root /work/mech-ai-scratch/alloy/refcoco_sam3
