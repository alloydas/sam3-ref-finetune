#!/bin/bash
set -euo pipefail
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate /work/mech-ai-scratch/alloy/.conda/envs/sam3
cd /work/mech-ai-scratch/alloy/sam3
export HF_TOKEN="$(tr -d '[:space:]' < ~/.cache/huggingface/token)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python sam3/train/train.py -c configs/refcoco/refl4_finetune.yaml --use-cluster 0 --num-gpus 1
