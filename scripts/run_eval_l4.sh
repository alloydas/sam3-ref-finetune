#!/bin/bash
set -euo pipefail
SPLIT="${1:-val}"; MAXREC="${2:--1}"
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate /work/mech-ai-scratch/alloy/.conda/envs/sam3
cd /work/mech-ai-scratch/alloy/sam3
export HF_TOKEN="$(tr -d '[:space:]' < ~/.cache/huggingface/token)"
ROOT=/work/mech-ai-scratch/alloy/refl4_sam3
python scripts_refcoco/eval_refcoco.py \
  --manifest "$ROOT/annotations/refl4_${SPLIT}_manifest.json" \
  --img-root "$ROOT/images" --resolution 1008 --amp-dtype float16 \
  --box-only --max-records "$MAXREC" --out "$ROOT/eval_${SPLIT}_zeroshot.json"
