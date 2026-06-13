#!/bin/bash
# Zero-shot SAM3 eval on RefCOCO. Usage: run_eval.sh <split> <max_records>
#   split:        val | testA | testB | test   (default val)
#   max_records:  cap for a quick pass; -1 = full split (default 2000)
set -euo pipefail
SPLIT="${1:-val}"
MAXREC="${2:-2000}"

source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate /work/mech-ai-scratch/alloy/.conda/envs/sam3
cd /work/mech-ai-scratch/alloy/sam3

# Force the token to be read regardless of HF_HOME inconsistencies.
export HF_TOKEN="$(tr -d '[:space:]' < ~/.cache/huggingface/token)"

ROOT=/work/mech-ai-scratch/alloy/refcoco_sam3
python scripts_refcoco/eval_refcoco.py \
  --manifest "$ROOT/annotations/refcoco_${SPLIT}_manifest.json" \
  --img-root "$ROOT/images" \
  --resolution 1008 \
  --amp-dtype float16 \
  --max-records "$MAXREC" \
  --out "$ROOT/eval_${SPLIT}_zeroshot.json"
