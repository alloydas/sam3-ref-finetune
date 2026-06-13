#!/bin/bash
set -euo pipefail
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate /work/mech-ai-scratch/alloy/.conda/envs/sam3
cd /work/mech-ai-scratch/alloy/sam3
python scripts_refcoco/convert_refcoco.py --dataset lmms-lab/RefCOCOplus --prefix refcocoplus \
  --out-root /work/mech-ai-scratch/alloy/refcocoplus_sam3 --splits val testA
