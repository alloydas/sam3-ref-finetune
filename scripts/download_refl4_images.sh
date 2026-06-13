#!/bin/bash
set -euo pipefail
source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh
conda activate /work/mech-ai-scratch/alloy/.conda/envs/sam3
export HF_TOKEN="$(tr -d '[:space:]' < ~/.cache/huggingface/token)"
ROOT=/work/mech-ai-scratch/alloy/refl4_sam3
mkdir -p "$ROOT/images"
echo "=== downloading images.tar.gz ==="
python - <<'PY'
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id="JierunChen/Ref-L4", repo_type="dataset", filename="images.tar.gz")
print("downloaded:", p)
open("/work/mech-ai-scratch/alloy/refl4_sam3/.tarpath","w").write(p)
PY
TAR=$(cat "$ROOT/.tarpath")
echo "=== extracting $TAR ==="
tar xzf "$TAR" -C "$ROOT/images"
echo "=== sample extracted layout ==="
find "$ROOT/images" -maxdepth 2 | head -15
echo "image count: $(find "$ROOT/images" -type f | wc -l)"
echo "=== DONE ==="
