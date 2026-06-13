#!/bin/bash
# Build a dedicated conda env for SAM3 training/eval on RefCOCO.
set -euo pipefail

ENV_PREFIX=/work/mech-ai-scratch/alloy/.conda/envs/sam3
REPO=/work/mech-ai-scratch/alloy/sam3

source /work/mech-ai/alloy/miniconda3/etc/profile.d/conda.sh

echo "=== [1/4] create env ==="
if [ ! -d "$ENV_PREFIX" ]; then
  conda create -y -p "$ENV_PREFIX" python=3.12
fi
conda activate "$ENV_PREFIX"
python --version

echo "=== [2/4] install torch (cu128) ==="
pip install --no-input torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

echo "=== [3/4] install sam3[train,notebooks] ==="
cd "$REPO"
pip install --no-input -e ".[train,notebooks]"

echo "=== [4/4] install datasets + hf ==="
pip install --no-input "datasets" "huggingface_hub[cli]"

echo "=== verify ==="
python - <<'PY'
import torch, sam3
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
print("sam3 OK", sam3.__file__)
PY
echo "=== DONE ==="
