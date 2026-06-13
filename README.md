# SAM 3 — Referring-Expression Fine-tuning & Evaluation (RefCOCO / RefCOCO+ / Ref-L4)

Pipeline to **evaluate and fine-tune [SAM 3](https://huggingface.co/facebook/sam3)**
on referring-expression datasets:

- [`lmms-lab/RefCOCO`](https://huggingface.co/datasets/lmms-lab/RefCOCO) — REC + RES
- [`lmms-lab/RefCOCOplus`](https://huggingface.co/datasets/lmms-lab/RefCOCOplus) — REC + RES
- [`JierunChen/Ref-L4`](https://huggingface.co/datasets/JierunChen/Ref-L4) — REC only (box, long compositional expressions)

The model is prompted with a referring sentence; we take the top-scoring predicted
instance and score it against the GT object. Fine-tuning **freezes the vision +
language backbones** and trains only the detection transformer + heads, so it fits on
a single 24 GB GPU.

## Results (full val, zero-shot vs fine-tuned)

**RefCOCO** (25,080 sentences) — REC + RES

| Metric | Zero-shot | Fine-tuned | Δ |
|---|---|---|---|
| REC Acc@0.5 | 68.13 | **79.73** | +11.60 |
| RES mIoU    | 59.83 | **68.86** | +9.03  |
| RES oIoU    | 53.62 | **63.03** | +9.41  |

**RefCOCO+** (10,758 sentences) — REC + RES

| Metric | Zero-shot | Fine-tuned | Δ |
|---|---|---|---|
| REC Acc@0.5 | 53.72 | **67.81** | +14.09 |
| RES mIoU    | 48.16 | **58.02** | +9.86  |
| RES oIoU    | 42.57 | **49.52** | +6.95  |

**Ref-L4** (13,420 expressions) — REC only

| Metric | Zero-shot | Fine-tuned | Δ |
|---|---|---|---|
| REC Acc@0.5  | 64.59 | **73.06** | +8.47 |
| REC Acc@0.75 | 58.33 | **66.93** | +8.60 |
| REC Acc@0.9  | 47.18 | **54.17** | +6.99 |
| mean box IoU | 62.14 | **69.21** | +7.07 |

Per-dataset details + machine-readable summaries are in `results/`.

> Fine-tunes are short demonstration runs (frozen backbone, fp32, ~1200 steps, batch 1,
> res 1008). None of these datasets ships a `train` split, so a held-out split (testA, or
> Ref-L4 `test`) is used as the training set — a cross-split generalization check, not the
> official train→val protocol.

## Repository layout

```
scripts/        data conversion, evaluation, and run wrappers
configs/        Hydra fine-tune configs (one per dataset)
sam3_modules/   new files to drop into the SAM 3 tree:
                  refcoco_loader.py  -> sam3/train/data/
                  refcoco_model.py   -> sam3/train/
sam3_patches/   library_edits.patch (2 edits to the SAM 3 repo, see below)
results/        metrics per dataset (RESULTS.md, CSV, JSON dumps)
```

## Setup

```bash
git clone https://github.com/facebookresearch/sam3.git && cd sam3
pip install -e ".[train,notebooks]"
pip install datasets wandb

# apply this repo's changes
cp <this-repo>/sam3_modules/refcoco_loader.py sam3/train/data/
cp <this-repo>/sam3_modules/refcoco_model.py  sam3/train/
cp <this-repo>/configs/*.yaml                 sam3/train/configs/refcoco/
cp <this-repo>/scripts/*                      scripts_refcoco/
git apply <this-repo>/sam3_patches/library_edits.patch
```

`library_edits.patch` makes two changes:
1. **`sam3/perflib/fused.py`** — the ViT fused MLP is inference-only (asserts grad
   disabled). Adds a differentiable fallback when grad is enabled so the model can be
   fine-tuned.
2. **`sam3/train/utils/logger.py`** — adds a `WandbLogger` + `make_wandb_logger` factory
   and wires Weights & Biases into the `Logger` facade (the repo only shipped TensorBoard).

## Usage

```bash
# 1. convert a dataset (downloads from HF; RefCOCO embeds images, Ref-L4 ships images.tar.gz)
python scripts/convert_refcoco.py --dataset lmms-lab/RefCOCO     --prefix refcoco     --out-root <root> --splits val testA
python scripts/convert_refcoco.py --dataset lmms-lab/RefCOCOplus --prefix refcocoplus --out-root <root> --splits val testA
bash   scripts/download_refl4_images.sh && python scripts/convert_refl4.py --splits val test

# 2. zero-shot eval
bash scripts/run_eval.sh val -1            # RefCOCO  (REC+RES)
bash scripts/run_eval_l4.sh val -1         # Ref-L4   (REC only, --box-only)

# 3. fine-tune (frozen backbone, wandb)
bash scripts/run_finetune.sh               # RefCOCO
bash scripts/run_finetune_plus.sh          # RefCOCO+
bash scripts/run_finetune_l4.sh            # Ref-L4

# 4. fine-tuned eval (overlays fine-tuned detection weights on the base model)
bash scripts/run_eval_ft.sh val -1
```

`scripts/eval_refcoco.py --box-only` builds without the seg head and reads boxes/scores
directly from `forward_grounding` — required for Ref-L4 (no masks; large images would OOM
the full-res mask interpolation).

## Notes
- GPU used: 1× Quadro RTX 6000 (24 GB, Turing). fp16 AMP diverges on Turing (NaN in the
  matcher), so training uses fp32 — cheap here because the backbone is frozen (~4 GB).
- SAM 3 weights are gated; request access on the HF repo and `hf auth login` first.
