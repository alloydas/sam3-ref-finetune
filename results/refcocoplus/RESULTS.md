# SAM3 on lmms-lab/RefCOCOplus (RefCOCO+) — Results

**Date:** 2026-06-12
**GPU:** 1× Quadro RTX 6000 (24 GB, Turing) — ISU Nova
**Model:** `facebook/sam3` (sam3.pt)
**Dataset:** `lmms-lab/RefCOCOplus` (eval-only: val / testA / testB — no train split)
**Task:** referring-expression detection (REC) + segmentation (RES)
**Protocol:** text prompt = referring sentence; top-1 predicted instance by score vs the
single GT object; metrics averaged over all sentences.
**wandb:** https://wandb.ai/sam3/sam3-refcocoplus/runs/99ay0i6m (account alloyuit, entity sam3)

## Headline comparison — full val (10,758 sentences)

| Metric | Zero-shot | Fine-tuned | Δ |
|---|---|---|---|
| REC Acc@0.5 (box) | 53.72 | **67.81** | **+14.09** |
| RES mIoU          | 48.16 | **58.02** | **+9.86**  |
| RES oIoU (cumul.) | 42.57 | **49.52** | **+6.95**  |
| RES Pr@0.5        | 51.98 | **65.22** | +13.24 |
| RES Pr@0.7        | 45.44 | **60.68** | +15.24 |
| RES Pr@0.9        | 22.31 | **30.89** | +8.58  |

RefCOCO+ is harder than RefCOCO (it forbids location words like "left"/"top", so
referring is purely appearance-based) — hence the lower absolute zero-shot numbers.

## Fine-tune setup

- Trained on **testA** (5,726 sentences; RefCOCO+ has no train split, so testA stands in
  — disjoint from val ⇒ cross-split generalization check, not the official protocol).
- **Frozen** vision + language backbones; only detection transformer + heads updated.
- 1,200 steps, batch 1, **fp32** (fp16 diverges on Turing), resolution 1008, ~35 min.
- Loss: running-avg `train_all_loss` ≈ **54 → 17**; peak GPU ~4 GB.
- Segmentation head NOT fine-tuned (kept from pretrained); RES still improves because the
  detection transformer feeds it better top-1 boxes.
- Logged to Weights & Biases (project `sam3-refcocoplus`, entity `sam3`).

## Artifacts

- `eval_val_zeroshot.json`, `eval_val_finetuned.json` — metric dumps
- `results_summary.csv` — table above, machine-readable
- `runs/finetune_testA/checkpoints/checkpoint.pt` — fine-tuned weights
- `runs/finetune_testA/wandb/` — wandb run
- `annotations/refcocoplus_{val,testA}_manifest.json`, `..._coco_gt.json`
- `images/` — RefCOCO+ images

## Reproduce

```bash
# convert
python scripts_refcoco/convert_refcoco.py --dataset lmms-lab/RefCOCOplus \
  --prefix refcocoplus --out-root /work/mech-ai-scratch/alloy/refcocoplus_sam3 --splits val testA
# zero-shot eval / fine-tune (wandb) / fine-tuned eval
bash scripts_refcoco/run_eval_plus.sh val -1
bash scripts_refcoco/run_finetune_plus.sh
bash scripts_refcoco/run_eval_ft_plus.sh val -1
```
