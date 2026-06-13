# SAM3 on JierunChen/Ref-L4 — Results

**Date:** 2026-06-13
**GPU:** 1× Quadro RTX 6000 (24 GB, Turing) — ISU Nova
**Model:** `facebook/sam3`
**Dataset:** `JierunChen/Ref-L4` (val / test; no train split). REC-only (box; no masks).
Long, compositional referring expressions; images from COCO + Objects365.
**Protocol:** text prompt = caption; top-1 predicted box by score vs GT box.
**wandb:** https://wandb.ai/sam3/sam3-refl4/runs/lkk8zth9 (account alloyuit, entity sam3)

## Headline comparison — full val (13,420 expressions)

| Metric | Zero-shot | Fine-tuned | Δ |
|---|---|---|---|
| REC Acc@0.5  | 64.59 | **73.06** | **+8.47** |
| REC Acc@0.75 | 58.33 | **66.93** | **+8.60** |
| REC Acc@0.9  | 47.18 | **54.17** | **+6.99** |
| mean box IoU | 62.14 | **69.21** | **+7.07** |

## Fine-tune setup

- Trained on **test** (31,921 expressions; Ref-L4 has no train split, so test stands in —
  disjoint from val ⇒ cross-split generalization check, not the official protocol).
- **Frozen** vision + language backbones; only detection transformer + heads updated.
- 1,200 steps (capped via limit_ids), batch 1, **fp32**, resolution 1008, ~42 min, ~4 GB.
- Loss: running-avg `train_all_loss` ≈ 133 → ~20 (smaller relative drop — Ref-L4 is harder).
- Logged to Weights & Biases (project `sam3-refl4`, entity `sam3`).

## Notes specific to Ref-L4
- Images shipped separately (`images.tar.gz`, 9,735 files) — not embedded in the parquet.
- REC-only: evaluated with `eval_refcoco.py --box-only`, which builds without the seg head
  and reads boxes/scores directly from `forward_grounding` (avoids the full-res mask
  interpolation that OOMs on large Objects365 images), plus restores `backbone_fpn` between
  expressions (the seg-head-less path pops it).

## Artifacts
- `eval_val_zeroshot.json`, `eval_val_finetuned.json` — metric dumps
- `results_summary.csv`
- `runs/finetune_test/checkpoints/checkpoint.pt` — fine-tuned weights
