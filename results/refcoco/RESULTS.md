# SAM3 on lmms-lab/RefCOCO — Results

**Date:** 2026-06-12
**GPU:** 1× Quadro RTX 6000 (24 GB, Turing) — ISU Nova
**Model:** `facebook/sam3` (sam3.pt, 3.45 GB)
**Dataset:** `lmms-lab/RefCOCO` (eval-only: val / testA / testB / test — no train split)
**Task:** referring-expression detection (REC) + segmentation (RES)
**Protocol:** text prompt = referring sentence; take the top-1 predicted instance by
score; compare to the single GT object. Metrics averaged over all sentences.

## Headline comparison — full val (25,080 sentences)

| Metric | Zero-shot | Fine-tuned | Δ |
|---|---|---|---|
| REC Acc@0.5 (box) | 68.13 | **79.73** | **+11.60** |
| RES mIoU          | 59.83 | **68.86** | **+9.03**  |
| RES oIoU (cumul.) | 53.62 | **63.03** | **+9.41**  |
| RES Pr@0.5        | 66.12 | **77.40** | +11.28 |
| RES Pr@0.7        | 59.61 | **72.32** | +12.71 |
| RES Pr@0.9        | 28.80 | **36.55** | +7.75  |

Both columns use the identical val set + protocol, so deltas are directly comparable.

## Fine-tune setup

- Trained on **testA** (5,657 sentences; lmms-lab/RefCOCO has no `train` split, so testA
  stands in — disjoint from val ⇒ valid cross-split generalization check, NOT the
  official RefCOCO train→val protocol).
- **Frozen** vision + language backbones; only the detection transformer + heads updated.
- 1,200 steps, batch size 1, **fp32** (fp16 diverges on Turing), resolution 1008, ~24 min.
- Loss: running-avg `train_all_loss` ≈ **165 → 17**; peak GPU ~4 GB.
- The **segmentation head was NOT fine-tuned** (kept from pretrained); RES still improved
  because the detection transformer feeds it better-localized top-1 boxes.

## Artifacts

- `eval_val_zeroshot.json`  — zero-shot metric dump
- `eval_val_finetuned.json` — fine-tuned metric dump
- `results_summary.csv`     — the table above, machine-readable
- `runs/finetune_testA/checkpoints/checkpoint.pt` — fine-tuned weights (3.6 GB)
- `annotations/refcoco_{val,testA}_manifest.json` — per-sentence manifests
- `annotations/refcoco_{val,testA}_coco_gt.json`  — COCO-format GT
- `images/` — 10,198 RefCOCO images

## Reproduce

```bash
# env: /work/mech-ai-scratch/alloy/.conda/envs/sam3 ; repo: /work/mech-ai-scratch/alloy/sam3
# zero-shot eval (full val)
bash scripts_refcoco/run_eval.sh val -1
# fine-tune (1 capped epoch on testA, fp32, frozen backbone)
bash scripts_refcoco/run_finetune.sh
# fine-tuned eval (overlays FT detection weights on pretrained seg head)
bash scripts_refcoco/run_eval_ft.sh val -1
```

## Code changes required to enable training

- `sam3/perflib/fused.py` — ViT fused MLP is inference-only (asserts grad disabled);
  added a differentiable fallback when grad is enabled.
- `sam3/train/refcoco_model.py` — builder wrapper that freezes the perception backbones
  (full model ≈1.7B params won't fully fine-tune on 24 GB).
- `sam3/train/refcoco_loader.py` — RefCOCO referring-expression loader (one datapoint =
  one image + sentence + GT object).
- `sam3/train/configs/refcoco/refcoco_finetune.yaml` — fp32 + frozen-backbone config.
