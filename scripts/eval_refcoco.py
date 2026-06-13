#!/usr/bin/env python
# Copyright (c) 2026. Standalone SAM3 RefCOCO REC/RES evaluation.
"""
Evaluate a SAM3 image model on a RefCOCO manifest produced by convert_refcoco.py.

For each (image, referring sentence) we run SAM3 with the sentence as the text
prompt, take the highest-scoring predicted instance (single-target RefCOCO
convention), and score it against the GT object:

  REC (box):  Acc@0.5  = fraction with box IoU >= 0.5
  RES (mask): mIoU     = mean per-sample mask IoU
              oIoU     = sum(intersection) / sum(union)  (cumulative IoU)
              Pr@X     = fraction with mask IoU >= X  (X in 0.5/0.7/0.9)

Image embeddings are cached per file_name so the many sentences that share an
image are encoded once.
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util
from tqdm import tqdm

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def poly_to_mask(segm, h, w):
    rles = mask_util.frPyObjects(segm, h, w)
    rle = mask_util.merge(rles)
    return mask_util.decode(rle).astype(bool)


def grounding_boxes_only(model, processor, state, prompt, device):
    """Run text-prompted grounding and return (boxes_xyxy_norm, scores) WITHOUT
    interpolating masks. Mirrors Sam3Processor._forward_grounding minus the seg
    head, so it is safe for large images / many instances (REC-only)."""
    from sam3.model import box_ops

    text_outputs = model.backbone.forward_text([prompt], device=device)
    state["backbone_out"].update(text_outputs)
    geom = model._get_dummy_prompt()
    outputs = model.forward_grounding(
        backbone_out=state["backbone_out"],
        find_input=processor.find_stage,
        geometric_prompt=geom,
        find_target=None,
    )
    out_bbox = outputs["pred_boxes"]
    out_probs = outputs["pred_logits"].sigmoid()
    presence = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
    out_probs = (out_probs * presence).squeeze(-1)
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(out_bbox)  # normalized [0,1]
    if boxes_xyxy.dim() == 3:      # drop batch dim -> [N, 4]
        boxes_xyxy = boxes_xyxy[0]
    if out_probs.dim() == 2:       # -> [N]
        out_probs = out_probs[0]
    return boxes_xyxy, out_probs


def box_iou_xyxy(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="Local checkpoint path; default downloads facebook/sam3.")
    ap.add_argument("--overlay-checkpoint", default=None,
                    help="Fine-tuned checkpoint to overlay (strict=False) on top of "
                         "the base model. Keeps the base seg head; replaces the "
                         "fine-tuned detection weights.")
    ap.add_argument("--resolution", type=int, default=1008)
    ap.add_argument("--amp-dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"],
                    help="Turing GPUs (RTX 6000) should use float16.")
    ap.add_argument("--max-records", type=int, default=-1)
    ap.add_argument("--out", default=None, help="Optional path to dump per-record results JSON.")
    ap.add_argument("--box-only", action="store_true",
                    help="REC-only datasets (e.g. Ref-L4): build without the seg head "
                         "and read boxes/scores directly from forward_grounding, "
                         "skipping the (memory-heavy) full-res mask interpolation.")
    args = ap.parse_args()

    with open(args.manifest) as f:
        records = json.load(f)
    if args.max_records > 0:
        records = records[: args.max_records]
    print(f"Loaded {len(records)} records from {args.manifest}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(
        checkpoint_path=args.checkpoint,
        load_from_HF=args.checkpoint is None,
        device=device,
        eval_mode=True,
        enable_segmentation=not args.box_only,
    )
    if args.overlay_checkpoint:
        ck = torch.load(args.overlay_checkpoint, map_location=device, weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[overlay] loaded {len(sd)} fine-tuned params | "
              f"missing(kept from base, e.g. seg head): {len(missing)} | "
              f"unexpected: {len(unexpected)}")

    # confidence_threshold=0.0 -> keep all candidates, then take top-1 by score.
    processor = Sam3Processor(model, resolution=args.resolution,
                              device=device, confidence_threshold=0.0)

    amp_dtype = {"float16": torch.float16,
                 "bfloat16": torch.bfloat16,
                 "float32": torch.float32}[args.amp_dtype]

    # Group records by image so we encode each image once.
    by_img = {}
    for r in records:
        by_img.setdefault(r["file_name"], []).append(r)

    box_ious, mask_ious = [], []
    tot_inter = tot_union = 0
    results = []
    # Records carry masks only for RES datasets (RefCOCO/+/g); Ref-L4 is box-only.
    has_seg = any(r.get("segmentation") for r in records)

    autocast = torch.autocast("cuda", dtype=amp_dtype) if device == "cuda" else \
        torch.autocast("cpu", dtype=torch.float32, enabled=False)

    with torch.inference_mode(), autocast:
        for file_name, recs in tqdm(by_img.items(), desc="eval"):
            image = Image.open(os.path.join(args.img_root, file_name)).convert("RGB")
            state = processor.set_image(image)
            # Without a seg head, forward_grounding pops backbone_fpn (line 424 of
            # sam3_image.py); cache it so each expression on this image can restore it.
            fpn_cache = state["backbone_out"].get("backbone_fpn")
            for r in recs:
                processor.reset_all_prompts(state)
                if args.box_only:
                    state["backbone_out"]["backbone_fpn"] = fpn_cache
                h, w = r["height"], r["width"]
                gt_box = [r["bbox"][0], r["bbox"][1],
                          r["bbox"][0] + r["bbox"][2], r["bbox"][1] + r["bbox"][3]]
                seg = r.get("segmentation")
                gt_mask = poly_to_mask(seg, h, w) if (seg and not args.box_only) else None

                if args.box_only:
                    boxes_norm, scores = grounding_boxes_only(
                        model, processor, state, r["expression"], device)
                    if scores.numel() == 0:
                        biou = 0.0
                    else:
                        top = int(torch.argmax(scores).item())
                        b = boxes_norm[top].float().cpu().tolist()
                        pred_box = [b[0]*w, b[1]*h, b[2]*w, b[3]*h]
                        biou = box_iou_xyxy(pred_box, gt_box)
                    box_ious.append(biou)
                    if args.out:
                        results.append({"id": r["id"], "box_iou": biou})
                    continue

                state = processor.set_text_prompt(prompt=r["expression"], state=state)
                scores = state["scores"]

                if scores.numel() == 0:
                    biou, miou = 0.0, 0.0
                    inter, union = 0, (int(gt_mask.sum()) if gt_mask is not None else 0)
                else:
                    top = int(torch.argmax(scores).item())
                    pred_box = state["boxes"][top].float().cpu().tolist()
                    biou = box_iou_xyxy(pred_box, gt_box)
                    if gt_mask is not None:
                        pred_mask = state["masks"][top, 0].cpu().numpy().astype(bool)
                        inter = int(np.logical_and(pred_mask, gt_mask).sum())
                        union = int(np.logical_or(pred_mask, gt_mask).sum())
                        miou = inter / union if union > 0 else 0.0
                    else:
                        inter = union = 0
                        miou = 0.0

                box_ious.append(biou)
                if gt_mask is not None:
                    mask_ious.append(miou)
                    tot_inter += inter
                    tot_union += union
                if args.out:
                    rec_out = {"id": r["id"], "box_iou": biou}
                    if gt_mask is not None:
                        rec_out["mask_iou"] = miou
                    results.append(rec_out)

    box_ious = np.array(box_ious)
    mask_ious = np.array(mask_ious)
    n = len(box_ious)
    print("\n================ Ref results ================")
    print(f"records evaluated : {n}")
    print(f"[REC]  Acc@0.5 (box)   : {100*(box_ious>=0.5).mean():.2f}")
    print(f"[REC]  Acc@0.75 (box)  : {100*(box_ious>=0.75).mean():.2f}")
    print(f"[REC]  Acc@0.9 (box)   : {100*(box_ious>=0.9).mean():.2f}")
    print(f"[REC]  mean box IoU    : {100*box_ious.mean():.2f}")
    summary = {
        "n": n,
        "box_acc@0.5": float((box_ious >= 0.5).mean()),
        "box_acc@0.75": float((box_ious >= 0.75).mean()),
        "box_acc@0.9": float((box_ious >= 0.9).mean()),
        "mean_box_iou": float(box_ious.mean()),
    }
    if has_seg and mask_ious.size:
        print(f"[RES]  mIoU            : {100*mask_ious.mean():.2f}")
        print(f"[RES]  oIoU (cumul.)   : {100*tot_inter/max(tot_union,1):.2f}")
        for t in (0.5, 0.7, 0.9):
            print(f"[RES]  Pr@{t}          : {100*(mask_ious>=t).mean():.2f}")
        summary["mIoU"] = float(mask_ious.mean())
        summary["oIoU"] = float(tot_inter / max(tot_union, 1))
    print("=================================================")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "records": results}, f)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
