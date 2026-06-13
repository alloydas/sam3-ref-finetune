#!/usr/bin/env python
# Copyright (c) 2026. JierunChen/Ref-L4 -> SAM3 box-only manifest converter.
"""
Ref-L4 is a REC (referring-expression comprehension) benchmark: one caption ->
one bounding box, NO segmentation masks, and images shipped separately in
images.tar.gz (COCO + Objects365 + ... sources). We emit the same flat manifest
format used by the RefCOCO pipeline, minus the `segmentation` field:

  { id, file_name, expression(=caption), bbox[xywh], height, width, object_id }

Splits: val / test (no train split).
"""
import argparse
import json
import os

from datasets import load_dataset
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="/work/mech-ai-scratch/alloy/refl4_sam3")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--max-rows", type=int, default=-1)
    args = ap.parse_args()

    ann_dir = os.path.join(args.out_root, "annotations")
    os.makedirs(ann_dir, exist_ok=True)

    for split in args.splits:
        print(f"\n=== JierunChen/Ref-L4 split: {split} ===")
        ds = load_dataset("JierunChen/Ref-L4", "ref_l4", split=split)
        if args.max_rows > 0:
            ds = ds.select(range(min(args.max_rows, len(ds))))

        manifest = []
        coco_images, coco_anns = [], []
        seen_imgs = {}
        for i, row in enumerate(tqdm(ds, desc=split)):
            fn = row["file_name"]
            h, w = int(row["height"]), int(row["width"])
            bbox = [float(v) for v in row["bbox"]]  # xywh

            if fn not in seen_imgs:
                seen_imgs[fn] = len(coco_images)
                coco_images.append({"id": seen_imgs[fn], "file_name": fn,
                                    "height": h, "width": w})
            coco_anns.append({
                "id": i, "image_id": seen_imgs[fn], "category_id": 1,
                "bbox": bbox, "area": float(bbox[2] * bbox[3]),
                "iscrowd": 0,
            })
            manifest.append({
                "id": i,
                "file_name": fn,
                "expression": row["caption"].strip(),
                "bbox": bbox,
                "height": h,
                "width": w,
                "iscrowd": 0,
                "object_id": i,
                # no "segmentation": Ref-L4 is box-only
            })

        coco_gt = {"images": coco_images, "annotations": coco_anns,
                   "categories": [{"id": 1, "name": "object"}]}
        man_path = os.path.join(ann_dir, f"refl4_{split}_manifest.json")
        gt_path = os.path.join(ann_dir, f"refl4_{split}_coco_gt.json")
        with open(man_path, "w") as f:
            json.dump(manifest, f)
        with open(gt_path, "w") as f:
            json.dump(coco_gt, f)
        print(f"  rows={len(manifest)}  unique_images={len(seen_imgs)}")
        print(f"  manifest -> {man_path}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
