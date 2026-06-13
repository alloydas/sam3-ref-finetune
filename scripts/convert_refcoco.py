#!/usr/bin/env python
# Copyright (c) 2026. RefCOCO (lmms-lab) -> SAM3 manifest + COCO GT converter.
"""
Download lmms-lab/RefCOCO from the HuggingFace Hub, dump the embedded images to
disk, and emit:

  1. A flat "manifest" JSON per split -- one record per (object, referring
     sentence).  This is what the standalone eval script and the SAM3 training
     loader consume.  Each record:
        {
          "id": int,                # global record id
          "file_name": str,         # image on disk (relative to images/)
          "expression": str,        # a single referring sentence (text prompt)
          "bbox": [x, y, w, h],     # absolute pixels, GT object
          "segmentation": [[...]],  # COCO polygon(s), absolute pixels
          "height": int, "width": int,
          "iscrowd": 0/1,
          "object_id": int          # unique GT object id (== row index)
        }

  2. A COCO-format GT JSON per split (images / annotations / categories) for
     reference and optional COCO-style metrics.

lmms-lab/RefCOCO only ships val/test/testA/testB (no train split): it is an
*evaluation* dataset.  We therefore treat it as eval-first; a held-out split can
be used as a stand-in "train" set for a pipeline-demonstration fine-tune.
"""
import argparse
import json
import os

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lmms-lab/RefCOCO",
                    help="HF dataset id, e.g. lmms-lab/RefCOCO or lmms-lab/RefCOCOplus")
    ap.add_argument("--prefix", default="refcoco",
                    help="Filename prefix for manifests/GT (e.g. refcoco, refcocoplus)")
    ap.add_argument("--out-root", default="/work/mech-ai-scratch/alloy/refcoco_sam3")
    ap.add_argument("--splits", nargs="+", default=["val", "testA", "testB", "test"])
    ap.add_argument(
        "--max-rows", type=int, default=-1,
        help="Cap rows per split (for quick smoke tests). -1 = all.",
    )
    ap.add_argument(
        "--one-sentence", action="store_true",
        help="Keep only the first referring sentence per object "
             "(instead of expanding every sentence into its own record).",
    )
    args = ap.parse_args()

    img_dir = os.path.join(args.out_root, "images")
    ann_dir = os.path.join(args.out_root, "annotations")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    for split in args.splits:
        print(f"\n=== {args.dataset} split: {split} ===")
        ds = load_dataset(args.dataset, split=split)
        if args.max_rows > 0:
            ds = ds.select(range(min(args.max_rows, len(ds))))

        manifest = []
        coco_images, coco_anns = [], []
        seen_files = set()
        rec_id = 0

        for obj_id, row in enumerate(tqdm(ds, desc=split)):
            file_name = row["file_name"]
            img = row["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            w, h = img.size

            img_path = os.path.join(img_dir, file_name)
            if file_name not in seen_files:
                img.save(img_path)
                seen_files.add(file_name)
                coco_images.append(
                    {"id": obj_id, "file_name": file_name, "height": h, "width": w}
                )

            bbox = [float(v) for v in row["bbox"]]
            seg = row["segmentation"]
            # COCO polygon must be list-of-lists of floats.
            if len(seg) > 0 and not isinstance(seg[0], (list, tuple)):
                seg = [[float(v) for v in seg]]
            else:
                seg = [[float(v) for v in poly] for poly in seg]
            iscrowd = int(row["iscrowd"])

            coco_anns.append({
                "id": obj_id,
                "image_id": obj_id,
                "category_id": 1,
                "bbox": bbox,
                "area": float(bbox[2] * bbox[3]),
                "segmentation": seg,
                "iscrowd": iscrowd,
            })

            sentences = row["answer"]
            if args.one_sentence and len(sentences) > 0:
                sentences = sentences[:1]
            for sent in sentences:
                manifest.append({
                    "id": rec_id,
                    "file_name": file_name,
                    "expression": sent.strip(),
                    "bbox": bbox,
                    "segmentation": seg,
                    "height": h,
                    "width": w,
                    "iscrowd": iscrowd,
                    "object_id": obj_id,
                })
                rec_id += 1

        coco_gt = {
            "images": coco_images,
            "annotations": coco_anns,
            "categories": [{"id": 1, "name": "object"}],
        }

        man_path = os.path.join(ann_dir, f"{args.prefix}_{split}_manifest.json")
        gt_path = os.path.join(ann_dir, f"{args.prefix}_{split}_coco_gt.json")
        with open(man_path, "w") as f:
            json.dump(manifest, f)
        with open(gt_path, "w") as f:
            json.dump(coco_gt, f)

        print(f"  rows(objects)={len(ds)}  records(sentences)={len(manifest)}  "
              f"unique_images={len(seen_files)}")
        print(f"  manifest -> {man_path}")
        print(f"  coco gt  -> {gt_path}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
