# Copyright (c) 2026. RefCOCO referring-expression loader for SAM3 training.
# pyre-unsafe
"""
A coco_json_loader compatible with sam3.train.data.sam3_image_dataset.Sam3ImageDataset.

Unlike COCO_FROM_JSON (which builds one text query per *category* per image),
RefCOCO needs one query per *referring expression*, each pointing at a single GT
object.  We consume the flat manifest emitted by
scripts_refcoco/convert_refcoco.py, where every record is one
(image, sentence, GT object) triple, and expose each record as its own datapoint.
"""
import json

from .coco_json_loaders import (
    ann_to_rle,
    convert_boxlist_to_normalized_tensor,
)


class RefCOCOManifestLoader:
    """One datapoint == one (image, referring sentence, single GT object)."""

    def __init__(self, annotation_file, include_negatives=False):
        with open(annotation_file, "r") as f:
            self._records = json.load(f)
        # include_negatives is accepted for config symmetry; RefCOCO records are
        # always positive (exactly one target), so it has no effect here.
        self.include_negatives = include_negatives

    def getDatapointIds(self):
        return list(range(len(self._records)))

    def loadImagesFromDatapoint(self, idx):
        rec = self._records[idx]
        return [{
            "id": 0,
            "file_name": rec["file_name"],
            "original_img_id": rec["object_id"],
            "coco_img_id": rec["object_id"],
        }]

    def loadQueriesAndAnnotationsFromDatapoint(self, idx):
        rec = self._records[idx]
        h, w = rec["height"], rec["width"]
        im_info = {"height": h, "width": w}

        normalized = convert_boxlist_to_normalized_tensor([rec["bbox"]], w, h)[0]

        annotation = {
            "id": 0,
            "object_id": 0,
            "image_id": 0,
            "is_crowd": int(rec.get("iscrowd", 0)),
            "bbox": normalized,  # normalized xywh tensor
            "area": (normalized[2] * normalized[3]).item(),
            "segmentation": None,
        }
        seg = rec.get("segmentation")
        if seg:
            annotation["segmentation"] = ann_to_rle(seg, im_info=im_info)

        query = {
            "id": 0,
            "original_cat_id": 0,
            "object_ids_output": [0],
            "query_text": rec["expression"],
            "query_processing_order": 0,
            "ptr_x_query_id": None,
            "ptr_y_query_id": None,
            "image_id": 0,
            "input_box": None,
            "input_box_label": None,
            "input_points": None,
            "is_exhaustive": True,
        }
        return [query], [annotation]
