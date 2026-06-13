# Copyright (c) 2026. RefCOCO fine-tune model builder (frozen perception backbone).
# pyre-unsafe
"""
Thin wrapper around build_sam3_image_model that freezes the vision and language
backbones, leaving the detection transformer + heads trainable.

Why: SAM3's released ViT uses an inference-only fused MLP (sam3.perflib.fused),
and the full model (~1.7B params) does not fit on a single 24GB GPU for full
fine-tuning. Freezing the perception backbones is the standard light fine-tuning
recipe and keeps optimizer/activation memory small enough for one GPU.
"""
import logging

from sam3.model_builder import build_sam3_image_model

logger = logging.getLogger(__name__)


def build_sam3_refcoco_finetune_model(freeze_vision=True, freeze_language=True, **kwargs):
    model = build_sam3_image_model(**kwargs)

    n_frozen = 0
    if freeze_vision:
        for p in model.backbone.vision_backbone.parameters():
            p.requires_grad_(False)
            n_frozen += p.numel()
    if freeze_language:
        for p in model.backbone.language_backbone.parameters():
            p.requires_grad_(False)
            n_frozen += p.numel()

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"[refcoco_ft] frozen params: {n_frozen/1e6:.1f}M | "
        f"trainable params: {n_train/1e6:.1f}M"
    )
    return model
