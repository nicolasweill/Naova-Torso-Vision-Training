from __future__ import annotations

from typing import Any

import torch


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Custom collate for variable-length bounding-box tensors.

    Returns:
      images:  Tensor [B, 3, H, W]
      boxes:   list[Tensor[N_i, 4]]
      labels:  list[Tensor[N_i]]
      image_paths: list[str]
      orig_sizes:  list[tuple[int,int]]
    """
    images = torch.stack([s["image"] for s in batch])
    boxes = [s["boxes"] for s in batch]
    labels = [s["labels"] for s in batch]
    image_paths = [s["image_path"] for s in batch]
    orig_sizes = [s["orig_size"] for s in batch]
    scales = [s["scale"] for s in batch]
    pads = [s["pad"] for s in batch]

    return {
        "images": images,
        "boxes": boxes,
        "labels": labels,
        "image_paths": image_paths,
        "orig_sizes": orig_sizes,
        "scales": scales,
        "pads": pads,
    }
