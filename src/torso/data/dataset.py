from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _letterbox(
    image: np.ndarray,
    target_size: tuple[int, int],
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Resize image with aspect-ratio-preserving letterboxing.

    Returns (letterboxed_image, scale, (pad_left, pad_top)).
    """
    h, w = image.shape[:2]
    th, tw = target_size
    scale = min(tw / w, th / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((th, tw, 3), color, dtype=np.uint8)
    pad_top = (th - new_h) // 2
    pad_left = (tw - new_w) // 2
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

    return canvas, scale, (pad_left, pad_top)


def _adjust_boxes_for_letterbox(
    boxes: np.ndarray,
    orig_size: tuple[int, int],
    target_size: tuple[int, int],
    scale: float,
    pad: tuple[int, int],
) -> np.ndarray:
    """
    Adjust normalised YOLO boxes (cx, cy, w, h) after letterboxing.

    orig_size: (H, W) before resize.
    target_size: (H, W) of the letterboxed canvas.
    """
    if len(boxes) == 0:
        return boxes

    oh, ow = orig_size
    th, tw = target_size
    pad_left, pad_top = pad

    boxes = boxes.copy().astype(np.float32)
    # convert to pixel coords in original image
    boxes[:, 0] *= ow
    boxes[:, 1] *= oh
    boxes[:, 2] *= ow
    boxes[:, 3] *= oh

    # scale + shift
    boxes[:, 0] = boxes[:, 0] * scale + pad_left
    boxes[:, 1] = boxes[:, 1] * scale + pad_top
    boxes[:, 2] = boxes[:, 2] * scale
    boxes[:, 3] = boxes[:, 3] * scale

    # back to normalised
    boxes[:, 0] /= tw
    boxes[:, 1] /= th
    boxes[:, 2] /= tw
    boxes[:, 3] /= th

    return np.clip(boxes, 0.0, 1.0)


def _parse_yolo_label(label_path: Path, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a YOLO .txt label file.

    Returns (boxes [N,4] float32, class_ids [N] int64).
    Empty file → arrays with shape (0, 4) and (0,).
    """
    boxes: list[list[float]] = []
    class_ids: list[int] = []

    try:
        text = label_path.read_text(encoding="utf-8").strip()
    except OSError as e:
        logger.warning("Cannot read label %s: %s", label_path, e)
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)

    if not text:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)

    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            logger.warning("Skipping malformed line in %s: %r", label_path, line)
            continue
        try:
            cid = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
        except ValueError:
            logger.warning("Skipping unparseable line in %s: %r", label_path, line)
            continue

        if not (0 <= cid < num_classes):
            logger.warning("Class id %d out of range [0,%d) in %s", cid, num_classes, label_path)
            continue
        if not all(0.0 <= v <= 1.0 for v in (cx, cy, bw, bh)):
            logger.warning("Coords out of [0,1] in %s: %r", label_path, line)
            continue

        boxes.append([cx, cy, bw, bh])
        class_ids.append(cid)

    return (
        np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32),
        np.array(class_ids, dtype=np.int64) if class_ids else np.zeros(0, dtype=np.int64),
    )


class Torso21Dataset(Dataset):
    """
    TORSO-21 detection dataset (YOLO-format labels).

    Expects:
      split_dir/images/<stem>.(png|jpg)
      split_dir/<stem>.txt            ← YOLO label

    Images without a matching label file are silently skipped.
    Images with an empty label file are kept (background frames).
    """

    def __init__(
        self,
        split_dir: str | Path,
        input_size: tuple[int, int] = (640, 640),
        num_classes: int = 7,
        image_extensions: tuple[str, ...] | None = None,
        transform: Optional[Callable] = None,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.images_dir = self.split_dir / "images"
        self.input_size = input_size  # (H, W)
        self.num_classes = num_classes
        self.transform = transform
        self.extensions = image_extensions or tuple(_IMAGE_EXTENSIONS)

        self.samples: list[tuple[Path, Path]] = []
        self._discover_samples()

    def _discover_samples(self) -> None:
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        found = skipped = 0
        for img_path in sorted(self.images_dir.iterdir()):
            if img_path.suffix.lower() not in self.extensions:
                continue
            label_path = self.split_dir / (img_path.stem + ".txt")
            if label_path.exists():
                self.samples.append((img_path, label_path))
                found += 1
            else:
                skipped += 1

        logger.info(
            "Dataset %s: %d valid pairs (%d images skipped — missing label)",
            self.split_dir.name,
            found,
            skipped,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, label_path = self.samples[idx]

        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        boxes, class_ids = _parse_yolo_label(label_path, self.num_classes)

        # Letterbox resize
        letterboxed, scale, pad = _letterbox(image, self.input_size)
        boxes = _adjust_boxes_for_letterbox(
            boxes, (orig_h, orig_w), self.input_size, scale, pad
        )

        if self.transform is not None:
            transformed = self.transform(
                image=letterboxed,
                bboxes=boxes.tolist(),
                class_labels=class_ids.tolist(),
            )
            letterboxed = transformed["image"]
            boxes_list = transformed["bboxes"]
            class_ids_list = transformed["class_labels"]

            boxes = (
                np.array(boxes_list, dtype=np.float32)
                if boxes_list
                else np.zeros((0, 4), dtype=np.float32)
            )
            class_ids = (
                np.array(class_ids_list, dtype=np.int64)
                if class_ids_list
                else np.zeros(0, dtype=np.int64)
            )

        # Convert image to tensor [3, H, W]
        if isinstance(letterboxed, np.ndarray):
            image_tensor = torch.from_numpy(letterboxed).permute(2, 0, 1).float() / 255.0
        else:
            image_tensor = letterboxed  # albumentations ToTensorV2 already did it

        return {
            "image": image_tensor,
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(class_ids),
            "image_path": str(img_path),
            "orig_size": (orig_h, orig_w),
            "scale": scale,
            "pad": pad,
        }
