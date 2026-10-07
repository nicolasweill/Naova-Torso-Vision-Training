from __future__ import annotations

import logging
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)

_FORMAT_WARNED = False


def _nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Pure-numpy NMS. Returns kept indices sorted by score."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    order = np.argsort(-scores)

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break

        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])

        inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-7)
        order = rest[iou < iou_threshold]

    return np.array(keep, dtype=np.int64)


def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    out = np.empty_like(boxes, dtype=np.float32)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def _detect_format(outputs: list[np.ndarray]) -> str:
    if len(outputs) >= 2:
        if outputs[0].ndim == 3 and outputs[0].shape[-1] == 4:
            return "ssd"

    raw = outputs[0]
    if raw.ndim == 3:
        _, n, c = raw.shape
        # YOLOv8: [B, 4+C, N_anchors] — small channel dim, large anchor dim
        if n < c and 4 < n <= 100:
            return "yolo_v8"
        # YOLOv5: [B, N_anchors, 5+C]
        if c > 5 and n > c:
            return "yolo_v5"
        if c > 5:
            return "yolo_v5"

    return "raw"


def _decode_yolo_v5(
    raw: np.ndarray,
    conf_threshold: float,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # raw: [1, N, 5 + C]
    raw = raw[0]  # [N, 5+C]
    objectness = 1.0 / (1.0 + np.exp(-raw[:, 4]))  # sigmoid
    class_probs = 1.0 / (1.0 + np.exp(-raw[:, 5:5 + num_classes]))
    combined = objectness[:, None] * class_probs
    labels = combined.argmax(axis=1)
    scores = combined.max(axis=1)

    mask = scores >= conf_threshold
    raw, scores, labels = raw[mask], scores[mask], labels[mask]

    cx, cy, bw, bh = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    return boxes, scores, labels


def _decode_yolo_v8(
    raw: np.ndarray,
    conf_threshold: float,
    num_classes: int,
    input_size: tuple[int, int] = (640, 640),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # raw: [1, 4+C, N] — boxes are pixel x1y1x2y2, class probs already sigmoided by model
    raw = raw[0].T  # [N, 4+C]
    class_probs = raw[:, 4:4 + num_classes]  # already sigmoided, no second sigmoid
    labels = class_probs.argmax(axis=1)
    scores = class_probs.max(axis=1)

    mask = scores >= conf_threshold
    raw, scores, labels = raw[mask], scores[mask], labels[mask]

    ih, iw = input_size
    boxes = raw[:, :4] / np.array([iw, ih, iw, ih], dtype=np.float32)  # normalize to [0,1]
    return boxes, scores, labels


def _decode_ssd(
    outputs: list[np.ndarray],
    conf_threshold: float,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes_raw = outputs[0][0]  # [N, 4]
    scores_raw = outputs[1][0]  # [N, C]
    labels = scores_raw.argmax(axis=1)
    scores = scores_raw.max(axis=1)
    mask = scores >= conf_threshold
    return boxes_raw[mask], scores[mask], labels[mask]


def decode_outputs(
    raw_outputs: list[np.ndarray],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    num_classes: int = 7,
    input_size: tuple[int, int] = (640, 640),
) -> dict:
    """
    Decode raw ONNX output into boxes/scores/labels with NMS.

    Supports YOLO-v5, YOLO-v8, SSD output formats.
    Returns {"boxes": [N,4] x1y1x2y2 normalised, "scores": [N], "labels": [N]}.
    """
    global _FORMAT_WARNED

    fmt = _detect_format(raw_outputs)

    if fmt == "yolo_v5":
        boxes, scores, labels = _decode_yolo_v5(raw_outputs[0], conf_threshold, num_classes)
    elif fmt == "yolo_v8":
        boxes, scores, labels = _decode_yolo_v8(raw_outputs[0], conf_threshold, num_classes, input_size)
    elif fmt == "ssd":
        boxes, scores, labels = _decode_ssd(raw_outputs, conf_threshold, num_classes)
    else:
        if not _FORMAT_WARNED:
            logger.warning(
                "Unknown output format (shape=%s). Returning empty detections.",
                [o.shape for o in raw_outputs],
            )
            _FORMAT_WARNED = True
        return {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": np.zeros(0, dtype=np.int64)}

    if len(boxes) == 0:
        return {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": np.zeros(0, dtype=np.int64)}

    # Per-class NMS
    keep_indices: list[int] = []
    for cls in np.unique(labels):
        cls_mask = labels == cls
        cls_idx = np.where(cls_mask)[0]
        kept = _nms(boxes[cls_mask], scores[cls_mask], iou_threshold)
        keep_indices.extend(cls_idx[kept].tolist())

    keep = np.array(keep_indices, dtype=np.int64)
    return {
        "boxes": np.clip(boxes[keep], 0.0, 1.0).astype(np.float32),
        "scores": scores[keep].astype(np.float32),
        "labels": labels[keep].astype(np.int64),
    }
