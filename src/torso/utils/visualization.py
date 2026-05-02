from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

_PALETTE = [
    (255, 56, 56),
    (255, 157, 151),
    (255, 112, 31),
    (255, 178, 29),
    (207, 210, 49),
    (72, 249, 10),
    (146, 204, 23),
]


def draw_detections(
    image: np.ndarray,
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    labels: Sequence[int],
    class_names: Sequence[str],
    conf_threshold: float = 0.0,
) -> np.ndarray:
    """
    Draw bounding boxes on a BGR image (numpy array).

    Boxes are expected in [x1, y1, x2, y2] format, normalised [0,1].
    Returns a copy of the image with boxes drawn.
    """
    img = image.copy()
    h, w = img.shape[:2]

    for box, score, label in zip(boxes, scores, labels):
        if score < conf_threshold:
            continue
        x1, y1, x2, y2 = box
        px1 = int(x1 * w)
        py1 = int(y1 * h)
        px2 = int(x2 * w)
        py2 = int(y2 * h)

        color = _PALETTE[int(label) % len(_PALETTE)]
        cv2.rectangle(img, (px1, py1), (px2, py2), color, 2)

        name = class_names[int(label)] if int(label) < len(class_names) else str(label)
        text = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (px1, py1 - th - 4), (px1 + tw, py1), color, -1)
        cv2.putText(
            img, text, (px1, py1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

    return img


def save_detection_result(
    image_path: str | Path,
    detections: dict,
    output_dir: str | Path,
    class_names: Sequence[str],
) -> Path:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    annotated = draw_detections(
        img,
        detections.get("boxes", []),
        detections.get("scores", []),
        detections.get("labels", []),
        class_names,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / Path(image_path).name
    cv2.imwrite(str(out_path), annotated)
    return out_path
