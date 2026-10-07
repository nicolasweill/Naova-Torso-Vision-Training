#!/usr/bin/env python3
"""
Visualize ground truth annotations on TORSO-21 dataset images.

Labels are read from YOLO .txt files (class_id cx cy w h, normalised [0,1]).
For images in a .../images/ directory, labels are looked up in the parent directory.

Usage:
    python scripts/visualize_gt.py --input dataset/test/images/
    python scripts/visualize_gt.py --input dataset/test/images/ --show
    python scripts/visualize_gt.py --input dataset/test/images/foo.png --show
    python scripts/visualize_gt.py --input dataset/train/images/ --output runs/gt_train
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
import yaml

from torso.utils.visualization import draw_detections

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ground truth annotations")
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--output", default="runs/gt_vis", help="Output directory for annotated images")
    parser.add_argument("--dataset", default="configs/dataset.yaml", help="Dataset YAML for class names")
    parser.add_argument("--show", action="store_true", help="Display results with cv2.imshow")
    parser.add_argument("--no-save", action="store_true", help="Do not save annotated images")
    return parser.parse_args()


def load_class_names(dataset_yaml: str) -> list[str]:
    try:
        with open(dataset_yaml, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        names = cfg.get("classes", {}).get("names", {})
        return [names.get(i, str(i)) for i in range(len(names))]
    except FileNotFoundError:
        logger.warning("Dataset config not found: %s. Using numeric class IDs.", dataset_yaml)
        return [str(i) for i in range(100)]


def gather_images(input_path: str) -> list[Path]:
    p = Path(input_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in _IMAGE_EXTENSIONS)
    return sorted(Path(".").glob(input_path))


def find_label(image_path: Path) -> Path:
    img_dir = image_path.parent
    label_dir = img_dir.parent if img_dir.name == "images" else img_dir
    return label_dir / (image_path.stem + ".txt")


def parse_yolo_label(label_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)

    boxes, labels = [], []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        boxes.append([x1, y1, x2, y2])
        labels.append(cls_id)

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)

    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)


def main() -> None:
    args = parse_args()

    class_names = load_class_names(args.dataset)
    image_paths = gather_images(args.input)

    if not image_paths:
        logger.error("No images found at: %s", args.input)
        sys.exit(1)

    output_dir = Path(args.output)
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Visualizing GT for %d image(s) ...", len(image_paths))

    n_saved = 0
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Cannot read image: %s", img_path)
            continue

        label_path = find_label(img_path)
        boxes, labels = parse_yolo_label(label_path)
        scores = np.ones(len(labels), dtype=np.float32)

        annotated = draw_detections(img, boxes, scores, labels, class_names, show_score=False)

        if not args.no_save:
            out_path = output_dir / img_path.name
            cv2.imwrite(str(out_path), annotated)
            n_saved += 1

        if args.show:
            window_title = f"GT: {img_path.name} ({len(labels)} objects)"
            cv2.imshow(window_title, annotated)
            key = cv2.waitKey(0)
            cv2.destroyWindow(window_title)
            if key == ord("q"):
                break

        logger.info("%s → %d object(s)%s", img_path.name, len(labels),
                    "" if label_path.exists() else " [no label file]")

    if args.show:
        cv2.destroyAllWindows()

    if not args.no_save:
        logger.info("Saved %d annotated image(s) → %s", n_saved, args.output)


if __name__ == "__main__":
    main()
