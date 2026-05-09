#!/usr/bin/env python3
"""
Run inference with any ONNX detection model on the TORSO-21 dataset or custom images.

Usage:
    python scripts/infer.py --model models/sim_data_det_0126.onnx --input dataset/test/images/
    python scripts/infer.py --model runs/exp1/best.onnx --input image.jpg --show
    python scripts/infer.py --model models/sim_data_det_0126.onnx --input dataset/test/images/ --benchmark
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml

from torso.inference.onnx_runner import OnnxRunner
from torso.utils.visualization import save_detection_result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ONNX detection inference")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--input", required=True, help="Image file, directory, or glob pattern")
    parser.add_argument("--output", default="runs/infer", help="Output directory for annotated images")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"],
        help="Inference device (requires onnxruntime-gpu for CUDA)"
    )
    parser.add_argument("--dataset", default="configs/dataset.yaml", help="Dataset YAML for class names")
    parser.add_argument("--show", action="store_true", help="Display results with cv2.imshow")
    parser.add_argument("--benchmark", action="store_true", help="Report mean inference latency")
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
    # Glob
    return sorted(Path(".").glob(input_path))


def main() -> None:
    args = parse_args()

    class_names = load_class_names(args.dataset)
    num_classes = len(class_names)

    runner = OnnxRunner(
        model_path=args.model,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        num_classes=num_classes,
    )

    image_paths = gather_images(args.input)
    if not image_paths:
        logger.error("No images found at: %s", args.input)
        sys.exit(1)

    logger.info("Running inference on %d image(s) ...", len(image_paths))

    if args.benchmark:
        ms = runner.benchmark(str(image_paths[0]))
        logger.info("Benchmark: %.2f ms/image (mean over 50 runs)", ms)

    n_saved = 0
    n_detections = 0

    for img_path in image_paths:
        detections = runner(str(img_path))
        n_det = len(detections.get("boxes", []))
        n_detections += n_det

        if not args.no_save:
            out_path = save_detection_result(
                img_path, detections, args.output, class_names
            )
            if n_det > 0:
                n_saved += 1

        if args.show:
            import cv2
            import numpy as np
            from torso.utils.visualization import draw_detections

            img = cv2.imread(str(img_path))
            if img is not None:
                annotated = draw_detections(
                    img,
                    detections.get("boxes", []),
                    detections.get("scores", []),
                    detections.get("labels", []),
                    class_names,
                )
                cv2.imshow("Detections", annotated)
                key = cv2.waitKey(0)
                if key == ord("q"):
                    break

        logger.info("%s → %d detection(s)", img_path.name, n_det)

    if args.show:
        import cv2
        cv2.destroyAllWindows()

    logger.info(
        "Done. %d total detections across %d images. Results saved to: %s",
        n_detections,
        len(image_paths),
        args.output if not args.no_save else "(not saved)",
    )


if __name__ == "__main__":
    main()
