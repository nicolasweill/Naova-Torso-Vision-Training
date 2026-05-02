from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from torso.inference.postprocess import decode_outputs

logger = logging.getLogger(__name__)


def _get_providers(device: str) -> list[str]:
    """Return ONNX Runtime execution providers."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    if device == "cpu":
        return ["CPUExecutionProvider"]

    # "auto" or "cuda"
    if "CUDAExecutionProvider" in available:
        logger.info("ONNX Runtime: using CUDAExecutionProvider")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    if device == "cuda":
        raise RuntimeError(
            "CUDAExecutionProvider not available. "
            "Install onnxruntime-gpu: pip install onnxruntime-gpu"
        )

    logger.info("ONNX Runtime: CUDA not available, using CPUExecutionProvider")
    return ["CPUExecutionProvider"]


class OnnxRunner:
    """
    ONNX Runtime-based inference engine.

    Handles preprocessing, inference and postprocessing.
    Supports any ONNX detection model (YOLO-v5, YOLO-v8, SSD).
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        num_classes: int = 7,
    ) -> None:
        import onnxruntime as ort

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        providers = _get_providers(device)
        self.session = ort.InferenceSession(str(path), providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        # Shape is typically [batch, 3, H, W]; use last two dims
        self.input_h = input_shape[2] if isinstance(input_shape[2], int) and input_shape[2] > 0 else 640
        self.input_w = input_shape[3] if isinstance(input_shape[3], int) and input_shape[3] > 0 else 640

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes

        logger.info(
            "OnnxRunner: %s | input %dx%d | providers: %s",
            path.name,
            self.input_h,
            self.input_w,
            providers,
        )

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        """
        BGR image → normalised float32 tensor + letterbox metadata.

        Returns (tensor [1,3,H,W], scale, (pad_left, pad_top)).
        """
        from torso.data.dataset import _letterbox

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        letterboxed, scale, pad = _letterbox(rgb, (self.input_h, self.input_w))
        tensor = letterboxed.astype(np.float32) / 255.0
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = (tensor - mean) / std
        tensor = tensor.transpose(2, 0, 1)[None]  # [1, 3, H, W]
        return tensor, scale, pad

    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        return self.session.run(None, {self.input_name: tensor})

    def postprocess(
        self,
        raw_outputs: list[np.ndarray],
        orig_shape: tuple[int, int],
        scale: float,
        pad: tuple[int, int],
    ) -> dict:
        """
        Decode and map boxes back to original image coordinates.

        orig_shape: (H, W) of original image.
        """
        detections = decode_outputs(
            raw_outputs,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            num_classes=self.num_classes,
        )

        boxes = detections["boxes"]
        if len(boxes) == 0:
            return detections

        oh, ow = orig_shape
        ih, iw = self.input_h, self.input_w
        pad_left, pad_top = pad

        # Undo letterbox: map from padded-input coords back to original image
        boxes_px = boxes.copy()
        boxes_px[:, [0, 2]] *= iw
        boxes_px[:, [1, 3]] *= ih

        boxes_px[:, [0, 2]] -= pad_left
        boxes_px[:, [1, 3]] -= pad_top
        boxes_px /= scale

        # Re-normalise to original image size
        boxes_px[:, [0, 2]] /= ow
        boxes_px[:, [1, 3]] /= oh
        detections["boxes"] = np.clip(boxes_px, 0.0, 1.0).astype(np.float32)

        return detections

    def __call__(self, image_path: str) -> dict:
        """Full pipeline: load → preprocess → run → postprocess."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        orig_h, orig_w = image.shape[:2]
        tensor, scale, pad = self.preprocess(image)
        raw = self.run(tensor)
        return self.postprocess(raw, (orig_h, orig_w), scale, pad)

    def benchmark(self, image_path: str, n_runs: int = 50) -> float:
        """Returns mean inference latency in milliseconds."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        tensor, _, _ = self.preprocess(image)
        # Warmup
        for _ in range(5):
            self.run(tensor)

        t0 = time.perf_counter()
        for _ in range(n_runs):
            self.run(tensor)
        elapsed_ms = (time.perf_counter() - t0) * 1000 / n_runs
        return elapsed_ms
