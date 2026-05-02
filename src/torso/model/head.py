from __future__ import annotations

import logging
from typing import Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_FORMAT_LOGGED = False  # log detection only once


def _detect_output_format(raw: Union[torch.Tensor, list, tuple]) -> str:
    """
    Heuristic to detect the output format of a detection model.

    Returns one of: "yolo_v5" | "yolo_v8" | "ssd" | "raw"
    """
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        # SSD-style: two tensors [boxes [B,N,4], scores [B,N,C]]
        if all(isinstance(t, torch.Tensor) for t in raw[:2]):
            shapes = [t.shape for t in raw[:2]]
            if len(shapes[0]) == 3 and len(shapes[1]) == 3:
                if shapes[0][-1] == 4:
                    return "ssd"
        return "raw"

    if not isinstance(raw, torch.Tensor):
        return "raw"

    if raw.dim() == 3:
        b, n, c = raw.shape
        # YOLO-v5: [B, N, 5 + num_classes] — objectness at index 4
        if c > 5:
            obj_col = raw[0, :, 4]
            if obj_col.min() >= 0.0 and obj_col.max() <= 1.0:
                return "yolo_v5"
        # YOLO-v8: [B, 4 + num_classes, N] — check if second dim looks like 4+C
        if n > c:
            # likely [B, 4+C, N] — transposed
            return "yolo_v8"

    return "raw"


class DetectionHead(nn.Module):
    """
    Thin wrapper around a converted ONNX model that normalises its output
    into a list of per-image detection dicts.

    Detected formats: yolo_v5, yolo_v8, ssd, raw.
    """

    def __init__(self, base_model: nn.Module, output_format: str = "auto") -> None:
        super().__init__()
        self.base_model = base_model
        self._output_format = output_format if output_format != "auto" else None

    @property
    def output_format(self) -> str | None:
        return self._output_format

    def forward(self, x: torch.Tensor) -> tuple[list[dict], torch.Tensor]:
        """
        Returns (decoded_list, raw_output).

        decoded_list: list of dicts {"boxes" [N,4], "scores" [N], "labels" [N]}
                      boxes in [x1,y1,x2,y2] normalised.
        raw_output:   the original model output (for loss computation).
        """
        global _FORMAT_LOGGED

        raw = self.base_model(x)

        if self._output_format is None:
            self._output_format = _detect_output_format(raw)
            if not _FORMAT_LOGGED:
                logger.info("Auto-detected output format: %s", self._output_format)
                _FORMAT_LOGGED = True

        decoded = self._decode(raw, x.shape[0])
        return decoded, raw

    def _decode(
        self, raw: Union[torch.Tensor, list, tuple], batch_size: int
    ) -> list[dict]:
        fmt = self._output_format

        if fmt == "yolo_v5":
            return self._decode_yolo_v5(raw, batch_size)
        if fmt == "yolo_v8":
            return self._decode_yolo_v8(raw, batch_size)
        if fmt == "ssd":
            return self._decode_ssd(raw, batch_size)

        # raw — cannot decode; return empty predictions
        return [{"boxes": torch.zeros(0, 4), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)} for _ in range(batch_size)]

    @staticmethod
    def _decode_yolo_v5(raw: torch.Tensor, batch_size: int) -> list[dict]:
        # raw: [B, N, 5 + C]
        objectness = raw[..., 4:5].sigmoid()
        class_probs = raw[..., 5:].sigmoid()
        scores, labels = (objectness * class_probs).max(dim=-1)

        # cx,cy,w,h → x1,y1,x2,y2
        cx = raw[..., 0]
        cy = raw[..., 1]
        w = raw[..., 2]
        h = raw[..., 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)

        return [
            {"boxes": boxes[i], "scores": scores[i], "labels": labels[i]}
            for i in range(batch_size)
        ]

    @staticmethod
    def _decode_yolo_v8(raw: torch.Tensor, batch_size: int) -> list[dict]:
        # raw: [B, 4 + C, N]
        raw = raw.permute(0, 2, 1)  # → [B, N, 4+C]
        boxes_raw = raw[..., :4]
        class_probs = raw[..., 4:].sigmoid()
        scores, labels = class_probs.max(dim=-1)

        cx = boxes_raw[..., 0]
        cy = boxes_raw[..., 1]
        w = boxes_raw[..., 2]
        h = boxes_raw[..., 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)

        return [
            {"boxes": boxes[i], "scores": scores[i], "labels": labels[i]}
            for i in range(batch_size)
        ]

    @staticmethod
    def _decode_ssd(
        raw: Union[list, tuple], batch_size: int
    ) -> list[dict]:
        boxes_out, scores_out = raw[0], raw[1]
        scores, labels = scores_out.max(dim=-1)
        return [
            {"boxes": boxes_out[i], "scores": scores[i], "labels": labels[i]}
            for i in range(batch_size)
        ]
