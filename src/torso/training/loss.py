from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_UNKNOWN_FORMAT_WARNED = False


@dataclass
class LossConfig:
    box_weight: float = 5.0
    cls_weight: float = 1.0
    obj_weight: float = 1.0
    label_smoothing: float = 0.1
    num_classes: int = 7


def _ciou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """
    CIoU loss between predicted and target boxes (normalised cx,cy,w,h).
    Returns mean CIoU loss.
    """
    if pred_boxes.numel() == 0:
        return torch.tensor(0.0, requires_grad=True, device=pred_boxes.device)

    # Convert cx,cy,w,h → x1,y1,x2,y2
    def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        return torch.stack([x1, y1, x2, y2], dim=1)

    p = cxcywh_to_xyxy(pred_boxes)
    t = cxcywh_to_xyxy(target_boxes)

    inter_x1 = torch.max(p[:, 0], t[:, 0])
    inter_y1 = torch.max(p[:, 1], t[:, 1])
    inter_x2 = torch.min(p[:, 2], t[:, 2])
    inter_y2 = torch.min(p[:, 3], t[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(0)
    inter_h = (inter_y2 - inter_y1).clamp(0)
    inter_area = inter_w * inter_h

    area_p = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
    area_t = (t[:, 2] - t[:, 0]) * (t[:, 3] - t[:, 1])
    union_area = area_p + area_t - inter_area + 1e-7

    iou = inter_area / union_area

    # Enclosing box
    enc_x1 = torch.min(p[:, 0], t[:, 0])
    enc_y1 = torch.min(p[:, 1], t[:, 1])
    enc_x2 = torch.max(p[:, 2], t[:, 2])
    enc_y2 = torch.max(p[:, 3], t[:, 3])
    enc_diag_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-7

    # Centre distance
    cx_p, cy_p = (p[:, 0] + p[:, 2]) / 2, (p[:, 1] + p[:, 3]) / 2
    cx_t, cy_t = (t[:, 0] + t[:, 2]) / 2, (t[:, 1] + t[:, 3]) / 2
    dist_sq = (cx_p - cx_t) ** 2 + (cy_p - cy_t) ** 2

    # Aspect ratio penalty
    w_p, h_p = p[:, 2] - p[:, 0], p[:, 3] - p[:, 1]
    w_t, h_t = t[:, 2] - t[:, 0], t[:, 3] - t[:, 1]
    v = (4 / (torch.pi**2)) * (torch.atan(w_t / (h_t + 1e-7)) - torch.atan(w_p / (h_p + 1e-7))) ** 2
    alpha = v / (1 - iou + v + 1e-7)

    ciou = iou - dist_sq / enc_diag_sq - alpha * v
    return (1.0 - ciou).mean()


class DetectionLoss(nn.Module):
    """
    Detection loss for YOLO-style decoded outputs.

    forward(predictions, targets) where:
      predictions: list[dict {"boxes":[N,4], "scores":[N], "labels":[N]}]
      targets:     list[dict {"boxes":[N,4], "labels":[N]}]
                   (boxes in normalised cx,cy,w,h)

    Returns dict {"total", "box", "cls", "obj"}.

    If output_format is "raw" (unknown), returns zero loss with a WARNING.
    """

    def __init__(self, cfg: LossConfig, output_format: str = "yolo_v5") -> None:
        super().__init__()
        self.cfg = cfg
        self.output_format = output_format

    def forward(
        self,
        predictions: list[dict],
        targets: list[dict],
    ) -> dict[str, torch.Tensor]:
        global _UNKNOWN_FORMAT_WARNED

        if self.output_format == "raw":
            if not _UNKNOWN_FORMAT_WARNED:
                logger.warning(
                    "DetectionLoss: output format is 'raw' — cannot compute loss. "
                    "Training will not converge. Please fix the output format detection."
                )
                _UNKNOWN_FORMAT_WARNED = True
            zero = torch.tensor(0.0, requires_grad=True)
            return {"total": zero, "box": zero, "cls": zero, "obj": zero}

        device = predictions[0]["boxes"].device if predictions else torch.device("cpu")

        total_box = torch.tensor(0.0, device=device)
        total_cls = torch.tensor(0.0, device=device)

        n_matched = 0
        for pred, tgt in zip(predictions, targets):
            gt_boxes = tgt["boxes"].to(device)
            gt_labels = tgt["labels"].to(device)

            if gt_boxes.numel() == 0:
                continue

            pred_boxes = pred["boxes"]
            pred_scores = pred["scores"]

            # Simple assignment: closest predicted box to each GT box by centre dist
            if pred_boxes.numel() > 0:
                gt_cx = gt_boxes[:, 0].unsqueeze(1)
                gt_cy = gt_boxes[:, 1].unsqueeze(1)
                pred_cx = pred_boxes[:, 0].unsqueeze(0) if pred_boxes.dim() == 2 else pred_boxes.unsqueeze(0)

                # Build a [GT, Pred] distance matrix
                dist = (
                    (gt_boxes[:, :2].unsqueeze(1) - pred_boxes[:, :2].unsqueeze(0))
                    .pow(2)
                    .sum(-1)
                    .sqrt()
                )
                matched_pred_idx = dist.argmin(dim=1)

                matched_boxes = pred_boxes[matched_pred_idx]
                total_box = total_box + _ciou_loss(matched_boxes, gt_boxes)

                # Classification loss (BCE)
                # Build a one-hot-style target for the matched predictions
                num_classes = self.cfg.num_classes
                smooth = self.cfg.label_smoothing
                cls_target = torch.full(
                    (len(gt_labels), num_classes),
                    smooth / num_classes,
                    device=device,
                )
                cls_target.scatter_(1, gt_labels.unsqueeze(1), 1.0 - smooth)

                pred_labels_scores = pred["scores"][matched_pred_idx]
                # Create a soft class distribution from scores (placeholder)
                cls_pred = torch.zeros(len(gt_labels), num_classes, device=device)
                cls_pred.scatter_(1, pred["labels"][matched_pred_idx].unsqueeze(1), pred_labels_scores.unsqueeze(1))

                total_cls = total_cls + F.binary_cross_entropy_with_logits(
                    cls_pred, cls_target, reduction="mean"
                )
                n_matched += len(gt_labels)

        box_loss = total_box * self.cfg.box_weight
        cls_loss = total_cls * self.cfg.cls_weight
        total = box_loss + cls_loss

        return {"total": total, "box": box_loss, "cls": cls_loss, "obj": torch.tensor(0.0, device=device)}
