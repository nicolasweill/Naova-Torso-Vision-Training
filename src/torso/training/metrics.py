from __future__ import annotations

import numpy as np
import torch


def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of boxes in [x1,y1,x2,y2] format.
    Returns [N, M] matrix.
    """
    x1 = np.maximum(box1[:, None, 0], box2[None, :, 0])
    y1 = np.maximum(box1[:, None, 1], box2[None, :, 1])
    x2 = np.minimum(box1[:, None, 2], box2[None, :, 2])
    y2 = np.minimum(box1[:, None, 3], box2[None, :, 3])

    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2[None, :] - inter + 1e-7
    return inter / union


def _average_precision(
    tp: np.ndarray,
    fp: np.ndarray,
    n_gt: int,
) -> float:
    """11-point interpolated AP."""
    if n_gt == 0:
        return float("nan")

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recalls = tp_cum / (n_gt + 1e-7)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-7)

    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recalls >= t
        ap += precisions[mask].max() if mask.any() else 0.0
    return ap / 11.0


def compute_map(
    predictions: list[dict],
    targets: list[dict],
    num_classes: int,
    iou_thresholds: list[float] | None = None,
) -> dict[str, float]:
    """
    Compute mAP at given IoU thresholds.

    predictions: list of {"boxes": Tensor/ndarray [N,4] x1y1x2y2 normalised,
                           "scores": [N], "labels": [N]}
    targets:     list of {"boxes": Tensor/ndarray [N,4] cx,cy,w,h normalised,
                           "labels": [N]}

    Returns {"mAP50", "mAP75", "mAP50_95", "per_class_AP_50": {...}}.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    def to_np(x) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float32)

    def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return boxes
        out = np.empty_like(boxes)
        out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return out

    # Convert targets to xyxy
    tgt_boxes_xyxy = []
    tgt_labels_list = []
    for tgt in targets:
        boxes = to_np(tgt["boxes"])
        labels = to_np(tgt["labels"]).astype(int)
        tgt_boxes_xyxy.append(cxcywh_to_xyxy(boxes))
        tgt_labels_list.append(labels)

    ap_by_threshold_class: dict[float, dict[int, float]] = {}

    for iou_thr in iou_thresholds:
        per_class_ap: dict[int, float] = {}
        for cls in range(num_classes):
            # Gather all predicted boxes for this class across all images
            all_scores: list[float] = []
            all_tp: list[int] = []
            all_fp: list[int] = []
            n_gt = 0

            for pred, gt_boxes, gt_labels in zip(predictions, tgt_boxes_xyxy, tgt_labels_list):
                pred_boxes = to_np(pred["boxes"])
                pred_scores = to_np(pred["scores"])
                pred_labels = to_np(pred["labels"]).astype(int)

                # GT boxes for this class
                gt_mask = gt_labels == cls
                gt_cls_boxes = gt_boxes[gt_mask]
                n_gt += len(gt_cls_boxes)
                gt_matched = np.zeros(len(gt_cls_boxes), dtype=bool)

                # Predicted boxes for this class
                pred_mask = pred_labels == cls
                if not pred_mask.any():
                    continue
                pred_cls_boxes = pred_boxes[pred_mask]
                pred_cls_scores = pred_scores[pred_mask]

                # Sort by score descending
                order = np.argsort(-pred_cls_scores)
                pred_cls_boxes = pred_cls_boxes[order]
                pred_cls_scores = pred_cls_scores[order]

                for pb, sc in zip(pred_cls_boxes, pred_cls_scores):
                    all_scores.append(float(sc))
                    if len(gt_cls_boxes) == 0:
                        all_tp.append(0)
                        all_fp.append(1)
                        continue

                    ious = _compute_iou(pb[None], gt_cls_boxes)[0]
                    best_iou_idx = ious.argmax()
                    best_iou = ious[best_iou_idx]

                    if best_iou >= iou_thr and not gt_matched[best_iou_idx]:
                        all_tp.append(1)
                        all_fp.append(0)
                        gt_matched[best_iou_idx] = True
                    else:
                        all_tp.append(0)
                        all_fp.append(1)

            if not all_scores:
                per_class_ap[cls] = float("nan")
                continue

            # Sort all detections by score
            order = np.argsort(-np.array(all_scores))
            tp_arr = np.array(all_tp)[order]
            fp_arr = np.array(all_fp)[order]
            per_class_ap[cls] = _average_precision(tp_arr, fp_arr, n_gt)

        ap_by_threshold_class[iou_thr] = per_class_ap

    def mean_ap(iou_thr: float) -> float:
        values = [v for v in ap_by_threshold_class[iou_thr].values() if not np.isnan(v)]
        return float(np.mean(values)) if values else 0.0

    mAP50 = mean_ap(0.5)
    mAP75 = mean_ap(0.75)
    mAP50_95 = float(np.mean([mean_ap(t) for t in iou_thresholds]))

    per_class_names_ap = {
        str(cls): ap_by_threshold_class[0.5][cls]
        for cls in range(num_classes)
    }

    return {
        "mAP50": mAP50,
        "mAP75": mAP75,
        "mAP50_95": mAP50_95,
        "per_class_AP_50": per_class_names_ap,
    }
