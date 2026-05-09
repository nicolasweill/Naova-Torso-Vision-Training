"""
Tests for inference postprocessing (NMS, output format detection, decoding).

Run with: pytest tests/test_postprocess.py -v
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from torso.inference.postprocess import (
    _detect_format,
    _nms,
    decode_outputs,
)


# ── NMS ───────────────────────────────────────────────────────────────────────

def test_nms_keeps_highest_score():
    # Two heavily overlapping boxes — only the higher-score one should survive
    boxes = np.array([[0.0, 0.0, 0.5, 0.5], [0.05, 0.05, 0.55, 0.55]], dtype=np.float32)
    scores = np.array([0.9, 0.6])
    kept = _nms(boxes, scores, iou_threshold=0.5)
    assert len(kept) == 1
    assert kept[0] == 0  # highest score kept


def test_nms_keeps_non_overlapping():
    # Completely separate boxes — both should survive
    boxes = np.array([[0.0, 0.0, 0.2, 0.2], [0.8, 0.8, 1.0, 1.0]], dtype=np.float32)
    scores = np.array([0.9, 0.8])
    kept = _nms(boxes, scores, iou_threshold=0.5)
    assert len(kept) == 2


def test_nms_empty_input():
    boxes = np.zeros((0, 4), dtype=np.float32)
    scores = np.zeros(0)
    kept = _nms(boxes, scores, iou_threshold=0.5)
    assert len(kept) == 0


# ── Format detection ─────────────────────────────────────────────────────────

def test_detect_format_yolo_v5():
    # [B, N, 5+C] with B=1, N=8400, C=12
    raw = np.random.rand(1, 8400, 12).astype(np.float32)
    fmt = _detect_format([raw])
    assert fmt == "yolo_v5"


def test_detect_format_yolo_v8():
    # [B, 4+C, N] — more anchors than channels
    raw = np.random.rand(1, 11, 8400).astype(np.float32)
    fmt = _detect_format([raw])
    assert fmt == "yolo_v8"


def test_detect_format_ssd():
    boxes = np.random.rand(1, 100, 4).astype(np.float32)
    scores = np.random.rand(1, 100, 7).astype(np.float32)
    fmt = _detect_format([boxes, scores])
    assert fmt == "ssd"


# ── Full decode ───────────────────────────────────────────────────────────────

def test_decode_yolo_v5_returns_valid_boxes():
    # Craft a YOLO-v5 output with one confident detection
    n_anchors = 100
    num_classes = 7
    raw = np.zeros((1, n_anchors, 5 + num_classes), dtype=np.float32)
    # Put a high-confidence box at anchor index 5
    raw[0, 5, 0] = 0.5   # cx
    raw[0, 5, 1] = 0.5   # cy
    raw[0, 5, 2] = 0.2   # w
    raw[0, 5, 3] = 0.2   # h
    raw[0, 5, 4] = 5.0   # objectness logit → sigmoid ≈ 0.99
    raw[0, 5, 5] = 5.0   # class 0 logit   → sigmoid ≈ 0.99

    result = decode_outputs([raw], conf_threshold=0.5, num_classes=num_classes)

    assert "boxes" in result
    assert "scores" in result
    assert "labels" in result
    assert len(result["boxes"]) >= 1
    assert result["scores"][0] > 0.5
    # All box coords should be in [0, 1]
    assert np.all(result["boxes"] >= 0.0)
    assert np.all(result["boxes"] <= 1.0)


def test_decode_empty_after_threshold():
    # All anchors below conf threshold → empty output
    raw = np.zeros((1, 100, 12), dtype=np.float32)
    result = decode_outputs([raw], conf_threshold=0.99, num_classes=7)
    assert len(result["boxes"]) == 0


def test_decode_raw_format_returns_empty():
    # A 2D array that doesn't match any known format
    raw = np.random.rand(10, 10).astype(np.float32)
    result = decode_outputs([raw], conf_threshold=0.25, num_classes=7)
    assert len(result["boxes"]) == 0
