"""
Smoke tests for Torso21Dataset.

Run with: pytest tests/test_dataset.py -v
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from torso.data.dataset import Torso21Dataset, _letterbox, _parse_yolo_label


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_fake_split(tmp_dir: Path, n_images: int = 5) -> Path:
    """Create a minimal fake split directory with images and labels."""
    images_dir = tmp_dir / "images"
    images_dir.mkdir(parents=True)

    for i in range(n_images):
        # Write a tiny valid PNG (1x1 white pixel)
        import cv2
        img = np.ones((48, 64, 3), dtype=np.uint8) * 128
        cv2.imwrite(str(images_dir / f"img_{i:03d}.png"), img)

        label_path = tmp_dir / f"img_{i:03d}.txt"
        if i == 0:
            # Empty label (background image)
            label_path.write_text("")
        else:
            # One detection per image
            label_path.write_text(f"{i % 7} 0.5 0.5 0.2 0.3\n")

    return tmp_dir


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_letterbox_preserves_aspect_ratio():
    img = np.zeros((480, 620, 3), dtype=np.uint8)
    result, scale, pad = _letterbox(img, (640, 640))
    assert result.shape == (640, 640, 3)
    # scale = min(640/620, 640/480)
    expected_scale = min(640 / 620, 640 / 480)
    assert abs(scale - expected_scale) < 1e-5


def test_parse_yolo_label_valid():
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("0 0.5 0.5 0.2 0.3\n")
        f.write("3 0.1 0.9 0.05 0.05\n")
        path = Path(f.name)

    boxes, labels = _parse_yolo_label(path, num_classes=7)
    assert len(boxes) == 2
    assert list(labels) == [0, 3]
    assert boxes.shape == (2, 4)
    path.unlink()


def test_parse_yolo_label_empty():
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        path = Path(f.name)

    boxes, labels = _parse_yolo_label(path, num_classes=7)
    assert boxes.shape == (0, 4)
    assert len(labels) == 0
    path.unlink()


def test_parse_yolo_label_malformed_skipped():
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("0 0.5 0.5 0.2 0.3\n")
        f.write("bad line here\n")
        f.write("2 0.3 0.3 0.1 0.1\n")
        path = Path(f.name)

    boxes, labels = _parse_yolo_label(path, num_classes=7)
    assert len(boxes) == 2  # malformed line skipped
    path.unlink()


def test_dataset_discovery_and_length():
    with tempfile.TemporaryDirectory() as tmp:
        split_dir = _make_fake_split(Path(tmp))
        ds = Torso21Dataset(split_dir, input_size=(64, 64), num_classes=7)
        assert len(ds) == 5


def test_dataset_getitem_returns_correct_types():
    with tempfile.TemporaryDirectory() as tmp:
        split_dir = _make_fake_split(Path(tmp))
        ds = Torso21Dataset(split_dir, input_size=(64, 64), num_classes=7)
        sample = ds[1]

        assert isinstance(sample["image"], torch.Tensor)
        assert sample["image"].shape == (3, 64, 64)
        assert isinstance(sample["boxes"], torch.Tensor)
        assert sample["boxes"].ndim == 2 and sample["boxes"].shape[1] == 4
        assert isinstance(sample["labels"], torch.Tensor)
        assert len(sample["labels"]) == len(sample["boxes"])


def test_dataset_empty_label_kept():
    """Background images (empty label) must not be dropped."""
    with tempfile.TemporaryDirectory() as tmp:
        split_dir = _make_fake_split(Path(tmp))
        ds = Torso21Dataset(split_dir, input_size=(64, 64), num_classes=7)
        sample = ds[0]  # first image has empty label
        assert sample["boxes"].shape == (0, 4)
        assert len(sample["labels"]) == 0


def test_dataset_skips_images_without_labels():
    with tempfile.TemporaryDirectory() as tmp:
        split_dir = Path(tmp)
        images_dir = split_dir / "images"
        images_dir.mkdir()

        import cv2
        # Image WITH label
        img = np.ones((48, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / "with_label.png"), img)
        (split_dir / "with_label.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        # Image WITHOUT label
        cv2.imwrite(str(images_dir / "no_label.png"), img)

        ds = Torso21Dataset(split_dir, input_size=(64, 64), num_classes=7)
        assert len(ds) == 1
