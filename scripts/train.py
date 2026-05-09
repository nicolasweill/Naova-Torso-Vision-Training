#!/usr/bin/env python3
"""
Train a detection model on TORSO-21 by fine-tuning an ONNX model.

Usage:
    python scripts/train.py [--config configs/train.yaml] [--dataset configs/dataset.yaml]
                            [--device auto|cuda|cpu] [--run-name my-run]
                            [--epochs 50] [--resume runs/exp/best.pt] [--dry-run]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as script without `pip install -e .`
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mlflow
import yaml
from torch.utils.data import DataLoader

from torso.data.dataset import Torso21Dataset
from torso.data.transforms import AugConfig, get_train_transforms, get_val_transforms
from torso.data.utils import collate_fn
from torso.model.loader import ModelConversionError, load_checkpoint, load_model_for_training
from torso.training.loss import LossConfig
from torso.training.trainer import Trainer, TrainingConfig
from torso.utils.device import get_device
from torso.utils.mlflow_utils import end_run, log_config, setup_experiment, start_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ONNX detection model on TORSO-21")
    parser.add_argument("--config", default="configs/train.yaml", help="Training config YAML")
    parser.add_argument("--dataset", default="configs/dataset.yaml", help="Dataset config YAML")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--run-name", default=None, help="MLflow run name")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume", default=None, help="Path to .pt checkpoint to resume from")
    parser.add_argument("--dry-run", action="store_true", help="Validate pipeline and exit")
    return parser.parse_args()


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_configs(train_cfg: dict, ds_cfg: dict, args: argparse.Namespace):
    """Parse YAML dicts into dataclasses."""
    t = train_cfg.get("training", {})
    m = train_cfg.get("model", {})
    l = train_cfg.get("loss", {})
    aug = train_cfg.get("augmentation", {})
    mlf = train_cfg.get("mlflow", {})
    ds = ds_cfg.get("dataset", {})
    classes = ds_cfg.get("classes", {})

    num_classes = classes.get("num_classes", 7)
    input_size = tuple(ds.get("input_size", [640, 640]))

    loss_cfg = LossConfig(
        box_weight=l.get("box_weight", 5.0),
        cls_weight=l.get("cls_weight", 1.0),
        obj_weight=l.get("obj_weight", 1.0),
        label_smoothing=l.get("label_smoothing", 0.1),
        num_classes=num_classes,
    )

    aug_cfg = AugConfig(
        enabled=aug.get("enabled", True),
        horizontal_flip=aug.get("horizontal_flip", 0.5),
        random_brightness_contrast=aug.get("random_brightness_contrast", 0.3),
        hue_saturation_value=aug.get("hue_saturation_value", 0.3),
        random_scale_min=aug.get("random_scale_min", 0.5),
        random_scale_max=aug.get("random_scale_max", 1.5),
    )

    train_config = TrainingConfig(
        epochs=args.epochs or t.get("epochs", 50),
        batch_size=args.batch_size or t.get("batch_size", 16),
        num_workers=t.get("num_workers", 4),
        learning_rate=t.get("learning_rate", 1e-4),
        weight_decay=t.get("weight_decay", 1e-4),
        lr_scheduler=t.get("lr_scheduler", "cosine"),
        warmup_epochs=t.get("warmup_epochs", 3),
        grad_clip_norm=t.get("grad_clip_norm", 10.0),
        val_every_n_epochs=t.get("val_every_n_epochs", 1),
        save_best_metric=t.get("save_best_metric", "val/mAP50"),
        output_dir=m.get("output_dir", "runs"),
        loss=loss_cfg,
        num_classes=num_classes,
        input_size=input_size,
    )

    return train_config, aug_cfg, mlf, m, ds, classes


def main() -> None:
    args = parse_args()

    train_cfg = load_yaml(args.config)
    ds_cfg = load_yaml(args.dataset)

    train_config, aug_cfg, mlf_cfg, model_cfg, ds_cfg_inner, classes_cfg = build_configs(
        train_cfg, ds_cfg, args
    )

    device = get_device(args.device)
    num_classes = classes_cfg.get("num_classes", 7)
    input_size = train_config.input_size
    root = Path(ds_cfg_inner.get("root", "dataset"))

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_split_dir = root / ds_cfg_inner["train"]["labels_dir"].split("/")[0]
    test_split_dir = root / ds_cfg_inner["test"]["labels_dir"].split("/")[0]

    train_transform = get_train_transforms(aug_cfg, input_size)
    val_transform = get_val_transforms(input_size)

    train_dataset = Torso21Dataset(
        train_split_dir, input_size=input_size, num_classes=num_classes, transform=train_transform
    )
    val_dataset = Torso21Dataset(
        test_split_dir, input_size=input_size, num_classes=num_classes, transform=val_transform
    )

    logger.info("Train: %d samples | Val: %d samples", len(train_dataset), len(val_dataset))

    if args.dry_run:
        logger.info("Dry run: dataset loaded successfully. Exiting.")
        # Quick single batch test
        sample = train_dataset[0]
        logger.info(
            "Sample image shape: %s | boxes: %s | labels: %s",
            sample["image"].shape,
            sample["boxes"].shape,
            sample["labels"],
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    onnx_path = model_cfg.get("onnx_path", "models/sim_data_det_0126.onnx")
    freeze_ratio = model_cfg.get("freeze_ratio", 0.7)

    try:
        model, metadata = load_model_for_training(onnx_path, device, freeze_ratio)
    except ModelConversionError as e:
        logger.error("Model conversion failed:\n%s", e)
        sys.exit(1)

    if args.resume:
        load_checkpoint(args.resume, model, device)

    if args.dry_run:
        import torch
        dummy = torch.zeros(1, 3, *input_size, device=device)
        with torch.no_grad():
            out = model(dummy)
        logger.info("Model forward pass OK. Output type: %s", type(out))
        logger.info("Dry run complete — all checks passed.")
        return

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    # ── MLflow ────────────────────────────────────────────────────────────────
    tracking_uri = mlf_cfg.get("tracking_uri", "mlruns/")
    experiment_name = mlf_cfg.get("experiment_name", "torso21-sim2real")
    extra_tags = mlf_cfg.get("tags", {})

    setup_experiment(tracking_uri, experiment_name)

    success = False
    with start_run(run_name=args.run_name, extra_tags=extra_tags):
        try:
            log_config({**train_cfg, **{"model_metadata": metadata}})

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                cfg=train_config,
                run_name=args.run_name,
            )
            trainer.train()
            success = True
        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
        finally:
            end_run(success)


if __name__ == "__main__":
    main()
