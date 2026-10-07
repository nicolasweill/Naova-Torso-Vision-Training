from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from torso.model.exporter import export_to_onnx
from torso.model.head import DetectionHead
from torso.training.loss import DetectionLoss, LossConfig
from torso.training.metrics import compute_map
from torso.utils.mlflow_utils import log_epoch_metrics, log_model_artifact

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"
    warmup_epochs: int = 3
    grad_clip_norm: float = 10.0
    val_every_n_epochs: int = 1
    save_best_metric: str = "val/mAP50"
    output_dir: str = "runs"
    num_workers: int = 4

    loss: LossConfig = field(default_factory=LossConfig)
    num_classes: int = 7
    input_size: tuple[int, int] = (640, 640)


def _build_optimizer(model: nn.Module, cfg: TrainingConfig) -> torch.optim.Optimizer:
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        trainable,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainingConfig,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if cfg.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.epochs - cfg.warmup_epochs,
            eta_min=cfg.learning_rate * 0.01,
        )
    if cfg.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    return None


def _warmup_lr(optimizer: torch.optim.Optimizer, epoch: int, warmup_epochs: int, base_lr: float) -> None:
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
        for pg in optimizer.param_groups:
            pg["lr"] = lr


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        cfg: TrainingConfig,
        run_name: str | None = None,
    ) -> None:
        self.model = model
        self.detection_head = DetectionHead(model, input_size=cfg.input_size)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cfg = cfg
        self.run_name = run_name

        self.output_dir = Path(cfg.output_dir) / (run_name or f"run_{int(time.time())}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = _build_optimizer(model, cfg)
        self.scheduler = _build_scheduler(self.optimizer, cfg)
        self.loss_fn = DetectionLoss(cfg.loss, output_format="auto")

        self.best_metric_value = -float("inf")
        self.best_checkpoint_path: str | None = None

    def train(self) -> None:
        logger.info("Starting training for %d epochs → %s", self.cfg.epochs, self.output_dir)

        for epoch in range(self.cfg.epochs):
            _warmup_lr(self.optimizer, epoch, self.cfg.warmup_epochs, self.cfg.learning_rate)

            train_metrics = self._train_epoch(epoch)

            val_metrics: dict[str, float] = {}
            if (epoch + 1) % self.cfg.val_every_n_epochs == 0:
                val_metrics = self._val_epoch(epoch)

            all_metrics = {
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()},
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            log_epoch_metrics(all_metrics, step=epoch)

            # ── Console summary ──────────────────────────────────────────────
            map50 = val_metrics.get("mAP50", float("nan"))
            logger.info(
                "Epoch %d/%d  train_loss=%.4f  val/mAP50=%.4f  lr=%.2e",
                epoch + 1,
                self.cfg.epochs,
                train_metrics.get("loss", float("nan")),
                map50,
                self.optimizer.param_groups[0]["lr"],
            )

            if val_metrics:
                self._maybe_save_best(epoch, val_metrics)

            if self.scheduler and epoch >= self.cfg.warmup_epochs:
                self.scheduler.step()

        self._finalize_run()

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        # Keep BN in eval to preserve sim-domain stats
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        total_loss = total_box = total_cls = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.cfg.epochs} [train]", leave=False)
        for batch in pbar:
            images = batch["images"].to(self.device)
            targets = [
                {"boxes": b.to(self.device), "labels": l.to(self.device)}
                for b, l in zip(batch["boxes"], batch["labels"])
            ]

            self.optimizer.zero_grad()
            decoded, raw = self.detection_head(images)

            # Update loss fn output format from head
            self.loss_fn.output_format = self.detection_head.output_format or "raw"

            losses = self.loss_fn(decoded, targets)
            losses["total"].backward()

            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.cfg.grad_clip_norm,
            )
            self.optimizer.step()

            total_loss += losses["total"].item()
            total_box += losses["box"].item()
            total_cls += losses["cls"].item()
            n_batches += 1

            pbar.set_postfix(
                loss=f"{losses['total'].item():.4f}",
                box=f"{losses['box'].item():.4f}",
                cls=f"{losses['cls'].item():.4f}",
            )

        n = max(n_batches, 1)
        return {"loss": total_loss / n, "box_loss": total_box / n, "cls_loss": total_cls / n}

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> dict[str, float]:
        self.model.eval()

        all_preds: list[dict] = []
        all_targets: list[dict] = []
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{self.cfg.epochs} [val]", leave=False)
        for batch in pbar:
            images = batch["images"].to(self.device)
            targets = [
                {"boxes": b.to(self.device), "labels": l.to(self.device)}
                for b, l in zip(batch["boxes"], batch["labels"])
            ]

            decoded, _ = self.detection_head(images)
            self.loss_fn.output_format = self.detection_head.output_format or "raw"
            losses = self.loss_fn(decoded, targets)

            total_loss += losses["total"].item()
            n_batches += 1

            # DetectionHead returns normalised cxcywh; convert to xyxy for mAP
            for pred in decoded:
                boxes = pred["boxes"].cpu()
                scores = pred["scores"].cpu()
                labels = pred["labels"].cpu()
                if self.detection_head.output_format in ("yolo_v5", "yolo_v8"):
                    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                    boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=-1)
                all_preds.append({"boxes": boxes, "scores": scores, "labels": labels})

            for tgt in targets:
                all_targets.append({"boxes": tgt["boxes"].cpu(), "labels": tgt["labels"].cpu()})

        map_results = compute_map(all_preds, all_targets, self.cfg.num_classes)
        n = max(n_batches, 1)

        metrics = {
            "loss": total_loss / n,
            "mAP50": map_results["mAP50"],
            "mAP75": map_results["mAP75"],
            "mAP50_95": map_results["mAP50_95"],
        }
        for cls_id, ap in map_results["per_class_AP_50"].items():
            if not (isinstance(ap, float) and math.isnan(ap)):
                metrics[f"AP50_cls{cls_id}"] = float(ap)

        return metrics

    def _maybe_save_best(self, epoch: int, val_metrics: dict[str, float]) -> None:
        metric_key = self.cfg.save_best_metric.replace("val/", "")
        value = val_metrics.get(metric_key, -float("inf"))

        if value > self.best_metric_value:
            self.best_metric_value = value
            ckpt_path = self.output_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    f"best_{metric_key}": value,
                },
                ckpt_path,
            )
            self.best_checkpoint_path = str(ckpt_path)
            logger.info(
                "Saved best checkpoint (epoch %d, %s=%.4f) → %s",
                epoch + 1,
                metric_key,
                value,
                ckpt_path,
            )
            mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")

    def _finalize_run(self) -> None:
        logger.info("Finalising run — exporting best model to ONNX ...")

        if self.best_checkpoint_path:
            ckpt = torch.load(self.best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])

        onnx_path = str(self.output_dir / "best.onnx")
        export_to_onnx(self.model, onnx_path, input_size=self.cfg.input_size)

        # Build a dummy example for MLflow signature
        dummy_input = np.zeros((1, 3, *self.cfg.input_size), dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            t_out = self.model(torch.from_numpy(dummy_input).to(self.device))
            if isinstance(t_out, (list, tuple)):
                dummy_output = t_out[0].cpu().numpy()
            else:
                dummy_output = t_out.cpu().numpy()

        log_model_artifact(
            onnx_path,
            input_example=dummy_input,
            output_example=dummy_output,
            registered_name="torso21-detector",
        )

        logger.info("Run complete. Best model: %s", onnx_path)
