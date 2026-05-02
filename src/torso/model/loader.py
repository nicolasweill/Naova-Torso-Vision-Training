from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelConversionError(RuntimeError):
    pass


def _read_onnx_metadata(onnx_path: str) -> dict:
    import onnx

    model = onnx.load(onnx_path)
    inputs = [
        {
            "name": inp.name,
            "shape": [
                d.dim_value if d.dim_value > 0 else None
                for d in inp.type.tensor_type.shape.dim
            ],
        }
        for inp in model.graph.input
    ]
    outputs = [
        {
            "name": out.name,
            "shape": [
                d.dim_value if d.dim_value > 0 else None
                for d in out.type.tensor_type.shape.dim
            ],
        }
        for out in model.graph.output
    ]
    opset = model.opset_import[0].version if model.opset_import else None
    return {"inputs": inputs, "outputs": outputs, "opset": opset}


def _freeze_sim_domain_layers(model: nn.Module, freeze_ratio: float = 0.7) -> dict:
    """
    Freeze strategy for sim-to-real transfer:
      1. All BatchNorm2d layers (carry sim-domain statistics).
      2. The first `freeze_ratio` fraction of named parameter groups.

    Returns a summary dict with frozen/total counts.
    """
    all_params = list(model.named_parameters())
    total = len(all_params)
    freeze_up_to = int(total * freeze_ratio)

    frozen = 0
    for i, (name, param) in enumerate(all_params):
        should_freeze = i < freeze_up_to
        param.requires_grad_(not should_freeze)
        if should_freeze:
            frozen += 1

    # Also freeze all BN layers regardless of position
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            for param in module.parameters():
                param.requires_grad_(False)
            module.eval()

    trainable = sum(p.requires_grad for p in model.parameters())
    logger.info(
        "Freeze: %d / %d params frozen (ratio=%.0f%%, BN layers always frozen). "
        "Trainable: %d",
        frozen,
        total,
        freeze_ratio * 100,
        trainable,
    )
    return {"frozen": frozen, "total": total, "trainable": trainable}


def load_model_for_training(
    onnx_path: str,
    device: torch.device,
    freeze_ratio: float = 0.7,
) -> tuple[nn.Module, dict]:
    """
    Convert an ONNX model to a trainable PyTorch nn.Module.

    Steps:
      1. Read ONNX metadata (always safe).
      2. Convert with onnx2torch.
      3. Apply sim-to-real freeze strategy.
      4. Move to device.

    Returns (model, metadata) where metadata includes input/output info
    and which layers are frozen.

    Raises ModelConversionError if onnx2torch cannot convert the model.
    """
    path = Path(onnx_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    logger.info("Reading ONNX metadata from %s ...", onnx_path)
    metadata = _read_onnx_metadata(onnx_path)
    logger.info(
        "ONNX — opset: %s  inputs: %s  outputs: %s",
        metadata["opset"],
        [i["name"] + str(i["shape"]) for i in metadata["inputs"]],
        [o["name"] + str(o["shape"]) for o in metadata["outputs"]],
    )

    logger.info("Converting ONNX → PyTorch via onnx2torch ...")
    try:
        import onnx2torch

        torch_model = onnx2torch.convert(onnx_path)
    except Exception as exc:
        raise ModelConversionError(
            f"onnx2torch could not convert '{onnx_path}'.\n"
            f"Underlying error: {type(exc).__name__}: {exc}\n\n"
            "Suggestions:\n"
            "  • Re-export the model from its original PyTorch source with a "
            "simpler opset (--opset 11 or 12).\n"
            "  • Check onnx2torch compatibility: https://github.com/ENOT-AutoDL/onnx2torch\n"
            "  • Use inference-only mode (scripts/infer.py) with the raw ONNX."
        ) from exc

    freeze_info = _freeze_sim_domain_layers(torch_model, freeze_ratio)
    metadata["freeze_info"] = freeze_info
    metadata["conversion_method"] = "onnx2torch"

    torch_model = torch_model.to(device)
    torch_model.train()

    # Keep BN in eval mode after freeze to prevent running-stat updates
    for module in torch_model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    return torch_model, metadata


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    device: torch.device,
) -> dict:
    """Load a .pt checkpoint back into `model`. Returns the checkpoint dict."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Loaded checkpoint from %s (epoch %d)", checkpoint_path, ckpt.get("epoch", -1))
    return ckpt
