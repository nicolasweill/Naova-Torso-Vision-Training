from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: nn.Module,
    save_path: str,
    input_size: tuple[int, int] = (640, 640),
    opset_version: int = 17,
    dynamic_batch: bool = True,
) -> str:
    """
    Export a PyTorch model to ONNX.

    Returns the path to the saved ONNX file.
    Validates the exported model with onnx.checker.
    """
    import onnx

    save_path = str(save_path)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    h, w = input_size
    dummy = torch.zeros(1, 3, h, w, device=next(model.parameters()).device)

    dynamic_axes: dict | None = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            save_path,
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            verbose=False,
        )

    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    logger.info("Exported and validated ONNX model → %s", save_path)
    return save_path
