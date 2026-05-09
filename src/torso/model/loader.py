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


def _register_missing_onnx2torch_converters() -> None:
    """Register onnx2torch converters missing for opset 18/19 ops used by YOLO models.

    - Split v18: added optional num_outputs attribute (same logic as v13).
    - Reshape v19: no semantic change from v14.
    - Resize v19: added antialias attribute (unused by YOLO; same logic as v13).
    """
    import warnings

    from onnx import defs
    from onnx2torch.node_converters.registry import (
        OperationDescription,
        _CONVERTER_REGISTRY,
        add_converter,
    )
    from onnx2torch.node_converters.reshape import OnnxReshape
    from onnx2torch.node_converters.resize import OnnxResize, _get_torch_align_corners
    from onnx2torch.node_converters.split import OnnxSplit13
    from onnx2torch.onnx_graph import OnnxGraph
    from onnx2torch.onnx_node import OnnxNode
    from onnx2torch.utils.common import OperationConverterResult, onnx_mapping_from_node

    def _missing(op: str, ver: int) -> bool:
        desc = OperationDescription(domain=defs.ONNX_DOMAIN, operation_type=op, version=ver)
        return desc not in _CONVERTER_REGISTRY

    if _missing("Split", 18):

        @add_converter(operation_type="Split", version=18)
        def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # noqa: F811
            axis = node.attributes.get("axis", 0)
            num_splits = node.attributes.get("num_outputs", len(node.output_values))
            return OperationConverterResult(
                torch_module=OnnxSplit13(axis=axis, num_splits=num_splits),
                onnx_mapping=onnx_mapping_from_node(node=node),
            )

    if _missing("Reshape", 19):

        @add_converter(operation_type="Reshape", version=19)
        def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # noqa: F811
            if node.attributes.get("allowzero", 0) == 1:
                raise NotImplementedError('"allowzero=1" is not implemented')
            return OperationConverterResult(
                torch_module=OnnxReshape(),
                onnx_mapping=onnx_mapping_from_node(node=node),
            )

    if _missing("Resize", 19):

        @add_converter(operation_type="Resize", version=19)
        def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # noqa: F811
            attrs = node.attributes
            coordinate_transformation_mode = attrs.get(
                "coordinate_transformation_mode", "half_pixel"
            )
            mode = attrs.get("mode", "nearest")
            nearest_mode = attrs.get("nearest_mode", "round_prefer_floor")
            if attrs.get("antialias", 0):
                warnings.warn("Resize antialias is not supported; ignoring.")
            if mode == "nearest" and nearest_mode != "floor":
                warnings.warn(
                    "PyTorch nearest interpolation uses 'floor' nearest_mode; "
                    "results may differ."
                )
            ignore_roi = coordinate_transformation_mode != "tf_crop_and_resize"
            return OperationConverterResult(
                torch_module=OnnxResize(
                    mode=mode,
                    align_corners=_get_torch_align_corners(
                        mode, coordinate_transformation_mode
                    ),
                    ignore_roi=ignore_roi,
                ),
                onnx_mapping=onnx_mapping_from_node(node=node),
            )


def _make_batch_dynamic(model: "onnx.ModelProto") -> "onnx.ModelProto":
    """Replace hardcoded batch_size=1 with -1 in Reshape shape initializers.

    Models exported with a fixed batch=1 embed '1' as the first element of the
    Reshape target shape (e.g. [1, 4, 16, 2100] in the DFL head).  Replacing
    that with -1 lets PyTorch infer the correct batch size at runtime.
    """
    import numpy as np
    from onnx import numpy_helper

    reshape_shape_inputs = {
        node.input[1]
        for node in model.graph.node
        if node.op_type == "Reshape" and len(node.input) >= 2
    }

    patched = 0
    for init in model.graph.initializer:
        if init.name not in reshape_shape_inputs:
            continue
        arr = numpy_helper.to_array(init).copy()
        if arr.ndim == 1 and len(arr) > 0 and arr[0] == 1:
            # Use 0 if shape already has a -1: OnnxReshape treats 0 as
            # "copy this dim from the input", avoiding two inferred dims.
            arr[0] = 0 if -1 in arr else -1
            init.CopyFrom(numpy_helper.from_array(arr, name=init.name))
            patched += 1

    if patched:
        logger.info("Patched %d Reshape initializer(s): static batch=1 → dynamic(-1)", patched)
    return model


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
        import onnx
        import onnx2torch

        _register_missing_onnx2torch_converters()
        onnx_model = _make_batch_dynamic(onnx.load(str(onnx_path)))
        torch_model = onnx2torch.convert(onnx_model)
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
