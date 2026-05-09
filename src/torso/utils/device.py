from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def get_device(requested: str = "auto") -> torch.device:
    """
    Resolve and return a torch.device.

    "auto"  → CUDA if available, else CPU
    "cuda"  → CUDA; raises RuntimeError if not available
    "cpu"   → always succeeds
    """
    requested = requested.lower()

    if requested == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            name = torch.cuda.get_device_name(0)
            logger.info("Device: CUDA (%s)", name)
        else:
            device = torch.device("cpu")
            logger.info("Device: CPU (CUDA not available)")
        return device

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build or use --device auto."
            )
        device = torch.device("cuda")
        logger.info("Device: CUDA (%s)", torch.cuda.get_device_name(0))
        return device

    if requested == "cpu":
        logger.info("Device: CPU (forced)")
        return torch.device("cpu")

    raise ValueError(f"Unknown device '{requested}'. Choose: auto | cuda | cpu")
