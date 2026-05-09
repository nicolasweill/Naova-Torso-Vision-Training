from __future__ import annotations

import logging
import platform
import socket
import subprocess
from typing import Any

import mlflow
import mlflow.onnx
import numpy as np
from mlflow.models import ModelSignature, infer_signature

logger = logging.getLogger(__name__)


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _flatten_dict(d: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, full_key))
        else:
            out[full_key] = v
    return out


def setup_experiment(tracking_uri: str, experiment_name: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow experiment: %s  (store: %s)", experiment_name, tracking_uri)


def start_run(
    run_name: str | None = None,
    extra_tags: dict[str, str] | None = None,
) -> mlflow.ActiveRun:
    tags: dict[str, str] = {
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "hostname": socket.gethostname(),
    }
    if extra_tags:
        tags.update(extra_tags)
    return mlflow.start_run(run_name=run_name, tags=tags)


def log_config(cfg_dict: dict) -> None:
    flat = _flatten_dict(cfg_dict)
    # MLflow param values are capped at 500 chars
    params = {k: str(v)[:500] for k, v in flat.items()}
    mlflow.log_params(params)


def log_epoch_metrics(metrics: dict[str, float], step: int) -> None:
    mlflow.log_metrics(metrics, step=step)


def log_model_artifact(
    onnx_model_path: str,
    input_example: np.ndarray,
    output_example: np.ndarray,
    registered_name: str | None = None,
) -> None:
    import onnx

    model = onnx.load(onnx_model_path)
    signature = infer_signature(input_example, output_example)

    mlflow.onnx.log_model(
        onnx_model=model,
        artifact_path="model",
        signature=signature,
        registered_model_name=registered_name,
    )
    # Also log the raw file for easy download
    mlflow.log_artifact(onnx_model_path, artifact_path="onnx_export")
    logger.info("MLflow: model artifact logged (registered_name=%s)", registered_name)


def end_run(success: bool) -> None:
    status = "FINISHED" if success else "FAILED"
    mlflow.end_run(status=status)
    logger.info("MLflow run ended with status: %s", status)
