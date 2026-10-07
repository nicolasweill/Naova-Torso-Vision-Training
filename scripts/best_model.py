#!/usr/bin/env python3
"""
Retrieve the run with the best metric from MLflow and download its model.

Usage:
    python scripts/best_model.py
    python scripts/best_model.py --metric val/mAP75 --output runs/best_retrieved
    python scripts/best_model.py --experiment torso21-sim2real --list
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch best MLflow model")
    parser.add_argument("--tracking-uri", default="mlruns/", help="MLflow tracking URI")
    parser.add_argument("--experiment", default="torso21-sim2real", help="Experiment name")
    parser.add_argument(
        "--metric", default="val/mAP50", help="Metric to maximise (default: val/mAP50)"
    )
    parser.add_argument(
        "--output", default="runs/best_retrieved", help="Directory to save the model"
    )
    parser.add_argument(
        "--artifact", default="onnx_export", choices=["onnx_export", "checkpoints", "model"],
        help="Which artifact to download (default: onnx_export)"
    )
    parser.add_argument(
        "--list", action="store_true", help="List all finished runs with their metrics and exit"
    )
    return parser.parse_args()


def get_experiment(client, name: str):
    exp = client.get_experiment_by_name(name)
    if exp is None:
        logger.error("Experiment '%s' not found. Available experiments:", name)
        for e in client.search_experiments():
            logger.error("  - %s", e.name)
        sys.exit(1)
    return exp


def get_max_metric(client, run_id: str, metric: str) -> tuple[float, int]:
    """Return (max_value, step) of a metric across all logged epochs for a run."""
    history = client.get_metric_history(run_id, metric)
    if not history:
        return float("nan"), -1
    best = max(history, key=lambda m: m.value)
    return best.value, best.step


def list_runs(client, experiment_id: str, metric: str) -> list[tuple]:
    """Return runs sorted by MAX metric value across all epochs (not just last)."""
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="attributes.status = 'FINISHED'",
    )
    scored = []
    for run in runs:
        max_val, best_step = get_max_metric(client, run.info.run_id, metric)
        scored.append((run, max_val, best_step))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def print_runs_table(scored: list[tuple], metric: str) -> None:
    if not scored:
        logger.info("No finished runs found.")
        return

    header = f"{'Rank':<5} {'Run ID':<12} {'Run Name':<25} {f'max({metric})':<18} {'Best epoch':<12} {'Date'}"
    print(header)
    print("-" * len(header))
    for i, (run, max_val, best_step) in enumerate(scored):
        from datetime import datetime, timezone
        run_id = run.info.run_id[:8]
        name = (run.info.run_name or "—")[:24]
        ts = datetime.fromtimestamp(run.info.start_time / 1000, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M"
        )
        marker = " ←" if i == 0 else ""
        epoch_str = str(best_step) if best_step >= 0 else "?"
        print(f"{i+1:<5} {run_id:<12} {name:<25} {max_val:<18.4f} {epoch_str:<12} {ts}{marker}")


def download_artifact(client, run_id: str, artifact_path: str, output_dir: str) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    local_path = client.download_artifacts(run_id, artifact_path, str(output))
    return Path(local_path)


def main() -> None:
    args = parse_args()

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    exp = get_experiment(client, args.experiment)
    logger.info("Experiment: %s (id=%s)", exp.name, exp.experiment_id)

    scored = list_runs(client, exp.experiment_id, args.metric)

    if not scored:
        logger.error("No finished runs found in experiment '%s'.", args.experiment)
        sys.exit(1)

    if args.list:
        print_runs_table(scored, args.metric)
        return

    best_run, best_value, best_epoch = scored[0]
    logger.info(
        "Best run: %s  (max %s=%.4f at epoch %s, name=%s)",
        best_run.info.run_id,
        args.metric,
        best_value,
        best_epoch,
        best_run.info.run_name or "—",
    )

    logger.info("Downloading artifact '%s' ...", args.artifact)
    local = download_artifact(client, best_run.info.run_id, args.artifact, args.output)
    logger.info("Saved to: %s", local)

    # Print all metric MAX values for the best run
    print(f"\n── Best run: {best_run.info.run_name} ({best_run.info.run_id[:8]}) ──")
    print(f"   Ranked by: max({args.metric}) = {best_value:.4f}  (epoch {best_epoch})\n")
    print(f"{'Metric':<32} {'Last value':>12}  {'Max value':>12}  {'Best epoch':>10}")
    print("-" * 72)
    for k in sorted(best_run.data.metrics.keys()):
        last_val = best_run.data.metrics[k]
        max_val, step = get_max_metric(client, best_run.info.run_id, k)
        print(f"  {k:<30} {last_val:>12.4f}  {max_val:>12.4f}  {step:>10}")
    print(f"\n── Artifact location: {local}")


if __name__ == "__main__":
    main()
