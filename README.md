# TrainOnnxDet — TORSO-21 Sim-to-Real Fine-Tuning

Fine-tune a detection model (ONNX) trained on simulated RoboCup data onto the **TORSO-21 reality** dataset.
Tracks all experiments with **MLflow**. Runs on **CUDA** by default, falls back to **CPU**.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -e .

# 2. Validate setup (no training, just loads data and model)
python3 scripts/train.py --dry-run

# 3. Launch training
python3 scripts/train.py --device auto

# 3.1 overnight training
ulimit -n 65536 && systemd-inhibit --what=sleep:idle --who="training" --why="60 epochs" python3 scripts/train.py --device auto

# 4. Inspect results
mlflow ui --backend-store-uri mlruns/
mlflow ui --backend-store-uri mlruns/ --port 5001

# 5. Run inference with any ONNX model
python3 scripts/infer.py \
    --model models/sim_data_det_0126.onnx \
    --input dataset/test/images/ \
    --conf 0.25
```

# best model
runs/run_1777849666/best.onnx

python3 scripts/infer.py \
    --model runs/run_1777849666/best.onnx \
    --input dataset/test/images/ \
    --conf 0.25 #0.5


---

## Dataset: TORSO-21 (reality subset)

| Split | Images | Labels |
|-------|--------|--------|
| Train | 4,345  | 8,894  |
| Test  | 1,570  | 1,570  |

**7 classes:** ball · goalpost · robot · obstacle · L-Intersection · T-Intersection · X-Intersection

Annotation format: YOLO (`class_id x_center y_center w h`, normalised).

---

## Project Structure

```
configs/          YAML configs for dataset and training hyperparameters
dataset/          Raw TORSO-21 data (images + YOLO labels, not tracked by git)
models/           Base ONNX models (sim_data_det_0126.onnx)
src/torso/        Python package
  data/           Dataset class, augmentation pipeline, utilities
  model/          ONNX→PyTorch loader, detection head, ONNX exporter
  training/       Training loop, loss, mAP metrics
  inference/      ONNX Runtime runner, output decoding + NMS
  utils/          Device selection, MLflow helpers, visualization
scripts/          CLI entry points (train.py, infer.py)
tests/            Unit tests
runs/             Training outputs — checkpoints + exported ONNX (gitignored)
mlruns/           MLflow tracking store (gitignored)
```

---

## Training

```bash
python scripts/train.py \
    --config   configs/train.yaml \
    --dataset  configs/dataset.yaml \
    --device   auto \               # auto | cuda | cpu
    --run-name "lr1e-4-freeze70"    # optional MLflow run name
    --resume   runs/exp1/best.pt    # optional: resume from checkpoint
    --epochs   50                   # override configs/train.yaml
    --dry-run                       # validate pipeline, then exit
```

The best checkpoint is automatically exported to ONNX and logged as an MLflow artifact.

---

## Inference

```bash
# Single image
python scripts/infer.py --model models/sim_data_det_0126.onnx --input image.jpg

# Directory of images
python scripts/infer.py --model runs/exp1/best.onnx --input dataset/test/images/

# With options
python scripts/infer.py \
    --model   models/sim_data_det_0126.onnx \
    --input   dataset/test/images/ \
    --conf    0.25 \
    --iou     0.45 \
    --output  runs/infer/ \
    --show \
    --benchmark
```

Works with **any** ONNX detection model (auto-detects YOLO-v5, YOLO-v8, SSD output formats).

---

## GPU / CPU Notes

- **Training**: `torch.cuda.is_available()` is checked automatically. Pass `--device cpu` to force CPU.
- **Inference (ONNX Runtime)**: `onnxruntime.get_available_providers()` is used. Install `onnxruntime-gpu` instead of `onnxruntime` to enable CUDA inference.

```bash
# GPU inference
pip uninstall onnxruntime && pip install onnxruntime-gpu
```

---

## MLflow

All runs are stored in `./mlruns/`. Launch the UI with:

```bash
mlflow ui --backend-store-uri mlruns/ --port 5000
```

Logged per run:
- **Params**: all hyperparameters from `train.yaml`
- **Metrics**: `train/loss`, `val/mAP50`, `val/mAP75`, per-class AP — one point per epoch
- **Artifacts**: best ONNX model with signature, config snapshots
- **Tags**: `git_commit`, `base_model`, `dataset`, `python_version`, `hostname`
