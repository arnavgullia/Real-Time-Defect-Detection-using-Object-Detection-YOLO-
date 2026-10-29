"""
Reproducible Training Pipeline for YOLOv8 Baseline on NEU-DET Surface Defect Dataset.
Supports CUDA, MPS, and CPU execution with automated hardware logging and fail-safe checkpoint management.
"""

import os
import sys
import json
import time
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any

# Add workspace root to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.utils import (
    setup_logger,
    get_system_hardware_info,
    get_best_device,
    load_yaml,
)


def train_baseline(
    data_yaml: str = "datasets/defect_dataset/data.yaml",
    model_name: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = "auto",
    project: str = "outputs/baseline",
    run_name: str = "baseline_run",
    seed: int = 42,
    patience: int = 15,
    workers: int = 2,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """Executes reproducible baseline YOLOv8 training."""
    logger = setup_logger("TrainBaseline", f"{project}/training.log")
    logger.info("Initializing YOLOv8 Baseline Training Pipeline...")

    # Hardware & Environment Logging
    hw_info = get_system_hardware_info()
    logger.info(f"System Configuration: {json.dumps(hw_info, indent=2)}")

    if device == "auto":
        target_device = get_best_device()
    else:
        target_device = device

    logger.info(f"Selected computation device: '{target_device}'")

    if target_device == "cpu":
        logger.warning(
            "CUDA is not available. Training will run on CPU. "
            "Batch size and workers will be optimized for host CPU stability."
        )
        # On Windows CPU, workers=0 or 2 prevents multiprocessing deadlocks
        workers = min(workers, 2)
        if batch_size > 16:
            batch_size = 16

    data_path = Path(data_yaml).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml does not exist at: {data_path}. Run src/prepare_dataset.py first.")

    project_dir = Path(project).resolve()
    run_dir = project_dir / run_name

    # Protect against accidental overwrite of completed baseline
    if run_dir.exists() and (run_dir / "weights" / "best.pt").exists():
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"{run_name}_{timestamp}"
        run_dir = project_dir / run_name
        logger.info(f"Previous completed run detected. Saving new run under: {run_name}")

    run_dir.mkdir(parents=True, exist_ok=True)

    # Ultralytics Import
    try:
        from ultralytics import YOLO
    except ImportError as e:
        logger.error(f"Failed to import ultralytics: {e}. Run 'pip install ultralytics'.")
        raise

    # Log training configuration
    train_config = {
        "model": model_name,
        "data": str(data_path),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch_size,
        "device": target_device,
        "seed": seed,
        "deterministic": deterministic,
        "patience": patience,
        "workers": workers,
        "project": str(project_dir),
        "name": run_name,
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "auto",
        "verbose": True,
    }

    with open(run_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=2)

    with open(run_dir / "system_info.json", "w", encoding="utf-8") as f:
        json.dump(hw_info, f, indent=2)

    logger.info(f"Starting model.train() with {model_name} for {epochs} epochs at imgsz={imgsz}...")
    start_time = time.time()

    try:
        model = YOLO(model_name)
        results = model.train(
            data=str(data_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=target_device,
            project=str(project_dir),
            name=run_name,
            seed=seed,
            deterministic=deterministic,
            patience=patience,
            workers=workers,
            exist_ok=True,
            save=True,
            plots=True,
        )
    except RuntimeError as re:
        if "out of memory" in str(re).lower():
            logger.critical("CUDA Out of Memory encountered! Recommendations:")
            logger.critical("1. Reduce batch size (e.g., --batch 8 or --batch 4).")
            logger.critical("2. Lower image resolution (e.g., --imgsz 320).")
        raise
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

    duration_sec = time.time() - start_time
    duration_min = duration_sec / 60.0
    logger.info(f"Training successfully completed in {duration_min:.2f} minutes ({duration_sec:.1f} s).")

    # Copy best weights to convenient top-level baseline location
    best_weights = run_dir / "weights" / "best.pt"
    top_best = project_dir / "best.pt"
    if best_weights.exists():
        shutil.copy2(best_weights, top_best)
        logger.info(f"Copied best model weights to: {top_best}")

    # Save summary metadata
    summary = {
        "status": "COMPLETED",
        "run_directory": str(run_dir),
        "best_weights": str(best_weights) if best_weights.exists() else None,
        "training_duration_sec": round(duration_sec, 2),
        "epochs_trained": epochs,
        "device_used": target_device,
    }
    with open(project_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 Baseline for Industrial Defect Detection")
    parser.add_argument("--data", type=str, default="datasets/defect_dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained weights or model yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image dimension")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('0', 'cpu', 'auto')")
    parser.add_argument("--project", type=str, default="outputs/baseline", help="Directory to save runs")
    parser.add_argument("--name", type=str, default="baseline_run", help="Run name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_baseline(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        project=args.project,
        run_name=args.name,
        seed=args.seed,
        workers=args.workers,
    )
