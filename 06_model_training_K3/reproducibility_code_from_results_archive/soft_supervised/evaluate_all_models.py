from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay

from embedding_models import build_embedding_model
from shared import NUM_CLASSES, REGIME_NAMES, TARGET_KIND, load_json, load_soft_manifest, resolve_dataset_paths, resolve_device, save_json
from train_embedding_model import load_embeddings, make_loader, run_epoch

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
EMBEDDING_MODELS = ("soft_logistic_regression", "embedding_mlp")


def evaluate_embedding_model(name, config, device, test_frame):
    output_root = Path(config["output_dir"]).expanduser().resolve()
    output = output_root / name
    checkpoint = torch.load(output / "best_model.pt", map_location=device)
    if checkpoint.get("target_kind") != TARGET_KIND or checkpoint.get("num_classes") != NUM_CLASSES:
        raise ValueError(f"Incompatible checkpoint: {output / 'best_model.pt'}")
    if checkpoint.get("model_name") != name:
        raise ValueError(f"Checkpoint model name mismatch for {name}.")
    model = build_embedding_model(name, config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    with np.load(output_root / "frozen_resnet18_embeddings" / "training_fitted_standardizer.npz") as scaler:
        mean, scale = scaler["mean"].copy(), scaler["scale"].copy()
    embeddings = load_embeddings(output_root / "frozen_resnet18_embeddings" / "test_embeddings.npz")
    loader = make_loader(
        embeddings, mean, scale,
        int(config.get("embedding_training_batch_size", 256)), False,
        int(config.get("random_seed", 42)),
    )
    with torch.inference_mode():
        metrics, raw = run_epoch(model, loader, device, None)
    metrics["soft_cross_entropy"] = metrics.pop("loss")
    save_json(metrics, output / "independent_test_evaluation.json")
    rows = test_frame.iloc[raw["indices"]].copy().reset_index(drop=True)
    rows["actual_target_cluster_id"] = raw["hard_targets"]
    rows["predicted_target_cluster_id"] = raw["predictions"]
    for cluster in range(NUM_CLASSES):
        rows[f"predicted_probability_cluster_{cluster}"] = raw["probabilities"][:, cluster]
    rows["prediction_correct"] = rows["actual_target_cluster_id"] == rows["predicted_target_cluster_id"]
    rows.to_csv(output / "independent_test_predictions.csv", index=False)
    display = ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(metrics["confusion_matrix"], dtype=int),
        display_labels=[f"{i}: {REGIME_NAMES[i]}" for i in range(NUM_CLASSES)],
    )
    display.plot(values_format="d")
    plt.title(f"{name} independent test")
    plt.tight_layout()
    plt.savefig(output / "independent_test_confusion_matrix.png", dpi=180)
    plt.close()
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Deliberately repeat test evaluation.")
    args = parser.parse_args()
    config = load_json(PACKAGE_ROOT / "configs" / "training_config.json")
    output_root = Path(config["output_dir"]).expanduser().resolve()
    lock = output_root / "TEST_EVALUATION_COMPLETE.json"
    in_progress = output_root / "TEST_EVALUATION_IN_PROGRESS.json"
    if lock.is_file() and not args.force:
        raise SystemExit(
            f"Test evaluation has already been completed: {lock}\n"
            "Refusing to rerun. Use --force only when a documented rerun is intentional."
        )
    if in_progress.is_file() and not args.force:
        raise SystemExit(
            f"An earlier final-test evaluation started but did not finish: {in_progress}\n"
            "Inspect the existing outputs, then use --force only for a documented recovery."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    save_json({
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "selection_metric": "validation_macro_f1",
        "test_used_for_tuning": False,
        "status": "in_progress",
    }, in_progress)
    subprocess.run(
        [sys.executable, "-u", str(SCRIPT_DIR / "extract_embeddings.py"), "--include-test"],
        check=True,
    )
    paths = resolve_dataset_paths(config)
    test_frame = load_soft_manifest(paths.test_manifest, "test")
    device = resolve_device(config.get("device", "auto"))
    rows = []
    for name in EMBEDDING_MODELS:
        metrics = evaluate_embedding_model(name, config, device, test_frame)
        rows.append({
            "model": name,
            "test_macro_f1": metrics["macro_f1"],
            "test_balanced_accuracy": metrics["balanced_accuracy"],
            "test_accuracy": metrics["accuracy"],
            "test_soft_cross_entropy": metrics["soft_cross_entropy"],
            "test_soft_brier_score": metrics["soft_brier_score"],
        })
    subprocess.run([sys.executable, "-u", str(SCRIPT_DIR / "evaluate_checkpoint.py")], check=True)
    layer4_metrics = load_json(
        output_root / "soft_membership_unfreeze_layer4" / "independent_test_evaluation.json"
    )
    rows.append({
        "model": "resnet18_unfreeze_layer4",
        "test_macro_f1": layer4_metrics["macro_f1"],
        "test_balanced_accuracy": layer4_metrics["balanced_accuracy"],
        "test_accuracy": layer4_metrics["accuracy"],
        "test_soft_cross_entropy": layer4_metrics["soft_cross_entropy"],
        "test_soft_brier_score": layer4_metrics["soft_brier_score"],
    })
    comparison = pd.DataFrame(rows).sort_values("test_macro_f1", ascending=False)
    comparison.to_csv(output_root / "independent_test_model_comparison.csv", index=False)
    completed = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "models_evaluated": [row["model"] for row in rows],
        "selection_metric": "validation_macro_f1",
        "test_used_for_tuning": False,
        "forced_rerun": bool(args.force),
    }
    save_json(completed, lock)
    in_progress.unlink(missing_ok=True)
    print("\nINDEPENDENT TEST COMPARISON")
    print(comparison.to_string(index=False))
    print(f"\nEvaluation lock written: {lock}")


if __name__ == "__main__":
    main()
