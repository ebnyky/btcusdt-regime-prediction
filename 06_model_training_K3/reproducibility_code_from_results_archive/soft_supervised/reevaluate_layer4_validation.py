from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import torch

from shared import (
    NUM_CLASSES,
    TARGET_KIND,
    SoftFutureRegimeImageDataset,
    build_unfreeze_layer4_model,
    load_json,
    load_soft_manifest,
    resolve_dataset_paths,
    resolve_device,
    save_json,
)
from train_classifier import (
    evaluate_model,
    make_eval_loader,
    prediction_frame,
    save_confusion,
)


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PACKAGE_ROOT / "configs" / "training_config.json"


def preserve_once(source: Path, destination: Path) -> None:
    if source.is_file() and not destination.exists():
        shutil.copy2(source, destination)


def main() -> None:
    config = load_json(CONFIG_PATH)
    device = resolve_device(config.get("device", "auto"))
    paths = resolve_dataset_paths(config)
    output = Path(config["output_dir"]).expanduser().resolve() / "soft_membership_unfreeze_layer4"
    checkpoint_path = output / "best_model.pt"
    report_path = output / "training_report.json"
    predictions_path = output / "validation_predictions.csv"
    confusion_path = output / "validation_confusion_matrix.png"

    if not checkpoint_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("Layer-4 checkpoint or training report is missing.")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("training_mode") != "unfreeze_layer4":
        raise ValueError("Checkpoint training mode is not unfreeze_layer4.")
    if checkpoint.get("target_kind") != TARGET_KIND or checkpoint.get("num_classes") != NUM_CLASSES:
        raise ValueError("Checkpoint is not compatible with K=3 soft membership.")

    preserve_once(report_path, output / "training_report_mixed_precision_probabilities_original.json")
    preserve_once(predictions_path, output / "validation_predictions_mixed_precision_original.csv")
    preserve_once(confusion_path, output / "validation_confusion_matrix_original.png")

    validation_frame = load_soft_manifest(paths.validation_manifest, "validation")
    model, transform = build_unfreeze_layer4_model(num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    dataset = SoftFutureRegimeImageDataset(validation_frame, paths.dataset_dir, transform)
    loader = make_eval_loader(dataset, config, device)
    metrics, raw = evaluate_model(model, loader, device, "Float32-probability validation reevaluation")

    probabilities = np.asarray(raw["probabilities"], dtype=np.float64)
    row_sums = probabilities.sum(axis=1)
    if not np.isfinite(probabilities).all():
        raise ValueError("Recomputed validation probabilities contain non-finite values.")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Recomputed validation probabilities fall outside [0, 1].")
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError(
            "Float32 probability normalization failed; maximum deviation="
            f"{float(np.abs(row_sums - 1.0).max())}"
        )

    report = load_json(report_path)
    report["validation"] = metrics
    report["validation_probability_dtype"] = "float32"
    report["validation_recomputed_from_best_checkpoint"] = True
    report["validation_probability_max_abs_sum_deviation"] = float(
        np.abs(row_sums - 1.0).max()
    )
    report["test_evaluated"] = False
    save_json(report, report_path)
    prediction_frame(validation_frame, raw).to_csv(predictions_path, index=False)
    save_confusion(metrics, confusion_path, "Best-checkpoint validation confusion matrix")

    print("FLOAT32 VALIDATION REEVALUATION COMPLETE")
    print(f"Rows: {len(validation_frame):,}")
    print(f"Best epoch: {checkpoint['best_epoch']}")
    print(f"Macro-F1: {metrics['macro_f1']:.10f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.10f}")
    print(f"Soft Brier score: {metrics['soft_brier_score']:.10f}")
    print(f"Maximum probability-sum deviation: {float(np.abs(row_sums - 1.0).max()):.12g}")
    print("Test evaluated: False")
    print(f"Outputs: {output}")


if __name__ == "__main__":
    main()
