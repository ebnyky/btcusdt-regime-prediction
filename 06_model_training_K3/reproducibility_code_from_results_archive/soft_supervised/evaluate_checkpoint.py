from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from shared import (
    NUM_CLASSES,
    REGIME_NAMES,
    TARGET_KIND,
    SoftFutureRegimeImageDataset,
    build_unfreeze_layer4_model,
    compute_metrics,
    load_json,
    load_soft_manifest,
    resolve_dataset_paths,
    resolve_device,
    save_json,
    soft_brier_score,
    soft_weighted_cross_entropy,
)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PACKAGE_ROOT / "configs" / "training_config.json"


def loader_kwargs(config, device):
    workers = int(config.get("num_workers", 2))
    kwargs = {
        "num_workers": workers,
        "pin_memory": bool(config.get("pin_memory", True)) and device.type == "cuda",
        "persistent_workers": bool(config.get("persistent_workers", True)) and workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = int(config.get("prefetch_factor", 2))
    return kwargs


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    hard_targets, predictions, probabilities, soft_targets = [], [], [], []
    certainties, indices = [], []
    running_loss = 0.0
    for images, batch_soft, batch_certainty, batch_hard, batch_indices in tqdm(
        loader, desc="Independent test evaluation", unit="batch"
    ):
        images = images.to(device, non_blocking=True)
        batch_soft_device = batch_soft.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = soft_weighted_cross_entropy(logits, batch_soft_device)
        logits_float = logits.float()
        probs = torch.softmax(logits_float, dim=1)
        preds = logits_float.argmax(dim=1)
        running_loss += loss.item() * images.size(0)
        hard_targets.extend(batch_hard.tolist())
        predictions.extend(preds.cpu().tolist())
        probabilities.extend(probs.cpu().tolist())
        soft_targets.extend(batch_soft.tolist())
        certainties.extend(batch_certainty.tolist())
        indices.extend(batch_indices.tolist())

    metrics = compute_metrics(hard_targets, predictions)
    metrics["soft_cross_entropy"] = running_loss / len(loader.dataset)
    metrics["soft_brier_score"] = soft_brier_score(
        np.asarray(probabilities, dtype=float), np.asarray(soft_targets, dtype=float)
    )
    return metrics, {
        "hard_targets": hard_targets,
        "predictions": predictions,
        "probabilities": probabilities,
        "soft_targets": soft_targets,
        "certainty_weights": certainties,
        "indices": indices,
    }


def uncertainty_strata(raw):
    frame = pd.DataFrame({
        "certainty_weight": raw["certainty_weights"],
        "actual": raw["hard_targets"],
        "predicted": raw["predictions"],
    })
    frame["certainty_stratum"] = pd.cut(
        frame["certainty_weight"],
        bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
        labels=["low", "medium", "high"],
    )
    rows = []
    for name, group in frame.groupby("certainty_stratum", observed=False):
        if group.empty:
            rows.append({"certainty_stratum": str(name), "samples": 0})
            continue
        metrics = compute_metrics(group["actual"].tolist(), group["predicted"].tolist())
        rows.append({
            "certainty_stratum": str(name),
            "samples": len(group),
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
        })
    return pd.DataFrame(rows)


def main():
    config = load_json(CONFIG_PATH)
    device = resolve_device(config.get("device", "auto"))
    paths = resolve_dataset_paths(config)
    output_dir = Path(config["output_dir"]).expanduser().resolve() / "soft_membership_unfreeze_layer4"
    checkpoint_path = output_dir / "best_model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("training_mode") != "unfreeze_layer4":
        raise ValueError("The checkpoint is not an unfreeze_layer4 checkpoint.")
    if checkpoint.get("target_kind") != TARGET_KIND or checkpoint.get("num_classes") != NUM_CLASSES:
        raise ValueError("The checkpoint is not a K=3 soft-membership checkpoint.")

    model, transform = build_unfreeze_layer4_model(num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    test_frame = load_soft_manifest(paths.test_manifest, "test")
    dataset = SoftFutureRegimeImageDataset(test_frame, paths.dataset_dir, transform)
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("evaluation_batch_size", 32)),
        shuffle=False,
        **loader_kwargs(config, device),
    )
    metrics, raw = evaluate(model, loader, device)
    save_json(metrics, output_dir / "independent_test_evaluation.json")

    results = test_frame.iloc[raw["indices"]].copy().reset_index(drop=True)
    results["actual_target_cluster_id"] = raw["hard_targets"]
    results["predicted_target_cluster_id"] = raw["predictions"]
    probabilities = np.asarray(raw["probabilities"], dtype=float)
    for cluster in range(NUM_CLASSES):
        results[f"predicted_probability_cluster_{cluster}"] = probabilities[:, cluster]
    results["prediction_correct"] = (
        results["actual_target_cluster_id"] == results["predicted_target_cluster_id"]
    )
    results.to_csv(output_dir / "independent_test_predictions.csv", index=False)
    pd.DataFrame([
        {"cluster_id": int(cluster), **values}
        for cluster, values in metrics["per_class"].items()
    ]).to_csv(output_dir / "independent_test_classification_report.csv", index=False)
    uncertainty_strata(raw).to_csv(
        output_dir / "independent_test_uncertainty_strata.csv", index=False
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(metrics["confusion_matrix"], dtype=int),
        display_labels=[f"{i}: {REGIME_NAMES[i]}" for i in range(NUM_CLASSES)],
    )
    display.plot(values_format="d")
    plt.title("Independent test confusion matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "independent_test_confusion_matrix.png", dpi=180)
    plt.close()

    print("=" * 72)
    print("INDEPENDENT K=3 SOFT-MEMBERSHIP TEST EVALUATION")
    print("=" * 72)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Soft cross-entropy: {metrics['soft_cross_entropy']:.4f}")
    print(f"Soft Brier score: {metrics['soft_brier_score']:.4f}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
