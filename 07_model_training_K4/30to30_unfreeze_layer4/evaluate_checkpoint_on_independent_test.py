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
    FutureRegimeImageDataset,
    build_unfreeze_layer4_model,
    compute_metrics,
    load_json,
    load_manifest,
    resolve_dataset_paths,
    resolve_device,
    save_json,
)

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"

@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    targets, predictions, probabilities, indices = [], [], [], []
    for images, batch_targets, batch_indices in tqdm(loader, desc="Independent test evaluation", unit="batch"):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        targets.extend(batch_targets.tolist())
        predictions.extend(preds.cpu().tolist())
        probabilities.extend(probs.cpu().tolist())
        indices.extend(batch_indices.tolist())
    return compute_metrics(targets, predictions), {
        "targets": targets,
        "predictions": predictions,
        "probabilities": probabilities,
        "indices": indices,
    }


def main():
    config = load_json(CONFIG_PATH)
    device = resolve_device(config.get("device", "auto"))
    paths = resolve_dataset_paths(config)
    output_dir = Path(config["output_dir"]).expanduser().resolve() / "unfreeze_layer4"
    checkpoint_path = output_dir / "best_model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("training_mode") != "unfreeze_layer4":
        raise ValueError("The checkpoint is not an unfreeze_layer4 checkpoint.")

    model, transform = build_unfreeze_layer4_model(num_classes=4)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    test_frame = load_manifest(paths.test_manifest)
    dataset = FutureRegimeImageDataset(test_frame, paths.dataset_dir, transform)
    workers = int(config.get("num_workers", 2))
    kwargs = {
        "num_workers": workers,
        "pin_memory": bool(config.get("pin_memory", True)) and device.type == "cuda",
        "persistent_workers": bool(config.get("persistent_workers", True)) and workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = int(config.get("prefetch_factor", 2))
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("evaluation_batch_size", 32)),
        shuffle=False,
        **kwargs,
    )
    metrics, raw = evaluate(model, loader, device)
    save_json(metrics, output_dir / "independent_test_evaluation.json")
    results = test_frame.iloc[raw["indices"]].copy().reset_index(drop=True)
    results["actual_target_cluster_id"] = raw["targets"]
    results["predicted_target_cluster_id"] = raw["predictions"]
    probabilities = np.asarray(raw["probabilities"], dtype=float)
    for cluster in range(4):
        results[f"probability_cluster_{cluster}"] = probabilities[:, cluster]
    results["prediction_correct"] = results["actual_target_cluster_id"] == results["predicted_target_cluster_id"]
    results.to_csv(output_dir / "independent_test_predictions.csv", index=False)
    pd.DataFrame([
        {"cluster_id": int(cluster), **values}
        for cluster, values in metrics["per_class"].items()
    ]).to_csv(output_dir / "independent_test_classification_report.csv", index=False)
    display = ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(metrics["confusion_matrix"], dtype=int),
        display_labels=["0", "1", "2", "3"],
    )
    display.plot(values_format="d")
    plt.title("Independent test confusion matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "independent_test_confusion_matrix.png", dpi=180)
    plt.close()
    print("=" * 72)
    print("INDEPENDENT UNFREEZE_LAYER4 CHECKPOINT EVALUATION")
    print("=" * 72)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Outputs: {output_dir}")

if __name__ == "__main__":
    main()
