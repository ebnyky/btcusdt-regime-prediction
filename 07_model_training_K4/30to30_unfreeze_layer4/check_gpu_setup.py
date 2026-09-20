from pathlib import Path
import json
import torch

ROOT = Path(__file__).resolve().parent
with (ROOT / "config.json").open("r", encoding="utf-8") as f:
    config = json.load(f)

paths = {
    "dataset_dir": Path(config["dataset_dir"]),
    "train_manifest": Path(config["dataset_dir"]) / "manifests" / "train_samples.csv",
    "validation_manifest": Path(config["dataset_dir"]) / "manifests" / "validation_samples.csv",
    "test_manifest": Path(config["dataset_dir"]) / "manifests" / "test_samples.csv",
    "pca_model": Path(config["pca_model"]),
    "kmeans_model": Path(config["kmeans_model"]),
}

print("Kaggle GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print()
failed = False
for name, path in paths.items():
    exists = path.exists()
    failed |= not exists
    print(f"{name}: {exists} — {path}")
if failed:
    raise SystemExit("One or more required paths do not exist.")
print("\nSetup is ready.")
