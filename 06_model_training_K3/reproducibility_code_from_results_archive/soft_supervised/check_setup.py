from pathlib import Path

import torch

from embedding_models import EmbeddingMLP, SoftLogisticRegression

from shared import (
    load_json,
    load_soft_class_weights,
    load_soft_manifest,
    resolve_dataset_paths,
    verify_split_boundaries,
)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
config = load_json(PACKAGE_ROOT / "configs" / "training_config.json")
paths = resolve_dataset_paths(config)

print("Kaggle GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print()

required = {
    "dataset_dir": paths.dataset_dir,
    "soft_membership_dir": paths.soft_membership_dir,
    "train_manifest": paths.train_manifest,
    "validation_manifest": paths.validation_manifest,
    "test_manifest": paths.test_manifest,
    "weights_report": paths.weights_report,
}
failed = False
for name, path in required.items():
    exists = path.exists()
    failed |= not exists
    print(f"{name}: {exists} — {path}")
if failed:
    raise SystemExit("One or more required paths do not exist.")

train = load_soft_manifest(paths.train_manifest, "train")
validation = load_soft_manifest(paths.validation_manifest, "validation")
test = load_soft_manifest(paths.test_manifest, "test")
boundary_audit = verify_split_boundaries(train, validation, test)
weights = load_soft_class_weights(paths.weights_report)

soft_mass = train[list(f"target_soft_cluster_{i}" for i in range(3))].sum(axis=0).to_numpy(dtype=float)
expected_weights = len(train) / (3.0 * soft_mass)
if not torch.allclose(weights, torch.tensor(expected_weights, dtype=torch.float32), rtol=2e-4, atol=2e-5):
    raise ValueError(
        "The saved soft class weights do not match the retained training soft-membership mass. "
        f"Saved={weights.tolist()}, expected={expected_weights.tolist()}"
    )

if bool(config.get("verify_all_images", False)):
    missing = []
    for frame in (train, validation, test):
        for value in frame["input_image"]:
            image_path = Path(str(value))
            if not image_path.is_absolute():
                image_path = paths.dataset_dir / image_path
            if not image_path.is_file():
                missing.append(str(image_path))
                if len(missing) == 10:
                    break
        if len(missing) == 10:
            break
    if missing:
        raise FileNotFoundError(f"Missing images (first {len(missing)}): {missing}")

print(f"\nTraining rows: {len(train):,}")
print(f"Validation rows: {len(validation):,}")
print(f"Test rows held out: {len(test):,}")
for boundary in boundary_audit:
    print(
        f"Boundary {boundary['boundary']}: {boundary['left_target_end']} -> "
        f"{boundary['right_input_start']} | gap={boundary['gap']} | "
        f"shared_sources={boundary['shared_sources']} | "
        f"index_check_applied={boundary['index_check_applied']}"
    )
print("Soft class-mass weights:", [round(float(x), 5) for x in weights])
print("Training soft mass:", [round(float(x), 3) for x in soft_mass])
print("Imbalance audit: PASSED (weights are training-only inverse soft-mass weights)")
logistic = SoftLogisticRegression()
mlp = EmbeddingMLP(
    hidden_units=tuple(config.get("mlp_hidden_units", [256, 64])),
    dropout=float(config.get("mlp_dropout", 0.3)),
)
dummy = torch.randn(4, 512)
assert logistic(dummy).shape == (4, 3)
assert mlp(dummy).shape == (4, 3)
print("Three-class embedding model forward checks: PASSED")
print("\nSetup is ready.")
