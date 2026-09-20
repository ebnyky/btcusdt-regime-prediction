from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent


def run(*arguments):
    command = [sys.executable, "-u", *map(str, arguments)]
    print("\n" + "=" * 80)
    print("RUNNING:", " ".join(command))
    print("=" * 80, flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    with (PACKAGE_ROOT / "configs" / "training_config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    output_root = Path(config["output_dir"]).expanduser().resolve()
    forbidden = [
        output_root / "TEST_EVALUATION_COMPLETE.json",
        output_root / "TEST_EVALUATION_IN_PROGRESS.json",
        output_root / "frozen_resnet18_embeddings" / "test_embeddings.npz",
    ]
    existing = [str(path) for path in forbidden if path.exists()]
    if existing:
        raise SystemExit(
            "Training is locked because final-test state already exists:\n- "
            + "\n- ".join(existing)
            + "\nUse a new output_dir for a fresh experiment; do not tune after viewing test results."
        )
    run(SCRIPT_DIR / "check_setup.py")
    run(SCRIPT_DIR / "extract_embeddings.py")
    run(SCRIPT_DIR / "train_embedding_model.py", "--model", "soft_logistic_regression")
    run(SCRIPT_DIR / "train_embedding_model.py", "--model", "embedding_mlp")
    run(SCRIPT_DIR / "train_classifier.py")
    run(SCRIPT_DIR / "compare_validation.py")
    print("\nAll training and validation selection is complete. The test set remains unevaluated.")
