"""Run fast, non-training checks on the reconstructed project package."""

from __future__ import annotations

import compileall
import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd

from kline_dataset_generator import render_original_bxp


ROOT = Path(__file__).resolve().parent
REFERENCE_HASH = "9551df8b81bd0b63ff071884c446525942e20e7856c70c198e3c423aabe80487"
MODEL_HASHES = {
    "cluster_regime_package/historical_k4_models/pca.joblib": (
        "27b25808c16887bd7d8033d8589f37b3085cc5196cfb01c5af03b50db699966c"
    ),
    "cluster_regime_package/historical_k4_models/kmeans.joblib": (
        "f515044878713dae6061d68da41053cf69dc4bb7407046cdeab21fbfa1a5d216"
    ),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if not compileall.compile_dir(ROOT, quiet=1):
        raise RuntimeError("One or more Python files failed syntax compilation.")

    json_files = list(ROOT.rglob("*.json"))
    for path in json_files:
        json.loads(path.read_text(encoding="utf-8"))

    reference = ROOT / "reference" / "sample_0_29.jpg"
    if digest(reference) != REFERENCE_HASH:
        raise RuntimeError("The supplied reference JPEG has changed.")
    frame = pd.read_csv(ROOT / "reference" / "BTCUSDT_1h_2020-01-01_first30.csv")
    with tempfile.TemporaryDirectory() as directory:
        generated = Path(directory) / "sample_0_29.jpg"
        render_original_bxp(
            frame,
            {"open": "open", "high": "high", "low": "low", "close": "close"},
            generated,
            {"width_px": 360, "height_px": 360, "dpi": 90},
        )
        if generated.read_bytes() != reference.read_bytes():
            raise RuntimeError("The active renderer no longer matches sample_0_29.jpg.")

    for relative, expected in MODEL_HASHES.items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f"Historical model changed: {relative}")

    print(f"Python syntax: PASS")
    print(f"JSON configurations: PASS ({len(json_files)} files)")
    print("Reference renderer: PASS (byte-for-byte)")
    print("Historical K=4 model identities: PASS")


if __name__ == "__main__":
    main()
