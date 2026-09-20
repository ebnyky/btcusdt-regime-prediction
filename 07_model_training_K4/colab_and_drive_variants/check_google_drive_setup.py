from __future__ import annotations

from pathlib import Path

from shared import load_json, load_manifest, resolve_dataset_paths

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"


def main() -> None:
    config = load_json(CONFIG_PATH)
    paths = resolve_dataset_paths(config)

    print("=" * 72, flush=True)
    print("MYDRIVE DIRECT SETUP CHECK", flush=True)
    print("=" * 72, flush=True)
    print(f"Package directory: {SCRIPT_DIR}", flush=True)
    print(f"Dataset directory: {paths.dataset_dir}", flush=True)
    print(f"Output directory: {config['output_dir']}", flush=True)

    required = {
        "dataset directory": paths.dataset_dir,
        "train manifest": paths.train_manifest,
        "validation manifest": paths.validation_manifest,
        "test manifest": paths.test_manifest,
        "PCA model": Path(config["pca_model"]),
        "K-Means model": Path(config["kmeans_model"]),
    }

    missing = []
    for label, path in required.items():
        exists = path.is_dir() if label == "dataset directory" else path.is_file()
        print(f"{label}: {'OK' if exists else 'MISSING'} — {path}", flush=True)
        if not exists:
            missing.append((label, path))

    if missing:
        details = "\n".join(f"- {label}: {path}" for label, path in missing)
        raise FileNotFoundError(f"Required paths are missing:\n{details}")

    for split_name, manifest_path in (
        ("train", paths.train_manifest),
        ("validation", paths.validation_manifest),
        ("test", paths.test_manifest),
    ):
        frame = load_manifest(manifest_path)
        first_relative = str(frame.iloc[0]["input_image"])
        first_image = paths.dataset_dir / first_relative
        print(
            f"{split_name}: rows={len(frame):,}; first image "
            f"{'OK' if first_image.is_file() else 'MISSING'} — {first_image}",
            flush=True,
        )
        if not first_image.is_file():
            raise FileNotFoundError(first_image)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = output_dir / ".write_test"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    print("Output directory is writable.", flush=True)
    print("Setup check passed.", flush=True)


if __name__ == "__main__":
    main()
