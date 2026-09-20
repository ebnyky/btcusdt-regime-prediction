
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import joblib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image, ImageOps, ImageDraw
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import (
        adjusted_rand_score,
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )
    from torch import Tensor, nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
except Exception:
    print(traceback.format_exc(), flush=True)
    print(
        "\nInstall the required packages with:\n"
        "    python -m pip install -r requirements.txt\n",
        flush=True,
    )
    raise SystemExit(1)


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"


@dataclass
class Paths:
    dataset_dir: Path
    output_dir: Path
    embeddings_dir: Path
    models_dir: Path
    reports_dir: Path
    plots_dir: Path
    clusters_dir: Path


class ManifestImageDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_column: str,
        transform: Any,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_column = image_column
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image_path = Path(str(self.frame.iloc[index][self.image_column]))

        if not image_path.is_file():
            raise FileNotFoundError(f"Image does not exist: {image_path}")

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                tensor = self.transform(image)
        except Exception as exc:
            raise RuntimeError(f"Could not load image: {image_path}") from exc

        return tensor, index


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Missing configuration file: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)

    return config


def resolve_config_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = CONFIG_PATH.parent / path
    return path.resolve()


def prepare_paths(config: dict[str, Any]) -> Paths:
    dataset_dir = resolve_config_path(config["dataset_dir"])
    output_dir = resolve_config_path(config["output_dir"])

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    if output_dir.exists() and any(output_dir.iterdir()):
        if not config.get("reuse_saved_embeddings", True):
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}\n"
                "Choose a new output_dir or set reuse_saved_embeddings=true."
            )

    paths = Paths(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        embeddings_dir=output_dir / "embeddings",
        models_dir=output_dir / "models",
        reports_dir=output_dir / "reports",
        plots_dir=output_dir / "plots",
        clusters_dir=output_dir / "clusters",
    )

    for folder in (
        paths.output_dir,
        paths.embeddings_dir,
        paths.models_dir,
        paths.reports_dir,
        paths.plots_dir,
        paths.clusters_dir,
    ):
        folder.mkdir(parents=True, exist_ok=True)

    return paths


def set_reproducibility(seed: int, cpu_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)

    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def resolve_device(value: str) -> torch.device:
    value = value.lower().strip()

    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(value)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    return device


def read_manifest(
    dataset_dir: Path,
    relative_path: str,
    image_column: str,
) -> pd.DataFrame:
    manifest_path = dataset_dir / relative_path

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    frame = pd.read_csv(manifest_path)

    required = {
        "sample_name",
        "start_index",
        "stop_index",
        image_column,
    }
    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"Manifest {manifest_path} is missing columns: {sorted(missing)}"
        )

    if frame.empty:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    frame = frame.sort_values(
        ["start_index", "stop_index"]
    ).reset_index(drop=True)

    def absolute_image_path(value: Any) -> str:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = dataset_dir / path
        return str(path.resolve())

    frame[image_column] = frame[image_column].map(absolute_image_path)

    return frame


def build_encoder(
    model_name: str,
    pretrained: bool,
    device: torch.device,
) -> tuple[nn.Module, Any, int]:
    model_name = model_name.lower().strip()

    if model_name != "resnet18":
        raise ValueError(
            "This package currently supports embedding_model='resnet18'."
        )

    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        transform = weights.transforms()
    else:
        model = models.resnet18(weights=None)
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    embedding_dim = int(model.fc.in_features)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model, transform, embedding_dim


def metadata_signature(frame: pd.DataFrame, image_column: str) -> list[str]:
    return [
        f"{row.sample_name}|{row.start_index}|{row.stop_index}|{getattr(row, image_column)}"
        for row in frame.itertuples(index=False)
    ]


def save_metadata(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def saved_embeddings_are_compatible(
    embedding_path: Path,
    metadata_path: Path,
    frame: pd.DataFrame,
    image_column: str,
) -> bool:
    if not embedding_path.is_file() or not metadata_path.is_file():
        return False

    try:
        saved_frame = pd.read_csv(metadata_path)
        embeddings = np.load(embedding_path, mmap_mode="r")
    except Exception:
        return False

    if embeddings.shape[0] != len(frame):
        return False

    needed = ["sample_name", "start_index", "stop_index", image_column]
    if any(column not in saved_frame.columns for column in needed):
        return False

    left = saved_frame[needed].astype(str).to_numpy()
    right = frame[needed].astype(str).to_numpy()

    return left.shape == right.shape and np.array_equal(left, right)


def extract_embeddings(
    name: str,
    frame: pd.DataFrame,
    image_column: str,
    model: nn.Module,
    transform: Any,
    embedding_dim: int,
    device: torch.device,
    batch_size: int,
    workers: int,
    embedding_path: Path,
    metadata_path: Path,
    reuse: bool,
) -> np.ndarray:
    if reuse and saved_embeddings_are_compatible(
        embedding_path,
        metadata_path,
        frame,
        image_column,
    ):
        print(f"Reusing saved {name} embeddings: {embedding_path}", flush=True)
        return np.load(embedding_path)

    dataset = ManifestImageDataset(frame, image_column, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
    )

    embeddings = np.empty(
        (len(dataset), embedding_dim),
        dtype=np.float32,
    )

    seen = 0

    print(
        f"\nExtracting {name} embeddings from {len(dataset):,} images...",
        flush=True,
    )

    with torch.inference_mode():
        for batch_number, (images, indices) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            output = model(images)

            if output.ndim != 2 or output.shape[1] != embedding_dim:
                raise ValueError(
                    f"Unexpected embedding shape: {tuple(output.shape)}"
                )

            output_np = output.detach().cpu().numpy().astype(
                np.float32,
                copy=False,
            )
            indices_np = indices.numpy()
            embeddings[indices_np] = output_np
            seen += len(indices_np)

            if batch_number % 25 == 0 or seen == len(dataset):
                print(
                    f"  {name}: {seen:,}/{len(dataset):,}",
                    flush=True,
                )

    if not np.isfinite(embeddings).all():
        raise ValueError(f"{name} embeddings contain NaN or Inf.")

    np.save(embedding_path, embeddings)
    save_metadata(frame, metadata_path)

    return embeddings


def choose_pca_components(
    configured: int | float,
    n_samples: int,
    n_features: int,
) -> int | float:
    if isinstance(configured, float) and 0.0 < configured < 1.0:
        return configured

    components = int(configured)
    return max(2, min(components, n_samples - 1, n_features))


def fit_pca(
    train_embeddings: np.ndarray,
    validation_embeddings: np.ndarray,
    configured_components: int | float,
    seed: int,
    model_path: Path,
) -> tuple[PCA, np.ndarray, np.ndarray]:
    n_components = choose_pca_components(
        configured_components,
        train_embeddings.shape[0],
        train_embeddings.shape[1],
    )

    print(
        f"\nFitting PCA on training embeddings only "
        f"(n_components={n_components})...",
        flush=True,
    )

    pca = PCA(
        n_components=n_components,
        random_state=seed,
        svd_solver="auto",
    )
    train_reduced = pca.fit_transform(train_embeddings)
    validation_reduced = pca.transform(validation_embeddings)

    joblib.dump(pca, model_path)

    explained = float(np.sum(pca.explained_variance_ratio_))
    print(
        f"PCA complete. Explained variance retained: {explained:.4%}",
        flush=True,
    )

    return pca, train_reduced.astype(np.float32), validation_reduced.astype(np.float32)


def sampled_indices(
    n_samples: int,
    maximum: int,
    seed: int,
) -> np.ndarray:
    if maximum <= 0 or n_samples <= maximum:
        return np.arange(n_samples)

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_samples, size=maximum, replace=False))


def cluster_balance_metrics(labels: np.ndarray, k: int) -> dict[str, Any]:
    counts = np.bincount(labels, minlength=k).astype(np.int64)
    proportions = counts / counts.sum()

    entropy = -np.sum(
        proportions[proportions > 0] * np.log(proportions[proportions > 0])
    )
    normalized_entropy = float(entropy / math.log(k)) if k > 1 else 0.0

    return {
        "cluster_sizes": counts.tolist(),
        "minimum_cluster_size": int(counts.min()),
        "maximum_cluster_size": int(counts.max()),
        "minimum_cluster_proportion": float(proportions.min()),
        "maximum_cluster_proportion": float(proportions.max()),
        "normalized_cluster_entropy": normalized_entropy,
    }


def clustering_stability(
    features: np.ndarray,
    k: int,
    repeats: int,
    subsample_limit: int,
    n_init: int,
    seed: int,
) -> tuple[float, list[float]]:
    indices = sampled_indices(
        len(features),
        subsample_limit,
        seed + 1000 + k,
    )
    sample = features[indices]

    label_sets: list[np.ndarray] = []

    for repeat in range(repeats):
        model = KMeans(
            n_clusters=k,
            n_init=n_init,
            random_state=seed + repeat,
        )
        label_sets.append(model.fit_predict(sample))

    pair_scores: list[float] = []

    for left in range(len(label_sets)):
        for right in range(left + 1, len(label_sets)):
            pair_scores.append(
                float(adjusted_rand_score(label_sets[left], label_sets[right]))
            )

    if not pair_scores:
        return 1.0, [1.0]

    return float(np.mean(pair_scores)), pair_scores


def evaluate_candidate_k(
    features: np.ndarray,
    candidate_k: list[int],
    n_init: int,
    stability_repeats: int,
    stability_subsample: int,
    silhouette_subsample: int,
    seed: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    silhouette_indices = sampled_indices(
        len(features),
        silhouette_subsample,
        seed,
    )

    for k in candidate_k:
        if k < 2:
            print(f"Skipping invalid K={k}", flush=True)
            continue

        if k >= len(features):
            print(f"Skipping K={k}; not enough training samples.", flush=True)
            continue

        print(f"\nEvaluating K={k}...", flush=True)

        model = KMeans(
            n_clusters=k,
            n_init=n_init,
            random_state=seed,
        )
        labels = model.fit_predict(features)

        sample_features = features[silhouette_indices]
        sample_labels = labels[silhouette_indices]

        if len(np.unique(sample_labels)) < 2:
            silhouette = float("nan")
        else:
            silhouette = float(
                silhouette_score(sample_features, sample_labels)
            )

        db = float(davies_bouldin_score(features, labels))
        ch = float(calinski_harabasz_score(features, labels))

        stability_mean, stability_pairs = clustering_stability(
            features,
            k,
            stability_repeats,
            stability_subsample,
            n_init,
            seed,
        )

        balance = cluster_balance_metrics(labels, k)

        result = {
            "k": int(k),
            "silhouette_score": silhouette,
            "davies_bouldin_score": db,
            "calinski_harabasz_score": ch,
            "inertia": float(model.inertia_),
            "stability_ari_mean": stability_mean,
            "stability_ari_pair_scores": stability_pairs,
            **balance,
        }
        results.append(result)

        print(
            f"  silhouette={silhouette:.6f} | "
            f"DB={db:.6f} | CH={ch:.2f} | "
            f"stability={stability_mean:.6f} | "
            f"min_cluster={balance['minimum_cluster_size']:,}",
            flush=True,
        )

    if not results:
        raise RuntimeError("No valid K values were evaluated.")

    return results


def minmax(values: np.ndarray, reverse: bool = False) -> np.ndarray:
    finite = np.isfinite(values)

    if not finite.any():
        return np.zeros_like(values, dtype=float)

    output = np.zeros_like(values, dtype=float)
    valid = values[finite]
    low = valid.min()
    high = valid.max()

    if math.isclose(float(low), float(high)):
        output[finite] = 1.0
    else:
        output[finite] = (valid - low) / (high - low)

    if reverse:
        output[finite] = 1.0 - output[finite]

    return output


def recommend_k(results: list[dict[str, Any]]) -> tuple[int, pd.DataFrame]:
    frame = pd.DataFrame(
        [
            {
                "k": row["k"],
                "silhouette_score": row["silhouette_score"],
                "davies_bouldin_score": row["davies_bouldin_score"],
                "calinski_harabasz_score": row["calinski_harabasz_score"],
                "inertia": row["inertia"],
                "stability_ari_mean": row["stability_ari_mean"],
                "minimum_cluster_proportion": row["minimum_cluster_proportion"],
                "normalized_cluster_entropy": row["normalized_cluster_entropy"],
            }
            for row in results
        ]
    )

    frame["score_silhouette"] = minmax(
        frame["silhouette_score"].to_numpy(float)
    )
    frame["score_davies_bouldin"] = minmax(
        frame["davies_bouldin_score"].to_numpy(float),
        reverse=True,
    )
    frame["score_calinski_harabasz"] = minmax(
        np.log1p(frame["calinski_harabasz_score"].to_numpy(float))
    )
    frame["score_stability"] = minmax(
        frame["stability_ari_mean"].to_numpy(float)
    )
    frame["score_min_cluster"] = minmax(
        frame["minimum_cluster_proportion"].to_numpy(float)
    )
    frame["score_balance"] = minmax(
        frame["normalized_cluster_entropy"].to_numpy(float)
    )

    weights = {
        "score_silhouette": 0.25,
        "score_davies_bouldin": 0.15,
        "score_calinski_harabasz": 0.10,
        "score_stability": 0.25,
        "score_min_cluster": 0.15,
        "score_balance": 0.10,
    }

    frame["recommendation_score"] = 0.0
    for column, weight in weights.items():
        frame["recommendation_score"] += frame[column] * weight

    frame = frame.sort_values(
        ["recommendation_score", "k"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return int(frame.iloc[0]["k"]), frame


def save_candidate_reports(
    results: list[dict[str, Any]],
    scoring_frame: pd.DataFrame,
    reports_dir: Path,
) -> None:
    with (reports_dir / "candidate_k_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=2)

    rows = []
    for row in results:
        clean = dict(row)
        clean["cluster_sizes"] = json.dumps(clean["cluster_sizes"])
        clean["stability_ari_pair_scores"] = json.dumps(
            clean["stability_ari_pair_scores"]
        )
        rows.append(clean)

    pd.DataFrame(rows).to_csv(
        reports_dir / "candidate_k_metrics.csv",
        index=False,
    )
    scoring_frame.to_csv(
        reports_dir / "candidate_k_recommendation_scores.csv",
        index=False,
    )


def plot_metric(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    ylabel: str,
    destination: Path,
) -> None:
    fig = plt.figure(figsize=(8, 5))
    plt.plot(frame[x], frame[y], marker="o")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(destination, dpi=160)
    plt.close(fig)


def save_k_plots(
    results: list[dict[str, Any]],
    scoring_frame: pd.DataFrame,
    plots_dir: Path,
) -> None:
    frame = pd.DataFrame(results)

    plot_metric(
        frame,
        "k",
        "silhouette_score",
        "Silhouette Score by K",
        "Silhouette score (higher is better)",
        plots_dir / "k_selection_silhouette.png",
    )
    plot_metric(
        frame,
        "k",
        "davies_bouldin_score",
        "Davies-Bouldin Index by K",
        "Davies-Bouldin index (lower is better)",
        plots_dir / "k_selection_davies_bouldin.png",
    )
    plot_metric(
        frame,
        "k",
        "calinski_harabasz_score",
        "Calinski-Harabasz Score by K",
        "Calinski-Harabasz score (higher is better)",
        plots_dir / "k_selection_calinski_harabasz.png",
    )
    plot_metric(
        frame,
        "k",
        "inertia",
        "K-Means Inertia by K",
        "Inertia (elbow method)",
        plots_dir / "k_selection_inertia.png",
    )
    plot_metric(
        frame,
        "k",
        "stability_ari_mean",
        "Cluster Stability by K",
        "Mean adjusted Rand index",
        plots_dir / "k_selection_stability.png",
    )

    ordered = scoring_frame.sort_values("k")
    plot_metric(
        ordered,
        "k",
        "recommendation_score",
        "Combined Recommendation Score",
        "Recommendation score",
        plots_dir / "k_selection_recommendation.png",
    )


def fit_final_model(
    train_features: np.ndarray,
    validation_features: np.ndarray,
    selected_k: int,
    n_init: int,
    seed: int,
    model_path: Path,
) -> tuple[KMeans, np.ndarray, np.ndarray]:
    print(f"\nFitting final K-Means model with K={selected_k}...", flush=True)

    model = KMeans(
        n_clusters=selected_k,
        n_init=n_init,
        random_state=seed,
    )
    train_labels = model.fit_predict(train_features)
    validation_labels = model.predict(validation_features)

    joblib.dump(model, model_path)

    return model, train_labels, validation_labels


def add_labels_and_distances(
    frame: pd.DataFrame,
    features: np.ndarray,
    labels: np.ndarray,
    model: KMeans,
) -> pd.DataFrame:
    result = frame.copy()
    distances = np.linalg.norm(
        features - model.cluster_centers_[labels],
        axis=1,
    )
    result["cluster_id"] = labels.astype(int)
    result["distance_to_centroid"] = distances.astype(float)
    return result


def safe_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def labelled_thumbnail(
    image_path: Path,
    label: str,
    tile_size: int = 256,
) -> Image.Image:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail((tile_size, tile_size - 28))
        canvas = Image.new("RGB", (tile_size, tile_size), "white")
        x = (tile_size - image.width) // 2
        y = 4 + (tile_size - 28 - image.height) // 2
        canvas.paste(image, (x, y))

    draw = ImageDraw.Draw(canvas)
    draw.text((6, tile_size - 22), label, fill="black")
    return canvas


def create_montage(
    rows: pd.DataFrame,
    image_column: str,
    destination: Path,
    columns: int,
    title_prefix: str,
) -> None:
    if rows.empty:
        return

    tile_size = 256
    count = len(rows)
    rows_count = math.ceil(count / columns)
    montage = Image.new(
        "RGB",
        (columns * tile_size, rows_count * tile_size),
        "white",
    )

    for number, (_, row) in enumerate(rows.iterrows()):
        image_path = Path(str(row[image_column]))
        label = (
            f"{title_prefix} {number + 1} | "
            f"{row['sample_name']} | "
            f"d={row['distance_to_centroid']:.3f}"
        )
        tile = labelled_thumbnail(image_path, label, tile_size)
        x = (number % columns) * tile_size
        y = (number // columns) * tile_size
        montage.paste(tile, (x, y))

    montage.save(destination)


def save_cluster_inspection(
    labelled_train: pd.DataFrame,
    image_column: str,
    selected_k: int,
    nearest_count: int,
    random_count: int,
    columns: int,
    seed: int,
    clusters_dir: Path,
) -> pd.DataFrame:
    if clusters_dir.exists():
        for child in clusters_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    clusters_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []

    for cluster_id in range(selected_k):
        cluster = labelled_train[
            labelled_train["cluster_id"] == cluster_id
        ].sort_values("distance_to_centroid")

        cluster_dir = clusters_dir / f"cluster_{cluster_id:02d}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        if cluster.empty:
            summary_rows.append(
                {
                    "cluster_id": cluster_id,
                    "training_size": 0,
                    "representative_sample": None,
                    "mean_distance_to_centroid": None,
                    "maximum_distance_to_centroid": None,
                }
            )
            continue

        representative = cluster.iloc[0]
        safe_copy(
            Path(str(representative[image_column])),
            cluster_dir / "representative.jpg",
        )

        nearest = cluster.head(nearest_count).copy()
        random_rows = cluster.sample(
            n=min(random_count, len(cluster)),
            random_state=seed + cluster_id,
        ).sort_values("start_index")

        nearest.to_csv(cluster_dir / "nearest_members.csv", index=False)
        random_rows.to_csv(cluster_dir / "random_members.csv", index=False)

        for number, (_, row) in enumerate(nearest.iterrows(), start=1):
            safe_copy(
                Path(str(row[image_column])),
                cluster_dir / f"nearest_{number:02d}_{row['sample_name']}.jpg",
            )

        for number, (_, row) in enumerate(random_rows.iterrows(), start=1):
            safe_copy(
                Path(str(row[image_column])),
                cluster_dir / f"random_{number:02d}_{row['sample_name']}.jpg",
            )

        create_montage(
            nearest,
            image_column,
            cluster_dir / "montage_nearest.png",
            columns,
            "nearest",
        )
        create_montage(
            random_rows,
            image_column,
            cluster_dir / "montage_random.png",
            columns,
            "random",
        )

        summary_rows.append(
            {
                "cluster_id": cluster_id,
                "training_size": int(len(cluster)),
                "representative_sample": str(representative["sample_name"]),
                "representative_start_index": int(
                    representative["start_index"]
                ),
                "representative_stop_index": int(
                    representative["stop_index"]
                ),
                "mean_distance_to_centroid": float(
                    cluster["distance_to_centroid"].mean()
                ),
                "maximum_distance_to_centroid": float(
                    cluster["distance_to_centroid"].max()
                ),
            }
        )

    return pd.DataFrame(summary_rows)


def save_assignment_reports(
    train_labelled: pd.DataFrame,
    validation_labelled: pd.DataFrame,
    selected_k: int,
    reports_dir: Path,
) -> pd.DataFrame:
    train_labelled.to_csv(
        reports_dir / "train_cluster_labels.csv",
        index=False,
    )
    validation_labelled.to_csv(
        reports_dir / "validation_cluster_labels.csv",
        index=False,
    )

    train_counts = train_labelled["cluster_id"].value_counts().sort_index()
    validation_counts = (
        validation_labelled["cluster_id"].value_counts().sort_index()
    )

    summary = pd.DataFrame({"cluster_id": range(selected_k)})
    summary["training_size"] = summary["cluster_id"].map(
        train_counts
    ).fillna(0).astype(int)
    summary["validation_size"] = summary["cluster_id"].map(
        validation_counts
    ).fillna(0).astype(int)
    summary["training_proportion"] = (
        summary["training_size"] / len(train_labelled)
    )
    summary["validation_proportion"] = (
        summary["validation_size"] / len(validation_labelled)
    )

    return summary


def main() -> None:
    config = load_config()
    paths = prepare_paths(config)

    seed = int(config.get("random_seed", 42))
    set_reproducibility(
        seed,
        int(config.get("cpu_threads", 4)),
    )

    device = resolve_device(str(config.get("device", "auto")))
    image_column = str(config.get("image_column", "exported_image"))

    print("=" * 80, flush=True)
    print("CANDLESTICK VISUAL-REGIME DISCOVERY", flush=True)
    print("=" * 80, flush=True)
    print(f"Dataset: {paths.dataset_dir}", flush=True)
    print(f"Output:  {paths.output_dir}", flush=True)
    print(f"Device:  {device}", flush=True)

    train_frame = read_manifest(
        paths.dataset_dir,
        str(config["train_manifest"]),
        image_column,
    )
    validation_frame = read_manifest(
        paths.dataset_dir,
        str(config["validation_manifest"]),
        image_column,
    )

    print(f"Training images:   {len(train_frame):,}", flush=True)
    print(f"Validation images: {len(validation_frame):,}", flush=True)

    model, transform, embedding_dim = build_encoder(
        str(config.get("embedding_model", "resnet18")),
        bool(config.get("use_pretrained_weights", True)),
        device,
    )

    train_embeddings = extract_embeddings(
        "training",
        train_frame,
        image_column,
        model,
        transform,
        embedding_dim,
        device,
        int(config.get("batch_size", 16)),
        int(config.get("num_workers", 0)),
        paths.embeddings_dir / "train_embeddings.npy",
        paths.embeddings_dir / "train_metadata.csv",
        bool(config.get("reuse_saved_embeddings", True)),
    )

    validation_embeddings = extract_embeddings(
        "validation",
        validation_frame,
        image_column,
        model,
        transform,
        embedding_dim,
        device,
        int(config.get("batch_size", 16)),
        int(config.get("num_workers", 0)),
        paths.embeddings_dir / "validation_embeddings.npy",
        paths.embeddings_dir / "validation_metadata.csv",
        bool(config.get("reuse_saved_embeddings", True)),
    )

    pca, train_reduced, validation_reduced = fit_pca(
        train_embeddings,
        validation_embeddings,
        config.get("pca_components", 50),
        seed,
        paths.models_dir / "pca.joblib",
    )

    np.save(
        paths.embeddings_dir / "train_pca_features.npy",
        train_reduced,
    )
    np.save(
        paths.embeddings_dir / "validation_pca_features.npy",
        validation_reduced,
    )

    candidate_k = sorted(
        {
            int(value)
            for value in config.get(
                "candidate_k",
                [4, 6, 8, 10, 12],
            )
        }
    )

    candidate_results = evaluate_candidate_k(
        train_reduced,
        candidate_k,
        int(config.get("kmeans_n_init", 20)),
        int(config.get("stability_repeats", 5)),
        int(config.get("stability_subsample", 4000)),
        int(config.get("silhouette_subsample", 5000)),
        seed,
    )

    recommended_k, scoring_frame = recommend_k(candidate_results)
    configured_selected = config.get("selected_k")
    selected_k = (
        int(configured_selected)
        if configured_selected is not None
        else recommended_k
    )

    valid_candidate_values = {row["k"] for row in candidate_results}
    if selected_k not in valid_candidate_values:
        raise ValueError(
            f"selected_k={selected_k} was not evaluated. "
            f"Available values: {sorted(valid_candidate_values)}"
        )

    save_candidate_reports(
        candidate_results,
        scoring_frame,
        paths.reports_dir,
    )
    save_k_plots(
        candidate_results,
        scoring_frame,
        paths.plots_dir,
    )

    with (paths.reports_dir / "selected_k.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "recommended_k": recommended_k,
                "selected_k": selected_k,
                "selection_source": (
                    "config"
                    if configured_selected is not None
                    else "automatic_recommendation"
                ),
                "warning": (
                    "Inspect cluster montages before interpreting clusters "
                    "as market regimes."
                ),
            },
            file,
            indent=2,
        )

    kmeans, train_labels, validation_labels = fit_final_model(
        train_reduced,
        validation_reduced,
        selected_k,
        int(config.get("kmeans_n_init", 20)),
        seed,
        paths.models_dir / "kmeans.joblib",
    )

    train_labelled = add_labels_and_distances(
        train_frame,
        train_reduced,
        train_labels,
        kmeans,
    )
    validation_labelled = add_labels_and_distances(
        validation_frame,
        validation_reduced,
        validation_labels,
        kmeans,
    )

    assignment_summary = save_assignment_reports(
        train_labelled,
        validation_labelled,
        selected_k,
        paths.reports_dir,
    )

    inspection_summary = save_cluster_inspection(
        train_labelled,
        image_column,
        selected_k,
        int(config.get("nearest_images_per_cluster", 16)),
        int(config.get("random_images_per_cluster", 16)),
        int(config.get("montage_columns", 4)),
        seed,
        paths.clusters_dir,
    )

    cluster_summary = assignment_summary.merge(
        inspection_summary,
        on=["cluster_id", "training_size"],
        how="left",
    )
    cluster_summary.to_csv(
        paths.reports_dir / "cluster_summary.csv",
        index=False,
    )

    run_summary = {
        "dataset_dir": str(paths.dataset_dir),
        "output_dir": str(paths.output_dir),
        "device": str(device),
        "embedding_model": str(
            config.get("embedding_model", "resnet18")
        ),
        "pretrained_weights": bool(
            config.get("use_pretrained_weights", True)
        ),
        "embedding_dimension": int(embedding_dim),
        "training_samples": int(len(train_frame)),
        "validation_samples": int(len(validation_frame)),
        "pca_components_fitted": int(pca.n_components_),
        "pca_explained_variance": float(
            np.sum(pca.explained_variance_ratio_)
        ),
        "candidate_k": candidate_k,
        "recommended_k": int(recommended_k),
        "selected_k": int(selected_k),
        "output_files": {
            "pca_model": str(paths.models_dir / "pca.joblib"),
            "kmeans_model": str(paths.models_dir / "kmeans.joblib"),
            "train_labels": str(
                paths.reports_dir / "train_cluster_labels.csv"
            ),
            "validation_labels": str(
                paths.reports_dir / "validation_cluster_labels.csv"
            ),
            "cluster_summary": str(
                paths.reports_dir / "cluster_summary.csv"
            ),
        },
    }

    with (paths.reports_dir / "run_summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(run_summary, file, indent=2)

    print("\n" + "=" * 80, flush=True)
    print("REGIME DISCOVERY COMPLETE", flush=True)
    print("=" * 80, flush=True)
    print(f"Recommended K: {recommended_k}", flush=True)
    print(f"Selected K:    {selected_k}", flush=True)
    print(
        f"Cluster montages:\n  {paths.clusters_dir}",
        flush=True,
    )
    print(
        f"Reports:\n  {paths.reports_dir}",
        flush=True,
    )
    print(
        "\nInspect montage_nearest.png and montage_random.png in every "
        "cluster folder before accepting the regime interpretation.",
        flush=True,
    )


def pause(config: dict[str, Any] | None) -> None:
    if not config or not config.get("pause_on_exit", True):
        return

    try:
        input("\nPress Enter to close this window...")
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    arguments = parser.parse_args()
    CONFIG_PATH = arguments.config.expanduser().resolve()
    loaded_config: dict[str, Any] | None = None
    try:
        loaded_config = load_config()
        main()
    except Exception:
        print("\nREGIME DISCOVERY FAILED\n", flush=True)
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        pause(loaded_config)
