# predict_new_image.py


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models


REGIME_NAMES = {
    0: "High-volatility consolidation",
    1: "Bullish trend and breakout",
    2: "Volatile sideways and reversal",
    3: "Bearish trend and breakdown",
}


def load_pipeline(
    pca_path: Path,
    kmeans_path: Path,
    device: torch.device,
) -> tuple[nn.Module, Any, Any, Any]:
    if not pca_path.is_file():
        raise FileNotFoundError(f"PCA model not found: {pca_path}")

    if not kmeans_path.is_file():
        raise FileNotFoundError(
            f"K-Means model not found: {kmeans_path}"
        )

    weights = models.ResNet18_Weights.DEFAULT
    transform = weights.transforms()

    encoder = models.resnet18(weights=weights)
    encoder.fc = nn.Identity()
    encoder.eval()
    encoder.to(device)

    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    pca = joblib.load(pca_path)
    kmeans = joblib.load(kmeans_path)

    if int(pca.n_features_in_) != 512:
        raise ValueError(
            "The PCA model does not expect a 512-dimensional "
            f"embedding. It expects {pca.n_features_in_}."
        )

    if int(kmeans.n_clusters) != 4:
        raise ValueError(
            f"Expected four clusters, found {kmeans.n_clusters}."
        )

    return encoder, transform, pca, kmeans


def predict_cluster(
    image_path: Path,
    encoder: nn.Module,
    transform: Any,
    pca: Any,
    kmeans: Any,
    device: torch.device,
) -> dict[str, Any]:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = transform(image).unsqueeze(0)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load image: {image_path}"
        ) from exc

    tensor = tensor.to(device)

    with torch.inference_mode():
        embedding = encoder(tensor)

    embedding_array = (
        embedding.detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )

    reduced_features = pca.transform(embedding_array)

    cluster_id = int(kmeans.predict(reduced_features)[0])
    distances = kmeans.transform(reduced_features)[0]

    ordered_clusters = np.argsort(distances)
    closest = float(distances[ordered_clusters[0]])
    second_closest = float(distances[ordered_clusters[1]])

    absolute_margin = second_closest - closest
    relative_margin = absolute_margin / max(
        second_closest,
        1e-8,
    )

    return {
        "image": str(image_path.resolve()),
        "cluster_id": cluster_id,
        "regime_name": REGIME_NAMES.get(
            cluster_id,
            f"Cluster {cluster_id}",
        ),
        "distance_to_assigned_centroid": float(
            distances[cluster_id]
        ),
        "distances_to_all_clusters": {
            str(index): float(distance)
            for index, distance in enumerate(distances)
        },
        "clusters_nearest_to_farthest": [
            int(index) for index in ordered_clusters
        ],
        "absolute_distance_margin": float(absolute_margin),
        "relative_distance_margin": float(relative_margin),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign a candlestick image to the saved "
            "four-cluster visual-regime model."
        )
    )

    parser.add_argument(
        "image",
        type=Path,
        help="Path to the new candlestick image.",
    )
    parser.add_argument(
        "--pca",
        type=Path,
        default=Path("models/pca.joblib"),
        help="Path to pca.joblib.",
    )
    parser.add_argument(
        "--kmeans",
        type=Path,
        default=Path("models/kmeans.joblib"),
        help="Path to kmeans.joblib.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional destination for a JSON prediction report.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    encoder, transform, pca, kmeans = load_pipeline(
        pca_path=args.pca,
        kmeans_path=args.kmeans,
        device=device,
    )

    prediction = predict_cluster(
        image_path=args.image,
        encoder=encoder,
        transform=transform,
        pca=pca,
        kmeans=kmeans,
        device=device,
    )

    print("\nPrediction")
    print("-" * 60)
    print(
        f"Cluster: {prediction['cluster_id']} — "
        f"{prediction['regime_name']}"
    )
    print(
        "Distance to assigned centroid: "
        f"{prediction['distance_to_assigned_centroid']:.6f}"
    )
    print(
        "Relative distance margin: "
        f"{prediction['relative_distance_margin']:.6f}"
    )
    print("\nFull result:")
    print(json.dumps(prediction, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with args.output.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(prediction, file, indent=2)

        print(f"\nSaved report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
