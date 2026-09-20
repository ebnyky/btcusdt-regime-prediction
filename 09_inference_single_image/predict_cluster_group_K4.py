# predict_image_cluster_group.py


from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models


# ============================================================
# CHANGE ONLY THIS PATH
# ============================================================
IMAGE_PATH = Path(
    r"C:\Users\m0zim\OneDrive\Desktop\local_regime_testing"
    r"\test_images\sample_209_238.jpg"
)


# The models folder should be beside this Python script.
PROJECT_DIR = Path(__file__).resolve().parent
PCA_PATH = PROJECT_DIR / "models" / "pca.joblib"
KMEANS_PATH = PROJECT_DIR / "models" / "kmeans.joblib"


REGIME_NAMES = {
    0: "High-volatility consolidation",
    1: "Bullish trend and breakout",
    2: "Volatile sideways and reversal",
    3: "Bearish trend and breakdown",
}


def main() -> None:
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    if not PCA_PATH.is_file():
        raise FileNotFoundError(f"PCA model not found: {PCA_PATH}")

    if not KMEANS_PATH.is_file():
        raise FileNotFoundError(
            f"K-Means model not found: {KMEANS_PATH}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Loading models...")

    weights = models.ResNet18_Weights.DEFAULT
    transform = weights.transforms()

    encoder = models.resnet18(weights=weights)
    encoder.fc = nn.Identity()
    encoder.eval()
    encoder.to(device)

    pca = joblib.load(PCA_PATH)
    kmeans = joblib.load(KMEANS_PATH)

    print("Processing image...")

    with Image.open(IMAGE_PATH) as image:
        image = image.convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.inference_mode():
        embedding = encoder(image_tensor)

    embedding = (
        embedding.cpu()
        .numpy()
        .astype(np.float32)
    )

    reduced_features = pca.transform(embedding)

    cluster_id = int(
        kmeans.predict(reduced_features)[0]
    )

    distances = kmeans.transform(
        reduced_features
    )[0]

    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"Image: {IMAGE_PATH.name}")
    print(f"Cluster: {cluster_id}")
    print(f"Regime: {REGIME_NAMES[cluster_id]}")
    print(
        "Distance to centroid: "
        f"{distances[cluster_id]:.6f}"
    )

    print("\nDistances to all clusters:")

    for index, distance in enumerate(distances):
        print(
            f"Cluster {index}: {distance:.6f} "
            f"({REGIME_NAMES[index]})"
        )


if __name__ == "__main__":
    main()
