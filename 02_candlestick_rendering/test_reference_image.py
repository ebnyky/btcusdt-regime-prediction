import hashlib
import json
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_supplied_reference_image_identity_and_shape() -> None:
    image_path = PROJECT_ROOT / "reference" / "sample_0_29.jpg"
    metadata_path = PROJECT_ROOT / "reference" / "REFERENCE_IMAGE.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()

    assert digest == metadata["sha256"]
    assert digest == "9551df8b81bd0b63ff071884c446525942e20e7856c70c198e3c423aabe80487"
    with Image.open(image_path) as image:
        assert image.size == (metadata["width_px"], metadata["height_px"])
        assert image.mode == metadata["mode"]
        assert image.format == "JPEG"
