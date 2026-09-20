import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image

from kline_dataset_generator import render_original_bxp


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_original_renderer_writes_360_rgb_jpeg(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "open": [100.0, 102.0, 101.0],
            "high": [103.0, 104.0, 103.0],
            "low": [99.0, 100.0, 98.0],
            "close": [102.0, 101.0, 99.0],
        }
    )
    destination = tmp_path / "sample_0_2.jpg"
    render_original_bxp(
        frame,
        {"open": "open", "high": "high", "low": "low", "close": "close"},
        destination,
    )
    with Image.open(destination) as image:
        assert image.size == (360, 360)
        assert image.mode == "RGB"
        assert image.format == "JPEG"


def test_first_historical_window_matches_reference_bytes(tmp_path: Path) -> None:
    frame = pd.read_csv(
        PROJECT_ROOT / "reference" / "BTCUSDT_1h_2020-01-01_first30.csv"
    )
    destination = tmp_path / "sample_0_29.jpg"
    render_original_bxp(
        frame,
        {"open": "open", "high": "high", "low": "low", "close": "close"},
        destination,
        {"width_px": 360, "height_px": 360, "dpi": 90},
    )
    generated = destination.read_bytes()
    reference = (PROJECT_ROOT / "reference" / "sample_0_29.jpg").read_bytes()

    assert generated == reference
    assert hashlib.sha256(generated).hexdigest() == (
        "9551df8b81bd0b63ff071884c446525942e20e7856c70c198e3c423aabe80487"
    )
