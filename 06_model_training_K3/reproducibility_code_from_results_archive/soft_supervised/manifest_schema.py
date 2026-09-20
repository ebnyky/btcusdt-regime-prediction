from __future__ import annotations

import pandas as pd


def canonicalize_input_image_column(frame: pd.DataFrame, manifest_name: str) -> pd.DataFrame:
    """Return a frame with the dataset generator's image-path field canonicalized."""
    if "input_image" not in frame.columns:
        if "input_image_path" not in frame.columns:
            raise ValueError(
                f"Manifest {manifest_name} is missing an image-path column. "
                "Expected 'input_image_path' or 'input_image'."
            )
        return frame.rename(columns={"input_image_path": "input_image"})

    if "input_image_path" in frame.columns:
        canonical = frame["input_image"].astype(str).str.replace("\\", "/", regex=False)
        generator = frame["input_image_path"].astype(str).str.replace("\\", "/", regex=False)
        if not canonical.equals(generator):
            raise ValueError(
                f"Manifest {manifest_name} contains conflicting 'input_image' and "
                "'input_image_path' values."
            )
    return frame


def verify_source_aware_split_boundaries(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> list[dict[str, object]]:
    """Verify chronological separation without comparing unrelated local indices."""
    partitions = {
        "train": train,
        "validation": validation,
        "test": test,
    }
    required = {
        "pair_id",
        "input_source_id",
        "target_source_id",
        "input_start_index",
        "target_stop_index",
        "input_start_time",
        "target_stop_time",
    }
    parsed: dict[str, tuple[pd.Series, pd.Series]] = {}

    for name, frame in partitions.items():
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"Cannot verify {name} split boundaries; missing columns: {sorted(missing)}"
            )
        if frame.empty:
            raise ValueError(f"Cannot verify an empty {name} partition.")
        if not frame["pair_id"].astype(str).is_unique:
            raise ValueError(f"Duplicate pair IDs detected within {name}.")
        source_mismatch = (
            frame["input_source_id"].astype(str)
            != frame["target_source_id"].astype(str)
        )
        if source_mismatch.any():
            raise ValueError(f"At least one {name} pair crosses source files.")

        input_times = pd.to_datetime(frame["input_start_time"], utc=True, errors="coerce")
        target_times = pd.to_datetime(frame["target_stop_time"], utc=True, errors="coerce")
        if input_times.isna().any() or target_times.isna().any():
            raise ValueError(f"Unparseable boundary timestamps detected in {name}.")
        if (target_times <= input_times).any():
            raise ValueError(f"At least one {name} target does not follow its input.")
        parsed[name] = (input_times, target_times)

    pair_sets = {
        name: set(frame["pair_id"].astype(str))
        for name, frame in partitions.items()
    }
    for left_name, right_name in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if pair_sets[left_name].intersection(pair_sets[right_name]):
            raise ValueError(
                f"Leakage detected: {left_name} and {right_name} share pair IDs."
            )

    audits: list[dict[str, object]] = []
    for left_name, right_name in (("train", "validation"), ("validation", "test")):
        left = partitions[left_name]
        right = partitions[right_name]
        left_end = parsed[left_name][1].max()
        right_start = parsed[right_name][0].min()
        if left_end >= right_start:
            raise ValueError(
                f"Leakage detected: final {left_name} target ({left_end}) is not before "
                f"first {right_name} input ({right_start})."
            )

        left_sources = set(left["target_source_id"].astype(str))
        right_sources = set(right["input_source_id"].astype(str))
        shared_sources = sorted(left_sources.intersection(right_sources))
        for source_id in shared_sources:
            left_indices = left.loc[
                left["target_source_id"].astype(str) == source_id,
                "target_stop_index",
            ]
            right_indices = right.loc[
                right["input_source_id"].astype(str) == source_id,
                "input_start_index",
            ]
            if int(left_indices.max()) >= int(right_indices.min()):
                raise ValueError(
                    f"Leakage detected within source {source_id!r}: final {left_name} "
                    f"target index is not before first {right_name} input index."
                )

        audits.append({
            "boundary": f"{left_name}_to_{right_name}",
            "left_target_end": left_end,
            "right_input_start": right_start,
            "gap": right_start - left_end,
            "shared_sources": shared_sources,
            "index_check_applied": bool(shared_sources),
        })

    return audits
