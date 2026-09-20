from __future__ import annotations
import json, os, random
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import ResNet18_Weights

REQUIRED_BASE_COLUMNS = {
    'input_image', 'target_cluster_id',
    'input_start_index', 'input_stop_index',
    'target_start_index', 'target_stop_index'
}


def load_config(path: str | Path) -> dict:
    p = Path(path).resolve()
    cfg = json.loads(p.read_text(encoding='utf-8'))
    cfg['_config_path'] = str(p)
    cfg['_package_root'] = str(p.parent.parent)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(name: str = 'auto') -> torch.device:
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def manifest_paths(cfg: dict) -> Dict[str, Path]:
    root = Path(cfg['dataset_root']).expanduser().resolve()
    mdir = root / cfg.get('manifest_dir', 'manifests')
    return {
        'train': mdir / cfg.get('train_manifest', 'train_samples.csv'),
        'validation': mdir / cfg.get('validation_manifest', 'validation_samples.csv'),
        'test': mdir / cfg.get('test_manifest', 'test_samples.csv'),
    }


def read_manifests(cfg: dict) -> Dict[str, pd.DataFrame]:
    paths = manifest_paths(cfg)
    out = {}
    for split, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f'Missing {split} manifest: {path}')
        df = pd.read_csv(path)
        missing = REQUIRED_BASE_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f'{split} manifest is missing columns: {sorted(missing)}')
        out[split] = df
    return out


def resolve_image_path(dataset_root: Path, value: str) -> Path:
    p = Path(str(value))
    return p if p.is_absolute() else dataset_root / p


def validate_schema(cfg: dict, dfs: Dict[str, pd.DataFrame], check_all_images: bool = True) -> dict:
    expected_input = int(cfg.get('expected_input_candles', 90))
    expected_target = int(cfg.get('expected_target_candles', 30))
    image_col = cfg.get('image_column', 'input_image')
    label_col = cfg.get('label_column', 'target_cluster_id')
    root = Path(cfg['dataset_root']).expanduser().resolve()
    issues: List[str] = []
    summary = {}
    previous_last_target = None
    for split in ['train', 'validation', 'test']:
        df = dfs[split].copy()
        if image_col not in df.columns or label_col not in df.columns:
            issues.append(f'{split}: missing configured image or label column.')
            continue
        input_len = df['input_stop_index'] - df['input_start_index'] + 1
        target_len = df['target_stop_index'] - df['target_start_index'] + 1
        bad_input = int((input_len != expected_input).sum())
        bad_target = int((target_len != expected_target).sum())
        bad_mapping = int((df['target_start_index'] != df['input_stop_index'] + 1).sum())
        bad_labels = int((~df[label_col].astype(int).isin(range(int(cfg.get('num_classes', 4))))).sum())
        if bad_input: issues.append(f'{split}: {bad_input} rows are not {expected_input}-candle inputs.')
        if bad_target: issues.append(f'{split}: {bad_target} rows are not {expected_target}-candle targets.')
        if bad_mapping: issues.append(f'{split}: {bad_mapping} rows have input-target overlap or a gap.')
        if bad_labels: issues.append(f'{split}: {bad_labels} labels are outside the expected class range.')
        if not df['input_start_index'].is_monotonic_increasing:
            issues.append(f'{split}: rows are not chronological by input_start_index.')
        first_input = int(df['input_start_index'].min())
        last_target = int(df['target_stop_index'].max())
        if previous_last_target is not None and first_input <= previous_last_target:
            issues.append(f'{split}: split begins at raw row {first_input}, overlapping the previous split through row {previous_last_target}.')
        previous_last_target = last_target
        missing_images = 0
        paths = df[image_col].astype(str)
        iterable = paths if check_all_images else paths.head(100)
        for value in iterable:
            if not resolve_image_path(root, value).exists():
                missing_images += 1
        if missing_images:
            issues.append(f'{split}: {missing_images} checked input images are missing.')
        if 'target_image' in df.columns and (df[image_col].astype(str) == df['target_image'].astype(str)).any():
            issues.append(f'{split}: an input_image path equals a target_image path.')
        summary[split] = {
            'rows': int(len(df)),
            'first_input_index': first_input,
            'last_target_index': last_target,
            'class_counts': {str(k): int(v) for k, v in df[label_col].astype(int).value_counts().sort_index().items()},
            'missing_images_checked': missing_images,
        }
    if issues:
        raise ValueError('\n'.join(issues))
    return summary


class ManifestImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, dataset_root: str | Path, image_col: str, label_col: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.root = Path(dataset_root).expanduser().resolve()
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform or ResNet18_Weights.DEFAULT.transforms()
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = resolve_image_path(self.root, row[self.image_col])
        with Image.open(path) as im:
            image = im.convert('RGB')
        return self.transform(image), int(row[self.label_col]), idx


def class_weights_from_labels(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels.astype(int), minlength=num_classes)
    weights = len(labels) / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def save_json(path: str | Path, obj) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')
