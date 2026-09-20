from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from tqdm.auto import tqdm

SCRIPT_DIR=Path(__file__).resolve().parent
CONFIG_PATH=SCRIPT_DIR/"config.json"


def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def save_config(config):
    CONFIG_PATH.write_text(json.dumps(config,indent=2),encoding="utf-8")


def same_file(src: Path, dst: Path) -> bool:
    try:
        return dst.is_file() and src.stat().st_size == dst.stat().st_size
    except OSError:
        return False


def copy_dataset(source: Path, destination: Path):
    if not source.is_dir():
        raise FileNotFoundError(f"Drive dataset folder not found: {source}")
    destination.mkdir(parents=True,exist_ok=True)
    files=[p for p in source.rglob('*') if p.is_file()]
    copied=0
    skipped=0
    for src in tqdm(files, desc="Copying dataset from Drive to /content", unit="file"):
        rel=src.relative_to(source)
        dst=destination/rel
        dst.parent.mkdir(parents=True,exist_ok=True)
        if same_file(src,dst):
            skipped+=1
            continue
        shutil.copy2(src,dst)
        copied+=1
    return copied,skipped,len(files)


def main():
    if not Path('/content/drive/MyDrive').is_dir():
        raise RuntimeError("Google Drive is not mounted. Run drive.mount('/content/drive') first.")
    config=load_config()
    source=Path(config.get('drive_dataset_dir','/content/drive/MyDrive/supervised_future_regime_dataset'))
    destination=Path(config.get('local_dataset_dir','/content/supervised_future_regime_dataset'))
    copied,skipped,total=copy_dataset(source,destination)
    config['dataset_dir']=str(destination)
    save_config(config)
    print("\nCOLAB PREPARATION COMPLETE")
    print(f"Dataset source: {source}")
    print(f"Local runtime dataset: {destination}")
    print(f"Files discovered: {total:,}; copied: {copied:,}; already present: {skipped:,}")
    print(f"Results will be stored in: {config['output_dir']}")
    print("The local dataset will disappear when the Colab runtime resets; rerun this script next session.")

if __name__=='__main__':
    main()
