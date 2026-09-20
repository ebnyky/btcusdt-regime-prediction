from __future__ import annotations

import json
from pathlib import Path


def code(source: str):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def markdown(source: str):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


cells = [
    markdown("# K=3 soft-membership three-model experiment — v1.3\n\nThis notebook trains **Soft Logistic Regression**, an **Embedding MLP**, and **ResNet18 with Layer 4 unfrozen** under the same chronological and soft-label protocol. Class imbalance is handled in every trainer with training-only inverse soft-membership-mass weights. Version 1.3 supports `input_image_path` and source-aware chronological boundary validation. Run the cells in order.\n"),
    code("import torch\n\nprint('PyTorch:', torch.__version__)\nprint('CUDA available:', torch.cuda.is_available())\nif torch.cuda.is_available():\n    print('GPU:', torch.cuda.get_device_name(0))\nelse:\n    print('A GPU is strongly recommended for embedding extraction and Layer-4 training.')\n"),
    code("!nvidia-smi\n"),
    markdown("## Discover the attached dataset and package\n\nThe dataset is identified by its validated soft-membership manifests. The package is identified by the v1.3 `soft_supervised/run_all_models.py` workflow.\n"),
    code("from pathlib import Path\n\ninput_root = Path('/kaggle/input')\ndataset_candidates = []\npackage_candidates = []\nfor path in input_root.rglob('*'):\n    if path.name == 'train_soft_shuffled.csv' and path.parent.name == 'manifests':\n        soft_dir = path.parent.parent\n        dataset_candidates.append(soft_dir.parent if soft_dir.name == 'soft_membership' else soft_dir)\n    if path.name == 'run_all_models.py' and path.parent.name == 'soft_supervised':\n        candidate = path.parent.parent\n        config_path = candidate / 'configs' / 'training_config.json'\n        if config_path.is_file():\n            import json\n            with config_path.open('r', encoding='utf-8') as handle:\n                candidate_config = json.load(handle)\n            if candidate_config.get('package_version') == '1.3':\n                package_candidates.append(candidate)\n\ndataset_candidates = sorted(set(dataset_candidates))\npackage_candidates = sorted(set(package_candidates))\nprint('Dataset candidates:')\nfor i, path in enumerate(dataset_candidates): print(i, path)\nprint('\\nPackage candidates:')\nfor i, path in enumerate(package_candidates): print(i, path)\nif not dataset_candidates or not package_candidates:\n    raise RuntimeError('Attach both the processed dataset and the v1.3 three-model package.')\n"),
    markdown("If more than one candidate appears, change the two indices below before continuing."),
    code("DATASET_ROOT = dataset_candidates[0]\nPACKAGE_INPUT = package_candidates[0]\nprint('Selected dataset:', DATASET_ROOT)\nprint('Selected package:', PACKAGE_INPUT)\n"),
    markdown("## Copy the package to Kaggle working storage and configure it"),
    code("import json\nimport shutil\n\nPACKAGE_WORKING = Path('/kaggle/working/k3_soft_membership_kaggle_three_model_v1_3')\nif PACKAGE_WORKING.exists():\n    shutil.rmtree(PACKAGE_WORKING)\nshutil.copytree(PACKAGE_INPUT, PACKAGE_WORKING)\nCONFIG_PATH = PACKAGE_WORKING / 'configs' / 'training_config.json'\nSCRIPTS = PACKAGE_WORKING / 'soft_supervised'\nRESULTS_ROOT = Path('/kaggle/working/k3_soft_three_model_results_v1_3')\nwith CONFIG_PATH.open('r', encoding='utf-8') as handle:\n    config = json.load(handle)\nconfig['dataset_dir'] = str(DATASET_ROOT)\nconfig['soft_membership_dir'] = ''\nconfig['output_dir'] = str(RESULTS_ROOT)\nconfig['device'] = 'auto'\nconfig['num_workers'] = 2\nwith CONFIG_PATH.open('w', encoding='utf-8') as handle:\n    json.dump(config, handle, indent=2)\nprint(json.dumps(config, indent=2))\n"),
    markdown("## Preflight check\n\nThis validates paths, manifests, split boundaries, weights, and all three output heads."),
    code("!python -u \"{SCRIPTS / 'check_setup.py'}\"\n"),
    markdown("## Train all models and select checkpoints using validation only\n\nThis cell caches frozen ResNet18 embeddings once, trains Soft Logistic Regression and the MLP on those shared embeddings, then trains ResNet18 Layer 4 from images. It does **not** evaluate test performance.\n"),
    code("!python -u \"{SCRIPTS / 'run_all_models.py'}\"\n"),
    markdown("## Inspect the validation-only comparison"),
    code("import pandas as pd\n\nvalidation_comparison = pd.read_csv(RESULTS_ROOT / 'validation_model_comparison.csv')\ndisplay(validation_comparison)\n"),
    markdown("## Independent test evaluation — run once\n\nRun this only after the model definitions, hyperparameters, and validation selection are final. The script evaluates all three validation-best checkpoints and writes a lock that prevents an accidental repeat.\n"),
    code("!python -u \"{SCRIPTS / 'evaluate_all_models.py'}\"\n"),
    code("test_comparison = pd.read_csv(RESULTS_ROOT / 'independent_test_model_comparison.csv')\ndisplay(test_comparison)\n"),
    markdown("## Archive the complete results"),
    code("import shutil\nfrom IPython.display import FileLink\n\narchive_base = Path('/kaggle/working/k3_soft_three_model_results')\narchive_path = shutil.make_archive(str(archive_base), 'zip', RESULTS_ROOT)\nprint('Created:', archive_path)\nFileLink(archive_path)\n"),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
        "kaggle": {"accelerator": "gpu", "isGpuEnabled": True},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

destination = Path(__file__).resolve().parent.parent / "notebooks" / "KAGGLE_K3_SOFT_THREE_MODELS.ipynb"
destination.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(destination)
