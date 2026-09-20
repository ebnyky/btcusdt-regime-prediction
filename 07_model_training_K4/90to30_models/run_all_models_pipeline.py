from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
base=Path(__file__).resolve().parent
steps=['preflight_dataset.py','run_baselines.py','train_frozen_resnet_head.py','extract_embeddings.py','train_logistic_regression.py','train_embedding_mlp.py','compare_results.py']
for step in steps:
 print('\n'+'='*78);print('RUNNING:',step);print('='*78)
 subprocess.run([sys.executable,str(base/step),'--config',a.config],check=True)
print('\nALL MODELS AND COMPARISONS COMPLETED.')
