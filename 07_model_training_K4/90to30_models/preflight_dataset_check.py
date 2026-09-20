from __future__ import annotations
import argparse, json
from pathlib import Path
from common import load_config, read_manifests, validate_schema, save_json

p=argparse.ArgumentParser()
p.add_argument('--config', required=True)
a=p.parse_args()
cfg=load_config(a.config)
if cfg.get('image_column') != 'input_image':
 raise ValueError("image_column must be 'input_image'; target_image is audit-only.")
dfs=read_manifests(cfg)
summary=validate_schema(cfg, dfs, check_all_images=True)
out=Path(cfg['output_root'])/'preflight'
out.mkdir(parents=True, exist_ok=True)
save_json(out/'preflight_summary.json', summary)
print('\nPREFLIGHT PASSED')
print('Confirmed: 90-candle input_image -> next non-overlapping 30-candle target_cluster_id.')
print('Confirmed: target_image is not configured as predictor input.')
for split,s in summary.items(): print(f"{split}: {s['rows']} rows | classes {s['class_counts']}")
print('Report:', out/'preflight_summary.json')
