from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from common import load_config
p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=load_config(a.config);root=Path(cfg['output_root'])
models={'Frozen ResNet-18 head':'frozen_resnet18_head','Logistic regression':'logistic_regression','Embedding MLP':'embedding_mlp'}
rows=[]
for name,folder in models.items():
 for split in ['validation','test']:
  path=root/folder/f'{split}_metrics.json'
  if path.exists():
   m=json.loads(path.read_text());rows.append({'model':name,'split':split,**{k:m[k] for k in ['accuracy','balanced_accuracy','macro_f1','weighted_f1']}})
for name,folder in [('Majority baseline','baselines/majority'),('Persistence baseline','baselines/persistence')]:
 for split in ['validation','test']:
  path=root/folder/f'{split}_metrics.json'
  if path.exists():
   m=json.loads(path.read_text());rows.append({'model':name,'split':split,**{k:m[k] for k in ['accuracy','balanced_accuracy','macro_f1','weighted_f1']}})
if not rows: raise SystemExit('No completed result files were found.')
df=pd.DataFrame(rows);df.to_csv(root/'model_comparison.csv',index=False)
for split in ['validation','test']:
 sub=df[df.split==split].sort_values('macro_f1',ascending=False)
 if sub.empty:continue
 fig,ax=plt.subplots(figsize=(8,4.8));ax.barh(sub.model,sub.macro_f1);ax.set_xlabel('Macro-F1');ax.set_title(f'{split.title()} macro-F1 comparison');ax.invert_yaxis();fig.tight_layout();fig.savefig(root/f'{split}_macro_f1_comparison.png',dpi=180);plt.close(fig)
print('\nMODEL COMPARISON')
print(df.sort_values(['split','macro_f1'],ascending=[True,False]).to_string(index=False))
print('\nSaved:',root/'model_comparison.csv')
