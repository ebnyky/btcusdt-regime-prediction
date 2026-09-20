from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser(description='Combine model_comparison.csv files from stride 5, 10 and 15 experiments.')
p.add_argument('--stride5',required=True);p.add_argument('--stride10',required=True);p.add_argument('--stride15',required=True);p.add_argument('--output',default='stride_comparison.csv');a=p.parse_args()
rows=[]
for stride,path in [(5,a.stride5),(10,a.stride10),(15,a.stride15)]:
 f=Path(path)/'model_comparison.csv'
 if not f.exists():raise FileNotFoundError(f)
 d=pd.read_csv(f);d.insert(0,'stride',stride);rows.append(d)
out=pd.concat(rows,ignore_index=True);out.to_csv(a.output,index=False);print(out.sort_values(['split','macro_f1'],ascending=[True,False]).to_string(index=False));print('Saved',a.output)
