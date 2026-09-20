from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from common import load_config, read_manifests, validate_schema, ManifestImageDataset, get_device, set_seed, save_json
from models import embedding_resnet18

p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--force',action='store_true'); a=p.parse_args()
cfg=load_config(a.config); set_seed(int(cfg['seed'])); dfs=read_manifests(cfg); validate_schema(cfg,dfs,check_all_images=False)
out=Path(cfg['output_root'])/'embeddings'; out.mkdir(parents=True,exist_ok=True)
device=get_device(cfg.get('device','auto')); model=embedding_resnet18().to(device)
for split,df in dfs.items():
    path=out/f'{split}_resnet18_embeddings.npz'
    if path.exists() and not a.force: print('Exists, skipping:',path); continue
    ds=ManifestImageDataset(df,cfg['dataset_root'],cfg['image_column'],cfg['label_column'])
    dl=DataLoader(ds,batch_size=int(cfg['embedding_batch_size']),shuffle=False,num_workers=int(cfg['num_workers']),pin_memory=(device.type=='cuda'))
    feats=[]; labels=[]; rows=[]
    with torch.inference_mode():
        for x,y,idx in tqdm(dl,desc=f'Embedding {split}'):
            z=model(x.to(device,non_blocking=True)).cpu().numpy().astype('float32')
            feats.append(z); labels.append(y.numpy()); rows.append(idx.numpy())
    np.savez_compressed(path,X=np.concatenate(feats),y=np.concatenate(labels),row_id=np.concatenate(rows))
    print('Saved',path)
save_json(out/'embedding_metadata.json',{'backbone':'torchvision ResNet18_Weights.DEFAULT','dimension':512,'predictor_column':cfg['image_column'],'target_column':cfg['label_column'],'target_image_used':False})
