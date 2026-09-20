from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from common import load_config, read_manifests, validate_schema, ManifestImageDataset, class_weights_from_labels, get_device, set_seed, save_json
from models import frozen_resnet18_head
from metrics_utils import save_evaluation

p=argparse.ArgumentParser(); p.add_argument('--config',required=True); a=p.parse_args()
cfg=load_config(a.config); set_seed(int(cfg['seed']))
dfs=read_manifests(cfg); validate_schema(cfg,dfs,check_all_images=False)
out=Path(cfg['output_root'])/'frozen_resnet18_head'; out.mkdir(parents=True,exist_ok=True)
device=get_device(cfg.get('device','auto')); print('Device:',device)
image_col=cfg['image_column']; label_col=cfg['label_column']
loaders={}
for split in dfs:
    ds=ManifestImageDataset(dfs[split],cfg['dataset_root'],image_col,label_col)
    loaders[split]=DataLoader(ds,batch_size=int(cfg['batch_size']),shuffle=(split=='train'),num_workers=int(cfg['num_workers']),pin_memory=(device.type=='cuda'))
model=frozen_resnet18_head(int(cfg['num_classes'])).to(device)
weights=class_weights_from_labels(dfs['train'][label_col].to_numpy(),int(cfg['num_classes'])).to(device) if cfg.get('use_class_weights',True) else None
criterion=nn.CrossEntropyLoss(weight=weights,label_smoothing=float(cfg.get('label_smoothing',0)))
opt=torch.optim.AdamW(model.fc.parameters(),lr=float(cfg['head_learning_rate']),weight_decay=float(cfg['weight_decay']))
scaler=torch.amp.GradScaler('cuda',enabled=(device.type=='cuda' and cfg.get('mixed_precision',True)))
patience=int(cfg['early_stopping_patience']); best=-1; wait=0; history=[]; start=time.time()

def run_epoch(loader,train):
    model.train(train)
    if train and hasattr(model,'_keep_frozen_bn_eval'): model._keep_frozen_bn_eval()
    ys=[]; ps=[]; probs=[]; losses=[]
    for x,y,_ in loader:
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        if train: opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda',enabled=(device.type=='cuda' and cfg.get('mixed_precision',True))):
            logits=model(x); loss=criterion(logits,y)
        if train:
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.fc.parameters(),1.0); scaler.step(opt); scaler.update()
        pr=torch.softmax(logits.detach(),1)
        losses.append(float(loss.detach().cpu())); ys.extend(y.cpu().numpy()); ps.extend(pr.argmax(1).cpu().numpy()); probs.append(pr.cpu().numpy())
    return float(np.mean(losses)),np.asarray(ys),np.asarray(ps),np.concatenate(probs)

for epoch in range(1,int(cfg['max_epochs'])+1):
    tr_loss,tr_y,tr_p,_=run_epoch(loaders['train'],True)
    va_loss,va_y,va_p,va_probs=run_epoch(loaders['validation'],False)
    row={'epoch':epoch,'train_loss':tr_loss,'validation_loss':va_loss,'train_macro_f1':float(f1_score(tr_y,tr_p,average='macro',zero_division=0)),'validation_macro_f1':float(f1_score(va_y,va_p,average='macro',zero_division=0))}
    history.append(row); print(row)
    pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
    torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'config':cfg,'history':history},out/'last_checkpoint.pt')
    if row['validation_macro_f1']>best+1e-6:
        best=row['validation_macro_f1']; wait=0
        torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'config':cfg,'validation_macro_f1':best},out/'best_model.pt')
        save_evaluation(out,'validation',va_y,va_p,va_probs,cfg['class_names'])
    else:
        wait+=1
        if wait>=patience: print('Early stopping.'); break
ck=torch.load(out/'best_model.pt',map_location=device); model.load_state_dict(ck['model_state_dict'])
for split in ['validation','test']:
    _,y,pred,probs=run_epoch(loaders[split],False)
    m=save_evaluation(out,split,y,pred,probs,cfg['class_names'])
    print(split,m['accuracy'],m['balanced_accuracy'],m['macro_f1'])
save_json(out/'run_metadata.json',{'model':'frozen_pretrained_resnet18_head_only','best_epoch':ck['epoch'],'best_validation_macro_f1':ck['validation_macro_f1'],'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'total_parameters':sum(p.numel() for p in model.parameters()),'elapsed_seconds':time.time()-start,'device':str(device)})
