from __future__ import annotations
import argparse,time
from pathlib import Path
import numpy as np, pandas as pd, joblib, torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from common import load_config,set_seed,get_device,class_weights_from_labels,save_json
from models import SmallMLP
from metrics_utils import save_evaluation
p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=load_config(a.config);set_seed(int(cfg['seed']))
emb=Path(cfg['output_root'])/'embeddings';out=Path(cfg['output_root'])/'embedding_mlp';out.mkdir(parents=True,exist_ok=True)
data={s:np.load(emb/f'{s}_resnet18_embeddings.npz') for s in ['train','validation','test']}
scaler=StandardScaler().fit(data['train']['X']);joblib.dump(scaler,out/'embedding_scaler.joblib')
X={s:scaler.transform(data[s]['X']).astype('float32') for s in data}; y={s:data[s]['y'].astype('int64') for s in data}
device=get_device(cfg.get('device','auto')); model=SmallMLP(512,tuple(cfg['hidden_units']),int(cfg['num_classes']),float(cfg['dropout'])).to(device)
weights=class_weights_from_labels(y['train'],int(cfg['num_classes'])).to(device) if cfg.get('use_class_weights',True) else None
crit=nn.CrossEntropyLoss(weight=weights,label_smoothing=float(cfg.get('label_smoothing',0)))
opt=torch.optim.AdamW(model.parameters(),lr=float(cfg['mlp_learning_rate']),weight_decay=float(cfg['weight_decay']))
loaders={s:DataLoader(TensorDataset(torch.from_numpy(X[s]),torch.from_numpy(y[s]),torch.from_numpy(data[s]['row_id'])),batch_size=int(cfg['batch_size']),shuffle=(s=='train')) for s in data}
pat=int(cfg['early_stopping_patience']);best=-1;wait=0;hist=[];start=time.time()
def run(dl,train):
 model.train(train);losses=[];ys=[];ps=[];prb=[];ids=[]
 for xb,yb,idx in dl:
  xb=xb.to(device);yb=yb.to(device)
  if train:opt.zero_grad(set_to_none=True)
  logits=model(xb);loss=crit(logits,yb)
  if train:loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
  p=torch.softmax(logits.detach(),1);losses.append(float(loss.detach().cpu()));ys.extend(yb.cpu().numpy());ps.extend(p.argmax(1).cpu().numpy());prb.append(p.cpu().numpy());ids.extend(idx.numpy())
 return np.mean(losses),np.asarray(ys),np.asarray(ps),np.concatenate(prb),np.asarray(ids)
for e in range(1,int(cfg['max_epochs'])+1):
 tl,ty,tp,_,_=run(loaders['train'],True);vl,vy,vp,vpr,vid=run(loaders['validation'],False)
 r={'epoch':e,'train_loss':float(tl),'validation_loss':float(vl),'train_macro_f1':float(f1_score(ty,tp,average='macro',zero_division=0)),'validation_macro_f1':float(f1_score(vy,vp,average='macro',zero_division=0))};hist.append(r);print(r);pd.DataFrame(hist).to_csv(out/'training_history.csv',index=False)
 torch.save({'epoch':e,'model_state_dict':model.state_dict(),'config':cfg},out/'last_checkpoint.pt')
 if r['validation_macro_f1']>best+1e-6:best=r['validation_macro_f1'];wait=0;torch.save({'epoch':e,'model_state_dict':model.state_dict(),'config':cfg,'validation_macro_f1':best},out/'best_model.pt');save_evaluation(out,'validation',vy,vp,vpr,cfg['class_names'],vid)
 else:
  wait+=1
  if wait>=pat:print('Early stopping.');break
ck=torch.load(out/'best_model.pt',map_location=device);model.load_state_dict(ck['model_state_dict'])
for s in ['validation','test']:
 _,yt,yp,pr,ids=run(loaders[s],False);m=save_evaluation(out,s,yt,yp,pr,cfg['class_names'],ids);print(s,m['accuracy'],m['balanced_accuracy'],m['macro_f1'])
save_json(out/'run_metadata.json',{'model':'small_mlp_on_frozen_resnet18_embeddings','best_epoch':ck['epoch'],'best_validation_macro_f1':ck['validation_macro_f1'],'scaler_fitted_on_train_only':True,'elapsed_seconds':time.time()-start,'device':str(device)})
