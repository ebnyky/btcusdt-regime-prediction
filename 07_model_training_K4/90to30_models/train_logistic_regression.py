from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from common import load_config, save_json
from metrics_utils import save_evaluation
p=argparse.ArgumentParser(); p.add_argument('--config',required=True); a=p.parse_args(); cfg=load_config(a.config)
emb=Path(cfg['output_root'])/'embeddings'; out=Path(cfg['output_root'])/'logistic_regression'; out.mkdir(parents=True,exist_ok=True)
data={s:np.load(emb/f'{s}_resnet18_embeddings.npz') for s in ['train','validation','test']}
model=Pipeline([('scaler',StandardScaler()),('classifier',LogisticRegression(max_iter=3000,class_weight='balanced',solver='lbfgs',multi_class='auto',random_state=int(cfg['seed'])))])
model.fit(data['train']['X'],data['train']['y'])
joblib.dump(model,out/'logistic_regression.joblib')
for split in ['validation','test']:
    pred=model.predict(data[split]['X']); probs=model.predict_proba(data[split]['X'])
    m=save_evaluation(out,split,data[split]['y'],pred,probs,cfg['class_names'],data[split]['row_id'])
    print(split,m['accuracy'],m['balanced_accuracy'],m['macro_f1'])
save_json(out/'run_metadata.json',{'model':'multiclass_logistic_regression_on_frozen_resnet18_embeddings','train_fitted_only':True,'scaler_fitted_on_train_only':True})
