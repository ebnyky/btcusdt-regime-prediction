from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix


def compute_metrics(y_true, y_pred, class_names=None):
    labels = sorted(set(map(int, y_true)) | set(map(int, y_pred)))
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'classification_report': classification_report(y_true, y_pred, labels=labels, target_names=[class_names.get(str(i), str(i)) if class_names else str(i) for i in labels], output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        'labels': labels,
    }


def save_evaluation(out_dir, split, y_true, y_pred, probs, class_names, row_ids=None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    m = compute_metrics(y_true, y_pred, class_names)
    (out / f'{split}_metrics.json').write_text(json.dumps(m, indent=2), encoding='utf-8')
    pd.DataFrame(m['classification_report']).T.to_csv(out / f'{split}_classification_report.csv')
    cm = np.asarray(m['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_xlabel('Predicted class'); ax.set_ylabel('True class'); ax.set_title(f'{split.title()} confusion matrix')
    ax.set_xticks(range(len(m['labels']))); ax.set_yticks(range(len(m['labels'])))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): ax.text(j, i, str(cm[i, j]), ha='center', va='center')
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(out / f'{split}_confusion_matrix.png', dpi=180); plt.close(fig)
    pred = pd.DataFrame({'row_id': np.arange(len(y_true)) if row_ids is None else row_ids, 'true_label': y_true, 'predicted_label': y_pred})
    if probs is not None:
        for i in range(probs.shape[1]): pred[f'prob_class_{i}'] = probs[:, i]
    pred.to_csv(out / f'{split}_predictions.csv', index=False)
    return m
