"""
Report cards for a model.

Besides the usual ML scores we track two numbers clinical microbiologists care about:

- VME (very major error): truly resistant, but we said "susceptible".
  The dangerous one: a patient gets a drug that won't work.
  A smoke alarm that stays silent during a fire.
- ME (major error): truly susceptible, but we said "resistant".
  Costly (a good drug gets ruled out) but not directly harmful.
  A smoke alarm that goes off because of burnt toast.
"""
import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)


def score(y_true, prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob)
    pred = (prob >= threshold).astype(int)
    both_classes = len(np.unique(y_true)) == 2  # AUROC is undefined otherwise
    return {
        "auroc": roc_auc_score(y_true, prob) if both_classes else np.nan,
        "auprc": average_precision_score(y_true, prob) if both_classes else np.nan,
        "mcc": matthews_corrcoef(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, pred),
        "vme": (pred[y_true == 1] == 0).mean(),
        "me": (pred[y_true == 0] == 1).mean(),
    }
