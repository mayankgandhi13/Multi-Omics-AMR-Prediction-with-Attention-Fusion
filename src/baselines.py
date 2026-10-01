"""
Step 2: classic machine-learning baselines.

    python -m src.baselines --antibiotic CIP --split lineage

Trains Logistic Regression, a linear SVM and a Random Forest (the same three
as Ren et al. 2022) with 5-fold cross-validation, on label and one-hot
encodings. These set the bar: if a fancy deep model can't beat a Random Forest,
it hasn't earned its complexity.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .data import get_labels, load_config, load_processed
from .encoding import one_hot
from .metrics import score
from .splits import make_folds


def make_model(name, seed):
    # class_weight="balanced": when resistant isolates are rare, each one counts
    # for more during training. Same goal as SMOTE, without inventing fake genomes.
    if name == "LR":
        return LogisticRegression(max_iter=5000, class_weight="balanced")
    if name == "SVM":
        return SVC(kernel="linear", class_weight="balanced", random_state=seed)
    if name == "RF":
        return RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=seed)
    raise ValueError(f"Unknown model {name!r}")


def resistance_score(model, X):
    """A 0-1 score per isolate; above 0.5 means "resistant"."""
    if isinstance(model, SVC):
        # An SVM measures how far each isolate sits from its dividing line, not a
        # probability. A sigmoid squashes that distance into 0-1 with the line
        # landing exactly on 0.5. (Platt calibration would instead pull scores
        # towards the real resistance rate, undo class_weight, and on rare-resistance
        # drugs like GEN call every isolate susceptible.)
        return expit(model.decision_function(X))
    return model.predict_proba(X)[:, 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--antibiotic", required=True)
    ap.add_argument("--split", choices=["random", "lineage"], default="lineage")
    ap.add_argument("--models", nargs="+", default=["LR", "SVM", "RF"])
    ap.add_argument("--encodings", nargs="+", choices=["label", "onehot"], default=["label", "onehot"])
    args = ap.parse_args()

    cfg = load_config()
    data = load_processed(cfg)
    y, tested = get_labels(data, args.antibiotic)
    y, X_raw = y[tested].astype(int), data["X"][tested]
    isolates = data["isolates"][tested]
    folds = make_folds(y, data["lineage"][tested], args.split, cfg["n_folds"], cfg["seed"])

    metrics, predictions = [], []
    for enc in args.encodings:
        X = X_raw.astype(np.float32) if enc == "label" else one_hot(X_raw)
        for name in args.models:
            for fold, (train, test) in enumerate(folds):
                model = make_model(name, cfg["seed"]).fit(X[train], y[train])
                prob = resistance_score(model, X[test])
                tags = {"antibiotic": args.antibiotic, "split": args.split, "model": name,
                        "encoding": enc, "fold": fold}
                metrics.append({**tags, **score(y[test], prob)})
                predictions.append(pd.DataFrame({**tags, "isolate": isolates[test],
                                                 "y_true": y[test], "prob": prob}))
                print(f"{args.antibiotic} {args.split:7s} {name:3s} {enc:6s} fold {fold}: "
                      f"AUROC {metrics[-1]['auroc']:.3f}  MCC {metrics[-1]['mcc']:.3f}", flush=True)

    out = cfg["results_dir"] / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.antibiotic}_{args.split}"
    pd.DataFrame(metrics).to_csv(out / f"{tag}_metrics.csv", index=False)
    pd.concat(predictions).to_csv(out / f"{tag}_predictions.csv", index=False)
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
