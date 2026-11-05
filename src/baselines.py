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
from sklearn.feature_selection import SelectKBest, chi2
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
    ap.add_argument("--dataset", choices=["giessen", "pangenome", "snps", "combined"], default="giessen")
    ap.add_argument("--encodings", nargs="+", choices=["label", "onehot", "genes"],
                    help="SNP encodings for giessen/snps (default: label onehot); pangenome/combined are 0/1 already")
    ap.add_argument("--top", type=int,
                    help="keep only the K features most linked to resistance, picked inside each training fold")
    args = ap.parse_args()
    if args.dataset in ("pangenome", "combined"):
        args.encodings = ["genes"]   # already a 0/1 table: nothing to encode
    elif not args.encodings:
        # The chi-squared test behind --top needs yes/no columns; label codes
        # (1=A ... 4=T) are names, not amounts, so only one-hot fits.
        args.encodings = ["onehot"] if args.top else ["label", "onehot"]

    cfg = load_config()
    data = load_processed(cfg, args.dataset)
    y, tested = get_labels(data, args.antibiotic)
    y, X_raw = y[tested].astype(int), data["X"][tested]
    isolates = data["isolates"][tested]
    folds = make_folds(y, data["lineage"][tested], args.split, cfg["n_folds"], cfg["seed"])

    metrics, predictions = [], []
    for enc in args.encodings:
        X = one_hot(X_raw) if enc == "onehot" else X_raw.astype(np.float32)
        for name in args.models:
            for fold, (train, test) in enumerate(folds):
                Xf = X
                if args.top:
                    # Hundreds of thousands of SNPs, a handful of real culprits:
                    # like a line-up where most faces just show which family
                    # someone is from. Shortlist the K most suspicious columns,
                    # judged on the training isolates only, so the test isolates
                    # never get a vote in their own exam.
                    picker = SelectKBest(chi2, k=min(args.top, X.shape[1])).fit(X[train], y[train])
                    Xf = X[:, picker.get_support(indices=True)]
                model = make_model(name, cfg["seed"]).fit(Xf[train], y[train])
                prob = resistance_score(model, Xf[test])
                tags = {"dataset": args.dataset, "antibiotic": args.antibiotic, "split": args.split, "model": name,
                        "encoding": f"{enc}+top{args.top}" if args.top else enc, "fold": fold}
                metrics.append({**tags, **score(y[test], prob)})
                predictions.append(pd.DataFrame({**tags, "isolate": isolates[test],
                                                 "y_true": y[test], "prob": prob}))
                print(f"{args.antibiotic} {args.split:7s} {name:3s} {enc:6s} fold {fold}: "
                      f"AUROC {metrics[-1]['auroc']:.3f}  MCC {metrics[-1]['mcc']:.3f}", flush=True)

    out = cfg["results_dir"] / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.antibiotic}_{args.split}" if args.dataset == "giessen" else f"{args.dataset}_{args.antibiotic}_{args.split}"
    if args.top:
        tag += f"_top{args.top}"
    pd.DataFrame(metrics).to_csv(out / f"{tag}_metrics.csv", index=False)
    pd.concat(predictions).to_csv(out / f"{tag}_predictions.csv", index=False)
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
