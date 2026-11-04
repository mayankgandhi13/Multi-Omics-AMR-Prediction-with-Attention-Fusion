"""
Which genes does the model lean on? A first sanity check before full SHAP.

    python -m src.top_features --dataset pangenome

For each drug, fit a Random Forest on every labelled isolate and list the
features it relies on most, with how common each one is among resistant (R)
and susceptible (S) isolates. If the model learned real biology, known
resistance genes should sit at the top: think blaCTX-M for cefotaxime or
aac(3) for gentamicin. If the top is full of anonymous "group_1234" genes,
it may be leaning on lineage markers instead.

Writes results/<dataset>_top_features.csv.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .data import get_labels, load_config, load_processed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["pangenome"], default="pangenome")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    cfg = load_config()
    data = load_processed(cfg, args.dataset)
    names = data["features"]
    rows = []
    for ab in cfg["antibiotics"]:
        y, tested = get_labels(data, ab)
        X, y = data["X"][tested], y[tested].astype(int)
        rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1,
                                    random_state=cfg["seed"]).fit(X, y)
        for rank, i in enumerate(np.argsort(rf.feature_importances_)[::-1][:args.top], 1):
            rows.append({"antibiotic": ab, "rank": rank, "importance": round(rf.feature_importances_[i], 4),
                         "in_R": round(X[y == 1, i].mean(), 2), "in_S": round(X[y == 0, i].mean(), 2),
                         "genes": names[i][:120]})
        print(pd.DataFrame([r for r in rows if r["antibiotic"] == ab]).to_string(index=False), "\n", flush=True)

    out = cfg["results_dir"] / f"{args.dataset}_top_features.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
