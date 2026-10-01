"""
Step 3: deep-learning baselines.

    python -m src.train_cnn --antibiotic CIP --split lineage --input fcgr

--input fcgr   2D CNN on the 64x64 FCGR picture (the future WGS branch)
--input label  1D CNN on the raw SNP strip (replicates the original paper)

Same folds and report card as the baselines, so the numbers line up.

In each fold, 90% of the training isolates are the lessons and 10% are a
practice exam used to pick the best epoch. The test fold stays sealed until
the very end. (The original code picked epochs by peeking at the test fold,
which flatters the scores.)
"""
import argparse
import copy

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import get_labels, load_config, load_processed
from .metrics import score
from .models import FCGRCNN, SNPCNN1D
from .splits import make_folds


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")  # Explorer GPU node
    if torch.backends.mps.is_available():
        return torch.device("mps")   # Apple-silicon laptop
    return torch.device("cpu")


@torch.no_grad()
def predict(model, X, device, batch_size=64):
    model.eval()
    batches = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=batch_size)
    return torch.cat([torch.sigmoid(model(xb.to(device))).cpu() for (xb,) in batches]).numpy()


def train_one_fold(make_model, X, y, hp, device, seed):
    X_fit, X_val, y_fit, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=seed)
    # The data is small (under 200 MB), so park all of it on the GPU once instead of
    # shipping it over batch by batch. Like keeping the ingredients on the counter
    # rather than walking to the pantry for every spoonful.
    X_fit, X_val, y_fit, y_val = (torch.from_numpy(a).to(device) for a in (X_fit, X_val, y_fit, y_val))

    # pos_weight: rarer resistant isolates count for more, like class_weight="balanced".
    pos_weight = (y_fit == 0).sum() / (y_fit == 1).sum().clamp(min=1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = make_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

    best_loss, best_state = float("inf"), None
    for epoch in range(hp["epochs"]):
        model.train()
        shuffled = torch.randperm(len(X_fit)).to(device)
        for batch in shuffled.split(hp["batch_size"]):
            if len(batch) < 2:  # BatchNorm can't learn from a batch of one isolate
                continue
            opt.zero_grad()
            loss_fn(model(X_fit[batch]), y_fit[batch]).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()
        if val_loss < best_loss:  # keep the version of the model that did best on the practice exam
            best_loss, best_state = val_loss, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--antibiotic", required=True)
    ap.add_argument("--split", choices=["random", "lineage"], default="lineage")
    ap.add_argument("--input", choices=["fcgr", "label"], default="fcgr")
    ap.add_argument("--epochs", type=int, help="override config.yaml (handy for a quick smoke test)")
    args = ap.parse_args()

    cfg = load_config()
    hp = {**cfg["cnn"], **({"epochs": args.epochs} if args.epochs else {})}
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True  # inputs never change shape, so let cuDNN find its fastest kernels once
    device = pick_device()
    print(f"Training on {device}")

    data = load_processed(cfg)
    y, tested = get_labels(data, args.antibiotic)
    y, isolates = y[tested].astype(np.float32), data["isolates"][tested]
    if args.input == "fcgr":
        X = data["fcgr"][tested]                                # (isolates, 1, 64, 64)
        make_model = FCGRCNN
    else:
        X = data["X"][tested].astype(np.float32)[:, None, :]    # (isolates, 1, positions)
        make_model = lambda: SNPCNN1D(X.shape[2])
    folds = make_folds(y.astype(int), data["lineage"][tested], args.split, cfg["n_folds"], cfg["seed"])

    metrics, predictions = [], []
    for fold, (train, test) in enumerate(folds):
        model = train_one_fold(make_model, X[train], y[train], hp, device, cfg["seed"])
        prob = predict(model, X[test], device)
        tags = {"antibiotic": args.antibiotic, "split": args.split, "model": "CNN",
                "encoding": args.input, "fold": fold}
        metrics.append({**tags, **score(y[test], prob)})
        predictions.append(pd.DataFrame({**tags, "isolate": isolates[test],
                                         "y_true": y[test].astype(int), "prob": prob}))
        print(f"{args.antibiotic} {args.split:7s} CNN {args.input:5s} fold {fold}: "
              f"AUROC {metrics[-1]['auroc']:.3f}  MCC {metrics[-1]['mcc']:.3f}", flush=True)

    out = cfg["results_dir"] / "cnn"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.antibiotic}_{args.split}_{args.input}"
    pd.DataFrame(metrics).to_csv(out / f"{tag}_metrics.csv", index=False)
    pd.concat(predictions).to_csv(out / f"{tag}_predictions.csv", index=False)
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
