"""
Phase 4: attention fusion of the gene table and the SNPs.

    python -m src.fusion --antibiotic CIP --split lineage

Phase 2 showed each drug leans on a different kind of evidence: CIP on SNPs
(gyrA, parC), CTX/CTZ/GEN on acquired genes (blaCTX-M, aac(3)). Simply gluing
both tables together drowned the useful columns. Here each data type gets its
own encoder, and an attention layer learns per isolate how much to trust each
one (see AttentionFusion in models.py).

Same isolates, folds and report card as the baselines, so the numbers line up
with `src.baselines --dataset combined --top 1000`. Inside each fold:
  1. shortlist the top K genes and top K SNPs, judged on the training isolates only
  2. train with the same 90/10 "lessons vs practice exam" scheme as train_cnn
  3. score the sealed test fold, and record the attention weights it used

Writes results/fusion/<antibiotic>_<split>_{metrics,predictions}.csv. The
predictions carry each isolate's attention weights (attn_genes, attn_snps).
"""
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import SelectKBest, chi2

from .data import get_labels, load_config, load_processed
from .metrics import score
from .models import AttentionFusion
from .splits import make_folds
from .train_cnn import pick_device, predict, train_one_fold

SOURCES = ["genes", "snps"]   # later: "rna", for the P. aeruginosa cohort


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--antibiotic", required=True)
    ap.add_argument("--split", choices=["random", "lineage"], default="lineage")
    ap.add_argument("--epochs", type=int, help="override config.yaml (handy for a quick smoke test)")
    args = ap.parse_args()

    cfg = load_config()
    hp = {**cfg["fusion"], **({"epochs": args.epochs} if args.epochs else {})}
    torch.manual_seed(cfg["seed"])
    device = pick_device()
    print(f"Training on {device}")

    genes, snps = load_processed(cfg, "pangenome"), load_processed(cfg, "snps")
    assert (genes["isolates"] == snps["isolates"]).all(), "gene table and SNPs must list the same isolates"
    y, tested = get_labels(genes, args.antibiotic)
    y, isolates = y[tested].astype(np.float32), genes["isolates"][tested]
    # Both as yes/no columns: "has this gene?" and "differs from K-12 here?"
    tables = [genes["X"][tested], (snps["X"][tested] != 0).astype(np.int8)]
    folds = make_folds(y.astype(int), genes["lineage"][tested], args.split, cfg["n_folds"], cfg["seed"])

    metrics, predictions = [], []
    for fold, (train, test) in enumerate(folds):
        # 1. Shortlist each data type on its own, so neither can crowd out the other.
        kept = [SelectKBest(chi2, k=min(hp["top"], t.shape[1])).fit(t[train], y[train]).get_support(indices=True)
                for t in tables]
        X = np.hstack([t[:, k] for t, k in zip(tables, kept)]).astype(np.float32)
        sizes = [len(k) for k in kept]

        # 2. Train.
        make_model = lambda: AttentionFusion(sizes, hp["hidden"], hp["embed_dim"], hp["dropout"])
        model = train_one_fold(make_model, X[train], y[train], hp, device, cfg["seed"])

        # 3. Score the test fold, and ask whose report the model believed.
        prob = predict(model, X[test], device)
        with torch.no_grad():
            model(torch.from_numpy(X[test]).to(device))
        weights = model.last_weights.cpu().numpy()

        tags = {"dataset": "fusion", "antibiotic": args.antibiotic, "split": args.split, "model": "AttnFusion",
                "encoding": "+".join(SOURCES), "fold": fold}
        attn = {f"attn_{s}": weights[:, i] for i, s in enumerate(SOURCES)}
        metrics.append({**tags, **score(y[test], prob), **{k: v.mean() for k, v in attn.items()}})
        predictions.append(pd.DataFrame({**tags, "isolate": isolates[test], "y_true": y[test].astype(int),
                                         "prob": prob, **attn}))
        print(f"{args.antibiotic} {args.split:7s} fusion fold {fold}: AUROC {metrics[-1]['auroc']:.3f}  "
              f"MCC {metrics[-1]['mcc']:.3f}  attention genes {weights[:, 0].mean():.2f} / "
              f"SNPs {weights[:, 1].mean():.2f}", flush=True)

    out = cfg["results_dir"] / "fusion"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.antibiotic}_{args.split}"
    pd.DataFrame(metrics).to_csv(out / f"{tag}_metrics.csv", index=False)
    pd.concat(predictions).to_csv(out / f"{tag}_predictions.csv", index=False)
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
