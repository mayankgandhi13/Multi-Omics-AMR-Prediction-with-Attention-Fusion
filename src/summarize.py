"""
Step 4: one table with every result.

    python -m src.summarize

Gathers every *_metrics.csv under results/, prints mean +/- std across folds
for each antibiotic x split x model, and saves results/summary.csv.
Read it side by side: the random-vs-lineage gap is the generalisation gap.
"""
import pandas as pd

from .data import load_config

KEYS = ["antibiotic", "split", "model", "encoding"]
METRICS = ["auroc", "auprc", "mcc", "f1", "vme", "me"]


def main():
    cfg = load_config()
    files = sorted(cfg["results_dir"].glob("*/*_metrics.csv"))
    if not files:
        print("No results yet. Run src.baselines or src.train_cnn first.")
        return

    folds = pd.concat(map(pd.read_csv, files))
    stats = folds.groupby(KEYS)[METRICS].agg(["mean", "std"])
    table = pd.DataFrame({m: stats[(m, "mean")].map("{:.3f}".format) + " ± " + stats[(m, "std")].map("{:.3f}".format)
                          for m in METRICS})
    print(table.to_string())
    table.to_csv(cfg["results_dir"] / "summary.csv")
    print(f"\nSaved {cfg['results_dir'] / 'summary.csv'}")


if __name__ == "__main__":
    main()
