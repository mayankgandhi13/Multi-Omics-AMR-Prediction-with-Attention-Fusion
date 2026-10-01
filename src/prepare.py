"""
Step 1: raw CSVs -> tidy, fast-loading arrays.

    python -m src.prepare

Like meal prep on a Sunday: do the slow chopping once (parsing a 100 MB CSV,
sorting isolates into families, drawing FCGR pictures) so every training run
afterwards just grabs a ready-made box from the fridge.
"""
import numpy as np

from .data import PROCESSED, load_config, read_raw
from .encoding import fcgr
from .splits import lineage_groups


def main():
    cfg = load_config()
    out = cfg["data_dir"] / PROCESSED
    out.parent.mkdir(parents=True, exist_ok=True)

    print("Reading raw SNP matrix and phenotypes...")
    snps, pheno = read_raw(cfg)
    X = snps.to_numpy(np.int8)
    labels = pheno[cfg["antibiotics"]].to_numpy(np.float32)
    print(f"  {X.shape[0]} isolates x {X.shape[1]:,} SNP positions")
    for ab, col in zip(cfg["antibiotics"], labels.T):
        print(f"  {ab}: {int(np.nansum(col))} resistant / {int((col == 0).sum())} susceptible")

    print("Sorting isolates into lineage families...")
    lineage = lineage_groups(X, cfg["lineage"]["max_distance"])
    sizes = np.bincount(lineage)[1:]
    print(f"  {len(sizes)} families; the biggest holds {sizes.max()} isolates ({sizes.max() / len(X):.0%})")

    print(f"Drawing FCGR images (k={cfg['fcgr']['k']})...")
    images = fcgr(X, cfg["fcgr"]["k"])

    np.savez_compressed(
        out, X=X, labels=labels, lineage=lineage, fcgr=images,
        isolates=snps.index.to_numpy(str), positions=snps.columns.to_numpy(str),
        antibiotics=np.array(cfg["antibiotics"]),
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
