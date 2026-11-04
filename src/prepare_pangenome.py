"""
Phase 2: Panaroo's gene table + phenotypes -> a model-ready dataset.

    python -m src.prepare_pangenome

Panaroo hands us a giant checklist: ~20,000 gene families x ~1,900 isolates,
1 if the isolate carries the gene, 0 if not. Most of it is useless for
prediction (genes everyone has can't separate resistant from susceptible), and
a lot of it is repetitive (genes that always travel together, like passengers
on the same plasmid bus). So we:

1. keep genes present in 1-99% of isolates,
2. merge genes with identical patterns into one feature,
3. attach R/S labels from the paper's table (intermediates are dropped),
4. sort isolates into lineage families, exactly as in Phase 1.

The result is saved in the same format as Phase 1's dataset, so
`python -m src.baselines --dataset pangenome` just works.
"""
import numpy as np
import pandas as pd

from .data import DATASETS, load_config
from .splits import lineage_groups

LABEL = {"R": 1.0, "S": 0.0}   # "I" (intermediate) and blanks become NaN: not used


def main():
    cfg = load_config()
    pg = cfg["pangenome"]
    morad = cfg["data_dir"] / "moradigaravand"

    print("Reading Panaroo gene table...")
    table = pd.read_csv(morad / "pangenome" / "gene_presence_absence.Rtab", sep="\t", index_col=0)
    meta = pd.read_csv(morad / "metadata.csv", encoding="utf-8-sig").set_index("Lane.accession")

    # A few rows of the paper's table point two isolates at the same ENA sample:
    # one genome, two (sometimes contradictory) phenotypes. We can't tell which
    # label belongs to the reads, so both are dropped. (pandas renames the second
    # copy's column "X.1", hence the split on ".".)
    shared = set(meta.index[meta.index.duplicated()])
    table = table[[c for c in table.columns if c.split(".")[0] not in shared]]
    meta = meta[~meta.index.isin(shared)]
    if shared:
        print(f"  dropped {', '.join(sorted(shared))}: one sample listed for two isolates")

    genes = table.T.astype(np.int8)          # rows = isolates (named by run), columns = gene families
    print(f"  {genes.shape[0]} isolates x {genes.shape[1]:,} gene families")

    print("Attaching phenotypes...")
    labels = meta.loc[genes.index, cfg["antibiotics"]].apply(lambda col: col.map(LABEL)).to_numpy(np.float32)
    for ab, col in zip(cfg["antibiotics"], labels.T):
        print(f"  {ab}: {int(np.nansum(col))} resistant / {int((col == 0).sum())} susceptible / {int(np.isnan(col).sum())} not used")

    print("Keeping genes that vary between isolates...")
    freq = genes.mean(axis=0)
    variable = genes.loc[:, (freq >= pg["min_frequency"]) & (freq <= pg["max_frequency"])]
    print(f"  {variable.shape[1]:,} genes in {pg['min_frequency']:.0%}-{pg['max_frequency']:.0%} of isolates")

    # Genes that are present/absent in exactly the same isolates carry the same
    # information: keep one column per pattern and remember who was merged.
    patterns, first, inverse = np.unique(variable.to_numpy().T, axis=0, return_index=True, return_inverse=True)
    names = variable.columns.to_numpy()
    groups = ["|".join(names[inverse.ravel() == k]) for k in range(len(patterns))]
    X = patterns.T.astype(np.int8)
    print(f"  {X.shape[1]:,} distinct patterns after merging genes that always travel together")

    print("Sorting isolates into lineage families...")
    lineage = lineage_groups(X, pg["lineage_max_distance"])
    sizes = np.bincount(lineage)[1:]
    print(f"  {len(sizes)} families; the biggest holds {sizes.max()} isolates ({sizes.max() / len(X):.0%})")

    out = cfg["data_dir"] / DATASETS["pangenome"]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, labels=labels, lineage=lineage,
                        isolates=genes.index.to_numpy(str), features=np.array(groups),
                        antibiotics=np.array(cfg["antibiotics"]))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
