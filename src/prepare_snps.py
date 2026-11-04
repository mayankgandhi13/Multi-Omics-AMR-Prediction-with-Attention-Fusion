"""
Phase 2: Snippy's SNP calls -> a model-ready dataset on the gene table's isolates.

    python -m src.prepare_snps

This rebuilds Phase 1's view of the genome (single-letter differences from the
K-12 reference) for the Moradigaravand isolates. It deliberately reuses the
pangenome dataset's isolates, labels and lineage families, so both views get
exactly the same folds: any difference in scores comes from the features alone.

Encoding matches Phase 1: 0 = same as reference (or not called), 1=A 2=G 3=C 4=T.
Each kept position is labelled with its gene and protein change from Snippy
(e.g. "2339162:gyrA:p.Ser83Leu"), so important SNPs can be read off directly.
"""
import numpy as np
import pandas as pd

from .data import DATASETS, load_config, load_processed

CODE = {"A": 1, "G": 2, "C": 3, "T": 4}


def read_calls(path):
    """
    Single-base changes of one isolate: positions, alt-base codes, annotations.

    Snippy reports neighbouring changes as one multi-base call ("mnp"/"complex"),
    and that's how key resistance mutations such as gyrA D87N and parC S80I
    often come out. Calls where REF and ALT have the same length are split back
    into their individual changed bases, so none of them are lost. (Insertions and
    deletions shift the sequence and are left out, as in Phase 1.)
    """
    calls = pd.read_csv(path, sep="\t", usecols=["POS", "TYPE", "REF", "ALT", "GENE", "EFFECT"], dtype=str)
    calls = calls[calls["REF"].str.len() == calls["ALT"].str.len()]
    rows = []
    for pos, ref, alt, gene, effect in calls[["POS", "REF", "ALT", "GENE", "EFFECT"]].itertuples(index=False):
        for k, (r, a) in enumerate(zip(ref, alt)):
            if r != a and a in CODE:
                rows.append((int(pos) + k, CODE[a], gene, effect))
    out = pd.DataFrame(rows, columns=["POS", "CODE", "GENE", "EFFECT"])
    return out["POS"].to_numpy(np.int32), out["CODE"].to_numpy(np.int8), out


def main():
    cfg = load_config()
    morad = cfg["data_dir"] / "moradigaravand"
    genes = load_processed(cfg, "pangenome")
    isolates = genes["isolates"]

    print(f"Reading SNP calls for {len(isolates)} isolates...")
    positions, codes, notes = [], [], {}
    for run in isolates:
        pos, code, calls = read_calls(morad / "snps" / f"{run}.tab")
        positions.append(pos)
        codes.append(code)
        for p, gene, effect in zip(pos, calls["GENE"].fillna(""), calls["EFFECT"].fillna("")):
            notes.setdefault(p, f"{gene}:{effect.split(' ')[-1] if effect else ''}")
    print(f"  {np.mean([len(p) for p in positions]):,.0f} SNPs per isolate on average")

    # Keep positions where enough isolates differ from the reference.
    everything = np.concatenate(positions)
    unique, counts = np.unique(everything, return_counts=True)
    keep = unique[counts >= cfg["snps"]["min_frequency"] * len(isolates)]
    print(f"  {len(unique):,} variable positions in total, {len(keep):,} seen in >= {cfg['snps']['min_frequency']:.0%} of isolates")

    X = np.zeros((len(isolates), len(keep)), np.int8)
    for i, (pos, code) in enumerate(zip(positions, codes)):
        idx = np.searchsorted(keep, pos)
        hit = (idx < len(keep)) & (keep[np.minimum(idx, len(keep) - 1)] == pos)
        X[i, idx[hit]] = code[hit]

    out = cfg["data_dir"] / DATASETS["snps"]
    np.savez_compressed(out, X=X, labels=genes["labels"], lineage=genes["lineage"], isolates=isolates,
                        features=np.array([f"{p}:{notes[p]}" for p in keep]), antibiotics=genes["antibiotics"])
    print(f"Saved {out}: {X.shape[0]} isolates x {X.shape[1]:,} SNP positions")


if __name__ == "__main__":
    main()
