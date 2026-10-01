"""
Loading data.

Think of this file as the librarian: it knows where every file lives, fetches
it, and hands you a tidy stack. Nothing clever happens here, on purpose.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent

RAW_SNPS = "raw/Giessen_dataset/cip_ctx_ctz_gen_multi_data.csv"
RAW_PHENO = "raw/Giessen_dataset/cip_ctx_ctz_gen_pheno.csv"
PROCESSED = "processed/dataset.npz"


def load_config(path=ROOT / "config.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # $AMR_DATA_DIR lets the HPC point us at /scratch without editing the config.
    cfg["data_dir"] = Path(os.environ.get("AMR_DATA_DIR", ROOT / cfg["data_dir"]))
    cfg["results_dir"] = ROOT / cfg["results_dir"]
    return cfg


def read_raw(cfg):
    """
    Read the original CSVs from Ren et al.

    snps:  one row per isolate, one column per SNP position, already
           label-encoded (0 = no variant, 1=A, 2=G, 3=C, 4=T).
    pheno: one row per isolate, one column per antibiotic (1 = resistant).
    """
    snps = pd.read_csv(cfg["data_dir"] / RAW_SNPS, index_col=0)
    pheno = pd.read_csv(cfg["data_dir"] / RAW_PHENO, index_col=0)
    # Line the labels up with the SNP rows. Mismatched rows are the classic
    # silent bug in genomics ML, so .loc fails loudly if an isolate is missing.
    pheno = pheno.loc[snps.index]
    return snps, pheno


def load_processed(cfg):
    """Load the arrays cached by `python -m src.prepare`."""
    path = cfg["data_dir"] / PROCESSED
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.prepare` first.")
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def get_labels(data, antibiotic):
    """Labels for one drug, plus a mask of isolates that were actually tested for it."""
    col = list(data["antibiotics"]).index(antibiotic)
    y = data["labels"][:, col]
    tested = ~np.isnan(y)
    return y, tested
