"""
Splitting isolates into train/test folds.

The trap: E. coli isolates come in families (clonal lineages). Close relatives
share almost all their SNPs *and* usually their resistance. If one sibling is
in the training set and its twin in the test set, the model can ace the test by
recognising the family face instead of learning resistance biology. It's like
acing an exam because your older sibling sat the same one last year.

Two ways to split:
- "random":  the classic shuffle. What most papers report; optimistic.
- "lineage": families always stay together in one fold. Harder, more honest.

The gap between the two scores is the generalisation gap this project is
trying to close.
"""
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def snp_distance(X):
    """For every pair of isolates: the fraction of SNP positions where they disagree."""
    same = np.zeros((len(X), len(X)), np.float32)
    for code in range(5):
        hit = (X == code).astype(np.float32)
        same += hit @ hit.T  # one matrix multiply counts matches for all pairs at once
    dist = 1 - same / X.shape[1]
    np.fill_diagonal(dist, 0)
    return dist


def lineage_groups(X, max_distance):
    """
    Group isolates into families: build a family tree (average linkage) and cut
    it wherever branches are more than `max_distance` apart.
    Returns one integer family ID per isolate.
    """
    tree = linkage(squareform(snp_distance(X), checks=False), method="average")
    return fcluster(tree, t=max_distance, criterion="distance")


def make_folds(y, groups, kind, n_folds, seed):
    """
    List of (train_idx, test_idx) pairs. Stratified, so every fold keeps roughly
    the same resistant/susceptible ratio as the full dataset.
    """
    if kind == "random":
        cv = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
        return list(cv.split(np.zeros(len(y)), y))
    if kind == "lineage":
        cv = StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed)
        return list(cv.split(np.zeros(len(y)), y, groups))
    raise ValueError(f"Unknown split {kind!r}: use 'random' or 'lineage'.")
