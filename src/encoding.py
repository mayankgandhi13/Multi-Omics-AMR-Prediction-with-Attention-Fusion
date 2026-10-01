"""
Turning DNA letters into numbers.

Models can't read "A, C, G, T"; they only eat numbers. Three recipes:

1. Label encoding:   each letter gets a number (N=0, A=1, G=2, C=3, T=4).
   The raw data already comes this way. Quick, but it quietly tells the model
   "T is bigger than A", which means nothing biologically.

2. One-hot encoding: each SNP position becomes four yes/no switches
   ("is it A?", "is it G?", ...). No fake ordering, but 4x the columns.

3. FCGR:             squash an isolate's whole SNP profile into a 64x64
   picture so a CNN can look at it like a photo.
"""
import numpy as np
from scipy import sparse

# Codes used in the raw data. N means "same as the reference genome here".
N, A, G, C, T = 0, 1, 2, 3, 4


def one_hot(X):
    """
    (isolates x positions) label matrix -> sparse (isolates x positions*4) switches.

    N is left as all switches off: matching the reference is the default, like
    a control panel where "all lights off" means "nothing unusual here". At least
    3 of every 4 switches are off, so a sparse matrix saves a lot of memory.
    """
    blocks = [sparse.csr_matrix(X == base, dtype=np.float32) for base in (A, G, C, T)]
    return sparse.hstack(blocks, format="csr")


def fcgr_one(seq, k):
    """
    Frequency Chaos Game Representation of one sequence.

    Picture a square with a letter in each corner: A bottom-left, C top-left,
    G top-right, T bottom-right. Every k-letter word lands in exactly one tiny
    cell. The last letter picks the quadrant, the letter before it picks the
    sub-quadrant, and so on, like zooming in on a map. With k=6 there are
    4^6 = 4,096 words, so the square is a 64x64 grid. Count how often each word
    appears and you get a heat-map fingerprint of the sequence.
    """
    size = 2 ** k
    x_bit = ((seq == G) | (seq == T)).astype(np.int64)  # right half of the square
    y_bit = ((seq == C) | (seq == G)).astype(np.int64)  # top half of the square
    valid = seq != N

    n_words = len(seq) - k + 1
    col = np.zeros(n_words, np.int64)
    row = np.zeros(n_words, np.int64)
    ok = np.ones(n_words, bool)
    for j in range(k):
        # Letter j of each word sets bit j, so the last letter is the most
        # significant bit: it decides the biggest quadrant.
        col += x_bit[j:j + n_words] << j
        row += y_bit[j:j + n_words] << j
        ok &= valid[j:j + n_words]  # a word containing N is broken; skip it

    counts = np.bincount(row[ok] * size + col[ok], minlength=size * size)
    return counts.reshape(size, size)


def fcgr(X, k=6):
    """
    FCGR image for every isolate -> float32 array (isolates x 1 x 2^k x 2^k).

    The "sequence" here is an isolate's SNP alleles read left to right, as in
    Ren et al. Honest caveat: neighbouring letters can be kilobases apart in the
    real genome, so this is a fingerprint of the SNP profile, not of real DNA words.

    Each image is scaled so 1.0 means "this word shows up as often as average".
    """
    images = np.stack([fcgr_one(seq, k) for seq in X]).astype(np.float32)
    totals = images.sum(axis=(1, 2), keepdims=True)
    images = images / np.maximum(totals, 1) * images[0].size
    return images[:, None]
