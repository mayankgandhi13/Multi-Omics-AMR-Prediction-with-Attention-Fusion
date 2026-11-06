"""
The neural networks.

- SNPCNN1D reads the SNP vector like a long strip of film, sliding a small
  window along it. A PyTorch port of the original paper's Keras model.
- FCGRCNN looks at the 64x64 FCGR picture like a photo. It becomes the WGS
  branch of the fusion model later, which is why it has `embed()`: fusion needs
  the model's summary of the genome, not its final verdict.
- AttentionFusion takes several kinds of evidence (gene table, SNPs, later RNA),
  summarises each one separately, and learns how much to trust each summary.

All output one raw score (a "logit") per isolate; sigmoid turns it into a
probability of resistance.
"""
import torch
from torch import nn


class SNPCNN1D(nn.Module):
    """Two conv blocks over the SNP strip, then a dense layer (Ren et al. 2022)."""

    def __init__(self, n_positions):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 8, 3), nn.ReLU(), nn.BatchNorm1d(8),
            nn.Conv1d(8, 8, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(8, 16, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(16),
            nn.Conv1d(16, 16, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Flatten(),
        )
        with torch.no_grad():  # run a dummy input through to learn the flattened size
            n_flat = self.features(torch.zeros(1, 1, n_positions)).shape[1]
        self.head = nn.Sequential(nn.Linear(n_flat, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


class FCGRCNN(nn.Module):
    """Three conv blocks over the FCGR image -> a 64-number summary -> resistance logit."""

    def __init__(self, embed_dim=64):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out),
                                 nn.ReLU(), nn.MaxPool2d(2))

        self.encoder = nn.Sequential(
            block(1, 16), block(16, 32), block(32, embed_dim),  # 64 -> 32 -> 16 -> 8 pixels
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(embed_dim, 1))

    def embed(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.head(self.embed(x)).squeeze(1)


class AttentionFusion(nn.Module):
    """
    One small encoder per data type -> attention weights -> one resistance logit.

    Like a panel of specialists: a gene expert and a SNP expert each write a
    short report (a 64-number summary) on the isolate, and a chair decides how
    much weight each report gets for *this* isolate before giving the verdict.
    Those weights are stored in `last_weights` (isolates x data types), so we
    can ask afterwards whose report the model believed, e.g. SNPs for CIP.

    The input is every data type's columns side by side; `sizes` says how many
    columns belong to each, in order.
    """

    def __init__(self, sizes, hidden=128, embed_dim=64, dropout=0.3):
        super().__init__()
        self.sizes = list(sizes)
        self.encoders = nn.ModuleList(
            nn.Sequential(nn.Linear(n, hidden), nn.ReLU(), nn.Dropout(dropout),
                          nn.Linear(hidden, embed_dim), nn.ReLU())
            for n in self.sizes
        )
        self.score = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Tanh(), nn.Linear(embed_dim, 1))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, 1))
        self.last_weights = None

    def forward(self, x):
        reports = torch.stack([enc(part) for enc, part in zip(self.encoders, x.split(self.sizes, dim=1))], dim=1)
        weights = torch.softmax(self.score(reports).squeeze(2), dim=1)   # (isolates, data types), rows sum to 1
        self.last_weights = weights.detach()
        fused = (weights.unsqueeze(2) * reports).sum(dim=1)
        return self.head(fused).squeeze(1)
