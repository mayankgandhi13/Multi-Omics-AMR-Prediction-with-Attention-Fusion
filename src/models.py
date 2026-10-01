"""
The neural networks.

- SNPCNN1D reads the SNP vector like a long strip of film, sliding a small
  window along it. A PyTorch port of the original paper's Keras model.
- FCGRCNN looks at the 64x64 FCGR picture like a photo. It becomes the WGS
  branch of the fusion model later, which is why it has `embed()`: fusion needs
  the model's summary of the genome, not its final verdict.

Both output one raw score (a "logit") per isolate; sigmoid turns it into a
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
