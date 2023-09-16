import torch.nn as nn
from model.module import AttentionBlock, normalization, View

class CELEBA64Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.latent_dim = kwargs["latent_dim"]

        self.encoder = nn.Sequential(

            nn.Conv2d(3, 64, (3, 3), (2, 2), 1),          # batch_size x 64 x 32 x 32

            normalization(64),
            nn.SiLU(True),
            nn.Conv2d(64, 128, (3, 3), (2, 2), 1),          # batch_size x 128 x 16 x 16

            AttentionBlock(128, 4, -1, False),

            normalization(128),
            nn.SiLU(True),
            nn.Conv2d(128, 128, (3, 3), (2, 2), 1),          # batch_size x 128 x 8 x 8

            normalization(128),
            nn.SiLU(True),
            nn.Conv2d(128, 128, (3, 3), (2, 2), 1),          # batch_size x 128 x 4 x 4

            normalization(128),
            nn.SiLU(True),
            View((-1, 128 * 4 * 4)),                  # batch_size x 2048
            nn.Linear(2048, self.latent_dim)
        )

    # x: batch_size x 3 x 64 x 64
    def forward(self, x):
        # batch_size x latent_dim
        return self.encoder(x)
