import math
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Splits a time series into non-overlapping (or strided) patches and projects
    each patch to d_model.  Works in channel-independent mode: the batch and
    channel dimensions are folded together before patching.
    """

    def __init__(self, seq_len, patch_len, stride, d_model, dropout=0.0):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        # Number of patches
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.projection = nn.Linear(patch_len, d_model)
        self.pos_enc = nn.Parameter(
            self._sinusoidal(self.n_patches, d_model), requires_grad=False
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _sinusoidal(n, d):
        pe = torch.zeros(n, d)
        pos = torch.arange(n, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d // 2])
        return pe

    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        # channel-independent: fold C into batch
        x = x.permute(0, 2, 1).contiguous().view(B * C, L)  # (B*C, L)
        # unfold into patches
        x = x.unfold(1, self.patch_len, self.stride)         # (B*C, N, P)
        x = self.projection(x)                                # (B*C, N, d_model)
        x = x + self.pos_enc                                  # broadcast positional enc
        return self.dropout(x), B, C
