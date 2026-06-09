import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _BaseAttention(nn.Module):
    """Shared QKV projection + output projection for all attention variants."""

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _qkv(self, x):
        B, N, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.q(x).view(B, N, H, dh).transpose(1, 2)  # (B, H, N, dh)
        k = self.k(x).view(B, N, H, dh).transpose(1, 2)
        v = self.v(x).view(B, N, H, dh).transpose(1, 2)
        return q, k, v

    def _attend(self, q, k, v, mask=None, bias=None):
        attn = (q @ k.transpose(-2, -1)) / self.scale  # (B, H, N, N)
        if bias is not None:
            attn = attn + bias
        if mask is not None:
            attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = attn.nan_to_num(0.0)
        return self.dropout(attn) @ v

    def _merge(self, x, B, N, D):
        return self.out(x.transpose(1, 2).contiguous().view(B, N, D))


class GlobalAttention(_BaseAttention):
    """Full (standard) self-attention — no inductive bias."""

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self._qkv(x)
        out = self._attend(q, k, v)
        return self._merge(out, B, N, D)


class CausalAttention(_BaseAttention):
    """Autoregressive masked attention — each patch attends only to earlier patches."""

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self._qkv(x)
        mask = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
        out = self._attend(q, k, v, mask=mask)
        return self._merge(out, B, N, D)


class LocalAttention(_BaseAttention):
    """Sliding-window attention — each patch attends within ±window_size positions."""

    def __init__(self, d_model, n_heads, window_size=3, dropout=0.0):
        super().__init__(d_model, n_heads, dropout)
        self.window_size = window_size

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self._qkv(x)
        i = torch.arange(N, device=x.device).unsqueeze(1)
        j = torch.arange(N, device=x.device).unsqueeze(0)
        mask = (i - j).abs() > self.window_size
        out = self._attend(q, k, v, mask=mask)
        return self._merge(out, B, N, D)


class PeriodicAttention(_BaseAttention):
    """
    Attention with learnable periodic positional bias.

    For each pair (i, j) the bias is indexed by |i−j| mod max_period,
    so the model can learn to up-weight attention at periodic lag distances.
    One bias vector per head, shared across all batch/token positions.
    """

    def __init__(self, d_model, n_heads, max_period=12, dropout=0.0):
        super().__init__(d_model, n_heads, dropout)
        self.max_period = max_period
        # (H, max_period) — one learnable scalar bias per (head, lag-mod-period)
        self.period_bias = nn.Parameter(torch.zeros(n_heads, max_period))

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self._qkv(x)
        i = torch.arange(N, device=x.device)
        j = torch.arange(N, device=x.device)
        lag = (i.unsqueeze(1) - j.unsqueeze(0)).abs() % self.max_period  # (N, N)
        # bias: (H, N, N)
        bias = self.period_bias[:, lag]
        out = self._attend(q, k, v, bias=bias.unsqueeze(0))
        return self._merge(out, B, N, D)


ATTENTION_REGISTRY = {
    'global': GlobalAttention,
    'causal': CausalAttention,
    'local': LocalAttention,
    'periodic': PeriodicAttention,
}


def build_attention(bias_type, d_model, n_heads, dropout=0.0, **kwargs):
    """Instantiate an attention module by name."""
    cls = ATTENTION_REGISTRY[bias_type]
    return cls(d_model, n_heads, dropout=dropout, **kwargs)
