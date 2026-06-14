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


# ---------------------------------------------------------------------------
# Expert 1: Global Context Expert
# ---------------------------------------------------------------------------

class GlobalAttention(_BaseAttention):
    """Full self-attention with no mask or positional bias (M=0, B=0)."""

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self._qkv(x)
        out = self._attend(q, k, v)
        return self._merge(out, B, N, D)


# ---------------------------------------------------------------------------
# Expert 2: Causal Expert
# ---------------------------------------------------------------------------

class CausalAttention(_BaseAttention):
    """Autoregressive causal mask — token i attends only to j <= i."""

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self._qkv(x)
        mask = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
        out = self._attend(q, k, v, mask=mask)
        return self._merge(out, B, N, D)


# ---------------------------------------------------------------------------
# Expert 3: Reverse-Causal Expert
# ---------------------------------------------------------------------------

class ReverseCausalAttention(_BaseAttention):
    """Reverse causal mask — token i attends only to j > i.

    Captures mean-reversion, turning points, and trend reversals.
    """

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self._qkv(x)
        # block j <= i (the diagonal and everything below it)
        mask = torch.tril(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=0)
        out = self._attend(q, k, v, mask=mask)
        return self._merge(out, B, N, D)


# ---------------------------------------------------------------------------
# Expert 4: Locality Expert (Window Attention)
# ---------------------------------------------------------------------------

class LocalAttention(_BaseAttention):
    """Hard sliding-window mask — token i attends only within ±window_size."""

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


# ---------------------------------------------------------------------------
# Expert 5: ALiBi Locality Expert
# ---------------------------------------------------------------------------

class ALiBiAttention(_BaseAttention):
    """ALiBi soft-locality bias — B_ij = -m_h * |i - j|.

    Slopes m_h are fixed per-head following the ALiBi geometric schedule:
    m_h = 2^(-8 * h / H) for h = 1, ..., H.

    slope_multiplier γ scales all heads uniformly (γ→0 = global, γ→∞ = very local).
    Used by Exp 4.3 to sweep mask-constraint stringency.
    """

    def __init__(self, d_model, n_heads, dropout=0.0, slope_multiplier=1.0):
        super().__init__(d_model, n_heads, dropout)
        slopes = slope_multiplier * torch.tensor(
            [2 ** (-8.0 * h / n_heads) for h in range(1, n_heads + 1)]
        )  # (H,)
        self.register_buffer('slopes', slopes)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self._qkv(x)
        i = torch.arange(N, device=x.device)
        j = torch.arange(N, device=x.device)
        dist = (i.unsqueeze(1) - j.unsqueeze(0)).abs().float()       # (N, N)
        bias = -self.slopes.view(-1, 1, 1) * dist.unsqueeze(0)        # (H, N, N)
        out = self._attend(q, k, v, bias=bias.unsqueeze(0))           # (1, H, N, N)
        return self._merge(out, B, N, D)


# ---------------------------------------------------------------------------
# Expert 6: Fixed Periodic Expert
# ---------------------------------------------------------------------------

class PeriodicAttention(_BaseAttention):
    """Periodic positional bias with a learnable period.

    Uses a differentiable cosine bias:
        B_ij = 1 - cos(2π * |i-j| / p),  p = exp(log_period)

    Tokens at the same phase (d=0) get zero penalty; tokens at opposite
    phases get a penalty of 2.  Gradients flow through p via log_period.
    """

    def __init__(self, d_model, n_heads, period=12, dropout=0.0):
        super().__init__(d_model, n_heads, dropout)
        # Store log so p stays positive without clamping
        self.log_period = nn.Parameter(torch.tensor(float(period)).log())

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self._qkv(x)
        idx = torch.arange(N, device=x.device, dtype=x.dtype)
        d = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()               # (N, N)
        p = self.log_period.exp()
        bias = torch.cos(2.0 * math.pi * d / p) - 1.0                 # (N, N)
        out = self._attend(q, k, v, bias=bias.unsqueeze(0).unsqueeze(0))
        return self._merge(out, B, N, D)



# ---------------------------------------------------------------------------
# Expert 7: Learnable Relative Position Expert
# ---------------------------------------------------------------------------

class RelativePositionAttention(_BaseAttention):
    """Learnable relative position bias — B_ij = f_theta(i - j).

    Uses a per-head embedding table of size 2*max_len - 1, one scalar per
    (head, relative-position) pair.
    """

    def __init__(self, d_model, n_heads, max_len=512, dropout=0.0):
        super().__init__(d_model, n_heads, dropout)
        self.max_len = max_len
        self.rpb = nn.Embedding(2 * max_len - 1, n_heads)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self._qkv(x)
        i = torch.arange(N, device=x.device)
        j = torch.arange(N, device=x.device)
        # relative positions in [-(N-1), N-1]; shift to valid embedding indices
        rel = (i.unsqueeze(1) - j.unsqueeze(0)) + (self.max_len - 1)  # (N, N)
        rel = rel.clamp(0, 2 * self.max_len - 2)
        bias = self.rpb(rel).permute(2, 0, 1).unsqueeze(0)            # (1, H, N, N)
        out = self._attend(q, k, v, bias=bias)
        return self._merge(out, B, N, D)


# ---------------------------------------------------------------------------
# Expert 8: Trend Expert
# ---------------------------------------------------------------------------

class TrendAttention(_BaseAttention):
    """Trend expert — QKV computed on the moving-average trend component T = MA(x)."""

    def __init__(self, d_model, n_heads, ma_kernel=25, dropout=0.0):
        super().__init__(d_model, n_heads, dropout)
        pad = ma_kernel // 2
        self.ma = nn.AvgPool1d(kernel_size=ma_kernel, stride=1, padding=pad)

    def _trend(self, x):
        N = x.shape[1]
        t = self.ma(x.permute(0, 2, 1))   # (B, D, N') — padding may add 1 extra
        return t[..., :N].permute(0, 2, 1)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self._qkv(self._trend(x))
        out = self._attend(q, k, v)
        return self._merge(out, B, N, D)


# ---------------------------------------------------------------------------
# Expert 9: Seasonal Expert
# ---------------------------------------------------------------------------

class SeasonalAttention(_BaseAttention):
    """Seasonal expert — QKV computed on the de-trended component S = x - MA(x)."""

    def __init__(self, d_model, n_heads, ma_kernel=25, dropout=0.0):
        super().__init__(d_model, n_heads, dropout)
        pad = ma_kernel // 2
        self.ma = nn.AvgPool1d(kernel_size=ma_kernel, stride=1, padding=pad)

    def _seasonal(self, x):
        N = x.shape[1]
        t = self.ma(x.permute(0, 2, 1))
        trend = t[..., :N].permute(0, 2, 1)
        return x - trend

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self._qkv(self._seasonal(x))
        out = self._attend(q, k, v)
        return self._merge(out, B, N, D)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ATTENTION_REGISTRY = {
    'global':          GlobalAttention,
    ####### ======== causal
    'causal':          CausalAttention,
    # 'reverse_causal':  ReverseCausalAttention,
    ###### ========= local
    # 'local':           LocalAttention,
    'alibi':           ALiBiAttention,
    ###### ========= periodic
    # 'periodic_fixed':  FixedPeriodicAttention,
    'periodic':        PeriodicAttention,   # backward-compat alias
    # 'relative':        RelativePositionAttention,
    ###### =========== trend/seasonal
    # 'trend':           TrendAttention,
    # 'seasonal':        SeasonalAttention,
}


def build_attention(bias_type, d_model, n_heads, dropout=0.0, **kwargs):
    """Instantiate an attention module by name."""
    cls = ATTENTION_REGISTRY[bias_type]
    return cls(d_model, n_heads, dropout=dropout, **kwargs)
