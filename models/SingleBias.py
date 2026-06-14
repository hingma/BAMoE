import torch
import torch.nn as nn
from models.layers.embed import PatchEmbedding
from models.layers.attention import build_attention


class _TransformerLayer(nn.Module):
    def __init__(self, attn, d_model, d_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = attn
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SingleBiasTransformer(nn.Module):
    """
    Patch-based channel-independent Transformer with a single fixed inductive bias.

    Used for:
    - Experiment 1 (preliminary study): causal / local / periodic / global variants
    - BAMoE ablation: homogeneous-expert baselines (all experts same type)

    The architecture is identical to BAMoE minus the MoE routing.
    """

    def __init__(self, args):
        super().__init__()
        self.pred_len = args.pred_len
        bias_type = getattr(args, 'bias_type', 'global')

        self.patch_embed = PatchEmbedding(
            seq_len=args.seq_len,
            patch_len=args.patch_len,
            stride=args.stride,
            d_model=args.d_model,
            dropout=args.dropout,
        )
        n_patches = self.patch_embed.n_patches

        attn_kwargs = {}
        if bias_type == 'local':
            attn_kwargs['window_size'] = getattr(args, 'local_window', 3)
        elif bias_type in ('periodic', 'periodic_fixed'):
            attn_kwargs['period'] = getattr(args, 'periodic_period', 12)
        elif bias_type == 'relative':
            attn_kwargs['max_len'] = n_patches
        elif bias_type in ('trend', 'seasonal'):
            attn_kwargs['ma_kernel'] = getattr(args, 'ma_kernel', 25)
        elif bias_type == 'alibi':
            mult = getattr(args, 'alibi_slope_mult', 1.0)
            if mult != 1.0:
                attn_kwargs['slope_multiplier'] = mult

        self.layers = nn.ModuleList([
            _TransformerLayer(
                attn=build_attention(bias_type, args.d_model, args.n_heads,
                                     dropout=args.dropout, **attn_kwargs),
                d_model=args.d_model,
                d_ff=args.d_ff,
                dropout=args.dropout,
            )
            for _ in range(args.n_layers)
        ])
        self.norm = nn.LayerNorm(args.d_model)
        self.head = nn.Linear(n_patches * args.d_model, args.pred_len)

    def forward(self, x, return_routing=False):
        """
        x : (B, seq_len, C)
        Returns pred (B, pred_len, C), aux_loss=0, routing_log=None
        (same signature as BAMoE for unified training loop)
        """
        tokens, B, C = self.patch_embed(x)
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)
        flat = tokens.contiguous().view(B * C, -1)
        pred = self.head(flat).view(B, C, self.pred_len).permute(0, 2, 1)
        return pred, torch.tensor(0.0, device=x.device), None
