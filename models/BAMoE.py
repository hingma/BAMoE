import torch
import torch.nn as nn
from models.layers.embed import PatchEmbedding
from models.layers.moe import BiasAwareMoE


class _TransformerLayer(nn.Module):
    """Pre-norm transformer layer wrapping either a plain or MoE attention."""

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
        self.is_moe = isinstance(attn, BiasAwareMoE)

    def forward(self, x):
        if self.is_moe:
            attn_out, lb_loss, router_logits = self.attn(self.norm1(x))
        else:
            attn_out = self.attn(self.norm1(x))
            lb_loss = torch.tensor(0.0, device=x.device)
            router_logits = None
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, lb_loss, router_logits


class BAMoE(nn.Module):
    """
    Bias-Aware Mixture-of-Experts Transformer for long-term time series forecasting.

    Architecture
    ------------
    - Patch embedding (channel-independent)
    - Stack of transformer layers each with a BiasAwareMoE attention block
    - Flatten + linear projection head → prediction

    Parameters controlled by `args`
    --------------------------------
    seq_len, pred_len, d_model, n_heads, n_layers, d_ff, dropout,
    patch_len, stride,
    expert_types     : comma-separated, e.g. 'causal,local,periodic,global'
    top_k            : int, number of active experts per token
    routing          : routing mode string (see BiasAwareMoE)
    load_balance_coef: float, weight for auxiliary balance loss
    local_window     : int, window size for local attention experts
    periodic_period  : int, period parameter for periodic attention experts
    """

    def __init__(self, args):
        super().__init__()
        self.pred_len = args.pred_len
        self.load_balance_coef = getattr(args, 'load_balance_coef', 0.01)

        expert_types = [e.strip() for e in args.expert_types.split(',')]

        self.patch_embed = PatchEmbedding(
            seq_len=args.seq_len,
            patch_len=args.patch_len,
            stride=args.stride,
            d_model=args.d_model,
            dropout=args.dropout,
        )
        n_patches = self.patch_embed.n_patches

        self.layers = nn.ModuleList([
            _TransformerLayer(
                attn=BiasAwareMoE(
                    d_model=args.d_model,
                    n_heads=args.n_heads,
                    expert_types=expert_types,
                    top_k=args.top_k,
                    routing=getattr(args, 'routing', 'learned_sparse'),
                    dropout=args.dropout,
                    local_window=getattr(args, 'local_window', 3),
                    periodic_period=getattr(args, 'periodic_period', 12),
                ),
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
        Returns
        -------
        pred        : (B, pred_len, C)
        aux_loss    : scalar (load-balance regularisation)
        routing_log : list of (B*C, N, n_experts) tensors per layer, or None
        """
        tokens, B, C = self.patch_embed(x)   # (B*C, N, d_model)
        total_lb = torch.tensor(0.0, device=x.device)
        routing_log = []

        for layer in self.layers:
            tokens, lb, router_logits = layer(tokens)
            total_lb = total_lb + lb
            if return_routing and router_logits is not None:
                routing_log.append(router_logits.detach())

        tokens = self.norm(tokens)                              # (B*C, N, d_model)
        flat = tokens.contiguous().view(B * C, -1)              # (B*C, N*d_model)
        pred = self.head(flat)                                  # (B*C, pred_len)
        pred = pred.view(B, C, self.pred_len).permute(0, 2, 1) # (B, pred_len, C)

        aux_loss = self.load_balance_coef * total_lb
        return pred, aux_loss, routing_log if return_routing else None
