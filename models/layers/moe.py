import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.attention import build_attention


class BiasAwareMoE(nn.Module):
    """
    Mixture-of-Experts attention layer with heterogeneous inductive biases.

    Each expert is a full attention module with a different temporal bias
    (global, causal, local, periodic).  A lightweight router produces
    per-token routing weights; Top-K experts are selected and their outputs
    are combined with normalized soft weights.

    routing modes
    -------------
    'learned_sparse'  — learned router, top-k selection  (default)
    'uniform'         — equal weight to all experts, no routing
    'random'          — Dirichlet-sampled random weights
    'top1'            — hard routing to single best expert
    'dense'           — soft routing over all experts
    """

    def __init__(self, d_model, n_heads, expert_types, top_k=2,
                 routing='learned_sparse', dropout=0.0,
                 local_window=3, periodic_period=12, max_len=512, ma_kernel=25):
        super().__init__()
        self.n_experts = len(expert_types)
        self.top_k = min(top_k, self.n_experts)
        self.routing = routing

        def _kwargs(et):
            kw = {}
            if et == 'local':
                kw['slope_multiplier'] = local_window
            elif et in ('periodic', 'periodic_fixed'):
                kw['period'] = periodic_period
            elif et == 'relative':
                kw['max_len'] = max_len
            elif et in ('trend', 'seasonal'):
                kw['ma_kernel'] = ma_kernel
            return kw

        self.expert_types = expert_types
        self.experts = nn.ModuleList([
            build_attention(et, d_model, n_heads, dropout=dropout, **_kwargs(et))
            for et in expert_types
        ])

        if routing in ('learned_sparse', 'top1', 'dense'):
            self.router = nn.Linear(d_model, self.n_experts, bias=False)
        else:
            self.router = None

    # ------------------------------------------------------------------
    def forward(self, x):
        """
        Parameters
        ----------
        x : (B, N, D)

        Returns
        -------
        out         : (B, N, D)
        lb_loss     : scalar auxiliary load-balancing loss
        router_logits : (B, N, n_experts) or None
        """
        B, N, D = x.shape
        E = self.n_experts

        # Compute every expert's output: (B, N, D) each → stack → (B, N, E, D)
        expert_outs = torch.stack([e(x) for e in self.experts], dim=2)

        router_logits = None

        if self.routing == 'uniform':
            weights = torch.ones(B, N, E, device=x.device) / E
            out = (weights.unsqueeze(-1) * expert_outs).sum(dim=2)
            lb_loss = torch.tensor(0.0, device=x.device)

        elif self.routing == 'random':
            alpha = torch.ones(E, device=x.device)
            weights = torch.distributions.Dirichlet(alpha).sample((B, N))  # (B, N, E)
            out = (weights.unsqueeze(-1) * expert_outs).sum(dim=2)
            lb_loss = torch.tensor(0.0, device=x.device)

        elif self.routing == 'dense':
            logits = self.router(x)  # (B, N, E)
            router_logits = logits
            weights = F.softmax(logits, dim=-1)
            out = (weights.unsqueeze(-1) * expert_outs).sum(dim=2)
            lb_loss = self._balance_loss(logits)

        else:  # 'learned_sparse' or 'top1'
            k = 1 if self.routing == 'top1' else self.top_k
            logits = self.router(x)  # (B, N, E)
            router_logits = logits
            top_vals, top_idx = torch.topk(logits, k, dim=-1)   # (B, N, k)
            weights = F.softmax(top_vals, dim=-1)                # (B, N, k)
            # gather selected expert outputs
            idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, -1, D)  # (B, N, k, D)
            selected = expert_outs.gather(2, idx_exp)               # (B, N, k, D)
            out = (weights.unsqueeze(-1) * selected).sum(dim=2)     # (B, N, D)
            lb_loss = self._balance_loss(logits)

        return out, lb_loss, router_logits

    # ------------------------------------------------------------------
    def get_expert_attn_maps(self):
        """Return {expert_type: (N, N) numpy array} from the last forward pass."""
        return {
            et: expert._last_attn
            for et, expert in zip(self.expert_types, self.experts)
            if hasattr(expert, '_last_attn')
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _balance_loss(logits):
        """Switch-Transformer auxiliary load-balancing loss."""
        E = logits.shape[-1]
        probs = F.softmax(logits, dim=-1)
        # fraction of tokens dispatched to each expert (via hard argmax)
        dispatch = F.one_hot(logits.argmax(dim=-1), num_classes=E).float()
        f = dispatch.mean(dim=[0, 1])      # (E,)
        p = probs.mean(dim=[0, 1])         # (E,)
        return E * (f * p).sum()
