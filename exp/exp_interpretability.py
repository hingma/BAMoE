"""
Experiment 4 — Interpretability Analysis

Figure 1 — Real-Time Routing Gate Trajectories
  Select a continuous test window containing a structural break (detected via
  CUSUM on the raw signal).  The top panel shows the actual time series; the
  bottom panel is a stacked area chart of the router's soft probability
  distribution G(X)_k across patches, perfectly aligned on the same time axis.

Figure 2 — Latent Regime Dimensionality Reduction
  Collect per-sample routing probability vectors (averaged over patches and
  channels) from the full out-of-sample evaluation set.  Apply t-SNE (or PCA
  when N < 30) to project the 4-D router outputs into 2-D.  The resulting
  scatter reveals tightly clustered "regime islands" — left panel coloured by
  the dominant expert, right panel coloured by input-signal volatility.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

from data_provider.data_factory import data_provider
from models.BAMoE import BAMoE
from utils.visualize import (
    plot_routing_gate_trajectories,
    plot_regime_embedding,
)


class ExpInterpretability:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            f'cuda:{args.gpu}' if (args.use_gpu and torch.cuda.is_available()) else 'cpu'
        )
        self.model = BAMoE(args).to(self.device)
        ckpt = os.path.join(args.checkpoints, args.exp_name, 'checkpoint.pth')
        if os.path.exists(ckpt):
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
            print(f'Loaded checkpoint: {ckpt}')
        else:
            print(f'WARNING: checkpoint not found at {ckpt}; using random weights.')

        self.expert_types = [e.strip() for e in args.expert_types.split(',')]
        self.out_dir = os.path.join(args.results, args.exp_name, 'interpretability')
        os.makedirs(self.out_dir, exist_ok=True)

        self.patch_len = args.patch_len
        self.stride = args.stride

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_raw_test_series(self):
        """Returns (T, C) float32 array of scaled test data (continuous)."""
        dataset, _ = data_provider(self.args, 'test')
        return dataset.data_x  # (T, C)

    def _infer_routing(self, window_np):
        """
        Run a single forward pass on a (seq_len, C) window.
        Returns (N_patches, n_experts) soft routing probs, averaged over C.
        """
        x = torch.tensor(window_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            _, _, routing_log = self.model(x, return_routing=True)
        if not routing_log:
            return None
        logits = routing_log[-1]                         # (C, N, n_experts)
        probs = F.softmax(logits, dim=-1).cpu().numpy()  # (C, N, n_experts)
        return probs.mean(axis=0)                        # (N, n_experts)

    def _find_break_window(self, raw_data):
        """
        Scan the test series for the seq_len window with the largest CUSUM
        peak-to-peak (classic level-shift detector).

        Returns
        -------
        win_start   : int — index into raw_data
        break_within: int — structural break position within the window
        """
        seq_len = self.args.seq_len
        T = raw_data.shape[0]
        search_stride = max(1, (T - seq_len) // 300)

        best_score = -np.inf
        best_start = 0
        for start in range(0, T - seq_len, search_stride):
            w = raw_data[start:start + seq_len, 0]
            cusum = np.cumsum(w - w.mean())
            score = float(np.ptp(cusum))   # peak-to-peak
            if score > best_score:
                best_score = score
                best_start = int(start)

        # Pinpoint the break: |CUSUM| maximum, clamped away from edges
        w = raw_data[best_start:best_start + seq_len, 0]
        cusum = np.cumsum(w - w.mean())
        break_within = int(np.argmax(np.abs(cusum)))
        lo, hi = seq_len // 5, 4 * seq_len // 5
        break_within = int(np.clip(break_within, lo, hi))

        return best_start, break_within

    def _patch_centers(self, n_patches):
        """Map patch index → centre timestep within the window."""
        return np.array([i * self.stride + self.patch_len // 2
                         for i in range(n_patches)])

    # ------------------------------------------------------------------
    # Figure 1 — Real-Time Routing Gate Trajectories
    # ------------------------------------------------------------------

    def run_routing_trajectories(self):
        raw_data = self._load_raw_test_series()
        win_start, break_idx = self._find_break_window(raw_data)
        window = raw_data[win_start:win_start + self.args.seq_len, :]

        routing_seq = self._infer_routing(window)       # (N, n_experts)
        if routing_seq is None:
            print('Figure 1 skipped: no routing logits (uniform/random routing mode).')
            return

        centers = self._patch_centers(routing_seq.shape[0])
        series = window[:, 0]                           # first channel for display

        save_path = os.path.join(self.out_dir, 'routing_gate_trajectories.pdf')
        plot_routing_gate_trajectories(
            series=series,
            routing_seq=routing_seq,
            patch_centers=centers,
            expert_labels=self.expert_types,
            break_idx=break_idx,
            save_path=save_path,
            title=(f'Routing Gate Trajectories — {self.args.data}'
                   f'  (pred_len={self.args.pred_len})'),
        )
        print(f'Saved: {save_path}')

    # ------------------------------------------------------------------
    # Figure 2 — Latent Regime Dimensionality Reduction
    # ------------------------------------------------------------------

    def run_regime_embedding(self):
        _, loader = data_provider(self.args, 'test')

        routing_vecs = []   # (n_experts,) per sample after averaging over N and C
        volatilities = []   # per-sample input σ, used as a second coloring axis

        n_experts = len(self.expert_types)
        self.model.eval()
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                _, _, routing_log = self.model(x, return_routing=True)
                if not routing_log:
                    continue

                logits = routing_log[-1]                         # (B*C, N, n_experts)
                probs = F.softmax(logits, dim=-1).cpu().numpy()  # (B*C, N, n_experts)

                B = x.shape[0]
                C = x.shape[2]
                probs_bc = probs.reshape(B, C, -1, n_experts)
                mean_probs = probs_bc.mean(axis=(1, 2))          # (B, n_experts)
                routing_vecs.append(mean_probs)

                x_np = x.cpu().numpy()
                volatilities.append(x_np.std(axis=(1, 2)))      # (B,)

        if not routing_vecs:
            print('Figure 2 skipped: no routing data collected.')
            return

        routing_vecs = np.concatenate(routing_vecs, axis=0)     # (N_total, n_experts)
        volatilities = np.concatenate(volatilities, axis=0)      # (N_total,)
        dominant_expert = routing_vecs.argmax(axis=1)            # (N_total,)

        save_path = os.path.join(self.out_dir, 'latent_regime_embedding.pdf')
        plot_regime_embedding(
            routing_vecs=routing_vecs,
            dominant_expert=dominant_expert,
            volatility=volatilities,
            expert_labels=self.expert_types,
            save_path=save_path,
            title=f'Latent Regime Embedding — {self.args.data}',
        )
        print(f'Saved: {save_path}')

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self):
        print('--- Figure 1: Routing Gate Trajectories ---')
        self.run_routing_trajectories()
        print('--- Figure 2: Latent Regime Embedding ---')
        self.run_regime_embedding()
        print('Interpretability analysis complete.')
