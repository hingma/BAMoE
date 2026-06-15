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
from models.SingleBias import SingleBiasTransformer
from utils.visualize import (
    plot_routing_gate_trajectories,
    plot_regime_embedding,
    plot_attention_snapshots,
    plot_best_case_comparison,
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
    # Helpers (attention maps)
    # ------------------------------------------------------------------

    def _infer_attention_maps(self, window_np):
        """Run a forward pass on (seq_len, C) window; return per-expert (N, N) maps."""
        x = torch.tensor(window_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            self.model(x)
        return self.model.get_expert_attn_maps()   # {expert_type: (N, N)}

    def _pick_snapshot_windows(self, raw_data, win_start, break_idx):
        """
        Derive 3 representative seq_len windows anchored in each regime.

        Returns
        -------
        windows : list of 3 (seq_len, C) arrays
        labels  : ['normal_operation', 'at_break', 'post_break']
        """
        seq_len = self.args.seq_len
        T = raw_data.shape[0]
        half = seq_len // 2

        abs_break = win_start + break_idx
        centers = [
            win_start + break_idx // 2,                              # normal regime centre
            abs_break,                                                # at the break
            abs_break + (seq_len - break_idx) // 2,                  # post-break centre
        ]
        labels = ['normal_operation', 'at_break', 'post_break']
        windows = []
        for c in centers:
            start = int(np.clip(c - half, 0, T - seq_len))
            windows.append(raw_data[start:start + seq_len, :])
        return windows, labels

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
    # Figure 3 — Per-Expert Attention Snapshots at 3 Regime Checkpoints
    # ------------------------------------------------------------------

    def run_attention_snapshots(self):
        raw_data = self._load_raw_test_series()
        win_start, break_idx = self._find_break_window(raw_data)

        windows, labels = self._pick_snapshot_windows(raw_data, win_start, break_idx)

        maps_per_period = []
        for window, label in zip(windows, labels):
            attn_maps = self._infer_attention_maps(window)
            if not attn_maps:
                print(f'Figure 3 skipped for {label}: no attention maps captured.')
                return
            maps_per_period.append((label, attn_maps))

        save_path = os.path.join(self.out_dir, 'attention_snapshots.pdf')
        plot_attention_snapshots(
            maps_per_period=maps_per_period,
            expert_labels=self.expert_types,
            save_path=save_path,
            title=(f'Per-Expert Attention Snapshots — {self.args.data}'
                   f'  (pred_len={self.args.pred_len})'),
        )
        print(f'Saved: {save_path}')

    # ------------------------------------------------------------------
    # Figure 4 — Best-Case Comparison: BAMoE vs. Single-Bias Models
    # ------------------------------------------------------------------

    def _load_single_bias_model(self, bias_type):
        """
        Instantiate a SingleBiasTransformer for *bias_type* on the same
        dataset / horizon as the current BAMoE run and load its checkpoint.

        Checkpoint path convention (matches make_exp_name in run.py):
          <checkpoints>/SingleBias_<bias_type>_<data>_sl<seq>_pl<pred>/checkpoint.pth
        """
        import copy
        sb_args = copy.copy(self.args)
        sb_args.bias_type = bias_type
        sb_args.exp_name = (
            f'SingleBias_{bias_type}_{self.args.data}'
            f'_sl{self.args.seq_len}_pl{self.args.pred_len}'
        )
        model = SingleBiasTransformer(sb_args).to(self.device)
        ckpt = os.path.join(self.args.checkpoints, sb_args.exp_name, 'checkpoint.pth')
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            print(f'  Loaded SingleBias ({bias_type}): {ckpt}')
        else:
            print(f'  WARNING: SingleBias ({bias_type}) checkpoint not found at {ckpt}; '
                  f'using random weights.')
        return model

    def _collect_predictions(self, model):
        """
        Run *model* over the full test set.

        Returns
        -------
        preds : (N, pred_len, C) float32 ndarray
        trues : (N, pred_len, C) float32 ndarray
        """
        _, loader = data_provider(self.args, 'test')
        all_preds, all_trues = [], []
        model.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                pred, _, _ = model(x)
                all_preds.append(pred.cpu().numpy())
                all_trues.append(y.numpy())
        return np.concatenate(all_preds, axis=0), np.concatenate(all_trues, axis=0)

    def run_best_case_comparison(self):
        """
        Figure 4 — Best-Case Forecast Comparison.

        Algorithm
        ---------
        1. Collect BAMoE predictions + per-sample MSE on the full test set.
        2. For every expert type, load the matching SingleBias checkpoint and
           collect its per-sample MSE.
        3. Select the test sample index *i* that maximises
               min_k MSE_SingleBias_k(i) − MSE_BAMoE(i)
           i.e. the moment where BAMoE beats *every* single-bias variant by
           the widest margin.
        4. Plot: input sequence (left) | all forecast paths + ground truth (right).
        """
        print('  Collecting BAMoE predictions …')
        bamoe_preds, trues = self._collect_predictions(self.model)
        bamoe_mse = np.mean((bamoe_preds - trues) ** 2, axis=(1, 2))   # (N,)

        sb_preds_dict: dict = {}
        sb_mse_dict:   dict = {}
        for bias_type in self.expert_types:
            print(f'  Collecting SingleBias ({bias_type}) predictions …')
            sb_model = self._load_single_bias_model(bias_type)
            preds, _ = self._collect_predictions(sb_model)
            sb_preds_dict[bias_type] = preds
            sb_mse_dict[bias_type]   = np.mean((preds - trues) ** 2, axis=(1, 2))

        # Best sample: largest BAMoE advantage over the strongest single-bias competitor
        sb_mse_stack = np.stack(list(sb_mse_dict.values()), axis=0)   # (E, N)
        best_sb_mse  = sb_mse_stack.min(axis=0)                        # (N,)
        advantage    = best_sb_mse - bamoe_mse                          # (N,)
        best_idx     = int(np.argmax(advantage))

        print(f'  Best sample: index={best_idx}, '
              f'BAMoE MSE={bamoe_mse[best_idx]:.4f}, '
              f'best SingleBias MSE={best_sb_mse[best_idx]:.4f}, '
              f'advantage ΔMSE={advantage[best_idx]:.4f}')

        # Retrieve the raw input for the chosen sample
        dataset, _ = data_provider(self.args, 'test')
        x_best, _ = dataset[best_idx]                 # (seq_len, C), (pred_len, C)
        x_best = np.asarray(x_best)                   # (seq_len, C)

        # Use the first channel for display
        channel = 0
        input_series  = x_best[:, channel]
        gt_series     = trues[best_idx, :, channel]
        bamoe_series  = bamoe_preds[best_idx, :, channel]
        sb_series     = {bt: sb_preds_dict[bt][best_idx, :, channel]
                         for bt in self.expert_types}

        save_path = os.path.join(self.out_dir, 'best_case_comparison.pdf')
        plot_best_case_comparison(
            input_series=input_series,
            gt_series=gt_series,
            bamoe_series=bamoe_series,
            sb_series=sb_series,
            expert_labels=self.expert_types,
            save_path=save_path,
            title=(f'Best-Case Forecast Comparison — {self.args.data}'
                   f'  (pred_len={self.args.pred_len}, sample #{best_idx})'),
            advantage=float(advantage[best_idx]),
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
        print('--- Figure 3: Attention Snapshots ---')
        self.run_attention_snapshots()
        print('--- Figure 4: Best-Case Comparison vs. Single-Bias Models ---')
        self.run_best_case_comparison()
        print('Interpretability analysis complete.')
