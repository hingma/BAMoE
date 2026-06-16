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

Figure 5 — KL-Divergence Structural Break Detection
  Compute D_KL(G_t || G_{t+1}) between consecutive patch routing distributions
  on the structural-break window and correlate with the CUSUM break indicator.
  A statistically significant Pearson r confirms context-aware regime discovery.

Figure 6 — Multinomial Logistic Regression Econometric Mapping
  Regress the dominant expert choice on per-sample rolling volatility (σ),
  trend strength (τ), and seasonality (γ).  Reports β, SE, and p-values as a
  publication-ready coefficient table for each expert vs. the baseline class.

Figure 7 — Mutual Information and Entropy Disentanglement
  Compute the temporal routing entropy H(G_t) over the full test set and the
  MI matrix I(G_k; Z) for Z ∈ {volatility, trend, seasonality}.  A diagonal-
  dominant MI matrix is mathematical proof of expert specialisation.
"""

import csv
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
    plot_kl_divergence_analysis,
    plot_logistic_regression_table,
    plot_mi_entropy_analysis,
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
    # Figure 5 — KL-Divergence Structural Break Detection
    # ------------------------------------------------------------------

    def _compute_kl_sequence(self, routing_seq):
        """D_KL(G_t || G_{t+1}) for consecutive patch routing distributions."""
        eps = 1e-9
        p = np.clip(routing_seq[:-1], eps, 1.0)
        q = np.clip(routing_seq[1:],  eps, 1.0)
        p = p / p.sum(axis=1, keepdims=True)
        q = q / q.sum(axis=1, keepdims=True)
        return (p * np.log(p / q)).sum(axis=1)   # (N-1,)

    def run_kl_divergence_analysis(self):
        """
        Figure 5 — KL-Divergence Structural Break Detection.

        Computes D_KL(G_t || G_{t+1}) across the CUSUM break window and reports
        the Pearson correlation between KL spikes and the CUSUM break indicator.
        """
        try:
            from scipy import stats as scipy_stats
        except ImportError:
            print('Figure 5 skipped: scipy is required.')
            return

        raw_data = self._load_raw_test_series()
        win_start, break_idx = self._find_break_window(raw_data)
        window = raw_data[win_start:win_start + self.args.seq_len, :]

        routing_seq = self._infer_routing(window)
        if routing_seq is None:
            print('Figure 5 skipped: no routing logits.')
            return

        kl_seq   = self._compute_kl_sequence(routing_seq)         # (N-1,)
        centers  = self._patch_centers(routing_seq.shape[0])       # (N,)
        kl_times = (centers[:-1] + centers[1:]) / 2.0             # (N-1,)

        series       = window[:, 0]
        cusum        = np.cumsum(series - series.mean())
        cusum_delta  = np.abs(np.diff(cusum))                      # (seq_len-1,)

        # Aggregate |Δcusum| over each inter-patch gap for correlation
        gap_break_score = np.zeros(len(kl_seq))
        for i in range(len(kl_seq)):
            t0 = max(0, int(centers[i]))
            t1 = min(len(cusum_delta), int(centers[i + 1]))
            if t0 < t1:
                gap_break_score[i] = cusum_delta[t0:t1].max()

        r, p_val = scipy_stats.pearsonr(kl_seq, gap_break_score)

        save_path = os.path.join(self.out_dir, 'kl_divergence_analysis.pdf')
        plot_kl_divergence_analysis(
            series=series,
            cusum=cusum,
            kl_seq=kl_seq,
            kl_timesteps=kl_times,
            break_idx=break_idx,
            r=r,
            p_val=p_val,
            save_path=save_path,
            title=(f'KL-Divergence Structural Break Analysis — {self.args.data}'
                   f'  (pred_len={self.args.pred_len})'),
        )
        print(f'Saved: {save_path}')
        print(f'  KL–CUSUM Pearson r={r:.4f}, p={p_val:.4e}')

        # Save KL sequence
        csv_seq = os.path.join(self.out_dir, 'kl_divergence_sequence.csv')
        with open(csv_seq, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['timestep', 'kl_divergence', 'cusum_break_score'])
            for t, kl, gb in zip(kl_times, kl_seq, gap_break_score):
                w.writerow([f'{t:.2f}', f'{kl:.8f}', f'{gb:.8f}'])
        print(f'Saved: {csv_seq}')

        # Save summary statistics
        csv_sum = os.path.join(self.out_dir, 'kl_divergence_summary.csv')
        peak_i  = int(np.argmax(kl_seq))
        with open(csv_sum, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['metric', 'value'])
            w.writerow(['pearson_r',        f'{r:.8f}'])
            w.writerow(['p_value',          f'{p_val:.8e}'])
            w.writerow(['break_idx',        break_idx])
            w.writerow(['kl_peak_timestep', f'{kl_times[peak_i]:.2f}'])
            w.writerow(['kl_peak_value',    f'{kl_seq[peak_i]:.8f}'])
            w.writerow(['kl_mean',          f'{kl_seq.mean():.8f}'])
            w.writerow(['kl_std',           f'{kl_seq.std():.8f}'])
        print(f'Saved: {csv_sum}')

    # ------------------------------------------------------------------
    # Figure 6 — Multinomial Logistic Regression Mapping
    # ------------------------------------------------------------------

    def _compute_sample_features(self, x_np):
        """
        Compute (volatility σ, trend τ, seasonality γ) for a (seq_len, C)
        window using channel 0.

        Returns
        -------
        sigma : float — rolling std of the full window
        tau   : float — OLS slope (directional velocity)
        gamma : float — autocorrelation at the primary seasonal lag
        """
        sig = x_np[:, 0].astype(np.float64)
        n   = len(sig)

        sigma = float(sig.std())
        tau   = float(np.polyfit(np.arange(n, dtype=np.float64), sig, 1)[0])

        freq    = getattr(self.args, 'freq', 'h').lower().rstrip('t')
        lag_map = {'h': 24, 'm': 60, 'd': 7, 'w': 4, 'b': 5, 's': 12}
        lag     = min(lag_map.get(freq, 24), n // 4)
        if lag > 0 and n > lag:
            gamma = float(np.corrcoef(sig[:-lag], sig[lag:])[0, 1])
            gamma = 0.0 if np.isnan(gamma) else gamma
        else:
            gamma = 0.0

        return sigma, tau, gamma

    def _fit_mnlogit(self, X, y, n_experts):
        """
        Fit a Multinomial Logistic Regression and return a coefficient table.

        Tries statsmodels (gives SE + p-values); falls back to sklearn
        (coefficients only).

        Returns
        -------
        list of dicts: {expert, coef:[intercept,f1,f2,...], se, pval}
        statsmodels convention: base class = last numeric label (n_experts-1).
        """
        n_feat   = X.shape[1]
        nan_row  = lambda: [np.nan] * (n_feat + 1)
        zero_row = lambda: [0.0]    * (n_feat + 1)

        try:
            import statsmodels.api as sm
            X_sm = sm.add_constant(X, has_constant='add')
            res  = sm.MNLogit(y, X_sm).fit(method='bfgs', maxiter=300, disp=False)
            # params/bse/pvalues: (n_feat+1, n_experts-1)
            # Column k → log-odds of class k vs base class (n_experts-1)
            table = [{
                'expert': self.expert_types[-1] + ' (base)',
                'coef': zero_row(), 'se': nan_row(), 'pval': nan_row(),
            }]
            for k in range(n_experts - 1):
                table.append({
                    'expert': self.expert_types[k],
                    'coef':   res.params[:, k].tolist(),
                    'se':     res.bse[:, k].tolist(),
                    'pval':   res.pvalues[:, k].tolist(),
                })
            return table

        except Exception as e_sm:
            print(f'  statsmodels failed ({e_sm}); using sklearn (no p-values).')
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(
                multi_class='multinomial', solver='lbfgs', max_iter=500, C=1e4
            )
            clf.fit(X, y)
            table = []
            for k in range(n_experts):
                coef_k = [float(clf.intercept_[k])] + clf.coef_[k].tolist()
                table.append({
                    'expert': self.expert_types[k],
                    'coef': coef_k, 'se': nan_row(), 'pval': nan_row(),
                })
            return table

    def run_logistic_regression_mapping(self):
        """
        Figure 6 — Multinomial Logistic Regression Econometric Mapping.

        Features : per-sample rolling volatility σ, trend strength τ,
                   seasonality γ (all standardised to μ=0, σ=1).
        Target   : argmax expert (dominant routing weight per test sample).
        Reports  : β, SE, p-values for each expert vs. the baseline class.
        """
        _, loader = data_provider(self.args, 'test')

        feature_rows, dominant_experts = [], []
        n_experts = len(self.expert_types)

        self.model.eval()
        with torch.no_grad():
            for x, _ in loader:
                x_dev = x.to(self.device)
                _, _, routing_log = self.model(x_dev, return_routing=True)
                if not routing_log:
                    continue
                logits = routing_log[-1]
                probs  = F.softmax(logits, dim=-1).cpu().numpy()
                B, C   = x.shape[0], x.shape[2]
                mean_probs = probs.reshape(B, C, -1, n_experts).mean(axis=(1, 2))
                x_np = x.numpy()
                for b in range(B):
                    feature_rows.append(list(self._compute_sample_features(x_np[b])))
                    dominant_experts.append(int(mean_probs[b].argmax()))

        if len(feature_rows) < max(10, n_experts * 3):
            print('Figure 6 skipped: insufficient test samples for regression.')
            return

        X = np.array(feature_rows,    dtype=np.float64)
        y = np.array(dominant_experts, dtype=int)

        if len(np.unique(y)) < 2:
            print('Figure 6 skipped: only one dominant expert class observed.')
            return

        X_mu, X_sd = X.mean(axis=0), X.std(axis=0)
        X_sd[X_sd < 1e-12] = 1.0
        X_norm = (X - X_mu) / X_sd

        coef_table = self._fit_mnlogit(X_norm, y, n_experts)

        save_path = os.path.join(self.out_dir, 'logistic_regression_mapping.pdf')
        plot_logistic_regression_table(
            coef_table=coef_table,
            expert_labels=self.expert_types,
            feature_names=['Volatility (σ)', 'Trend (τ)', 'Seasonality (γ)'],
            save_path=save_path,
            title=(f'Multinomial Logistic Regression — {self.args.data}'
                   f'  (pred_len={self.args.pred_len}, N={len(y)})'),
        )
        print(f'Saved: {save_path}')

        # Save coefficient table in long (tidy) format
        feat_full = ['Intercept', 'Volatility (σ)', 'Trend (τ)', 'Seasonality (γ)']
        csv_coef  = os.path.join(self.out_dir, 'logistic_regression_coefficients.csv')
        with open(csv_coef, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['expert', 'feature', 'coef', 'se', 'pval', 'significance'])
            for entry in coef_table:
                for feat, c, s, p in zip(feat_full,
                                          entry['coef'], entry['se'], entry['pval']):
                    if np.isnan(p):
                        sig, p_str, s_str = 'N/A', 'N/A', 'N/A'
                    else:
                        sig   = ('***' if p < 0.001 else
                                 '**'  if p < 0.01  else
                                 '*'   if p < 0.05  else 'n.s.')
                        p_str = f'{p:.8e}'
                        s_str = f'{s:.8f}' if not np.isnan(s) else 'N/A'
                    w.writerow([entry['expert'], feat, f'{c:.8f}', s_str, p_str, sig])
        print(f'Saved: {csv_coef}')

    # ------------------------------------------------------------------
    # Figure 7 — Mutual Information & Entropy Disentanglement
    # ------------------------------------------------------------------

    def run_mi_entropy_analysis(self):
        """
        Figure 7 — Mutual Information and Entropy Disentanglement.

        Computes:
          • Per-sample routing entropy H(G_t) = −Σ G_k log₂ G_k
          • MI matrix I(G_k; Z) for Z ∈ {volatility, trend, seasonality}
        """
        try:
            from sklearn.feature_selection import mutual_info_regression
        except ImportError:
            print('Figure 7 skipped: scikit-learn is required.')
            return

        _, loader = data_provider(self.args, 'test')

        routing_vecs, feature_rows = [], []
        n_experts = len(self.expert_types)

        self.model.eval()
        with torch.no_grad():
            for x, _ in loader:
                x_dev = x.to(self.device)
                _, _, routing_log = self.model(x_dev, return_routing=True)
                if not routing_log:
                    continue
                logits = routing_log[-1]
                probs  = F.softmax(logits, dim=-1).cpu().numpy()
                B, C   = x.shape[0], x.shape[2]
                mean_probs = probs.reshape(B, C, -1, n_experts).mean(axis=(1, 2))
                routing_vecs.append(mean_probs)
                x_np = x.numpy()
                for b in range(B):
                    feature_rows.append(list(self._compute_sample_features(x_np[b])))

        if not routing_vecs:
            print('Figure 7 skipped: no routing data collected.')
            return

        routing_vecs = np.concatenate(routing_vecs, axis=0)  # (N, n_experts)
        features     = np.array(feature_rows, dtype=np.float64)   # (N, 3)

        eps      = 1e-9
        entropy  = -(routing_vecs * np.log2(np.clip(routing_vecs, eps, 1.0))).sum(axis=1)
        max_ent  = np.log2(n_experts)

        mi_matrix = np.zeros((n_experts, 3))
        for k in range(n_experts):
            mi_matrix[k] = mutual_info_regression(
                features, routing_vecs[:, k], random_state=42, n_neighbors=5
            )

        save_path = os.path.join(self.out_dir, 'mi_entropy_analysis.pdf')
        plot_mi_entropy_analysis(
            entropy=entropy,
            max_entropy=max_ent,
            mi_matrix=mi_matrix,
            expert_labels=self.expert_types,
            feature_names=['Volatility (σ)', 'Trend (τ)', 'Seasonality (γ)'],
            save_path=save_path,
            title=(f'MI & Entropy Disentanglement — {self.args.data}'
                   f'  (pred_len={self.args.pred_len})'),
        )
        print(f'Saved: {save_path}')
        print(f'  Mean H(G_t) = {entropy.mean():.4f} bits  '
              f'(max = {max_ent:.4f}, utilisation = {entropy.mean() / max_ent:.1%})')

        feat_names = ['Volatility (σ)', 'Trend (τ)', 'Seasonality (γ)']

        # Per-sample entropy
        csv_ent = os.path.join(self.out_dir, 'routing_entropy_samples.csv')
        with open(csv_ent, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['sample_idx', 'entropy_bits'])
            for i, h in enumerate(entropy):
                w.writerow([i, f'{h:.8f}'])
        print(f'Saved: {csv_ent}')

        # MI matrix
        csv_mi = os.path.join(self.out_dir, 'mi_matrix.csv')
        with open(csv_mi, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['expert'] + feat_names)
            for k, et in enumerate(self.expert_types):
                w.writerow([et] + [f'{v:.8f}' for v in mi_matrix[k]])
        print(f'Saved: {csv_mi}')

        # Summary stats
        csv_sum = os.path.join(self.out_dir, 'mi_entropy_summary.csv')
        with open(csv_sum, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['metric', 'value'])
            w.writerow(['mean_entropy_bits',   f'{entropy.mean():.8f}'])
            w.writerow(['std_entropy_bits',    f'{entropy.std():.8f}'])
            w.writerow(['median_entropy_bits', f'{np.median(entropy):.8f}'])
            w.writerow(['max_entropy_bits',    f'{max_ent:.8f}'])
            w.writerow(['entropy_utilisation', f'{entropy.mean() / max_ent:.8f}'])
            for k, et in enumerate(self.expert_types):
                for j, fn in enumerate(['volatility', 'trend', 'seasonality']):
                    w.writerow([f'MI_{et}_{fn}', f'{mi_matrix[k, j]:.8f}'])
        print(f'Saved: {csv_sum}')

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
        print('--- Figure 5: KL-Divergence Structural Break Detection ---')
        self.run_kl_divergence_analysis()
        print('--- Figure 6: Multinomial Logistic Regression Mapping ---')
        self.run_logistic_regression_mapping()
        print('--- Figure 7: Mutual Information & Entropy Disentanglement ---')
        self.run_mi_entropy_analysis()
        print('Interpretability analysis complete.')
