"""
Experiment 4: Synthetic Validation for BAMoE

4.1  Controlled Isolation Test
     4×4 MSE table: each expert evaluated on each DGP, trained on that DGP's
     training split.  Diagonal (matched pair) must be lowest in each column.

4.2  Dynamic Regime Shift Test
     A regime-switching series (cyclical ↔ volatile shock) is used to show
     that isolated experts fail catastrophically when the regime flips, while
     BAMoE's dynamic router maintains low error throughout.

4.3  ALiBi Mask-Density Sensitivity Analysis
     Sweeps the ALiBi slope multiplier γ and noise level σ² on the Transient
     Shock DGP to produce an MSE-vs-γ curve that justifies the architectural
     hyperparameter choice for real-world benchmarks.
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from data_provider.synthetic import (
    SyntheticDataset,
    generate_cyclical_dgp,
    generate_mackey_glass,
    generate_regime_switching,
    generate_shock_dgp,
    generate_trend_dgp,
)
from models.BAMoE import BAMoE
from models.SingleBias import SingleBiasTransformer
from utils.metrics import mse as mse_fn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maps DGP name → (generator_fn, default_kwargs, matched_expert_type)
DGP_REGISTRY = {
    'trend':     (generate_trend_dgp,    {},                        'causal'),
    'shock':     (generate_shock_dgp,    {},                        'local'),
    'cyclical':  (generate_cyclical_dgp, {},                        'periodic'),
    'longrange': (generate_mackey_glass, {},                        'global'),
}

# Expert types in display order (must match DGP_REGISTRY key order for the
# diagonal-is-best result in Exp 4.1)
EXPERT_ORDER = ['causal', 'local', 'periodic', 'global']
DGP_ORDER    = ['trend', 'shock', 'cyclical', 'longrange']


# ---------------------------------------------------------------------------
# Lightweight training helper
# ---------------------------------------------------------------------------

class _Trainer:
    """
    Minimal train / evaluate loop for synthetic experiments.

    Avoids the full ExpForecast machinery (which requires CSV datasets) while
    keeping the same training contract: early stopping on validation loss,
    gradient clipping, best-model restoration.
    """

    def __init__(self, model, device, lr=1e-3, epochs=30, patience=5,
                 batch_size=64):
        self.model      = model.to(device)
        self.device     = device
        self.lr         = lr
        self.epochs     = epochs
        self.patience   = patience
        self.batch_size = batch_size

    def fit(self, series: np.ndarray, seq_len: int, pred_len: int,
            verbose: bool = False):
        """Train on the training split of *series* (70 %) with val-based early stopping."""
        train_ds = SyntheticDataset(series, seq_len, pred_len, 'train')
        val_ds   = SyntheticDataset(series, seq_len, pred_len, 'val')
        train_dl = DataLoader(train_ds, batch_size=self.batch_size,
                              shuffle=True,  drop_last=True,  num_workers=0)
        val_dl   = DataLoader(val_ds,   batch_size=self.batch_size * 2,
                              shuffle=False, drop_last=False, num_workers=0)

        optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.MSELoss()
        best_val  = float('inf')
        best_state: dict | None = None
        wait = 0

        for epoch in range(self.epochs):
            self.model.train()
            for x, y in train_dl:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                pred, aux, _ = self.model(x)
                loss = criterion(pred, y) + aux
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

            val_loss = self._eval_loss(val_dl, criterion)
            if verbose:
                print(f'      epoch {epoch + 1:3d}  val={val_loss:.5f}')
            if val_loss < best_val - 1e-6:
                best_val   = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, series: np.ndarray, seq_len: int, pred_len: int,
                flag: str = 'test') -> tuple[np.ndarray, np.ndarray]:
        """Return (preds, trues) numpy arrays over the requested split."""
        ds = SyntheticDataset(series, seq_len, pred_len, flag)
        dl = DataLoader(ds, batch_size=256, shuffle=False,
                        drop_last=False, num_workers=0)
        preds, trues = [], []
        self.model.eval()
        with torch.no_grad():
            for x, y in dl:
                x = x.to(self.device)
                pred, _, _ = self.model(x)
                preds.append(pred.cpu().numpy())
                trues.append(y.numpy())
        return np.concatenate(preds, 0), np.concatenate(trues, 0)

    def evaluate_mse(self, series: np.ndarray, seq_len: int, pred_len: int,
                     flag: str = 'test') -> float:
        preds, trues = self.predict(series, seq_len, pred_len, flag)
        return float(mse_fn(preds, trues))

    def _eval_loss(self, dl, criterion) -> float:
        self.model.eval()
        losses = []
        with torch.no_grad():
            for x, y in dl:
                x, y = x.to(self.device), y.to(self.device)
                pred, aux, _ = self.model(x)
                losses.append((criterion(pred, y) + aux).item())
        self.model.train()
        return float(np.mean(losses))


# ---------------------------------------------------------------------------
# Main experiment class
# ---------------------------------------------------------------------------

class ExpSynthetic:
    """Runs Experiments 4.1, 4.2, and 4.3."""

    def __init__(self, args):
        self.args = args
        self.device = self._get_device()
        self.out_dir = os.path.join(args.results, 'synthetic')
        os.makedirs(self.out_dir, exist_ok=True)
        self.T = getattr(args, 'T_synth', 5000)

    def _get_device(self):
        if getattr(self.args, 'use_gpu', True):
            if torch.cuda.is_available():
                return torch.device('cuda:0')
            if torch.backends.mps.is_available():
                return torch.device('mps')
        return torch.device('cpu')

    def _trainer(self, model) -> _Trainer:
        a = self.args
        return _Trainer(
            model, self.device,
            lr=getattr(a, 'learning_rate', 1e-3),
            epochs=getattr(a, 'train_epochs', 30),
            patience=getattr(a, 'patience', 5),
            batch_size=getattr(a, 'batch_size', 64),
        )

    # ------------------------------------------------------------------
    # Model factories
    # ------------------------------------------------------------------

    def _single_bias(self, bias_type: str, slope_mult: float = 1.0):
        """Create a fresh SingleBiasTransformer for the given bias type."""
        a = self.args
        ns = argparse.Namespace(
            model='SingleBias',
            bias_type=bias_type,
            seq_len=a.seq_len,
            pred_len=a.pred_len,
            d_model=a.d_model,
            n_heads=a.n_heads,
            n_layers=a.n_layers,
            d_ff=a.d_ff,
            dropout=a.dropout,
            patch_len=a.patch_len,
            stride=a.stride,
            local_window=getattr(a, 'local_window', 3),
            periodic_period=getattr(a, 'periodic_period', 24),
            ma_kernel=getattr(a, 'ma_kernel', 25),
            alibi_slope_mult=slope_mult,
        )
        return SingleBiasTransformer(ns)

    def _bamoe(self):
        """Create a fresh BAMoE with the 4 canonical expert types."""
        a = self.args
        ns = argparse.Namespace(
            model='BAMoE',
            expert_types='causal,local,periodic,global',
            top_k=2,
            routing='dense',
            load_balance_coef=0.01,
            local_window=getattr(a, 'local_window', 3),
            periodic_period=getattr(a, 'periodic_period', 24),
            ma_kernel=getattr(a, 'ma_kernel', 25),
            seq_len=a.seq_len,
            pred_len=a.pred_len,
            d_model=a.d_model,
            n_heads=a.n_heads,
            n_layers=a.n_layers,
            d_ff=a.d_ff,
            dropout=a.dropout,
            patch_len=a.patch_len,
            stride=a.stride,
        )
        return BAMoE(ns)

    # ------------------------------------------------------------------
    # Exp 4.1: Controlled Isolation Test
    # ------------------------------------------------------------------

    def run_exp41(self):
        """
        For each (expert, DGP) pair: train on that DGP's training split, evaluate
        on its test split.  Produces a 4×4 MSE matrix where the diagonal (matched
        expert ↔ matched DGP) must be the lowest in every column.
        """
        print('\n=== Exp 4.1: Controlled Isolation Test ===')
        a = self.args

        # Generate all four DGPs once, reuse across expert loop
        dgps = {}
        for name in DGP_ORDER:
            fn, kwargs, _ = DGP_REGISTRY[name]
            dgps[name] = fn(self.T, **kwargs, seed=42)

        mse_matrix = np.full((len(EXPERT_ORDER), len(DGP_ORDER)), np.nan)

        for ei, expert_type in enumerate(EXPERT_ORDER):
            print(f'  Expert: {expert_type}')
            for di, dgp_name in enumerate(DGP_ORDER):
                model   = self._single_bias(expert_type)
                trainer = self._trainer(model)
                trainer.fit(dgps[dgp_name], a.seq_len, a.pred_len)
                mse_val = trainer.evaluate_mse(dgps[dgp_name], a.seq_len, a.pred_len)
                mse_matrix[ei, di] = mse_val
                matched = '✓' if expert_type == DGP_REGISTRY[dgp_name][2] else ' '
                print(f'    [{matched}] {expert_type} on {dgp_name:9s}  MSE={mse_val:.4f}')

        self._save_csv_41(mse_matrix)
        self._plot_41(mse_matrix)
        return mse_matrix

    def _save_csv_41(self, matrix):
        path = os.path.join(self.out_dir, 'exp41_isolation_matrix.csv')
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['expert'] + DGP_ORDER)
            for ei, exp in enumerate(EXPERT_ORDER):
                w.writerow([exp] + [f'{matrix[ei, di]:.4f}' for di in range(len(DGP_ORDER))])
        print(f'  Saved: {path}')

    def _plot_41(self, matrix):
        path = os.path.join(self.out_dir, 'exp41_isolation_heatmap.png')

        fig, ax = plt.subplots(figsize=(7, 5))
        vmax = np.nanmax(matrix)
        im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd_r', vmin=0, vmax=vmax)

        ax.set_xticks(range(len(DGP_ORDER)))
        ax.set_yticks(range(len(EXPERT_ORDER)))
        ax.set_xticklabels(DGP_ORDER, rotation=20, ha='right', fontsize=10)
        ax.set_yticklabels(EXPERT_ORDER, fontsize=10)

        for i in range(len(EXPERT_ORDER)):
            for j in range(len(DGP_ORDER)):
                val = matrix[i, j]
                text = f'{val:.3f}'
                # Bold the matched (diagonal) cells
                fw = 'bold' if EXPERT_ORDER[i] == DGP_REGISTRY[DGP_ORDER[j]][2] else 'normal'
                ax.text(j, i, text, ha='center', va='center',
                        fontsize=9, fontweight=fw,
                        color='white' if val > vmax * 0.6 else 'black')

        ax.set_title('Exp 4.1: Isolation Test — MSE Matrix\n'
                     '(bold = matched expert ↔ DGP; ↓ lower is better)', pad=10)
        ax.set_xlabel('DGP (synthetic dataset)')
        ax.set_ylabel('Expert (isolated)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {path}')

    # ------------------------------------------------------------------
    # Exp 4.2: Dynamic Regime Shift Test
    # ------------------------------------------------------------------

    def run_exp42(self):
        """
        All models are trained on the same regime-switching series.
        They are then evaluated on three test sets:
          - pure cyclical series   (E_periodic should dominate)
          - pure shock series      (E_local should dominate)
          - regime-switching test  (BAMoE should dominate overall)
        """
        print('\n=== Exp 4.2: Dynamic Regime Shift Test ===')
        a = self.args

        # Series for training (regime-switching)
        regime_series, regime_labels = generate_regime_switching(
            self.T, cycle_len=500, seed=42
        )
        # Pure-domain evaluation series — full length so the test split (last 20%)
        # has enough points for sliding windows
        pure_cyclical = generate_cyclical_dgp(self.T, seed=99)
        pure_shock    = generate_shock_dgp(self.T, seed=99)

        models_cfg = [
            ('SingleBias-periodic', self._single_bias('periodic')),
            ('SingleBias-local',    self._single_bias('local')),
            ('SingleBias-causal',   self._single_bias('causal')),
            ('BAMoE',               self._bamoe()),
        ]

        test_series = {
            'regime_switch': regime_series,
            'pure_cyclical': pure_cyclical,
            'pure_shock':    pure_shock,
        }

        results = {}  # model_name -> {test_name: mse}

        for model_name, model in models_cfg:
            print(f'  Training {model_name}...')
            trainer = self._trainer(model)
            trainer.fit(regime_series, a.seq_len, a.pred_len)

            results[model_name] = {}
            for ts_name, ts in test_series.items():
                mse_val = trainer.evaluate_mse(ts, a.seq_len, a.pred_len)
                results[model_name][ts_name] = mse_val
                print(f'    {ts_name:18s}  MSE={mse_val:.4f}')

        self._save_csv_42(results)
        self._plot_42(results, regime_series, regime_labels)
        return results

    def _save_csv_42(self, results):
        path = os.path.join(self.out_dir, 'exp42_regime_shift.csv')
        test_names = ['regime_switch', 'pure_cyclical', 'pure_shock']
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['model'] + test_names)
            for m_name, mse_dict in results.items():
                w.writerow([m_name] + [f'{mse_dict[t]:.4f}' for t in test_names])
        print(f'  Saved: {path}')

    def _plot_42(self, results, series, regime_labels):
        path = os.path.join(self.out_dir, 'exp42_regime_comparison.png')

        model_names = list(results.keys())
        test_names  = ['regime_switch', 'pure_cyclical', 'pure_shock']
        colors      = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Left: the regime-switching series itself
        T = len(series)
        axes[0].plot(np.arange(T), series, lw=0.5, color='#333', label='y_t')
        for start in range(0, T, 1000):
            axes[0].axvspan(start + 500, min(start + 1000, T),
                            alpha=0.14, color='#e34')
        axes[0].set_title('Regime-Switching Series\n(shaded = shock regime)')
        axes[0].set_xlabel('t'); axes[0].set_ylabel('y_t')

        # Right: grouped bar chart
        x  = np.arange(len(test_names))
        w  = 0.18
        for mi, m_name in enumerate(model_names):
            vals   = [results[m_name][t] for t in test_names]
            offset = (mi - (len(model_names) - 1) / 2) * w
            bars = axes[1].bar(x + offset, vals, width=w,
                               color=colors[mi % len(colors)], label=m_name)
            for bar, v in zip(bars, vals):
                axes[1].text(bar.get_x() + bar.get_width() / 2,
                             bar.get_height() + 0.002,
                             f'{v:.3f}', ha='center', va='bottom', fontsize=7)

        axes[1].set_xticks(x)
        axes[1].set_xticklabels(['Regime-Switch\n(overall)',
                                  'Pure Cyclical\n(regime 0)',
                                  'Pure Shock\n(regime 1)'])
        axes[1].set_ylabel('Test MSE')
        axes[1].set_title('Exp 4.2: Regime Shift — Per-Domain MSE')
        axes[1].legend(fontsize=8)

        plt.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {path}')

    # ------------------------------------------------------------------
    # Exp 4.3: ALiBi Mask-Density Sensitivity Analysis
    # ------------------------------------------------------------------

    def run_exp43(self):
        """
        Fixes the Transient Shock DGP and sweeps:
          - slope_multiplier γ ∈ slope_multipliers   (x-axis)
          - noise level σ ∈ noise_levels             (separate curves)

        Expected outcome: U-shaped MSE curves; optimal γ shifts right as σ
        increases (noisier signals need tighter locality constraints).
        """
        print('\n=== Exp 4.3: ALiBi Mask-Density Sensitivity Analysis ===')
        a = self.args

        slope_mults  = getattr(a, 'slope_multipliers', [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
        noise_levels = getattr(a, 'noise_levels',      [0.5, 1.0, 2.0])

        results = {}  # (slope, sigma) -> mse

        for sigma in noise_levels:
            series = generate_shock_dgp(self.T, sigma=sigma, seed=42)
            for slope in slope_mults:
                print(f'  σ={sigma:.1f}  γ={slope:.2f}', end='  ')
                model   = self._single_bias('local', slope_mult=slope)
                trainer = self._trainer(model)
                trainer.fit(series, a.seq_len, a.pred_len)
                mse_val = trainer.evaluate_mse(series, a.seq_len, a.pred_len)
                results[(slope, sigma)] = mse_val
                print(f'MSE={mse_val:.4f}')

        self._save_csv_43(results, slope_mults, noise_levels)
        self._plot_43(results, slope_mults, noise_levels)
        return results

    def _save_csv_43(self, results, slope_mults, noise_levels):
        path = os.path.join(self.out_dir, 'exp43_slope_sensitivity.csv')
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['slope_multiplier', 'noise_sigma', 'mse'])
            for sigma in noise_levels:
                for slope in slope_mults:
                    w.writerow([slope, sigma, f'{results[(slope, sigma)]:.4f}'])
        print(f'  Saved: {path}')

    def _plot_43(self, results, slope_mults, noise_levels):
        path = os.path.join(self.out_dir, 'exp43_slope_sensitivity.png')

        fig, ax = plt.subplots(figsize=(8, 5))
        palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for ci, sigma in enumerate(noise_levels):
            mses = [results[(s, sigma)] for s in slope_mults]
            ax.plot(slope_mults, mses, marker='o',
                    label=f'σ={sigma:.1f}',
                    color=palette[ci % len(palette)])
            # Mark the optimal slope
            best_idx = int(np.argmin(mses))
            ax.annotate(
                f'γ*={slope_mults[best_idx]:.1f}',
                xy=(slope_mults[best_idx], mses[best_idx]),
                xytext=(0, -18), textcoords='offset points',
                ha='center', fontsize=8, color=palette[ci % len(palette)],
                arrowprops=dict(arrowstyle='->', color=palette[ci % len(palette)], lw=0.9),
            )

        ax.set_xscale('log')
        ax.set_xlabel('ALiBi Slope Multiplier γ  (log scale)')
        ax.set_ylabel('Test MSE  (Transient Shock DGP)')
        ax.set_title('Exp 4.3: Mask-Density Sensitivity Analysis\n'
                     '(optimal γ* shifts right as noise σ increases)')
        ax.legend(title='Noise level σ')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {path}')
