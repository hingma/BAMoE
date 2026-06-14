"""
Synthetic Data-Generating Processes (DGPs) for BAMoE Experiment 4.

Each DGP is designed so that exactly one of the four BAMoE experts is the
mathematically optimal forecaster while the others act as distractors:

  generate_trend_dgp    →  Causal / momentum expert
  generate_shock_dgp    →  Fine-grained locality / ALiBi expert
  generate_cyclical_dgp →  Periodicity expert
  generate_mackey_glass →  Global-context expert

generate_regime_switching creates a long series that alternates between
the cyclical and shock regimes to test dynamic router adaptation (Exp 4.2).
"""

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# DGP generators
# ---------------------------------------------------------------------------

def generate_trend_dgp(T, phi_c=0.9, sigma_eps=1.0, sigma_eta=0.1, seed=None):
    """
    ARIMA(1,1,0)-like random walk with AR(1) stochastic drift.

        y_t = y_{t-1} + c_t + ε_t,   ε_t ~ N(0, σ_eps²)
        c_t = φ_c * c_{t-1} + η_t,   η_t ~ N(0, σ_eta²)

    Designed for: CausalAttention (momentum / directional dependence).
    No seasonality; purely directional trend with varying velocity.
    """
    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    c = 0.0
    for t in range(1, T):
        c = phi_c * c + rng.normal(0, sigma_eta)
        y[t] = y[t - 1] + c + rng.normal(0, sigma_eps)
    return y


def generate_shock_dgp(T, phi=0.8, sigma=1.0, dof=3, seed=None):
    """
    Mean-reverting AR(1) with heavy-tailed Student-t noise.

        y_t = φ * y_{t-1} + ξ_t,   ξ_t ~ t_{dof}(0, σ²)

    Designed for: ALiBiAttention (fine-grained locality).
    Sudden unpredicted transient shocks; distant history is pure distractor.
    """
    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    for t in range(1, T):
        xi = rng.standard_t(dof) * sigma
        y[t] = phi * y[t - 1] + xi
    return y


def generate_cyclical_dgp(T, P1=24, P2=168, alpha=1.0, beta=0.5, sigma=0.1, seed=None):
    """
    Additive superposition of non-sinusoidal waveforms.

        y_t = α * Square(2π t / P1) + β * Sawtooth(2π t / P2) + ε_t

    Designed for: PeriodicAttention (long-range cyclical patterns).
    Forces the model to learn multi-period structure across distant patches.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=float)
    square   = alpha * np.sign(np.sin(2 * np.pi * t / P1))
    sawtooth = beta * 2.0 * (t / P2 - np.floor(t / P2 + 0.5))
    return square + sawtooth + rng.normal(0, sigma, T)


def generate_mackey_glass(T, tau=17, n=10, beta=0.2, gamma=0.1, seed=None):
    """
    Mackey-Glass delay-differential equation (chaotic attractor).

        dy/dt = β * y(t-τ) / (1 + y(t-τ)^n) - γ * y(t)

    Approximated via forward Euler with dt=1.

    Designed for: GlobalAttention (long-range cross-patch correlations).
    No short-range structure; requires full sequence visibility.
    """
    rng = np.random.default_rng(seed)
    burn = tau + 1
    total = T + burn
    y = np.zeros(total)
    y[:burn] = 0.9 + rng.normal(0, 0.05, burn)
    for t in range(burn, total):
        y_tau = y[t - tau]
        y[t] = y[t - 1] + beta * y_tau / (1.0 + y_tau ** n) - gamma * y[t - 1]
    return y[burn:]


def generate_regime_switching(T, cycle_len=500, P1=24, phi=0.8,
                              sigma_shock=1.0, sigma_cycle=0.1, seed=None):
    """
    Markov-switching series alternating every `cycle_len` steps between:
      - Regime 0 (cyclical)  : pure square-wave waveform (E_periodic optimal)
      - Regime 1 (shock)     : volatile AR(1) with heavy tails (E_local optimal)

    Returns
    -------
    y      : (T,) float64 series
    regime : (T,) int array, 0 = cyclical, 1 = shock
    """
    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    regime = np.zeros(T, dtype=int)

    for t in range(T):
        seg = (t // cycle_len) % 2
        regime[t] = seg
        if seg == 0:
            phase = t % cycle_len
            y[t] = np.sign(np.sin(2 * np.pi * phase / P1)) + rng.normal(0, sigma_cycle)
        else:
            prev = y[t - 1] if t > 0 else 0.0
            xi = rng.standard_t(3) * sigma_shock
            y[t] = phi * prev + xi

    return y, regime


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SyntheticDataset(Dataset):
    """
    Sliding-window dataset over a 1-D univariate synthetic series.

    Output shapes:
      x : (seq_len, 1)   — input window
      y : (pred_len, 1)  — target horizon

    Compatible with SingleBiasTransformer and BAMoE which expect (B, T, C)
    with C channels (here C=1).
    """

    def __init__(self, series: np.ndarray, seq_len: int, pred_len: int,
                 flag: str = 'train', train_ratio: float = 0.7,
                 val_ratio: float = 0.1, scale: bool = True):
        self.seq_len  = seq_len
        self.pred_len = pred_len

        T = len(series)
        n_train = int(T * train_ratio)
        n_val   = int(T * val_ratio)

        if flag == 'train':
            raw = series[:n_train]
        elif flag == 'val':
            raw = series[n_train: n_train + n_val]
        else:
            raw = series[n_train + n_val:]

        if scale:
            mu  = series[:n_train].mean()
            std = series[:n_train].std() + 1e-8
            raw = (raw - mu) / std

        self.data = raw.astype(np.float32)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len, np.newaxis]                         # (seq_len, 1)
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len, np.newaxis]  # (pred_len, 1)
        return torch.from_numpy(x), torch.from_numpy(y)
