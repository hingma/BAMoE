"""
utils/stat_tests.py
-------------------
Statistical significance tests for comparing forecast models.

Functions
---------
newey_west_var     : Newey-West HAC long-run variance estimator.
dm_test            : Diebold-Mariano test with Harvey-Leybourne-Newbold (1997)
                     small-sample correction.
mcs                : Model Confidence Set (Hansen, Lunde & Nason 2011)
                     via circular-block bootstrap and the TR range statistic.
per_sample_losses  : Helper — reduce (N, H, C) preds/trues to (N,) loss series.
"""

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Long-run variance
# ---------------------------------------------------------------------------

def newey_west_var(x: np.ndarray, max_lag: int | None = None) -> float:
    """
    Newey-West (Bartlett-kernel) heteroskedasticity- and autocorrelation-
    consistent (HAC) long-run variance of a 1-D series x.

    max_lag : number of lags; defaults to 4*(T/100)^(2/9) (Newey-West rule).
    """
    x = np.asarray(x, dtype=float)
    T = len(x)
    if max_lag is None:
        max_lag = max(1, int(np.floor(4 * (T / 100) ** (2 / 9))))

    xc = x - x.mean()
    gamma0 = np.dot(xc, xc) / T
    lrv = gamma0
    for j in range(1, max_lag + 1):
        w = 1.0 - j / (max_lag + 1)          # Bartlett weight
        gamma_j = np.dot(xc[j:], xc[:-j]) / T
        lrv += 2.0 * w * gamma_j
    return float(lrv)


# ---------------------------------------------------------------------------
# Diebold-Mariano test
# ---------------------------------------------------------------------------

def dm_test(
    losses_a: np.ndarray,
    losses_b: np.ndarray,
    h: int = 1,
    alpha: float = 0.05,
) -> dict:
    """
    Diebold-Mariano test with Harvey-Leybourne-Newbold (1997) correction.

    H0 : models A and B have equal predictive accuracy  (E[d_t] = 0)
    where d_t = loss_a_t – loss_b_t.
    A positive DM stat means A is *worse* than B (A has higher average loss).

    Parameters
    ----------
    losses_a, losses_b : (T,) per-sample scalar losses (e.g., MSE per window).
    h     : forecast horizon (# steps ahead).  For aggregated multi-step errors
            h=1 is conservative; h=pred_len is the theoretical choice.
    alpha : significance level for the reject flags.

    Returns
    -------
    dict with keys:
        dm_stat      — HLN-corrected test statistic (t-distributed under H0)
        p_two        — two-sided p-value
        p_one        — one-sided p-value: P(B better than A) = P(d > 0)
        reject_two   — bool, two-sided rejection at `alpha`
        reject_one   — bool, one-sided rejection at `alpha`
        d_bar        — mean loss difference (A – B)
    """
    losses_a = np.asarray(losses_a, dtype=float)
    losses_b = np.asarray(losses_b, dtype=float)
    if losses_a.shape != losses_b.shape or losses_a.ndim != 1:
        raise ValueError("losses_a and losses_b must be 1-D arrays of equal length.")

    d = losses_a - losses_b
    T = len(d)
    d_bar = d.mean()

    # Newey-West LRV with lag truncation = h-1 (covers MA(h-1) autocorrelation)
    lrv = newey_west_var(d, max_lag=max(h - 1, 1))
    lrv = max(lrv, 1e-15)

    dm_raw = d_bar / np.sqrt(lrv / T)

    # HLN small-sample correction
    hlnc = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_stat = dm_raw * hlnc

    # t(T-1) reference distribution
    p_two = float(2 * stats.t.sf(abs(dm_stat), df=T - 1))
    p_one = float(stats.t.sf(dm_stat, df=T - 1))   # P(B better than A)

    return {
        'dm_stat':    float(dm_stat),
        'p_two':      p_two,
        'p_one':      p_one,
        'reject_two': p_two < alpha,
        'reject_one': p_one < alpha,
        'd_bar':      float(d_bar),
    }


# ---------------------------------------------------------------------------
# Model Confidence Set
# ---------------------------------------------------------------------------

def mcs(
    losses_dict: dict,
    alpha: float = 0.05,
    n_boot: int = 1000,
    seed: int = 42,
) -> list:
    """
    Model Confidence Set (Hansen, Lunde & Nason 2011).

    Iteratively eliminates models whose mean loss is significantly higher than
    the best model, using the TR (range) statistic and a circular-block
    bootstrap for critical values.  The surviving set M̂* is returned.

    Parameters
    ----------
    losses_dict : {model_name: (T,) array of per-sample scalar losses}
    alpha       : MCS confidence level (e.g., 0.10 → 90% MCS)
    n_boot      : number of bootstrap replications
    seed        : RNG seed for reproducibility

    Returns
    -------
    List of model names that belong to the MCS at level alpha.
    """
    rng = np.random.default_rng(seed)
    models = list(losses_dict.keys())
    M = len(models)
    if M == 0:
        return []
    if M == 1:
        return models[:]

    # L : (T, M) loss matrix
    L = np.stack([np.asarray(losses_dict[m], dtype=float) for m in models], axis=1)
    T = L.shape[0]
    block_size = max(1, int(np.ceil(T ** (1 / 3))))

    surviving = list(range(M))

    while len(surviving) > 1:
        m = len(surviving)
        Ls = L[:, surviving]                            # (T, m)

        # Pairwise loss-difference series: D[t, i, j] = l_{i,t} - l_{j,t}
        D = Ls[:, :, None] - Ls[:, None, :]            # (T, m, m)
        d_bar = D.mean(axis=0)                          # (m, m)

        # Circular-block bootstrap to estimate Var(d̄_{ij})
        boot_d_bars = np.empty((n_boot, m, m))
        for b in range(n_boot):
            n_blocks = int(np.ceil(T / block_size))
            starts = rng.integers(0, T, size=n_blocks)
            idx = np.concatenate(
                [np.arange(s, s + block_size) % T for s in starts]
            )[:T]
            boot_d_bars[b] = D[idx].mean(axis=0)

        # Variance of d̄_{ij} across bootstrap replications
        var_d_bar = boot_d_bars.var(axis=0, ddof=1)    # (m, m)
        var_d_bar = np.maximum(var_d_bar, 1e-15)

        # TR statistic on original sample
        t_mat = d_bar / np.sqrt(var_d_bar)
        np.fill_diagonal(t_mat, 0.0)
        tr_stat = float(np.max(np.abs(t_mat)))

        # Bootstrap TR statistics (centred on d_bar)
        boot_tr = np.max(
            np.abs((boot_d_bars - d_bar) / np.sqrt(var_d_bar)).reshape(n_boot, -1),
            axis=1,
        )
        c_alpha = float(np.quantile(boot_tr, 1.0 - alpha))

        if tr_stat <= c_alpha:
            break   # All remaining models in M̂*

        # Eliminate model with the highest average relative loss: d_{i·}
        d_mean = d_bar.mean(axis=1)                     # (m,)
        worst_local = int(np.argmax(d_mean))
        surviving.pop(worst_local)

    return [models[i] for i in surviving]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def per_sample_losses(
    preds: np.ndarray,
    trues: np.ndarray,
    loss: str = 'mse',
) -> np.ndarray:
    """
    Reduce (N, H, C) prediction and truth arrays to a (N,) loss series
    suitable for DM / MCS tests.

    loss : 'mse' | 'mae'
    """
    preds = np.asarray(preds, dtype=float)
    trues = np.asarray(trues, dtype=float)
    err = preds - trues
    if loss == 'mse':
        return np.mean(err ** 2, axis=tuple(range(1, err.ndim)))
    elif loss == 'mae':
        return np.mean(np.abs(err), axis=tuple(range(1, err.ndim)))
    else:
        raise ValueError(f"Unknown loss type: {loss!r}. Choose 'mse' or 'mae'.")
