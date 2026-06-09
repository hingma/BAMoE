import numpy as np


def mse(pred, true):
    return float(np.mean((pred - true) ** 2))


def mae(pred, true):
    return float(np.mean(np.abs(pred - true)))


def crps(pred, true):
    """
    Continuous Ranked Probability Score (energy-score formulation).

    pred : (...) for a point forecast, or (M, ...) for an M-member ensemble
           where the ensemble axis is axis 0.
    true : (...)

    For a point forecast (no ensemble axis) this reduces to MAE.
    For an ensemble it computes:
        CRPS = E|Y - y| - 0.5 * E|Y - Y'|
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)

    if pred.shape == true.shape:
        # Point forecast: CRPS = MAE
        return float(np.mean(np.abs(pred - true)))

    # Ensemble forecast: pred is (M, *true.shape)
    M = pred.shape[0]
    spread = pred[np.newaxis] - pred[:, np.newaxis]   # (M, M, ...)
    energy_spread = float(np.mean(np.abs(spread))) * 0.5
    energy_error = float(np.mean(np.abs(pred - true[np.newaxis])))
    return energy_error - energy_spread


def mase(pred, true, naive_scale):
    """
    Mean Absolute Scaled Error.

    naive_scale : mean |y_t - y_{t-1}| over the training series
                  (computed by naive_forecast_scale on the training dataset).
    """
    if naive_scale == 0:
        return float('nan')
    return float(np.mean(np.abs(pred - true)) / naive_scale)


def naive_forecast_scale(train_data):
    """
    Compute the MASE denominator: mean absolute first difference of the
    training series.  train_data is a 2-D array (T, C); result is a scalar
    averaged over all channels.
    """
    diffs = np.abs(np.diff(train_data, axis=0))   # (T-1, C)
    scale = np.mean(diffs)
    return float(scale) if scale > 0 else 1.0


def metric(pred, true, naive_scale=None):
    """Return (mse, mae, crps, mase).  mase is nan when naive_scale is None."""
    mse_val  = mse(pred, true)
    mae_val  = mae(pred, true)
    crps_val = crps(pred, true)
    mase_val = mase(pred, true, naive_scale) if naive_scale is not None else float('nan')
    return mse_val, mae_val, crps_val, mase_val
