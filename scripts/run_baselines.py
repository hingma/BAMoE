#!/usr/bin/env python
"""
scripts/run_baselines.py
------------------------
Train / evaluate baseline model families and write per-sample loss CSVs
compatible with scripts/compute_significance.py.

Model families
--------------
  stat   : Auto-ARIMA, AutoETS, AutoTheta     (statsforecast)
  dl     : PatchTST, iTransformer, TimeMixer  (neuralforecast)
  timemoe: TimeMoE-50M                        (HuggingFace, zero-shot)

Output per model
----------------
  results/<exp_name>/losses.csv   (columns: sample_idx, mse, mae)
  results/summary.csv             (aggregate metrics, appended)

Install prerequisites (once)
----------------------------
  pip install statsforecast neuralforecast
  # transformers is already present

Usage
-----
  # All models, ETTh1, pred_len=96:
  python scripts/run_baselines.py --data ETTh1 --pred_len 96

  # Single family:
  python scripts/run_baselines.py --data ETTh1 --pred_len 96 --family dl

  # Single model:
  python scripts/run_baselines.py --data ETTh1 --pred_len 96 --models AutoARIMA

  # Limit stat-model rolling windows (faster):
  python scripts/run_baselines.py --data ETTh1 --pred_len 96 --family stat --n_stat_windows 200
"""

import sys
import os
import csv
import argparse
import warnings
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider.data_factory import DATA_MAP
from utils.metrics import naive_forecast_scale

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _build_test_windows(dataset_cls, data_path, root_path, seq_len, pred_len, features, target):
    """
    Return (X, Y, train_data_x) using the same scalers / splits as BAMoE.
    X : (N, seq_len, C)  –– context windows
    Y : (N, pred_len, C) –– targets
    train_data_x : (T_train, C)  –– for MASE denominator
    """
    from torch.utils.data import DataLoader as _DL

    def _ds(flag):
        return dataset_cls(
            root_path=root_path, flag=flag,
            size=[seq_len, pred_len],
            features=features, data_path=data_path,
            target=target, scale=True,
        )

    train_ds = _ds('train')
    test_ds  = _ds('test')

    N = len(test_ds)
    X = np.stack([test_ds.data_x[i: i + seq_len]               for i in range(N)])
    Y = np.stack([test_ds.data_y[i + seq_len: i + seq_len + pred_len] for i in range(N)])
    return X, Y, train_ds.data_x


def _save(exp_name, preds, trues, train_data_x, results_dir):
    """
    preds, trues : (N, pred_len, C)
    Writes losses.csv and appends to summary.csv.
    """
    from utils.metrics import metric, naive_forecast_scale
    out_dir = os.path.join(results_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    err = preds - trues
    per_mse = np.mean(err ** 2,     axis=tuple(range(1, err.ndim)))   # (N,)
    per_mae = np.mean(np.abs(err),  axis=tuple(range(1, err.ndim)))   # (N,)

    with open(os.path.join(out_dir, 'losses.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample_idx', 'mse', 'mae'])
        for i, (m, a) in enumerate(zip(per_mse, per_mae)):
            w.writerow([i, f'{m:.8f}', f'{a:.8f}'])

    scale = naive_forecast_scale(train_data_x)
    mse_v, mae_v, crps_v, mase_v = metric(preds, trues, naive_scale=scale)
    print(f'  MSE={mse_v:.4f}  MAE={mae_v:.4f}  CRPS={crps_v:.4f}  MASE={mase_v:.4f}')

    csv_path = os.path.join(results_dir, 'summary.csv')
    write_hdr = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        if write_hdr:
            w.writerow(['exp_name', 'model', 'data', 'pred_len', 'mse', 'mae', 'crps', 'mase'])
        parts = exp_name.split('_')
        model_tag = parts[0] if parts else exp_name
        data_tag  = parts[1] if len(parts) > 1 else ''
        w.writerow([exp_name, model_tag, data_tag,
                    f'{preds.shape[1]}',
                    f'{mse_v:.4f}', f'{mae_v:.4f}', f'{crps_v:.4f}', f'{mase_v:.4f}'])
    print(f'  Saved → {out_dir}/losses.csv')


def _exp_name(model_name, data, seq_len, pred_len):
    return f'{model_name}_{data}_sl{seq_len}_pl{pred_len}'


# ---------------------------------------------------------------------------
# 1. Statistical family  (statsforecast)
# ---------------------------------------------------------------------------

STAT_MODELS = ['AutoARIMA', 'AutoETS', 'AutoTheta']


def _to_long_df(data_x, freq_offset=0):
    """
    data_x : (T, C) numpy array → long-format DataFrame for statsforecast.
    unique_id = 'ch_{c}', ds = integer index starting from freq_offset.
    """
    T, C = data_x.shape
    rows = {
        'unique_id': np.repeat([f'ch_{c}' for c in range(C)], T),
        'ds':        np.tile(np.arange(freq_offset, freq_offset + T), C),
        'y':         data_x.T.ravel(),
    }
    return pd.DataFrame(rows)


def run_stat(model_name, X, Y, train_data_x, data, seq_len, pred_len,
             results_dir, n_windows_limit):
    """
    Channel-independent rolling forecast using statsforecast.
    Uses cross_validation with expanding window (refit each fold).
    n_windows_limit : cap number of test windows for speed (None = all).
    """
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA, AutoETS, AutoTheta
    except ImportError:
        print('[ERROR] statsforecast not installed. Run: pip install statsforecast')
        return

    model_cls = {'AutoARIMA': AutoARIMA, 'AutoETS': AutoETS, 'AutoTheta': AutoTheta}[model_name]

    N, H, C = Y.shape

    # Determine which test windows to evaluate
    if n_windows_limit and n_windows_limit < N:
        # Evenly spaced sample across the test period
        eval_idx = np.linspace(0, N - 1, n_windows_limit, dtype=int)
        print(f'  [stat] Evaluating {n_windows_limit}/{N} windows (evenly spaced)')
    else:
        eval_idx = np.arange(N)

    all_preds = np.full((N, H, C), np.nan)
    t0 = time.time()

    # Build long DataFrame: train series + test context up to each eval window.
    # For speed: train once on train_data_x, then predict from test context windows.
    # (Full cross_validation with N_test refits is prohibitively slow for large datasets.)
    for c in range(C):
        train_series = train_data_x[:, c]

        # Build DataFrame per channel: train portion only (fit base model once)
        df_train = pd.DataFrame({
            'unique_id': 'series',
            'ds': np.arange(len(train_series)),
            'y': train_series,
        })
        sf = StatsForecast(models=[model_cls()], freq=1, n_jobs=1, verbose=False)

        for n in eval_idx:
            # Provide seq_len context from the test window as "new history"
            context = X[n, :, c]
            history = np.concatenate([train_series, context])
            df_ctx = pd.DataFrame({
                'unique_id': 'series',
                'ds': np.arange(len(history)),
                'y': history,
            })
            try:
                sf.fit(df_ctx)
                fc = sf.predict(h=pred_len)['mean'].values  # (H,)
            except Exception as e:
                # Fall back to last value if model fails (e.g. constant series)
                fc = np.full(H, context[-1])
            all_preds[n, :, c] = fc

        elapsed = time.time() - t0
        print(f'    channel {c+1}/{C} done  ({elapsed:.1f}s)', end='\r')

    print()

    # Only keep evaluated windows (drop nan rows)
    mask = ~np.isnan(all_preds[:, 0, 0])
    preds_eval = all_preds[mask]
    trues_eval = Y[mask]

    exp = _exp_name(model_name, data, seq_len, pred_len)
    _save(exp, preds_eval, trues_eval, train_data_x, results_dir)


# ---------------------------------------------------------------------------
# 2. DL family  (neuralforecast)
# ---------------------------------------------------------------------------

DL_MODELS = ['PatchTST', 'iTransformer', 'TimeMixer']


def _to_nf_df(data_x, n_channels, start_dt=None):
    """
    data_x : (T, C) → neuralforecast long-format DataFrame.
    ds uses hourly DatetimeIndex starting at start_dt (or 2020-01-01 if None).
    """
    T, C = data_x.shape
    if start_dt is None:
        start_dt = pd.Timestamp('2020-01-01')
    dates = pd.date_range(start_dt, periods=T, freq='h')

    frames = []
    for c in range(C):
        frames.append(pd.DataFrame({
            'unique_id': f'ch_{c}',
            'ds': dates,
            'y': data_x[:, c].astype(np.float32),
        }))
    return pd.concat(frames, ignore_index=True)


def _nf_cv_to_array(cv_df, C, H, N):
    """
    Reshape neuralforecast cross_validation output → (N, H, C).
    cv_df has columns: unique_id, ds, cutoff, <model_name>
    We use the last model-name column as predictions.
    """
    model_col = [c for c in cv_df.columns if c not in ('unique_id', 'ds', 'cutoff', 'y')][-1]
    preds = np.zeros((N, H, C), dtype=np.float32)
    for c_idx in range(C):
        ch = cv_df[cv_df['unique_id'] == f'ch_{c_idx}'][model_col].values
        # ch has shape (N * H,) — each cutoff has H rows
        preds[:, :, c_idx] = ch.reshape(N, H)
    return preds


def run_dl(model_name, X, Y, train_data_x, data, seq_len, pred_len,
           results_dir, batch_size, max_steps, device_str):
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import PatchTST, iTransformer, TimeMixer
    except ImportError:
        print('[ERROR] neuralforecast not installed. Run: pip install neuralforecast')
        return

    model_map = {
        'PatchTST':    PatchTST,
        'iTransformer': iTransformer,
        'TimeMixer':   TimeMixer,
    }
    ModelCls = model_map[model_name]

    N, H, C = Y.shape

    # Common kwargs for all models
    common = dict(
        h=pred_len,
        input_size=seq_len,
        batch_size=batch_size,
        max_steps=max_steps,
        val_check_steps=50,
        early_stop_patience_steps=5,
    )

    # Model-specific kwargs
    extra = {}
    if model_name == 'PatchTST':
        extra = dict(patch_len=16, stride=8, d_model=128, n_heads=8,
                     encoder_layers=3, dropout=0.1)
    elif model_name == 'iTransformer':
        extra = dict(n_series=C, d_model=128, n_heads=8,
                     e_layers=3, d_ff=256, dropout=0.1)
    elif model_name == 'TimeMixer':
        extra = dict(n_series=C, d_model=64, d_ff=128,
                     dropout=0.1, e_layers=3)

    model = ModelCls(**common, **extra)
    nf = NeuralForecast(models=[model], freq='h')

    # Build full training df (train split only) per channel
    train_df = _to_nf_df(train_data_x, C)

    print(f'  Training {model_name} …')
    nf.fit(train_df)

    # Build test context df — for each test window, provide seq_len history
    # neuralforecast predict() needs history up to cutoff point
    # We use cross_validation over the combined train+test data
    # Combine train data + test context portion
    test_data_x = X[:, :, :]   # (N, seq_len, C) — these are consecutive windows
    # Reconstruct the flat test array (T_test, C) from sliding windows
    # The first window gives rows 0..seq_len-1; each subsequent window adds 1 row
    flat_test = np.concatenate(
        [X[0, :, :]] + [X[n, -1:, :] for n in range(1, N)], axis=0
    )  # (seq_len + N - 1, C)

    combined = np.concatenate([train_data_x, flat_test], axis=0)
    combined_df = _to_nf_df(combined, C)

    print(f'  Running cross_validation (n_windows={N}, h={pred_len}) …')
    cv_df = nf.cross_validation(
        combined_df,
        n_windows=N,
        step_size=1,
        refit=False,   # train once, evaluate N times
    )

    preds = _nf_cv_to_array(cv_df, C, H, N)
    exp = _exp_name(model_name, data, seq_len, pred_len)
    _save(exp, preds, Y, train_data_x, results_dir)


# ---------------------------------------------------------------------------
# 3. TimeMoE (HuggingFace, zero-shot)
# ---------------------------------------------------------------------------

TIMEMOE_REPO = 'Maple728/TimeMoE-50M'


def run_timemoe(X, Y, train_data_x, data, seq_len, pred_len, results_dir, device_str):
    """
    Zero-shot multi-step forecasting with TimeMoE-50M.
    Processes each channel independently (channel-independent).
    Input normalisation: per-window mean/std (RevIN-style, as used in TimeMoE paper).
    """
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        print('[ERROR] transformers not installed.')
        return

    device = torch.device(device_str)

    print(f'  Loading {TIMEMOE_REPO} …')
    model = AutoModelForCausalLM.from_pretrained(
        TIMEMOE_REPO,
        device_map='auto' if device_str != 'cpu' else None,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    if device_str == 'cpu' or not torch.cuda.is_available():
        model = model.to(device)

    N, H, C = Y.shape
    preds = np.zeros((N, H, C), dtype=np.float32)

    batch_size = 32   # reduce if OOM

    for c in range(C):
        print(f'    channel {c+1}/{C} …', end='\r')
        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            ctx = X[batch_start:batch_end, :, c]  # (B, seq_len)

            # Per-instance normalisation (mean-std, eps guard)
            mu  = ctx.mean(axis=1, keepdims=True)
            sig = ctx.std(axis=1, keepdims=True) + 1e-6
            ctx_norm = (ctx - mu) / sig

            ctx_t = torch.tensor(ctx_norm, dtype=torch.float32).to(device)  # (B, seq_len)

            with torch.no_grad():
                # TimeMoE generate: returns (B, seq_len + pred_len)
                out = model.generate(
                    inputs=ctx_t,
                    max_new_tokens=pred_len,
                )
                fc_norm = out[:, -pred_len:].cpu().numpy()  # (B, pred_len)

            # Denormalise
            fc = fc_norm * sig + mu
            preds[batch_start:batch_end, :, c] = fc

    print()
    exp = _exp_name('TimeMoE', data, seq_len, pred_len)
    _save(exp, preds, Y, train_data_x, results_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_MODELS = STAT_MODELS + DL_MODELS + ['TimeMoE']
FAMILY_MAP = {
    'stat':    STAT_MODELS,
    'dl':      DL_MODELS,
    'timemoe': ['TimeMoE'],
    'all':     ALL_MODELS,
}


def main():
    p = argparse.ArgumentParser(description='Run baseline models for significance testing.')
    p.add_argument('--data',     type=str, default='ETTh1',
                   choices=list(DATA_MAP.keys()))
    p.add_argument('--pred_len', type=int, default=96)
    p.add_argument('--seq_len',  type=int, default=336)
    p.add_argument('--features', type=str, default='M', choices=['M', 'MS', 'S'])
    p.add_argument('--target',   type=str, default='OT')
    p.add_argument('--root_path',type=str, default='./data/')
    p.add_argument('--results',  type=str, default='./results/')

    p.add_argument('--family', type=str, default='all', choices=list(FAMILY_MAP.keys()),
                   help='Model family to run.')
    p.add_argument('--models', type=str, default=None,
                   help='Comma-separated model names (overrides --family). '
                        f'Choose from: {", ".join(ALL_MODELS)}')

    # Statistical model options
    p.add_argument('--n_stat_windows', type=int, default=200,
                   help='Max rolling windows for stat models (None=all, but very slow). '
                        'Uses evenly-spaced windows across the test period.')

    # DL model options
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--max_steps',  type=int, default=1000,
                   help='Max training steps for neuralforecast DL models.')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--skip_existing', action='store_true',
                   help='Skip any model whose losses.csv already exists in results/.')

    args = p.parse_args()

    # Resolve model list
    if args.models:
        model_list = [m.strip() for m in args.models.split(',')]
    else:
        model_list = FAMILY_MAP[args.family]

    device_str = (f'cuda:{args.gpu}' if torch.cuda.is_available()
                  else 'mps' if torch.backends.mps.is_available()
                  else 'cpu')
    print(f'Device: {device_str}')

    # Load data once (shared across all models for this dataset/horizon)
    dataset_cls, default_path = DATA_MAP[args.data]
    print(f'\nLoading {args.data} (seq_len={args.seq_len}, pred_len={args.pred_len}) …')
    X, Y, train_data_x = _build_test_windows(
        dataset_cls, default_path, args.root_path,
        args.seq_len, args.pred_len, args.features, args.target,
    )
    print(f'Test windows: {X.shape[0]}  |  Context: {X.shape}  |  Target: {Y.shape}')

    # Run each requested model
    for model_name in model_list:
        exp = _exp_name(model_name, args.data, args.seq_len, args.pred_len)

        if args.skip_existing:
            losses_path = os.path.join(args.results, exp, 'losses.csv')
            if os.path.exists(losses_path):
                print(f'\n[SKIP] {exp} — losses.csv already exists.')
                continue

        print(f'\n{"=" * 60}')
        print(f'Model: {model_name}  |  Data: {args.data}  |  pred_len: {args.pred_len}')
        print(f'{"=" * 60}')
        t0 = time.time()

        if model_name in STAT_MODELS:
            run_stat(model_name, X, Y, train_data_x,
                     args.data, args.seq_len, args.pred_len,
                     args.results, args.n_stat_windows)

        elif model_name in DL_MODELS:
            run_dl(model_name, X, Y, train_data_x,
                   args.data, args.seq_len, args.pred_len,
                   args.results, args.batch_size, args.max_steps, device_str)

        elif model_name == 'TimeMoE':
            run_timemoe(X, Y, train_data_x,
                        args.data, args.seq_len, args.pred_len,
                        args.results, device_str)

        else:
            print(f'[WARN] Unknown model: {model_name}')
            continue

        print(f'  Wall time: {time.time() - t0:.1f}s')

    print(f'\nDone. Results under {args.results}')


if __name__ == '__main__':
    main()
