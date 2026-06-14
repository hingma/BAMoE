#!/usr/bin/env python
"""
scripts/compute_significance.py
--------------------------------
Load per-sample loss CSVs (produced by scripts/collect_preds.py) and run:

  1. Diebold-Mariano (DM) tests comparing every model against a reference.
  2. Model Confidence Set (MCS) across all loaded models.

Each experiment must have a losses.csv under results/<exp_name>/ with columns:
    sample_idx, mse, mae

Results are printed as a formatted table and appended to
results/significance_summary.csv.

Usage
-----
    python scripts/compute_significance.py

    python scripts/compute_significance.py \\
        --results_dir ./results \\
        --reference "BAMoE_K10_learned_sparse_ETTh1_sl336_pl96" \\
        --loss mse \\
        --pred_len 96 \\
        --alpha 0.05 \\
        --n_boot 1000
"""

import sys
import os
import csv
import glob
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.stat_tests import dm_test, mcs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_experiment_losses(results_dir: str, loss: str) -> dict[str, np.ndarray]:
    """
    Scan results_dir for sub-directories containing losses.csv.
    Returns {exp_name: (N,) array of per-sample losses for the chosen metric}.
    """
    data = {}
    for csv_path in sorted(glob.glob(
        os.path.join(results_dir, '**', 'losses.csv'), recursive=True
    )):
        exp_name = os.path.relpath(os.path.dirname(csv_path), results_dir)
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            if loss not in reader.fieldnames:
                print(f'[WARN] Column "{loss}" missing in {csv_path}, skipping.')
                continue
            values = np.array([float(row[loss]) for row in reader])
        if len(values) == 0:
            print(f'[WARN] Empty losses.csv at {csv_path}, skipping.')
            continue
        data[exp_name] = values
    return data


def _sig_stars(p: float) -> str:
    if p < 0.01:
        return '***'
    if p < 0.05:
        return '** '
    if p < 0.10:
        return '*  '
    return '   '


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    results_dir: str,
    reference: str | None,
    loss: str,
    pred_len: int,
    alpha: float,
    n_boot: int,
) -> None:
    # ------------------------------------------------------------------
    # 1. Load per-sample losses
    # ------------------------------------------------------------------
    losses = load_experiment_losses(results_dir, loss)
    if not losses:
        print(f'No losses.csv files found under {results_dir}.')
        print('Run  python scripts/collect_preds.py  first.')
        return

    print(f'\nLoaded {len(losses)} experiment(s):')
    for name in losses:
        print(f'  {name}  (N={len(losses[name])})')

    # Determine reference model (first match containing "BAMoE" + "learned_sparse")
    if reference is None:
        for name in losses:
            if 'BAMoE' in name and 'learned_sparse' in name:
                reference = name
                break
        if reference is None:
            reference = next(iter(losses))
    if reference not in losses:
        print(f'\n[ERROR] Reference model "{reference}" not found.')
        print(f'Available: {list(losses.keys())}')
        return

    mean_losses = {name: float(l.mean()) for name, l in losses.items()}
    ref_losses  = losses[reference]
    ref_mean    = mean_losses[reference]

    # ------------------------------------------------------------------
    # 2. Diebold-Mariano tests  (each model vs reference)
    # ------------------------------------------------------------------
    dm_results: dict[str, dict] = {}
    for name, l in losses.items():
        if name == reference:
            continue
        dm_results[name] = dm_test(l, ref_losses, h=pred_len, alpha=alpha)

    # ------------------------------------------------------------------
    # 3. Model Confidence Set
    # ------------------------------------------------------------------
    print(f'\nRunning MCS (B={n_boot}, α={alpha}) …', flush=True)
    mcs_members = set(mcs(losses, alpha=alpha, n_boot=n_boot))

    # ------------------------------------------------------------------
    # 4. Print results table
    # ------------------------------------------------------------------
    sep = '─' * 84
    print(f'\n{"Statistical Significance Results":^84}')
    print(sep)
    print(f'  Loss metric : {loss.upper()}   |   α = {alpha}   |   h = {pred_len}   |   B = {n_boot}')
    print(sep)

    print(f'\nDiebold-Mariano Tests  (baseline vs reference: {reference})')
    col_w = max(len(n) for n in losses) + 2
    hdr = (f'{"Model":<{col_w}}  {"Mean":>8}  {"Δ Mean":>8}'
           f'  {"DM stat":>8}  {"p-val":>7}  Sig  MCS')
    print(hdr)
    print('─' * len(hdr))

    # Reference row
    ref_mcs = '✓' if reference in mcs_members else ' '
    print(f'{"[REF] " + reference:<{col_w}}  {ref_mean:>8.5f}  {"—":>8}  {"—":>8}  {"—":>7}       {ref_mcs}')

    rows_for_csv = []
    for name in sorted(dm_results):
        r      = dm_results[name]
        delta  = mean_losses[name] - ref_mean
        stars  = _sig_stars(r['p_two'])
        in_mcs = '✓' if name in mcs_members else ' '
        winner = 'REF' if r['d_bar'] > 0 else 'MOD'
        print(
            f'{name:<{col_w}}  {mean_losses[name]:>8.5f}  {delta:>+8.5f}'
            f'  {r["dm_stat"]:>8.3f}  {r["p_two"]:>7.4f}  {stars}  {in_mcs}'
        )
        rows_for_csv.append({
            'exp_name':   name,
            'reference':  reference,
            'loss':       loss,
            'n_samples':  len(losses[name]),
            'mean_loss':  mean_losses[name],
            'ref_mean':   ref_mean,
            'delta_mean': delta,
            'dm_stat':    r['dm_stat'],
            'p_two':      r['p_two'],
            'p_one':      r['p_one'],
            'reject_two': r['reject_two'],
            'reject_one': r['reject_one'],
            'in_mcs':     name in mcs_members,
        })

    # --- MCS summary ---
    print(f'\nModel Confidence Set  (α={alpha}, B={n_boot})')
    print(sep)
    print(f'  MCS members ({len(mcs_members)}):')
    for m in sorted(mcs_members):
        marker = ' ← reference' if m == reference else ''
        print(f'    • {m}{marker}')

    print(f'\nLegend: *** p<0.01  ** p<0.05  * p<0.10  (two-sided DM)  |  ✓ = in MCS')
    print(sep)

    # ------------------------------------------------------------------
    # 5. Append to CSV summary
    # ------------------------------------------------------------------
    csv_path = os.path.join(results_dir, 'significance_summary.csv')
    write_header = not os.path.exists(csv_path)
    fieldnames = [
        'exp_name', 'reference', 'loss', 'n_samples',
        'mean_loss', 'ref_mean', 'delta_mean',
        'dm_stat', 'p_two', 'p_one',
        'reject_two', 'reject_one', 'in_mcs',
    ]
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerows(rows_for_csv)
    print(f'\nResults appended to {csv_path}')


def main():
    p = argparse.ArgumentParser(
        description='Compute DM tests and MCS from per-sample loss CSVs.'
    )
    p.add_argument('--results_dir', type=str, default='./results',
                   help='Directory containing per-experiment losses.csv files.')
    p.add_argument('--reference', type=str, default=None,
                   help='exp_name of the reference model '
                        '(default: first BAMoE+learned_sparse match).')
    p.add_argument('--loss', type=str, default='mse', choices=['mse', 'mae'],
                   help='Loss column from losses.csv to use for tests.')
    p.add_argument('--pred_len', type=int, default=1,
                   help='Forecast horizon h for DM test autocorrelation lag '
                        '(set to the prediction length used in experiments).')
    p.add_argument('--alpha', type=float, default=0.05,
                   help='Significance level for DM tests and MCS.')
    p.add_argument('--n_boot', type=int, default=1000,
                   help='Bootstrap replications for MCS critical values.')
    cli = p.parse_args()
    run(cli.results_dir, cli.reference, cli.loss, cli.pred_len, cli.alpha, cli.n_boot)


if __name__ == '__main__':
    main()
