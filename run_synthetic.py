"""
Entry point for BAMoE Experiment 4 (Synthetic Validation).

Usage
-----
# Run all three sub-experiments
python run_synthetic.py

# Run a single sub-experiment
python run_synthetic.py --exp 41
python run_synthetic.py --exp 42
python run_synthetic.py --exp 43

# Lightweight smoke-test (small model, short series, few epochs)
python run_synthetic.py --smoke_test

Key arguments
-------------
--T_synth          Total synthetic series length (default 5000)
--seq_len          Input window length           (default 336)
--pred_len         Forecast horizon              (default 96)
--d_model / --n_heads / --n_layers / --d_ff
                   Model size (defaults: 64 / 4 / 2 / 128 — smaller than
                   real-world runs to keep synthetic training fast)
--train_epochs     Max training epochs per run   (default 30)
--slope_multipliers  Comma-separated γ values for Exp 4.3
                     (default: 0.1,0.5,1.0,2.0,5.0,10.0)
--noise_levels       Comma-separated σ values for Exp 4.3
                     (default: 0.5,1.0,2.0)
"""

import argparse
import random

import numpy as np
import torch

from exp.exp_synthetic import ExpSynthetic


def _parse_float_list(s: str) -> list[float]:
    return [float(x) for x in s.split(',')]


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser('BAMoE — Synthetic Experiments (Exp 4)')

    # --- which experiments to run ---
    p.add_argument('--exp', type=str, default='all',
                   choices=['all', '41', '42', '43'],
                   help='Sub-experiment to run (default: all).')

    # --- synthetic data ---
    p.add_argument('--T_synth', type=int, default=5000,
                   help='Total synthetic series length.')
    p.add_argument('--seq_len', type=int, default=336)
    p.add_argument('--pred_len', type=int, default=96)

    # --- model architecture ---
    p.add_argument('--d_model',  type=int,   default=64)
    p.add_argument('--n_heads',  type=int,   default=4)
    p.add_argument('--n_layers', type=int,   default=2)
    p.add_argument('--d_ff',     type=int,   default=128)
    p.add_argument('--dropout',  type=float, default=0.1)
    p.add_argument('--patch_len', type=int,  default=16)
    p.add_argument('--stride',    type=int,  default=8)
    p.add_argument('--local_window',    type=int, default=3)
    p.add_argument('--periodic_period', type=int, default=24)

    # --- training ---
    p.add_argument('--train_epochs',  type=int,   default=30)
    p.add_argument('--patience',      type=int,   default=5)
    p.add_argument('--batch_size',    type=int,   default=64)
    p.add_argument('--learning_rate', type=float, default=1e-3)

    # --- Exp 4.3 sweeps ---
    p.add_argument('--slope_multipliers', type=_parse_float_list,
                   default=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                   help='Comma-separated ALiBi slope multipliers γ for Exp 4.3.')
    p.add_argument('--noise_levels', type=_parse_float_list,
                   default=[0.5, 1.0, 2.0],
                   help='Comma-separated noise σ values for Exp 4.3.')

    # --- device / output ---
    p.add_argument('--use_gpu', type=int, default=1)
    p.add_argument('--results', type=str, default='./results/')
    p.add_argument('--seed',    type=int, default=2024)

    # --- quick smoke test ---
    p.add_argument('--smoke_test', action='store_true',
                   help='Tiny config for CI / quick sanity check.')

    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_smoke_test(args: argparse.Namespace) -> argparse.Namespace:
    """Override args with minimal settings for a fast sanity check."""
    args.T_synth        = 1500
    args.seq_len        = 96
    args.pred_len       = 24
    args.d_model        = 32
    args.n_heads        = 2
    args.n_layers       = 1
    args.d_ff           = 64
    args.patch_len      = 8
    args.stride         = 4
    args.train_epochs   = 3
    args.patience       = 2
    args.batch_size     = 32
    args.slope_multipliers = [0.5, 1.0, 5.0]
    args.noise_levels      = [0.5, 1.0]
    return args


def main():
    args = get_args()
    if args.smoke_test:
        args = apply_smoke_test(args)
        print('Smoke-test mode: reduced config.')
    set_seed(args.seed)

    print(f'\nDevice config: use_gpu={args.use_gpu}')
    print(f'Series length: T={args.T_synth},  seq_len={args.seq_len},  '
          f'pred_len={args.pred_len}')
    print(f'Model:         d_model={args.d_model},  n_heads={args.n_heads},  '
          f'n_layers={args.n_layers}')
    print(f'Training:      epochs={args.train_epochs},  bs={args.batch_size},  '
          f'lr={args.learning_rate}')

    exp = ExpSynthetic(args)

    if args.exp in ('all', '41'):
        exp.run_exp41()

    if args.exp in ('all', '42'):
        exp.run_exp42()

    if args.exp in ('all', '43'):
        exp.run_exp43()

    print('\nExperiment 4 complete.  Results saved to:', exp.out_dir)


if __name__ == '__main__':
    main()
