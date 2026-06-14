#!/usr/bin/env python
"""
scripts/collect_preds.py
------------------------
Run inference for each experiment listed in EXPERIMENTS, load the saved
checkpoint, and write raw predictions + ground truth to

    results/<exp_name>/preds.npy
    results/<exp_name>/trues.npy

These files are consumed by scripts/compute_significance.py.

Prerequisites
-------------
Each model must already be trained (checkpoint present under ./checkpoints/).
Pass --train to train any missing checkpoints before collecting.

Usage
-----
    # Collect predictions using existing checkpoints:
    python scripts/collect_preds.py

    # Also train any missing checkpoints first:
    python scripts/collect_preds.py --train

    # Override the base config file:
    python scripts/collect_preds.py --config configs/main.yaml

Edit EXPERIMENTS below to define which models / baselines to compare.
"""

import sys
import os
import argparse
import random

import numpy as np
import torch

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import build_args, set_seed, make_exp_name
from exp.exp_forecast import ExpForecast


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
# Edit DATASETS and PRED_LENS to control which (dataset, horizon) pairs are
# collected.  Model variants are fixed below and applied to every combination.
# ---------------------------------------------------------------------------

DATASETS  = ['ETTh1', 'ETTm2', 'Exchange', 'Electricity', 'ILI']
PRED_LENS = [96, 192, 336, 720]

EXPERIMENTS = []
for _data in DATASETS:
    for _pred_len in PRED_LENS:
        _base = dict(data=_data, pred_len=_pred_len)
        # Proposed model
        EXPERIMENTS.append(dict(model='BAMoE', routing='learned_sparse', **_base))
        # SingleBias baselines (Experiment 1)
        for _bias in ('global', 'causal', 'local', 'periodic'):
            EXPERIMENTS.append(dict(model='SingleBias', bias_type=_bias, **_base))
        # BAMoE routing ablations
        for _routing in ('uniform', 'random', 'dense'):
            EXPERIMENTS.append(dict(model='BAMoE', routing=_routing, **_base))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def collect(config_file: str | None, do_train: bool) -> None:
    for exp_kwargs in EXPERIMENTS:
        args = build_args(config_file, **exp_kwargs)
        # save_losses is not a CLI flag; set it programmatically
        args.save_losses = True

        set_seed(args.seed)
        print(f'\n{"=" * 60}')
        print(f'Experiment : {args.exp_name}')
        print(f'Model      : {args.model}')
        print(f'{"=" * 60}')

        exp = ExpForecast(args)

        ckpt = exp._checkpoint_path()
        if not os.path.exists(ckpt):
            if do_train:
                print('No checkpoint found — training first.')
                exp.train()
            else:
                print(f'[SKIP] No checkpoint at {ckpt}. Re-run with --train to train first.')
                continue

        exp.test(load_ckpt=True)
        out_dir = os.path.join(args.results, args.exp_name)
        print(f'Predictions saved to {out_dir}/')


def main():
    p = argparse.ArgumentParser(description='Collect test predictions for significance testing.')
    p.add_argument('--config', type=str, default=None,
                   help='Base YAML config file (e.g. configs/main.yaml).')
    p.add_argument('--train', action='store_true',
                   help='Train any experiment whose checkpoint is missing.')
    cli = p.parse_args()
    collect(cli.config, cli.train)


if __name__ == '__main__':
    main()
