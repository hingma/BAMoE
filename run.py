"""
BAMoE — main entry point

Usage examples
--------------
# Experiment 1: single-bias preliminary study
python run.py --model SingleBias --bias_type causal \
    --data ETTh1 --pred_len 96

# Experiment 2: main BAMoE results
python run.py --model BAMoE --expert_types causal,local,periodic,global \
    --data Weather --pred_len 336

# Experiment 3a: homogeneous-expert ablation
python run.py --model BAMoE --expert_types causal,causal,causal,causal \
    --data ETTh1 --pred_len 96 --exp_name ablation_homo_causal_ETTh1_96

# Experiment 3b: routing mechanism ablation
python run.py --model BAMoE --routing uniform \
    --data ETTh1 --pred_len 96

# Experiment 3c: number-of-experts sweep
python run.py --model BAMoE \
    --expert_types causal,local,periodic,global,causal,local \
    --data ETTh1 --pred_len 96

# Interpretability (after training)
python run.py --model BAMoE --mode interpretability \
    --data Weather --pred_len 96
"""

import argparse
import random
import numpy as np
import torch
import yaml
import wandb

from exp.exp_forecast import ExpForecast
from exp.exp_interpretability import ExpInterpretability


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser('BAMoE')

    # ---- config file ----
    p.add_argument('--config', type=str, default=None,
                   help='YAML config file. CLI args override any key set here.')

    # ---- experiment meta ----
    p.add_argument('--mode', type=str, default='train_test',
                   choices=['train', 'test', 'train_test', 'interpretability'])
    p.add_argument('--exp_name', type=str, default='',
                   help='Unique run identifier; auto-generated if empty.')
    p.add_argument('--seed', type=int, default=2024)

    # ---- data ----
    p.add_argument('--data', type=str, default='ETTh1',
                   choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2',
                            'Weather', 'Traffic', 'Electricity', 'Exchange'])
    p.add_argument('--root_path', type=str, default='./data/',
                   help='Directory containing dataset CSV files.')
    p.add_argument('--data_path', type=str, default='',
                   help='CSV filename; inferred from --data if empty.')
    p.add_argument('--features', type=str, default='M',
                   choices=['M', 'MS', 'S'])
    p.add_argument('--target', type=str, default='OT')
    p.add_argument('--seq_len', type=int, default=336)
    p.add_argument('--pred_len', type=int, default=96)
    p.add_argument('--num_workers', type=int, default=0)

    # ---- model ----
    p.add_argument('--model', type=str, default='BAMoE',
                   choices=['BAMoE', 'SingleBias'])
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--n_layers', type=int, default=3)
    p.add_argument('--d_ff', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--patch_len', type=int, default=16)
    p.add_argument('--stride', type=int, default=8)

    # ---- SingleBias args (Experiment 1) ----
    p.add_argument('--bias_type', type=str, default='global',
                   choices=['global', 'causal', 'local', 'periodic','periodic_fixed', 'reverse_causal', 'alibi', 'relative', 'trend', 'seasonal'],
                   help='Inductive bias for SingleBiasTransformer.')

    # ---- BAMoE args (Experiments 2, 3) ----
    p.add_argument('--expert_types', type=str,
                   default='local,global,periodic,periodic_fixed,causal,reverse_causal,trend,seasonal,alibi,relative',
                   help='Comma-separated list of expert bias types.')
    p.add_argument('--top_k', type=int, default=2,
                   help='Number of active experts per token.')
    p.add_argument('--routing', type=str, default='learned_sparse',
                   choices=['learned_sparse', 'uniform', 'random', 'top1', 'dense'],
                   help='Routing mechanism.')
    p.add_argument('--load_balance_coef', type=float, default=0.01,
                   help='Weight for auxiliary load-balancing loss.')
    p.add_argument('--local_window', type=int, default=3,
                   help='Window size for local attention experts.')
    p.add_argument('--periodic_period', type=int, default=12,
                   help='Period parameter for periodic attention experts.')

    # ---- training ----
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--train_epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--lradj', type=str, default='cosine')

    # ---- device ----
    p.add_argument('--use_gpu', type=int, default=1)
    p.add_argument('--gpu', type=int, default=0)

    # ---- paths ----
    p.add_argument('--resume', action='store_true',
                   help='Skip training if a checkpoint already exists for this exp_name.')
    p.add_argument('--checkpoints', type=str, default='./checkpoints/')
    p.add_argument('--results', type=str, default='./results/')

    # ---- wandb ----
    p.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging.')
    p.add_argument('--wandb_project', type=str, default='BAMoE')
    p.add_argument('--wandb_entity', type=str, default=None)

    return p


def make_exp_name(args):
    if args.exp_name:
        return args.exp_name
    if args.model == 'SingleBias':
        tag = f'SingleBias_{args.bias_type}'
    else:
        n_exp = len(args.expert_types.split(','))
        tag = f'BAMoE_K{n_exp}_{args.routing}'
    return f'{tag}_{args.data}_sl{args.seq_len}_pl{args.pred_len}'


def build_args(config: str = None, **overrides) -> argparse.Namespace:
    """Programmatic entry point for notebooks and tests.

    Equivalent to: python run.py --config <config> [--key value ...]
    but callable from Python without touching sys.argv.

    Example:
        args = build_args('configs/main.yaml', data='ETTh1', pred_len=96)
    """
    p = _make_parser()
    if config:
        with open(config) as f:
            p.set_defaults(**yaml.safe_load(f))
    args = p.parse_args([])
    for k, v in overrides.items():
        setattr(args, k, v)
    args.exp_name = make_exp_name(args)
    return args


def get_args() -> argparse.Namespace:
    """CLI entry point — two-pass so explicit flags always beat the YAML."""
    p = _make_parser()
    pre, _ = p.parse_known_args()
    if pre.config:
        with open(pre.config) as f:
            p.set_defaults(**yaml.safe_load(f))
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = get_args()
    print(f'{args.expert_types}')
    set_seed(args.seed)
    args.exp_name = make_exp_name(args)
    print(f'\n=== {args.exp_name} ===')

    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.exp_name,
            config=vars(args),
        )

    if args.mode == 'interpretability':
        # print(f'exporting experts: {args.expert_types}')
        exp = ExpInterpretability(args)
        exp.run()
        if args.wandb:
            wandb.finish()
        return

    exp = ExpForecast(args)
    if args.mode in ('train', 'train_test'):
        exp.train()
    if args.mode in ('test', 'train_test'):
        exp.test()

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
