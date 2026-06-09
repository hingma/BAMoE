"""
Visualisation utilities for Experiment 4 (Interpretability Analysis).
All plotting functions save to disk; they do not display interactively.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# 4a — Expert activation heatmap across datasets / horizons
# ---------------------------------------------------------------------------

def plot_expert_heatmap(routing_matrix, expert_labels, row_labels, save_path):
    """
    routing_matrix : (n_rows, n_experts) — average routing weight per setting
    row_labels     : list of strings, e.g. ['ETTh1/96', 'ETTh1/192', ...]
    expert_labels  : list of strings, e.g. ['causal', 'local', 'periodic', 'global']
    """
    fig, ax = plt.subplots(figsize=(max(6, len(expert_labels) * 1.5),
                                    max(4, len(row_labels) * 0.4 + 1)))
    sns.heatmap(routing_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=expert_labels, yticklabels=row_labels,
                linewidths=0.5, ax=ax, vmin=0, vmax=1)
    ax.set_title('Average Expert Activation Weight')
    ax.set_xlabel('Expert')
    ax.set_ylabel('Dataset / Horizon')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4b — Routing dynamics for a single time series
# ---------------------------------------------------------------------------

def plot_routing_dynamics(routing_seq, expert_labels, save_path, title=''):
    """
    routing_seq : (n_patches, n_experts) — routing weights for one sample
    """
    n_patches, n_experts = routing_seq.shape
    fig, ax = plt.subplots(figsize=(max(8, n_patches * 0.15), 3))
    x = np.arange(n_patches)
    bottom = np.zeros(n_patches)
    colors = plt.cm.tab10(np.linspace(0, 0.8, n_experts))
    for e, (label, color) in enumerate(zip(expert_labels, colors)):
        ax.bar(x, routing_seq[:, e], bottom=bottom, label=label, color=color, width=1.0)
        bottom += routing_seq[:, e]
    ax.set_xlim(0, n_patches)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Patch index')
    ax.set_ylabel('Routing weight')
    ax.set_title(title or 'Expert routing dynamics')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4c — Correlation between routing weights and temporal statistics
# ---------------------------------------------------------------------------

def plot_routing_vs_temporal_stats(stats_dict, routing_dict, expert_labels, save_path):
    """
    stats_dict   : {dataset_name: {'dominant_freq': float, 'autocorr_lag1': float,
                                   'trend_strength': float}}
    routing_dict : {dataset_name: np.array of shape (n_experts,)}
    """
    stat_names = ['dominant_freq', 'autocorr_lag1', 'trend_strength']
    n_stats = len(stat_names)
    n_experts = len(expert_labels)
    datasets = list(stats_dict.keys())

    fig, axes = plt.subplots(n_experts, n_stats, figsize=(4 * n_stats, 3 * n_experts),
                             squeeze=False)
    for e, expert in enumerate(expert_labels):
        for s, stat in enumerate(stat_names):
            ax = axes[e][s]
            xs = [stats_dict[d][stat] for d in datasets]
            ys = [routing_dict[d][e] for d in datasets]
            ax.scatter(xs, ys, s=50, zorder=3)
            for i, d in enumerate(datasets):
                ax.annotate(d, (xs[i], ys[i]), fontsize=6, ha='left')
            ax.set_xlabel(stat)
            ax.set_ylabel(f'{expert} weight')
    plt.suptitle('Routing weight vs. temporal statistics', y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4d — Expert attention pattern visualisation
# ---------------------------------------------------------------------------

def plot_attention_patterns(attn_maps, expert_labels, save_path):
    """
    attn_maps : list of (N, N) numpy arrays, one per expert
    """
    n_experts = len(expert_labels)
    fig, axes = plt.subplots(1, n_experts, figsize=(4 * n_experts, 4))
    if n_experts == 1:
        axes = [axes]
    for ax, attn, label in zip(axes, attn_maps, expert_labels):
        im = ax.imshow(attn, aspect='auto', cmap='Blues', vmin=0)
        ax.set_title(label)
        ax.set_xlabel('Key patch')
        ax.set_ylabel('Query patch')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Efficiency Pareto plot (Experiment 5)
# ---------------------------------------------------------------------------

def plot_pareto(records, save_path):
    """
    records : list of dicts with keys 'label', 'params', 'mse'
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in records:
        ax.scatter(r['params'] / 1e6, r['mse'], s=80, zorder=3)
        ax.annotate(r['label'], (r['params'] / 1e6, r['mse']),
                    textcoords='offset points', xytext=(4, 4), fontsize=8)
    ax.set_xlabel('Parameters (M)')
    ax.set_ylabel('Average MSE')
    ax.set_title('Parameter count vs. forecasting performance')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
