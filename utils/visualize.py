"""
Visualisation utilities for Experiment 4 (Interpretability Analysis).
All plotting functions save to disk; they do not display interactively.

Experts are grouped into five categories and always rendered in this order:
  Global → Directionality → Locality → Periodicity → Decomposition
Each category receives a consistent colour across all plots.
"""

import os
import colorsys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import seaborn as sns


# ---------------------------------------------------------------------------
# Expert category taxonomy
# ---------------------------------------------------------------------------

EXPERT_CATEGORIES = [
    # (category_name, [registry_keys...],           hex_color)
    ("Global",         ["global"],                                  "#7f7f7f"),
    ("Directionality", ["causal", "reverse_causal"],                "#1f77b4"),
    ("Locality",       ["local", "alibi"],                          "#2ca02c"),
    ("Periodicity",    ["periodic_fixed", "periodic", "relative"],  "#ff7f0e"),
    ("Decomposition",  ["trend", "seasonal"],                       "#9467bd"),
]

# Flat lookup: normalised key -> (category_name, color, canonical_rank)
_EXPERT_META: dict = {}
_rank = 0
for _cname, _members, _col in EXPERT_CATEGORIES:
    for _m in _members:
        _EXPERT_META[_m] = (_cname, _col, _rank)
        _rank += 1
del _rank, _cname, _members, _col, _m

_HATCHES = ['', '//', 'xx', '\\\\', '..']


def _norm(label: str) -> str:
    return label.strip().lower().replace(' ', '_')


def _expert_color(label: str) -> str:
    return _EXPERT_META.get(_norm(label), ("", "#333333", 0))[1]


def _lighten(hex_color: str, amount: float = 0.80) -> str:
    """Return a lighter version of a hex color (amount=1.0 → white)."""
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    hue, lum, sat = colorsys.rgb_to_hls(r, g, b)
    lum = min(1.0, lum + (1.0 - lum) * amount)
    r2, g2, b2 = colorsys.hls_to_rgb(hue, lum, sat)
    return '#{:02x}{:02x}{:02x}'.format(int(r2 * 255), int(g2 * 255), int(b2 * 255))


def _sort_by_category(expert_labels):
    """
    Sort experts into canonical category order.

    Returns
    -------
    order : list[int]
        Indices into the original ``expert_labels`` list.
    sorted_labels : list[str]
    spans : list of (start, end, cat_name, color)
        Inclusive column range in the *sorted* layout, one entry per
        category that is actually present.
    """
    order = sorted(
        range(len(expert_labels)),
        key=lambda i: _EXPERT_META.get(_norm(expert_labels[i]), ("", "", 9999))[2],
    )
    sorted_labels = [expert_labels[i] for i in order]
    spans = []
    for cname, _members, color in EXPERT_CATEGORIES:
        cols = [j for j, lbl in enumerate(sorted_labels)
                if _EXPERT_META.get(_norm(lbl), ("",))[0] == cname]
        if cols:
            spans.append((min(cols), max(cols), cname, color))
    return order, sorted_labels, spans


def _draw_x_category_brackets(ax, spans, n_cols):
    """Draw coloured bracket lines + labels above the x-axis (axes coordinates)."""
    for start, end, cname, color in spans:
        x0   = (start + 0.08) / n_cols
        x1   = (end   + 0.92) / n_cols
        xmid = (start + end + 1) / 2 / n_cols
        ax.plot([x0, x1], [1.02, 1.02],
                transform=ax.transAxes, color=color, lw=2.5, clip_on=False)
        ax.text(xmid, 1.046, cname,
                transform=ax.transAxes,
                ha='center', va='bottom',
                color=color, fontsize=8.5, fontweight='bold', clip_on=False)


# ---------------------------------------------------------------------------
# 4a — Expert activation heatmap across datasets / horizons
# ---------------------------------------------------------------------------

def plot_expert_heatmap(routing_matrix, expert_labels, row_labels, save_path):
    """
    routing_matrix : (n_rows, n_experts) — average routing weight per setting
    row_labels     : list of strings, e.g. ['ETTh1/96', 'ETTh1/192', ...]
    expert_labels  : list of strings, e.g. ['causal', 'local', 'periodic', 'global']

    Experts are sorted into category groups (Global → Directionality → Locality
    → Periodicity → Decomposition).  Coloured bracket annotations appear above
    the corresponding column groups; x-tick labels are tinted to match.
    """
    order, sorted_labels, spans = _sort_by_category(expert_labels)
    sorted_matrix = routing_matrix[:, order]
    n_experts = len(sorted_labels)

    fig, ax = plt.subplots(figsize=(max(6, n_experts * 1.6),
                                    max(4, len(row_labels) * 0.4 + 1.5)))
    sns.heatmap(sorted_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=sorted_labels, yticklabels=row_labels,
                linewidths=0.5, ax=ax, vmin=0, vmax=1)
    ax.set_title('Average Expert Activation Weight', pad=32)
    ax.set_xlabel('Expert')
    ax.set_ylabel('Dataset / Horizon')

    # Colour x-tick labels by category
    plt.draw()
    for tick, lbl in zip(ax.get_xticklabels(), sorted_labels):
        tick.set_color(_expert_color(lbl))

    # White vertical separators at category boundaries
    for start, _end, _cname, _color in spans:
        if start > 0:
            ax.axvline(x=start, color='white', linewidth=3, zorder=5)

    # Coloured bracket annotations above the column groups
    _draw_x_category_brackets(ax, spans, n_experts)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4b — Routing dynamics for a single time series
# ---------------------------------------------------------------------------

def plot_routing_dynamics(routing_seq, expert_labels, save_path, title=''):
    """
    routing_seq : (n_patches, n_experts) — routing weights for one sample

    Experts are reordered by category.  Bars share the category colour and are
    distinguished within a category by hatch pattern.  The legend groups
    experts under a coloured category heading.
    """
    order, sorted_labels, spans = _sort_by_category(expert_labels)
    sorted_seq = routing_seq[:, order]
    n_patches, n_experts = sorted_seq.shape

    # Per-expert colour (category) and hatch (within-category index)
    cat_count: dict = {}
    bar_colors, bar_hatches = [], []
    for lbl in sorted_labels:
        color = _expert_color(lbl)
        cat   = _EXPERT_META.get(_norm(lbl), ("",))[0]
        idx   = cat_count.get(cat, 0)
        cat_count[cat] = idx + 1
        bar_colors.append(color)
        bar_hatches.append(_HATCHES[idx % len(_HATCHES)])

    fig, ax = plt.subplots(figsize=(max(8, n_patches * 0.15), 3.8))
    x      = np.arange(n_patches)
    bottom = np.zeros(n_patches)

    for e in range(n_experts):
        ax.bar(x, sorted_seq[:, e], bottom=bottom,
               color=bar_colors[e], hatch=bar_hatches[e],
               width=1.0, edgecolor='white', linewidth=0.3)
        bottom += sorted_seq[:, e]

    ax.set_xlim(0, n_patches)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Patch index')
    ax.set_ylabel('Routing weight')
    ax.set_title(title or 'Expert routing dynamics')

    # Grouped legend: coloured Line2D header per category + indented expert patches
    legend_handles: list = []
    legend_text: list    = []
    for start, end, cname, color in spans:
        legend_handles.append(Line2D([0], [0], color=color, lw=3))
        legend_text.append(cname)
        for j in range(start, end + 1):
            legend_handles.append(
                mpatches.Patch(facecolor=bar_colors[j], hatch=bar_hatches[j],
                               edgecolor='grey', linewidth=0.5)
            )
            legend_text.append(f'  {sorted_labels[j]}')

    ax.legend(legend_handles, legend_text,
              loc='upper right', fontsize=7.5, ncol=2,
              framealpha=0.85, edgecolor='#cccccc')

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

    Rows (experts) are sorted into category groups.  Each category block
    receives a light-tinted background; a rotated category label appears on
    the left margin and the y-axis label is coloured to match.
    """
    order, sorted_labels, spans = _sort_by_category(expert_labels)
    stat_names = ['dominant_freq', 'autocorr_lag1', 'trend_strength']
    n_stats    = len(stat_names)
    n_experts  = len(sorted_labels)
    datasets   = list(stats_dict.keys())

    fig, axes = plt.subplots(n_experts, n_stats,
                             figsize=(4 * n_stats, 3 * n_experts),
                             squeeze=False)

    for row_idx, orig_idx in enumerate(order):
        expert    = sorted_labels[row_idx]
        cat_color = _expert_color(expert)
        bg_color  = _lighten(cat_color, 0.82)

        for s, stat in enumerate(stat_names):
            ax = axes[row_idx][s]
            ax.set_facecolor(bg_color)
            xs = [stats_dict[d][stat] for d in datasets]
            ys = [routing_dict[d][orig_idx] for d in datasets]
            ax.scatter(xs, ys, s=50, color=cat_color, zorder=3)
            for i, d in enumerate(datasets):
                ax.annotate(d, (xs[i], ys[i]), fontsize=6, ha='left')
            ax.set_xlabel(stat, fontsize=8)
            if s == 0:
                ax.set_ylabel(f'{expert} weight',
                              color=cat_color, fontsize=8, fontweight='bold')

    # Rotated category labels along the left margin (one per group)
    for start, end, cname, color in spans:
        y_frac = 1.0 - (start + (end - start) / 2 + 0.5) / n_experts
        fig.text(0.005, y_frac, cname,
                 ha='left', va='center',
                 color=color, fontsize=9, fontweight='bold',
                 rotation=90, transform=fig.transFigure)

    plt.suptitle('Routing weight vs. temporal statistics', y=1.01)
    plt.tight_layout(rect=[0.03, 0, 1, 1])
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4d — Expert attention pattern visualisation
# ---------------------------------------------------------------------------

def plot_attention_patterns(attn_maps, expert_labels, save_path):
    """
    attn_maps : list of (N, N) numpy arrays, one per expert

    Experts are sorted into category groups.  A solid coloured header strip
    spans each category's columns; individual subplot titles are tinted to
    match their category colour.
    """
    order, sorted_labels, spans = _sort_by_category(expert_labels)
    sorted_maps = [attn_maps[i] for i in order]
    n_experts   = len(sorted_labels)

    # Two-row GridSpec: thin coloured header + attention map subplots
    fig = plt.figure(figsize=(4 * n_experts, 5.4))
    gs  = GridSpec(2, n_experts, figure=fig,
                   height_ratios=[0.10, 1], hspace=0.08, wspace=0.35)

    # Category header strips (top row)
    for start, end, cname, color in spans:
        ax_hdr = fig.add_subplot(gs[0, start: end + 1])
        ax_hdr.set_facecolor(color)
        ax_hdr.text(0.5, 0.5, cname,
                    transform=ax_hdr.transAxes,
                    ha='center', va='center',
                    color='white', fontsize=10, fontweight='bold')
        ax_hdr.set_xticks([])
        ax_hdr.set_yticks([])
        for spine in ax_hdr.spines.values():
            spine.set_visible(False)

    # Attention map subplots (bottom row)
    for j, (attn, label) in enumerate(zip(sorted_maps, sorted_labels)):
        ax = fig.add_subplot(gs[1, j])
        im = ax.imshow(attn, aspect='auto', cmap='Blues', vmin=0)
        ax.set_title(label, color=_expert_color(label), fontsize=9, pad=4)
        ax.set_xlabel('Key patch', fontsize=8)
        if j == 0:
            ax.set_ylabel('Query patch', fontsize=8)
        else:
            ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4-new-1 — Real-Time Routing Gate Trajectories
# ---------------------------------------------------------------------------

def plot_routing_gate_trajectories(series, routing_seq, patch_centers,
                                   expert_labels, break_idx, save_path, title=''):
    """
    series        : (seq_len,) time series values
    routing_seq   : (n_patches, n_experts) soft routing probabilities
    patch_centers : (n_patches,) timestep of each patch midpoint
    expert_labels : list of expert names
    break_idx     : int, timestep of the detected structural break
    """
    order, sorted_labels, spans = _sort_by_category(expert_labels)
    sorted_seq = routing_seq[:, order]
    n_experts = len(sorted_labels)
    colors = [_expert_color(lbl) for lbl in sorted_labels]

    seq_len = len(series)
    t = np.arange(seq_len)

    fig, (ax_ts, ax_rt) = plt.subplots(
        2, 1, figsize=(13, 6), sharex=True,
        gridspec_kw={'height_ratios': [1, 1.5], 'hspace': 0.04},
    )

    # Top panel: raw time series
    ax_ts.plot(t, series, color='#2c3e50', lw=1.2, alpha=0.9, zorder=3)
    ax_ts.axvline(break_idx, color='#e74c3c', lw=1.8, ls='--', alpha=0.85,
                  label='Structural break', zorder=4)
    ax_ts.axvspan(0, break_idx, alpha=0.06, color='#3498db')
    ax_ts.axvspan(break_idx, seq_len, alpha=0.06, color='#e74c3c')
    ax_ts.set_ylabel('Signal (normalised)', fontsize=10)
    ax_ts.legend(loc='upper right', fontsize=8.5)
    ax_ts.set_title(title, fontsize=11, pad=7)
    ax_ts.set_xlim(0, seq_len - 1)

    # Bottom panel: stacked area of routing weights
    bottom = np.zeros(len(patch_centers))
    for e in range(n_experts):
        y = sorted_seq[:, e]
        ax_rt.fill_between(patch_centers, bottom, bottom + y,
                           color=colors[e], alpha=0.82, zorder=2)
        bottom += y

    ax_rt.axvline(break_idx, color='#e74c3c', lw=1.8, ls='--', alpha=0.85, zorder=5)
    ax_rt.set_xlim(0, seq_len - 1)
    ax_rt.set_ylim(0, 1.03)
    ax_rt.set_xlabel('Timestep', fontsize=10)
    ax_rt.set_ylabel(r'$G(\mathbf{X})_k$  —  routing weight', fontsize=10)

    # Region labels just above the stacked area
    xform = ax_rt.get_xaxis_transform()
    ax_rt.text(break_idx * 0.5, 1.025, 'Normal operation',
               ha='center', va='bottom', fontsize=8.5, color='#2471a3',
               transform=xform, clip_on=False)
    ax_rt.text(break_idx + (seq_len - break_idx) * 0.5, 1.025,
               'Post-break regime',
               ha='center', va='bottom', fontsize=8.5, color='#c0392b',
               transform=xform, clip_on=False)

    # Expert legend
    handles = [
        mpatches.Patch(facecolor=colors[e], alpha=0.85, label=sorted_labels[e])
        for e in range(n_experts)
    ]
    ax_rt.legend(handles=handles, loc='lower right', fontsize=8,
                 ncol=min(n_experts, 4), framealpha=0.88, edgecolor='#cccccc')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4-new-2 — Latent Regime Dimensionality Reduction
# ---------------------------------------------------------------------------

def plot_regime_embedding(routing_vecs, dominant_expert, volatility,
                          expert_labels, save_path, title=''):
    """
    routing_vecs    : (N, n_experts) routing probability vectors
    dominant_expert : (N,) int — argmax expert index per sample
    volatility      : (N,) float — per-sample input σ
    expert_labels   : list of expert names

    Left panel: points coloured by the dominant expert (regime type).
    Right panel: same coordinates, coloured by input-signal volatility
                 (reveals whether structural intensity maps onto router space).
    """
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise ImportError('scikit-learn is required for regime embedding.') from exc

    N = routing_vecs.shape[0]

    if N >= 30:
        reducer_name = 't-SNE'
        perplexity = min(30, max(5, N // 10))
        coords = TSNE(n_components=2, perplexity=perplexity,
                      random_state=42, n_iter=1000,
                      init='pca').fit_transform(routing_vecs)
    else:
        reducer_name = 'PCA'
        coords = PCA(n_components=2, random_state=42).fit_transform(routing_vecs)

    colors = [_expert_color(lbl) for lbl in expert_labels]

    fig, (ax_exp, ax_vol) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: coloured by dominant expert
    for e, lbl in enumerate(expert_labels):
        mask = dominant_expert == e
        if mask.any():
            ax_exp.scatter(coords[mask, 0], coords[mask, 1],
                           c=colors[e], s=18, alpha=0.65, label=lbl,
                           edgecolors='none', zorder=3)
    ax_exp.set_title(f'Regime islands — dominant expert  ({reducer_name})',
                     fontsize=10)
    ax_exp.set_xlabel(f'{reducer_name}-1', fontsize=9)
    ax_exp.set_ylabel(f'{reducer_name}-2', fontsize=9)
    ax_exp.legend(fontsize=8.5, framealpha=0.88, edgecolor='#cccccc', loc='best')
    ax_exp.set_aspect('equal', 'datalim')
    ax_exp.grid(True, lw=0.4, alpha=0.4)

    # Right: coloured by input volatility
    vmin = float(np.percentile(volatility, 5))
    vmax = float(np.percentile(volatility, 95))
    sc = ax_vol.scatter(coords[:, 0], coords[:, 1],
                        c=volatility, cmap='RdYlGn_r',
                        vmin=vmin, vmax=vmax,
                        s=18, alpha=0.65, edgecolors='none', zorder=3)
    cbar = plt.colorbar(sc, ax=ax_vol, shrink=0.85, pad=0.02)
    cbar.set_label('Input volatility (σ)', fontsize=9)
    ax_vol.set_title(f'Regime islands — input volatility  ({reducer_name})',
                     fontsize=10)
    ax_vol.set_xlabel(f'{reducer_name}-1', fontsize=9)
    ax_vol.set_ylabel(f'{reducer_name}-2', fontsize=9)
    ax_vol.set_aspect('equal', 'datalim')
    ax_vol.grid(True, lw=0.4, alpha=0.4)

    fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
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
