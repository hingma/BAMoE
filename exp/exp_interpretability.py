"""
Experiment 4 — Interpretability Analysis

Runs BAMoE on the test set with routing capture enabled, then produces:
  4a. Expert activation heatmap (per dataset/horizon)
  4b. Routing dynamics for a representative sample
  4c. Correlation with temporal characteristics (FFT, autocorr, trend)
  4d. Expert attention-pattern visualisation (hooks into the first-layer experts)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats

from data_provider.data_factory import data_provider
from models.BAMoE import BAMoE
from utils.tools import count_parameters
from utils.visualize import (
    plot_expert_heatmap, plot_routing_dynamics,
    plot_routing_vs_temporal_stats, plot_attention_patterns,
)


class ExpInterpretability:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            f'cuda:{args.gpu}' if (args.use_gpu and torch.cuda.is_available()) else 'cpu'
        )
        self.model = BAMoE(args).to(self.device)
        ckpt = os.path.join(args.checkpoints, args.exp_name, 'checkpoint.pth')
        if os.path.exists(ckpt):
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
            print(f'Loaded checkpoint: {ckpt}')
        else:
            print(f'WARNING: checkpoint not found at {ckpt}; using random weights.')

        self.expert_types = [e.strip() for e in args.expert_types.split(',')]
        self.out_dir = os.path.join(args.results, args.exp_name, 'interpretability')
        os.makedirs(self.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 4a + 4b: collect routing weights from test set
    # ------------------------------------------------------------------

    def collect_routing(self):
        _, loader = data_provider(self.args, 'test')
        all_routing = []  # each entry: (B*C, N, n_experts) from last layer
        self.model.eval()
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                _, _, routing_log = self.model(x, return_routing=True)
                if routing_log:
                    # Use last layer's routing logits; softmax to get weights
                    last = routing_log[-1]          # (B*C, N, n_experts)
                    weights = F.softmax(last, dim=-1).cpu().numpy()
                    all_routing.append(weights)

        if not all_routing:
            print('No routing logits captured (model may use uniform/random routing).')
            return None

        routing_arr = np.concatenate(all_routing, axis=0)  # (total_BC, N, E)
        np.save(os.path.join(self.out_dir, 'routing_weights.npy'), routing_arr)
        return routing_arr

    # ------------------------------------------------------------------
    # 4a — heatmap
    # ------------------------------------------------------------------

    def plot_4a(self, routing_arr):
        avg = routing_arr.mean(axis=(0, 1))           # (n_experts,)
        label = f"{self.args.data}/{self.args.pred_len}"
        matrix = avg[np.newaxis, :]                   # (1, n_experts)
        save_path = os.path.join(self.out_dir, 'expert_heatmap.pdf')
        plot_expert_heatmap(matrix, self.expert_types, [label], save_path)
        print(f'Saved: {save_path}')

    # ------------------------------------------------------------------
    # 4b — routing dynamics for first test sample
    # ------------------------------------------------------------------

    def plot_4b(self, routing_arr):
        sample = routing_arr[0]                       # (N, n_experts)
        save_path = os.path.join(self.out_dir, 'routing_dynamics.pdf')
        title = f"Routing dynamics — {self.args.data} pred_len={self.args.pred_len}"
        plot_routing_dynamics(sample, self.expert_types, save_path, title=title)
        print(f'Saved: {save_path}')

    # ------------------------------------------------------------------
    # 4c — temporal statistics
    # ------------------------------------------------------------------

    def plot_4c(self, routing_arr):
        _, loader = data_provider(self.args, 'test')
        series_list = []
        for x, _ in loader:
            series_list.append(x.numpy())
            if len(series_list) * x.shape[0] > 200:
                break
        series = np.concatenate(series_list, axis=0)  # (N_samples, seq_len, C)

        # aggregate over channels and samples for a single representative series
        flat = series[:, :, 0]  # (N, seq_len)

        def dominant_freq(s):
            fft = np.abs(np.fft.rfft(s - s.mean()))
            return np.argmax(fft[1:]) + 1

        def autocorr_lag1(s):
            return float(np.corrcoef(s[:-1], s[1:])[0, 1])

        def trend_strength(s):
            _, _, r, _, _ = scipy_stats.linregress(np.arange(len(s)), s)
            return float(abs(r))

        stats = {
            self.args.data: {
                'dominant_freq': float(np.median([dominant_freq(s) for s in flat])),
                'autocorr_lag1': float(np.median([autocorr_lag1(s) for s in flat])),
                'trend_strength': float(np.median([trend_strength(s) for s in flat])),
            }
        }
        routing_dict = {
            self.args.data: routing_arr.mean(axis=(0, 1))
        }
        save_path = os.path.join(self.out_dir, 'routing_vs_temporal_stats.pdf')
        plot_routing_vs_temporal_stats(stats, routing_dict, self.expert_types, save_path)
        print(f'Saved: {save_path}')

    # ------------------------------------------------------------------
    # 4d — attention patterns (hooks into first-layer expert forward passes)
    # ------------------------------------------------------------------

    def plot_4d(self):
        _, loader = data_provider(self.args, 'test')
        x, _ = next(iter(loader))
        x = x[:1].to(self.device)                    # single sample

        # Channel-independent: take first channel only
        B, L, C = x.shape
        x_ci = x[:, :, :1]                           # (1, L, 1)

        attn_maps = []

        def make_hook(container):
            def hook(module, inp, out):
                # inp[0]: (B*C, N, D), compute attention map for display
                xi = inp[0]
                bci, n, d = xi.shape
                q = module.q(xi)
                k = module.k(xi)
                dh = module.d_head
                H = module.n_heads
                q = q.view(bci, n, H, dh).transpose(1, 2)
                k = k.view(bci, n, H, dh).transpose(1, 2)
                attn = (q @ k.transpose(-2, -1)) / module.scale  # (bci, H, N, N)
                attn = torch.softmax(attn, dim=-1)
                container.append(attn[0, 0].detach().cpu().numpy())  # first head
            return hook

        first_moe = self.model.layers[0].attn
        hooks = []
        for expert in first_moe.experts:
            container = []
            hooks.append((expert.register_forward_hook(make_hook(container)), container))

        self.model.eval()
        with torch.no_grad():
            self.model(x_ci)

        for h, _ in hooks:
            h.remove()

        maps = [c for _, c in hooks if c]
        if maps:
            attn_maps = [c[0] for c in [cont for _, cont in hooks]]
            save_path = os.path.join(self.out_dir, 'expert_attention_patterns.pdf')
            plot_attention_patterns(attn_maps, self.expert_types, save_path)
            print(f'Saved: {save_path}')

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self):
        routing_arr = self.collect_routing()
        if routing_arr is not None:
            self.plot_4a(routing_arr)
            self.plot_4b(routing_arr)
            self.plot_4c(routing_arr)
        self.plot_4d()
        print('Interpretability analysis complete.')
