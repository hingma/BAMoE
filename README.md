Based on the introduction, here's a structured experiment design covering all three contributions:

---

## Experiment Design for BAMoE

### 1. Preliminary Study: Temporal Inductive Bias Analysis
*Supports Contribution 1 — the empirical study*

**Goal:** Show that different biases work better on different datasets/horizons, motivating adaptive selection.

**Setup:** Train isolated single-bias Transformer variants (matching TIPS's bias-specialized teachers):
- **Causal** — standard autoregressive attention mask
- **Local** — sliding window / locality-aware attention
- **Periodic** — decomposition-based or periodic positional bias
- **Global** — full attention (no bias, baseline)

**Datasets:** Use standard long-term forecasting benchmarks: ETTh1/h2, ETTm1/m2, Weather, Traffic, Electricity, Exchange-Rate. These cover varying dominant patterns (trend-heavy, seasonal, noisy).

**Metrics:** MSE and MAE at prediction horizons {96, 192, 336, 720}.

**Expected finding:** No single bias dominates across all datasets/horizons — this is your key motivation table.

---

### 2. Main Forecasting Results
*Supports the core BAMoE proposal*

**Baselines to compare against:**

| Category | Models |
|---|---|
| Single-bias Transformers | Informer, Autoformer, ETSformer, iTransformer |
| Strong general baselines | PatchTST, Crossformer, TimesNet, DLinear |
| MoE forecasting | TimeMoE, SegMoE |
| Multi-bias distillation | TIPS (most critical baseline) |

**Evaluation:** Same 8 datasets × 4 horizons as above. Report average rank in addition to raw MSE/MAE to summarize across datasets.

**Key comparisons to highlight:**
- BAMoE vs. TIPS: demonstrates adaptive routing > static distillation
- BAMoE vs. TimeMoE/SegMoE: demonstrates heterogeneous > homogeneous experts
- BAMoE vs. best single-bias: demonstrates adaptive > fixed bias

---

### 3. Ablation Studies
*Supports Contribution 3*

Run on a representative subset (e.g., ETTh1, Weather, Traffic) at all horizons.

**3a. Expert diversity ablation**

Replace heterogeneous experts with homogeneous variants:

| Variant | Description |
|---|---|
| BAMoE-homo-causal | All experts use causal bias |
| BAMoE-homo-local | All experts use local bias |
| BAMoE-homo-periodic | All experts use periodic bias |
| BAMoE-K1 | Single expert (no MoE) |
| **BAMoE (full)** | Heterogeneous experts |

This directly validates the claim that *expert diversity is critical*.

**3b. Routing mechanism ablation**

| Variant | Description |
|---|---|
| BAMoE-uniform | Equal routing weights (no learned routing) |
| BAMoE-random | Random sparse routing |
| BAMoE-top1 | Hard routing (select 1 expert) |
| BAMoE-top2 | Top-2 sparse routing |
| BAMoE-dense | All experts active (dense routing) |
| **BAMoE (full)** | Learned sparse routing |

**3c. Number of experts K**

Sweep K ∈ {2, 3, 4, 6, 8} with heterogeneous biases. Shows the sweet spot between diversity and parameter cost.

---

### 4. Interpretability Analysis
*Supports Contribution 3 — understanding what the model learns*

**4a. Expert activation frequency**
Plot routing weights per dataset/horizon as a heatmap. Expected: periodic-biased experts activate more on seasonal datasets (Weather); local experts activate more at short horizons.

**4b. Routing dynamics over time**
For a single time series, visualize which expert is selected at each input window position. Shows that routing adapts to local temporal dynamics within a sequence.

**4c. Correlation with temporal characteristics**
Compute statistics per dataset (e.g., dominant frequency via FFT, autocorrelation at lag-1, trend strength) and correlate with which expert dominates. This quantitatively links routing behavior to temporal dynamics.

**4d. Expert attention pattern visualization**
Visualize learned attention maps for each expert on the same input. Confirms experts maintain distinct inductive biases after training (they don't collapse).

---

### 5. Efficiency Analysis

Since MoE adds parameters but uses sparse routing, report:
- Parameter count vs. forecasting performance (Pareto plot)
- FLOPs / inference time vs. performance
- Comparison against TIPS and iTransformer at matched parameter budgets

---

### Suggested Dataset Priority

| Priority | Datasets | Why |
|---|---|---|
| Primary | ETTh1, ETTh2, Weather, Traffic | Standard benchmarks, widely reported |
| Secondary | ETTm1, ETTm2, Electricity | Fill out the table |
| Optional | Exchange-Rate, Solar-Energy | Edge cases (noisy / strong periodicity) |

---

### Summary Table of Experiments

| # | Experiment | Contribution |
|---|---|---|
| 1 | Single-bias comparison across datasets | C1 (empirical study) |
| 2 | Main results vs. all baselines | C2 (BAMoE proposal) |
| 3a | Expert diversity ablation | C3 (diversity is critical) |
| 3b | Routing mechanism ablation | C3 (dynamic routing matters) |
| 3c | Number of experts sweep | C3 |
| 4a–d | Routing interpretability | C3 (interpretability) |
| 5 | Efficiency analysis | C2 support |

The most critical experiment to get right is **Experiment 1** — it establishes the premise of the whole paper — and **3a**, which is the direct empirical proof of your core design choice.