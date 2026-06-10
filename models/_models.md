# Temporal Inductive Bias Experts for BAMoE

## Overview

The core assumption of BAMoE is that different forecasting scenarios favor different temporal inductive biases. Rather than forcing a single Transformer to model all temporal dynamics, BAMoE constructs a set of heterogeneous experts, each explicitly encoding a specific temporal inductive bias through attention masks or positional bias functions.

Formally, the self-attention of expert $e$ is defined as

$$
\text{Attention}_e(Q,K,V)
=
\text{Softmax}
\left(
\frac{QK^\top}{\sqrt{d}}
+
M_e
+
B_e
\right)V
$$

where

- $M_e$ is an expert-specific attention mask,
- $B_e$ is an expert-specific positional bias matrix.

Different experts differ only in $(M_e, B_e)$, while sharing the same Transformer architecture.

---

# Expert 1: Global Context Expert

## Motivation

Many forecasting tasks require information from the entire historical window.

Examples:

- Long-term trends
- Structural breaks
- Multi-scale interactions
- Multivariate dependencies

The Global Expert acts as the unbiased baseline.

---

## Implementation

No mask and no positional bias:

$$
M = 0
$$

$$
B = 0
$$

Attention becomes

$$
A
=
\text{Softmax}
\left(
\frac{QK^\top}{\sqrt{d}}
\right)
$$

---

## Temporal Prior

Assumption:

> Any historical observation may be useful.

Equivalent to:

- Vanilla Transformer
- iTransformer
- Informer

---

## Strength

Useful when

- Long-range dependencies exist
- No clear temporal structure exists

---

# Expert 2: Causal Expert

## Motivation

Many time series exhibit autoregressive dynamics.

Examples:

- Energy demand
- Traffic flow
- Inflation
- Financial momentum

Recent observations influence future observations.

---

## Implementation

Use standard causal masking:

$$
M_{ij}
=
\begin{cases}
0 & i \ge j \\
-\infty & i < j
\end{cases}
$$

Thus

$$
A_{ij} = 0
$$

for future positions.

---

## Temporal Prior

Assumption:

> Future states are generated sequentially from previous states.

Equivalent to:

- RNN
- GRU
- LSTM
- Mamba

---

## Strength

Useful when

- Strong autoregression exists
- Momentum dominates
- Temporal ordering is critical

---

# Expert 3: Reverse-Causal Expert

## Motivation

Not all temporal dynamics are momentum-driven.

Some systems exhibit mean reversion.

Examples:

- Financial prices
- Inventory systems
- Control systems

Reverse attention often captures complementary information.

---

## Implementation

Reverse causal mask:

$$
M_{ij}
=
\begin{cases}
0 & i < j \\
-\infty & i \ge j
\end{cases}
$$

Within the historical window, tokens attend only to later observations.

---

## Temporal Prior

Assumption:

> Future observations within the lookback window help explain earlier observations.

---

## Strength

Useful when

- Mean-reversion exists
- Turning points are important
- Trend reversals occur

---

# Expert 4: Locality Expert (Window Attention)

## Motivation

Many forecasting tasks depend primarily on recent observations.

Examples:

- Electricity load
- Traffic forecasting
- Weather prediction

Distant observations often contain noise.

---

## Implementation

Fixed local window mask:

$$
M_{ij}
=
\begin{cases}
0 & |i - j| \le w \\
-\infty & \text{otherwise}
\end{cases}
$$

where $w$ is the locality window.

Common choices:

- 8
- 16
- 32

---

## Temporal Prior

Assumption:

> Nearby observations are more informative.

Equivalent to:

- CNN
- TCN
- Local Transformer

---

## Strength

Useful when

- Short-term dependencies dominate
- Noise increases with distance

---

# Expert 5: ALiBi Locality Expert

## Motivation

Hard windows may be too restrictive.

Instead, locality can be imposed softly.

---

## Implementation

Add distance-dependent penalty:

$$
B_{ij}
=
-m_h |i - j|
$$

where $m_h$ is the slope for attention head $h$.

Attention becomes

$$
A
=
\text{Softmax}
\left(
\frac{QK^\top}{\sqrt{d}}
-
m_h |i - j|
\right)
$$

---

## Temporal Prior

Assumption:

> Importance decays gradually with temporal distance.

---

## Strength

Useful when

- Recent observations matter most
- Long-range information remains occasionally useful

---

# Expert 6: Fixed Periodic Expert

## Motivation

Many time series contain explicit periodicity.

Examples:

- Hourly electricity demand
- Daily traffic
- Weekly sales
- Seasonal climate patterns

---

## Implementation

Define period $p$ and periodic distance

$$
d_{ij} = |i - j| \bmod p
$$

Periodic bias:

$$
B_{ij}
=
-\min(d_{ij},\; p - d_{ij})
$$

Observations with similar phases receive larger attention scores.

---

## Temporal Prior

Assumption:

> Similar temporal phases behave similarly.

---

## Strength

Useful when

- Seasonality dominates
- Stable periodic cycles exist

---

# Expert 7: Learnable Relative Position Expert

## Motivation

Real-world periodicity is rarely fixed.

The model should learn useful relative distances.

---

## Implementation

Relative Position Bias (RPB):

$$
B_{ij}
=
f_\theta(i - j)
$$

where $f_\theta$ is a learnable embedding table.

Common implementation:

```python
nn.Embedding(2 * max_len - 1, num_heads)
```

---

## Temporal Prior

Assumption:

> Useful temporal relationships can be learned from relative distances.

---

## Strength

Useful when

- Multiple seasonalities exist
- Temporal patterns are non-stationary

---

# Expert 8: Trend Expert (Recommended Extension)

## Motivation

Forecasting tasks often depend on long-term trends rather than raw observations.

TIPS does not explicitly encode trend structure.

---

## Implementation

Apply trend decomposition:

$$
x_t = T_t + R_t
$$

using moving average:

$$
T_t = \text{MA}(x_t)
$$

Self-attention operates on $T_t$ instead of the raw series.

---

## Temporal Prior

Assumption:

> Trend information is more important than short-term fluctuations.

---

## Strength

Useful when

- Strong trend exists
- Long-horizon forecasting

---

# Expert 9: Seasonal Expert (Recommended Extension)

## Motivation

Autoformer and ETSformer demonstrate that explicit seasonal modeling improves forecasting.

---

## Implementation

Extract seasonal component:

$$
S_t = x_t - T_t
$$

Attention is performed on $S_t$ alone.

---

## Temporal Prior

Assumption:

> Repeating seasonal structure dominates forecasting performance.

---

## Strength

Useful when

- Strong seasonality exists
- Periodic signals dominate

---

# Recommended BAMoE Expert Set

For a first paper, the recommended expert set is

$$
\mathcal{E}
=
\{
E_{\text{global}},\;
E_{\text{causal}},\;
E_{\text{local}},\;
E_{\text{ALiBi}},\;
E_{\text{fixed}},\;
E_{\text{relative}}
\}
$$

(6 experts)

because:

1. All experts share identical architecture.
2. Only masks/biases differ.
3. Parameter count remains comparable.
4. The router's behavior is easy to interpret.
5. Directly extends TIPS while introducing dynamic bias selection.

After the main model works, Trend and Seasonal experts can be added as a second-stage extension.
