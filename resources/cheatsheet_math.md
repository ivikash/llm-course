# Math Cheatsheet

A quick reference for the math vocabulary used in this course. Keep this tab open.

## Shapes

- Scalar: `()` - a single number.
- Vector: `(n,)` - 1-D array.
- Matrix: `(m, n)` - 2-D array, m rows × n cols.
- 3D tensor: `(B, T, C)` - batch × sequence × channels.
- 4D tensor: `(B, n_head, T, head_dim)` - multi-head attention.

## Matrix multiplication

`(m, k) @ (k, n) -> (m, n)`

Inner dims must match. Outer dims remain.

In PyTorch: `A @ B` or `torch.matmul(A, B)`.

## Common operations

| Op | Formula | PyTorch |
|----|---------|---------|
| Dot product | `sum(a_i * b_i)` | `a @ b` or `(a*b).sum()` |
| Norm of vector | `sqrt(sum(a_i^2))` | `a.norm()` |
| Mean | `sum(x_i) / n` | `x.mean()` |
| Std | `sqrt(E[(x - mean)^2])` | `x.std()` |
| Softmax | `exp(x_i) / sum_j exp(x_j)` | `F.softmax(x, dim=-1)` |
| LogSoftmax | `log(softmax(x))` (stable) | `F.log_softmax(x, dim=-1)` |
| Cross-entropy | `-log(p_true)` | `F.cross_entropy(logits, targets)` |
| MSE | `mean((pred - target)^2)` | `F.mse_loss(pred, target)` |

## Derivatives (conceptual)

| Function | Derivative (w.r.t. x) |
|----------|------------------------|
| `x` | `1` |
| `a*x` | `a` |
| `x^2` | `2x` |
| `x^n` | `n*x^(n-1)` |
| `exp(x)` | `exp(x)` |
| `log(x)` | `1/x` |
| `sin(x)` | `cos(x)` |
| `cos(x)` | `-sin(x)` |
| `1 / (1+exp(-x))` (sigmoid) | `sigmoid(x) * (1 - sigmoid(x))` |
| `relu(x)` | `1 if x > 0 else 0` |
| `tanh(x)` | `1 - tanh(x)^2` |

In practice: **you don't compute these.** PyTorch does. But knowing the shapes is useful.

## Gradient (the only calculus you use daily)

For a loss `L(w_1, ..., w_n)`, the **gradient** is the vector `(dL/dw_1, ..., dL/dw_n)`.

Gradient descent step: `w_i <- w_i - lr * (dL/dw_i)`

In PyTorch: `loss.backward()` fills `.grad` on every `requires_grad=True` param.

## Probability cheats

- Probability distribution: non-negative numbers summing to 1.
- `log(prob)` is always ≤ 0 (since prob ≤ 1). Bigger absolute value = smaller probability = more surprise.
- `-log(prob)` is "surprise" in nats. Lower is better.
- Cross-entropy loss: `-log(prob_of_correct_answer)`.
- Perplexity: `exp(cross_entropy_loss)`. Interpretable as "effective number of choices."

## Attention

```
score = Q @ K^T / sqrt(d)        # (T, T)
score = masked_fill(score, mask, -inf)
weights = softmax(score, dim=-1) # (T, T), rows sum to 1
output = weights @ V             # (T, d_v)
```

`Q, K, V` come from linear projections of the input.

## Parameter count rules of thumb (transformer)

- Per block: `12 * C^2` (where C = n_embd).
- Token embedding: `vocab_size * C`.
- Whole model: `vocab_size * C + L * 12 * C^2` (ignoring biases, which are small).

For GPT-2 small (C=768, L=12, V=50257): ≈ 124M.
For GPT-3 (C=12288, L=96, V=50257): ≈ 175B.

## Flops rules of thumb

- Forward pass: `~2 * N * T` FLOPs, where N = params, T = tokens in the batch. (6N per token for forward+backward combined.)
- Training FLOPs: `~6 * N * D`, where D = total tokens trained on.
- Example: GPT-3 training = `6 * 175B * 300B` = `3.15e23` FLOPs.
- An H100 does `~1e15` fp16 FLOPS. So GPT-3 training = `3.15e8` H100-seconds = ~100 H100-days. Of course they used 10,000 GPUs to train it in days.

## Scaling laws (Chinchilla, 2022)

Compute-optimal:
- `D ≈ 20 * N` (train on ~20 tokens per parameter).
- Loss: `L(N, D) ≈ 1.69 + (406.4 / N^0.34) + (410.7 / D^0.28)` (approximate - don't memorize exact constants).

For your own experiments: if you have FLOPs budget C, set `N ≈ sqrt(C / 120)` and `D ≈ sqrt(C * 20 / 6)`. Or just trust whatever nanochat's depth dial gives you.

## Memory estimation

Per-parameter memory costs (during training):
- fp32: 4 bytes for params, 4 for grad, 8 for AdamW state (m, v). Total 16 bytes/param.
- bf16 with fp32 master + AdamW: 6 bytes params (4 master + 2 bf16 copy), 4 grad, 8 state. ~18 bytes/param.
- fp16 with AMP: similar to above, ~16-20 bytes/param.

Activation memory: `~B * T * C * L` with a constant factor depending on implementation. Gradient checkpointing trades compute for this.

## Conversions

- 1K = 1000 (when talking about parameters/tokens, usually).
- 1M = 1e6
- 1B = 1e9 (parameters) or 1 GB (bytes)
- 1T = 1e12 (parameters: trillion) or 1 TB (bytes)
- 1 FLOP = one floating-point operation.
- 1 GFLOP = 1e9, 1 TFLOP = 1e12, 1 PFLOP = 1e15.

The "MFU" (Model FLOPs Utilization) metric: actual FLOPs / theoretical peak FLOPs. 40-50% is good. 60% is excellent. 80%+ only on custom kernels.
