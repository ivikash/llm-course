# 2.5 - Optimizers: From SGD to AdamW

Plain SGD works but is slow. A zoo of smarter optimizers exists. You need to know three: **SGD+momentum**, **Adam**, **AdamW**. And for context, two newer ones used in nanochat: **Muon** and the research landscape.

## SGD (vanilla)

```
w <- w - lr * grad
```

Simple. Works. Slow. No memory of past gradients.

## SGD + momentum

Keep a running "velocity" `v` and step in its direction:

```
v <- beta * v + grad
w <- w - lr * v
```

`beta` is typically 0.9. Intuition: like a ball rolling downhill, it builds up speed in consistent directions, damps oscillations.

## RMSprop (for context)

Keeps a running average of *squared* gradients per parameter, divides updates by its square root:

```
s <- beta2 * s + (1-beta2) * grad**2
w <- w - lr * grad / sqrt(s + eps)
```

Effect: parameters with small gradient history get big updates; parameters with large gradient history get small updates. A "per-parameter learning rate."

## Adam = momentum + RMSprop

```
m <- beta1 * m + (1-beta1) * grad       # momentum
s <- beta2 * s + (1-beta2) * grad**2    # RMS
# bias correction (small initialization fix):
m_hat = m / (1 - beta1**t)
s_hat = s / (1 - beta2**t)
w <- w - lr * m_hat / (sqrt(s_hat) + eps)
```

Default hyperparams: `beta1=0.9, beta2=0.999, eps=1e-8`. This is the workhorse optimizer of deep learning. Fast, robust, mostly "just works."

## AdamW = Adam with correct weight decay

Adam has a problem when combined with "weight decay" (L2 regularization on weights). Classical Adam entangles weight decay with the gradient-normalization, which is wrong in a subtle way. **AdamW** decouples them:

```
# Adam update (same as above)
w <- w - lr * m_hat / (sqrt(s_hat) + eps)
# PLUS explicit weight decay (separate from Adam):
w <- w - lr * weight_decay * w
```

This small fix meaningfully improves generalization. Since ~2019, AdamW is the default for training transformers. **This is what nanoGPT uses.** **This is what nanochat uses** (with some exceptions for specific tensors).

## In nanoGPT

`model.py`, `configure_optimizers`:

```python
# separate out parameters into decay (2D tensors) and no-decay (1D: biases, layernorms)
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
optim_groups = [
    {'params': decay_params, 'weight_decay': weight_decay},
    {'params': nodecay_params, 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, ...)
```

Key detail: **weight decay is only applied to 2D weight matrices**, not to biases or LayerNorm gains. This is standard practice.

## Muon (nanochat uses this on some params)

Muon is a 2024 optimizer based on orthogonalization of momentum via Newton-Schulz iteration. It's used in `nanochat/nanochat/optim.py`. Good at higher learning rates, faster per-token progress. Competitive with AdamW. Still newer / less battle-tested.

Open `~/workspace/nanochat/nanochat/optim.py` and skim. You'll see a `Muon` class alongside `AdamW`.

## Learning rate is everything

No matter the optimizer, the **learning rate schedule** dominates training. Too big: diverges. Too small: crawls. Standard recipe (which nanoGPT and nanochat both use):
- **Warmup**: start at 0, ramp up over the first few hundred/thousand steps.
- **Cosine decay**: ramp down smoothly to a small min lr by the end.

Covered in Module 4 Lesson 5.

## Visualize this

**SGD vs SGD+Momentum vs Adam, on a loss surface**:

```
                    ╲                         ╱
                     ╲                       ╱
                      ╲        ────         ╱
   SGD: bounces ───▶   ╲╱╲ ╱╲ ╱╲ ╱╲ ╱╲ ╱╲ ╱╲╱        zigzag
                              valley
                      ╲      ────        ╱
   SGD+momentum:       ●──●──●──●──●──●──●            smooths, goes faster
                       (builds up velocity)
                      ╲      ────        ╱
   Adam: like momentum but with per-parameter learning rates
                       ●──●──●──●──●──●──●            best of both
   (adapts to steep vs shallow directions)
```

Hard to draw well in ASCII. Watch **Sebastian Ruder's optimizer comparison gif**: https://ruder.io/optimizing-gradient-descent/ - animated 2D loss surfaces show each optimizer tracing a different path to the minimum. Highly recommended.

**Or watch them race right here:**

```viz
{"viz": "optimizers_compare"}
```

Press **Run**. SGD (red), SGD+momentum (orange), and Adam (blue) all start from the same point. They race through a curved valley to the minimum at (3,3). You'll see:
- SGD zigzags across the narrow direction, slowly.
- SGD+momentum smooths the oscillation, gets there faster.
- Adam's per-parameter learning rates glide through the curved valley efficiently.

Try changing the learning rate. Too big → bounce out. Too small → crawl.

**The "why AdamW decouples weight decay" picture**:

```
  Classical Adam + L2 penalty (wrong):
    gradient includes L2 penalty
         ↓
    Adam's per-param scaling normalizes the gradient, which also scales the penalty
         ↓
    weights with small gradients don't get decayed as much → subtle miscalibration

  AdamW (fixed):
    gradient: just the loss gradient
         ↓
    Adam scales only the loss gradient
         ↓
    separately subtract lr × weight_decay × w  (outside of Adam)
         ↓
    every weight gets the same decay, regardless of gradient magnitude → clean
```

This small fix (Loshchilov & Hutter 2017) improved generalization across the board. Every modern LLM uses AdamW.

**Which params get weight decay (2D yes, 1D no)**:

```
  Weight decay applied to:         Weight decay NOT applied to:
  ┌───┬───┬───┐                    ┌─┐   (bias vectors)
  │ . │ . │ . │  nn.Linear weights │.│   (LayerNorm gains/biases)
  │ . │ . │ . │  embeddings        └─┘
  │ . │ . │ . │  attention Q,K,V    ┌─┐
  └───┴───┴───┘                     │.│
                                    │.│
                                    └─┘
```

Rule: only 2D (and higher) tensors get weight decay. Biases and LayerNorm parameters stay untouched. Codified in every modern trainer's `configure_optimizers` (see nanoGPT line ~260).

## Exercise

1. Run the same gradient-descent example from Lesson 1.3 with `torch.optim.SGD`, `torch.optim.Adam`, `torch.optim.AdamW`. Compare convergence speed.

```python
import torch
w = torch.tensor([0.0], requires_grad=True)
opt = torch.optim.AdamW([w], lr=0.3, weight_decay=0.0)  # try each
for step in range(100):
    loss = (w * 2 - 10) ** 2
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(step, w.item(), loss.item())
```

2. In nanoGPT/model.py's `configure_optimizers`, identify which parameters get weight decay and which don't. Match the 2D/1D rule above.

## Next

`06_regularization_dropout_weightdecay.md` - stop the model from memorizing.
