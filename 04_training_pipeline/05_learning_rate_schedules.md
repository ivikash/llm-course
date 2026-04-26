# 4.5 - Learning Rate Schedules

Getting the learning rate (LR) wrong is the single most common reason training fails. This lesson covers how to schedule LR over time - the standard recipe that everyone uses.

## The problem

- Too high: training explodes (loss goes to NaN).
- Too low: training crawls (months to converge).
- Sweet spot: varies by model size, batch size, architecture, optimizer.

And: you don't want the **same** LR throughout training. Early in training, weights are random - you want to take careful steps. Late in training, you want to fine-tune small adjustments. Hence: **scheduling**.

## The standard recipe: warmup + cosine decay

```
LR
 ^
max|      ___________
   |    /            \___
   |   /                 \___
min|_/___________________________\___
   0  warmup              decay_end    step
```

Two phases:
1. **Warmup**: linearly increase LR from 0 (or near 0) to max over the first few hundred/thousand steps.
2. **Cosine decay**: smoothly decrease LR from max to min following a cosine curve.

### Why warmup?

At step 0, weights are random. A big LR applies big updates based on noisy gradients. The model can diverge before ever learning anything. Warmup lets AdamW build up its running statistics and lets the model stabilize before taking full-size steps.

Typical: 100-2000 steps of warmup. For a big model, more (GPT-3 used 375M tokens of warmup).

### Why cosine decay?

Empirically it works. A smoothly decreasing LR lets the model do big exploration early, fine refinement late. Competitors:
- **Linear decay**: works similarly, slightly inferior.
- **Step decay**: drop by 10x at preset milestones. Classical, mostly replaced.
- **WSD (Warmup-Stable-Decay)**: hold LR flat most of training, short decay at the end. Used in some modern runs (Llama-3 variants).

## In nanoGPT

`train.py`:

```python
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))   # 1 → 0
    return min_lr + coeff * (learning_rate - min_lr)

# then at each step:
lr = get_lr(iter_num) if decay_lr else learning_rate
for param_group in optimizer.param_groups:
    param_group['lr'] = lr
```

Typical values (GPT-2 config):
- `learning_rate = 6e-4`
- `min_lr = 6e-5` (10x smaller)
- `warmup_iters = 2000`
- `lr_decay_iters = 600000` (= `max_iters`, so decay ends when training ends)

## Picking the max LR

Rule of thumb: as model gets bigger, use smaller LR.

Approximate formulas from the Chinchilla paper and others:
- GPT-2 small (124M): 6e-4
- GPT-2 medium (350M): 3e-4
- GPT-2 large (770M): 2.5e-4
- GPT-3 (175B): 3e-5

"Small model, big LR. Big model, small LR." Larger models have more parameters, each updated less aggressively.

nanochat automates this based on `depth`. See `scripts/base_train.py` - the `get_lr_init` function computes it from depth.

## Picking batch-size scaling (LR scaling)

If you double batch size, should you double LR? **Square-root rule** for SGD/AdamW: when batch doubles, LR increases by sqrt(2). In practice people often do linear (double), and it works okay up to a point.

**Linear scaling rule** (Goyal et al, 2017): LR proportional to batch size, up to ~8K global batch. Beyond that diminishing returns.

Most LLM training happens at batch sizes where critical-batch-size is saturated, so the linear rule holds.

## Warning signs

During training, watch:
- **Loss goes to NaN**: LR too high, or bad data, or mixed-precision issue.
- **Loss oscillates wildly**: LR too high. Reduce.
- **Loss plateaus early**: LR might be too low. Or you've hit the noise floor.
- **Grad norm explodes**: LR too high. Clip gradients and/or reduce LR.

## Gradient clipping: the safety net

Always combine LR scheduling with gradient clipping. nanoGPT:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)   # grad_clip = 1.0
```

If the total gradient norm across all params exceeds `grad_clip`, scale all gradients down proportionally. Prevents rare big updates from wrecking training. Next lesson covers this.

## In nanochat

nanochat's `scripts/base_train.py` does the same cosine schedule but computes max_lr from depth:

```python
def get_init_lr(depth: int) -> float:
    # empirical fit
    return ...
```

Also uses per-parameter-group LRs - embedding and head get different LR than the body. An important refinement used in modern LLM training.

## Visualize this

**The canonical LR schedule (warmup + cosine decay)**:

```
  LR
   ^
   │
   │       ╱──────╲╮
   │      ╱         ╲
   │     ╱            ╲
   │    ╱               ╲
   │   ╱                  ╲╮
   │  ╱                     ╲
   │ ╱                        ╲
   │╱                            ╲
   │   warmup        cosine decay  ╲____
   │ (linear)       (smooth)          ─────
   │
   └──────────────────────────────────────────► step
   0                                    max_iters
```

- Warmup: 100 to 2000 steps, LR climbs linearly from 0 to max.
- Peak: hold briefly (or not at all).
- Decay: cosine curve down to min_lr (usually 10% of max).

**Plot nanoGPT's schedule yourself**:

```python
import math
import matplotlib.pyplot as plt

def get_lr(it, warmup=2000, max_iters=600000, lr=6e-4, min_lr=6e-5):
    if it < warmup:
        return lr * (it + 1) / (warmup + 1)
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup) / (max_iters - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)

steps = list(range(0, 600000, 100))
lrs = [get_lr(s) for s in steps]
plt.plot(steps, lrs)
plt.xlabel("step"); plt.ylabel("learning rate")
plt.title("nanoGPT: warmup + cosine decay"); plt.grid()
plt.savefig("lr_schedule.png")
```

**Max LR by model size** (empirical rule):

```
  model size       max LR
  10M params       3e-3
  124M (GPT-2)     6e-4
  350M             3e-4
  770M             2.5e-4
  1.3B             2e-4
  6.7B             1.2e-4
  13B              1e-4
  175B (GPT-3)     6e-5
  1T               ~1e-5

  rough rule: LR ∝ 1 / sqrt(model_size)
```

Bigger model = smaller LR.

**Alternative schedules (awareness)**:

```
  Linear:                         Step decay:
      ───╲                            ────        ────        ────
          ╲                               ╲___    ╲___
           ╲                                        ╲___
            ╲___

  WSD (Warmup-Stable-Decay):      Reverse cosine:
      ────────────╮                    ╱╮
                  ╲                   ╱   ╲
                   ╲                 ╱      ╲
                    ╲___            ╱         ╲___

  (hold then drop)              (rare)
```

WSD is newer, sometimes preferred for long runs - stays at max LR longer.

**Warning: this is the single most important hyperparameter**:

```
  LR too high:         Loss diverges.
                         │ NaN
                         │╱
                         │
                         │
                         │  ●
                         │ ╱  ●
                         │●     ●●
                         ├────────────────
                         └──── step ──────

  LR too low:          Loss crawls.
                         │ ●
                         │  ●
                         │   ●
                         │    ●
                         │     ●
                         │      ● ← still has long way to go
                         └──── step ──────

  LR just right:       Loss drops smoothly, flattens appropriately.
```

Always log your grad_norm too - a spike often precedes divergence.

## Exercises

1. Plot nanoGPT's `get_lr(it)` for `warmup=2000, max_iters=600000, lr=6e-4, min_lr=6e-5`. Should look like the ASCII drawing above.

2. Train the Shakespeare toy with `decay_lr=True` (default) and `decay_lr=False` (constant LR). Compare final val loss.

3. Train with LR 10x higher than the default. Watch training blow up.

4. Open `nanochat/scripts/base_train.py` and find the LR function. How does it differ from nanoGPT's? (It parameterizes by depth and uses different groups.)

## Next

`06_gradient_clipping_nan_debugging.md` - when training goes bad, how to save it.
