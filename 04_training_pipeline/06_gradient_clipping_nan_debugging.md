# 4.6 - Gradient Clipping and NaN Debugging

Training goes bad. The loss becomes `nan`, or explodes. What do you do?

## Gradient clipping: the standard safety net

Before the optimizer step, compute the total gradient norm (across all parameters). If it exceeds a threshold, scale all gradients down proportionally so the norm equals the threshold.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Standard value: `max_norm = 1.0` for LLM training.

**Why**: rare, spiky batches can produce huge gradients that send weights flying. Without clipping, one bad step can ruin a days-long training run. With clipping, the damage is bounded.

**Frequency**: almost every modern LLM training run uses it. nanoGPT: yes. nanochat: yes.

## What NaN means

"NaN" (Not a Number) is a special float value that arises from:
- `0/0`
- `log(0)` or `log(negative)`
- `sqrt(negative)`
- `inf - inf`
- Overflow in fp16

Once a NaN appears in a weight or gradient, every subsequent operation with it also produces NaN. Training is done.

## Common causes

### 1. Learning rate too high

Most common cause. Halve the LR and try again.

### 2. Uninitialized or poorly initialized weights

Models are usually ok with default init, but if you hand-roll something and forget to init properly, first forward pass can produce huge values. Watch the very first loss - should be roughly `-log(1/vocab_size)` for an LM.

### 3. Bad data

A malformed example (a sequence of zeros, a super-long token ID, wrong encoding) can produce weird gradients. Check your `get_batch`.

### 4. fp16 overflow

In fp16, values above ~65K become inf. Activations in unhealthy ranges cause this. Switch to bf16, or use loss scaling, or tune init.

### 5. Division by zero

Somewhere in your model. `x / x.sum()` with `x.sum() == 0`. Add an eps: `x / (x.sum() + 1e-8)`.

### 6. Numerical instability in custom code

If you write your own attention or loss function, you can easily create instability. Use PyTorch's built-in `F.softmax`, `F.cross_entropy`, `F.scaled_dot_product_attention` - they use numerically stable implementations.

## Debugging workflow

1. **Check if loss was ever okay**: if loss was 10 at step 0 and then NaN at step 100, something went wrong during training. If NaN at step 0, it's an init problem.

2. **Enable anomaly detection**:
```python
torch.autograd.set_detect_anomaly(True)  # slower but shows where NaN appeared
```

3. **Add assertions**:
```python
if torch.isnan(loss).any():
    print("NaN detected at step", step)
    print("x:", x.min(), x.max())
    print("logits:", logits.min(), logits.max())
    breakpoint()
```

4. **Reduce LR**. First thing. Halve it.

5. **Try bf16 if using fp16**. Solves range issues.

6. **Re-check your data prep**. Any malformed examples?

7. **Inspect weight magnitudes**:
```python
for name, p in model.named_parameters():
    print(name, p.abs().mean().item(), p.abs().max().item())
```

Healthy weights are small, ~0.01-1.0 range. Anything over 100 is suspicious.

## Loss scale gotchas

When using fp16 + GradScaler, if the scaler detects overflow, it *skips* that step and reduces the scale. For the first dozen steps you might see:

```
scale decreased to 16384
scale decreased to 8192
...
scale stabilized
```

That's normal. If it never stabilizes, LR is too high or there's another issue.

## Gradient norm monitoring

Log the gradient norm every step. It should be O(1). If it's 100+, clipping is bailing you out - investigate. If it's 0, something's wrong (dead weights, gradient flow blocked).

```python
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
if step % 100 == 0:
    print(f"grad_norm = {total_norm.item():.4f}")
wandb.log({"grad_norm": total_norm.item()})
```

## In nanoGPT

```python
if grad_clip != 0.0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
scaler.step(optimizer)
scaler.update()
```

Note: `scaler.unscale_` undoes the fp16 loss scaling before clipping, so the clip threshold is in the original gradient units.

## nanochat's MFU / grad-norm logging

Open `~/workspace/nanochat/scripts/base_train.py`. Find where it logs `train/grad_norm` to wandb. That's what you watch during long training runs to detect instability early.

## Your checklist when training goes bad

1. Did loss NaN from step 0, or later? (Init vs runtime bug.)
2. Did grad norm spike just before NaN? (Bad batch.)
3. Is LR appropriate for model size? (Table in Lesson 4.5.)
4. Is mixed precision set right? (bf16 if possible.)
5. Any custom layers or losses? (Suspect them first.)
6. Is data sane? (Check tokenizer output.)
7. When was the last known good run? (Diff the changes.)

## Visualize this

**The sign of unstable training (log this!)**:

```
  grad_norm
     ^
     │                 │ ← spike! (investigate!)
     │                 │
     │                 │      ●
     │                 │      │
     │                 │      │
     │                 │      ▼ training diverges here if unclipped
     │     ─────────────
     │  ─── stable at ~1.0 ──
     │
     └───────────────────────────► step
```

Healthy: grad_norm stays near 1.0 (after clipping kicks in). Spikes to 10+ = something bad.

**Gradient clipping, pictorially**:

```
  Without clipping:
  gradient vector:
  ●──────────────────────────────────►  (norm = 50, huge!)
  weights updated by: -lr × 50 = chaos

  With clip_grad_norm(max_norm=1.0):
  gradient vector (original):
  ●──────────────────────────────────►  (norm = 50)

  scale factor = 1.0 / 50 = 0.02
  clipped gradient:
  ●─►                                     (norm = 1.0)
  weights updated by: -lr × 1.0 = reasonable step
```

Preserves direction, bounds magnitude.

**NaN investigation flowchart**:

```
  loss.item() is NaN
         │
         ▼
  Was loss OK at step 0?
      │       │
      │       ▼
      │   yes (started OK, NaN'd later)
      │       │
      │       ▼
      │   Check: did grad_norm spike right before?
      │          │
      │          ├─ yes → likely LR too high or bad batch
      │          └─ no  → likely fp16 underflow (switch to bf16)
      │
      ▼
   no (NaN from step 0)
          │
          ▼
      Check: is initial loss reasonable?
         expected: ~log(vocab_size), e.g. 10.8 for 50257 vocab
              │
              ├─ yes (10.8) → check first backward grads for NaN
              └─ no  → init problem, bad weights
```

**Common causes, in order of likelihood**:

1. **LR too high** (70% of cases). Halve it.
2. **Mixed precision overflow** (fp16). Switch to bf16.
3. **Bad data** (a weird token ID, zero-length sequence). Check your tokenizer.
4. **Custom layer with division by zero**. Add `eps=1e-8`.
5. **Catastrophic optimizer state**. Fresh optimizer, resume from last good ckpt.

**Healthy vs unhealthy training curves**:

```viz
{"viz": "training_curve_simulator"}
```

Slide the LR factor from 0.1 to 5. Watch the shape change:
- LR 0.2 → slow, hasn't plateaued (undertrained).
- LR 1.0 → clean exponential decay, nice plateau (healthy).
- LR 3.0 → oscillations (unstable).
- LR 4.0 → diverges (loss explodes).
- Multiple seeds with noise → see variance in the curves.

This is **exactly the dashboard you'll stare at for weeks** while training.

```
  Healthy:
    loss
     │●
     │ ●
     │  ●
     │   ●●
     │     ●●●●●●●
     │           ●●●●●●●●●●●  (smooth decay, then plateau)
     └─────────────────────── step

  Unhealthy (LR too high):
    loss
     │●
     │ ●             NaN
     │  ●             │
     │   ●●           │
     │     ●●  ●●●●●  │
     │           ●●●  ↓ ..........
     └─────────────────────── step

  Unhealthy (overfitting):
    train ●●●●●●●●●●●●●●
    val   ●●●●●●●●●●●
                      ●●  ← diverging from train
                        ●●
                          ●●
```

Spot the shape early; save hours.

## Exercises

1. Train Shakespeare with `grad_clip = 0.0` (disable clipping) and LR 5x higher than default. Watch training explode.

2. Add NaN detection to nanoGPT's training loop. Trigger it artificially by setting one weight to `nan`. Verify your detector works.

3. Read `scaler.unscale_` vs `scaler.scale`. Understand why you call one before clipping and the other before backward.

## Next

`07_checkpointing_and_resume.md` - save/restore, without losing a week of training.
