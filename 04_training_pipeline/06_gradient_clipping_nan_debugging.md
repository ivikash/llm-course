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

## Exercises

1. Train Shakespeare with `grad_clip = 0.0` (disable clipping) and LR 5x higher than default. Watch training explode.

2. Add NaN detection to nanoGPT's training loop. Trigger it artificially by setting one weight to `nan`. Verify your detector works.

3. Read `scaler.unscale_` vs `scaler.scale`. Understand why you call one before clipping and the other before backward.

## Next

`07_checkpointing_and_resume.md` - save/restore, without losing a week of training.
