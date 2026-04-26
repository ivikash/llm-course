# 4.4 - Gradient Accumulation

Large batch sizes help training stability and scaling - but they can't always fit in GPU memory. **Gradient accumulation** lets you simulate a big batch by accumulating gradients over several smaller "micro-batches" before doing one optimizer step.

## The idea

Normally:

```python
for step in range(N):
    x, y = get_batch(batch_size=64)
    loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

With gradient accumulation:

```python
ACCUM = 4
for step in range(N):
    optimizer.zero_grad()
    for micro in range(ACCUM):
        x, y = get_batch(batch_size=16)          # smaller, fits in memory
        loss = model(x, y) / ACCUM                # divide to keep scale right
        loss.backward()                          # gradients accumulate
    optimizer.step()                             # step after ACCUM micro-batches
```

Effect: effective batch is `16 * 4 = 64`, same gradient as a single big batch, same step count. Only difference: slower (ACCUM sequential micro-passes instead of one parallel).

**Why divide the loss by ACCUM?** Because when we call `.backward()` multiple times, gradients sum. Dividing the loss gives us the average, which is what a single batch of size 64 would have produced.

## In nanoGPT

```python
# config
batch_size = 12
gradient_accumulation_steps = 5 * 8     # = 40
# effective batch = 12 * 40 = 480 sequences * 1024 tokens = ~500K tokens per step

# training step
optimizer.zero_grad(set_to_none=True)
for micro_step in range(gradient_accumulation_steps):
    if ddp:
        model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
    with ctx:
        logits, loss = model(X, Y)
        loss = loss / gradient_accumulation_steps
    X, Y = get_batch('train')   # prefetch next batch
    scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Two tricks here:
1. `loss = loss / gradient_accumulation_steps` - scale for averaging.
2. `model.require_backward_grad_sync = (micro_step == last)` - in DDP, don't all-reduce gradients on every micro-step (expensive). Only sync on the last one.

## The effective-batch formula

```
effective_batch = micro_batch_size * gradient_accumulation_steps * num_gpus
effective_tokens_per_step = effective_batch * block_size
```

For nanoGPT's GPT-2 config:
- `batch_size = 12` (per GPU)
- `grad_accum = 40`
- 8 GPUs (torchrun --nproc_per_node=8)
- `block_size = 1024`

Effective batch: `12 * 40 * 8 = 3840 sequences`. Tokens per step: `3840 * 1024 ≈ 3.9M`. That's the famous ~500K token batch of GPT-2/3-era models.

## Why you want big batches

- **Stability**: gradient noise decreases with batch size.
- **Optimizer math**: Adam's running averages smooth out with bigger batches.
- **GPU utilization**: bigger batches = more arithmetic per memory access.

But bigger-is-not-always-better. The "critical batch size" depends on the problem; past it, you get no gains from more parallelism, just wasted compute.

In nanochat, see `runs/speedrun.sh`'s 2026-02-05 update:

> "bump total batch size to 1M tokens"

It's an active hyperparameter they tune.

## Limits

- Doesn't solve OOM for activation memory if a single micro-batch is too big. You'd also need gradient checkpointing.
- Slows down wall clock (ACCUM sequential passes).
- Useful specifically when: your batch target > GPU capacity, but micro-batch fits.

## Gradient checkpointing (brief mention)

Separate trick: trade compute for activation memory. During forward, don't save activations of intermediate layers. During backward, recompute them.

PyTorch: `torch.utils.checkpoint.checkpoint(layer, x)`. Typically 30% slower but halves activation memory. Combined with gradient accumulation, you can train much bigger models than naive fit.

nanoGPT doesn't use this; nanochat also doesn't (they tune other knobs). Standard in FSDP setups for big models.

## Exercises

1. Train a small model with `grad_accum=1` vs `grad_accum=4` (same effective batch). Wall clock should be similar or slightly slower for ACCUM=4. Loss curves should overlap.

2. Compute the effective batch for nanochat's speedrun:
   - See `runs/speedrun.sh` for `--device-batch-size=16`, 8 GPUs.
   - `grad_accum` auto-computed to reach a target total batch. Look in `scripts/base_train.py` for the target.

3. In nanoGPT, reduce `batch_size` by half and double `grad_accum`. Keep effective batch constant. Verify training is equivalent (loss curves match).

4. Why does DDP all-reduce only on the last micro-step? Would always syncing give wrong results? (No, just slow - gradients sum anyway, but you'd pay the network cost N times instead of once.)

## Next

`05_learning_rate_schedules.md` - the single most important hyperparameter to get right.
