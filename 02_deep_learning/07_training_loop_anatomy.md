# 2.7 - Anatomy of a Training Loop

The standard training loop is 10 lines. Every deep-learning trainer - from your hobby project to the full pipeline training GPT-4 - is this 10-line skeleton wrapped in layers of infrastructure.

## The 10 lines

```python
model.train()                      # set layers like dropout to training mode
for step in range(num_steps):
    batch_x, batch_y = get_batch()           # sample mini-batch
    logits = model(batch_x)                  # forward
    loss = F.cross_entropy(logits, batch_y)  # loss
    optimizer.zero_grad()                    # clear old grads
    loss.backward()                          # backprop
    optimizer.step()                         # update weights
    if step % log_interval == 0:
        print(f"step {step}: loss={loss.item():.4f}")
```

That's it. Every model. Always.

## What real trainers wrap around this

In `nanoGPT/train.py`, you'll find:
1. **Config loading** - Python-based config system.
2. **Distributed setup** - `torch.distributed.init_process_group`, rank/world-size logic.
3. **Dataset loading** - memory-mapped binary files for fast access.
4. **Model creation** - load from scratch, from checkpoint, or from GPT-2 pretrained.
5. **Mixed-precision setup** - `torch.amp` for bf16/fp16.
6. **Optimizer creation** - AdamW with per-param-group weight decay.
7. **Learning rate scheduler** - warmup + cosine decay.
8. **Gradient accumulation** - simulate a larger batch by accumulating grads over multiple micro-batches.
9. **Gradient clipping** - clamp gradient norm to avoid explosions.
10. **Validation loop** - periodically compute loss on held-out data.
11. **Checkpointing** - save model + optimizer state every N steps.
12. **Logging** - to wandb or stdout.
13. **torch.compile** - optional JIT for speed.

All of that is infrastructure. The 10-line core is unchanged.

Open `~/workspace/nanoGPT/train.py` and scroll through. Look for the lines:
```python
logits, loss = model(X, Y)
```
and
```python
scaler.scale(loss).backward()
```

Everything else is scaffolding.

## The validation loop

Periodically (e.g. every 250 steps), you stop training and compute val loss:

```python
model.eval()                        # turn off dropout
with torch.no_grad():               # no gradient tracking - saves memory
    val_losses = []
    for batch in val_batches:
        _, loss = model(batch.x, batch.y)
        val_losses.append(loss.item())
    val_loss = sum(val_losses) / len(val_losses)
model.train()                       # back to training mode
```

Val loss tells you whether you're genuinely learning or just memorizing. Watch it carefully.

## The pattern everywhere

Once this skeleton is in your bones, every training script you encounter - HuggingFace Trainer, Lightning, TRL's PPOTrainer, Megatron - becomes "this same loop, plus some bells." You'll be fine.

## Exercise

1. Open `nanoGPT/train.py`. Find the main `while True:` loop (the training loop). Identify:
   - Where is the forward pass?
   - Where is `backward()`?
   - Where is `optimizer.step()`?
   - Where is validation?
   - Where is the checkpoint save?
   Bookmark these lines mentally - we'll walk through them in Module 3.

2. Open `nanochat/scripts/base_train.py`. Much longer and more feature-rich. Find the same things. Notice how gradient accumulation and distributed wrap around the core.

## Next

`capstone_mlp.md` - Module 2 capstone: train a 2-layer MLP on character sequences. One step more capable than the bigram model.
