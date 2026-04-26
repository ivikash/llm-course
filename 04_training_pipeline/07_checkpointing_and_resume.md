# 4.7 - Checkpointing and Resume

Long training runs fail. GPUs crash. Clouds preempt your spot instance. Your power goes out. If you can't resume from where you stopped, you lose days or weeks of compute.

## What to save

A **checkpoint** must include everything needed to resume training exactly:

1. **Model weights** (`model.state_dict()`)
2. **Optimizer state** (`optimizer.state_dict()`) - includes AdamW's `m` and `v` tensors
3. **LR scheduler state** (if you use one)
4. **Current step number**
5. **Model config** (so you can reconstruct the model class)
6. **Training config** (so you know what hyperparameters were being used)
7. **Best val loss so far** (to decide which checkpoint to keep)
8. **RNG state** (optional, for exact reproducibility)

Missing any of these and your "resume" is actually "restart from scratch, hope for the best".

## Writing a checkpoint (nanoGPT-style)

```python
checkpoint = {
    'model': raw_model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': model_args,
    'iter_num': iter_num,
    'best_val_loss': best_val_loss,
    'config': config,
}
torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
```

`raw_model` (vs `model`): if using DDP, `model` is a `DistributedDataParallel` wrapper. `raw_model = model.module` extracts the underlying nn.Module. Save the inner one - DDP wrappers are recreated on resume.

## Reading a checkpoint (nanoGPT-style)

```python
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
model_args = checkpoint['model_args']
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
state_dict = checkpoint['model']

# unwrap DDP-prefix keys if present
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
iter_num = checkpoint['iter_num']
best_val_loss = checkpoint['best_val_loss']

model.to(device)
optimizer = model.configure_optimizers(...)
optimizer.load_state_dict(checkpoint['optimizer'])
```

The `_orig_mod.` unwrapping is for models wrapped in `torch.compile`. Small detail you'll hit eventually.

## When to checkpoint

Trade-offs:
- Too often: slows training (each save takes seconds), fills disk.
- Too rarely: lose more work on crash.

Standard: every N iterations where `save_every_N_hours ≈ 1`. For a run where each iteration is 100ms, that's ~36,000 iterations. nanoGPT's default: every 2000 iterations.

Also: save whenever val loss improves:

```python
if losses['val'] < best_val_loss or always_save_checkpoint:
    best_val_loss = losses['val']
    # ... save checkpoint ...
```

## Atomic writes

The dreaded failure mode: crash in the middle of writing the checkpoint file. You get a half-written file, and your old good checkpoint is already overwritten. Training run dead.

**Atomic write pattern**:
1. Save to `ckpt.pt.tmp`.
2. When done, rename `ckpt.pt.tmp` → `ckpt.pt` (renames are atomic on most filesystems).

PyTorch's `torch.save` does NOT do this by default. For long runs, wrap it:

```python
def atomic_save(obj, path):
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)
```

nanochat's `nanochat/checkpoint_manager.py` handles this (and more). Read it.

## Keeping multiple checkpoints

For research, you often want:
- The latest checkpoint (to resume).
- The best-val-loss checkpoint (to evaluate on).
- Historical checkpoints at intervals (for post-hoc analysis).

Simple scheme:
```
ckpts/
  ckpt_latest.pt
  ckpt_best.pt
  ckpt_step_10000.pt
  ckpt_step_20000.pt
  ckpt_step_30000.pt
```

Rotate old ones to stay within disk budget.

## Checkpoint size

A checkpoint is roughly `model_size_in_bytes * 3`: weights + AdamW m + AdamW v.

- GPT-2 small (124M params, fp32): ~1.5GB per checkpoint.
- Llama-2 7B (fp32): ~85GB per checkpoint.
- Llama-3 70B (fp32): ~850GB.

For huge models, you save in fp16/bf16 and reconstitute fp32 state as needed, or use specialized sharded formats (Zarr, safetensors).

## Safetensors (modern format)

HuggingFace's `safetensors` format is zero-copy, type-safe, and safer than pickle-based `torch.save` (which can execute arbitrary code on load).

```python
from safetensors.torch import save_file, load_file
save_file(model.state_dict(), "model.safetensors")
state = load_file("model.safetensors")
```

You'll see this everywhere on HuggingFace Hub. For production, prefer it over `torch.save`.

## Resuming after hardware change

If you saved on an 8xA100 run and want to resume on an 8xH100 run, mostly works. Gotchas:
- CUDA version mismatches can cause weird errors.
- Different world-sizes: you're fine if you save the raw state_dict (not DDP-wrapped).
- Different precisions: cast weights as needed.

## nanochat's checkpoint system

`~/workspace/nanochat/nanochat/checkpoint_manager.py`: atomic writes, multiple checkpoints per training phase (pretrain, SFT, RL), config JSON alongside weights. Worth reading - it's 200 lines of practical code.

## Visualize this

**What's in a checkpoint**:

```
  ckpt.pt (PyTorch pickle):
  ┌───────────────────────────────────────────────┐
  │ {                                              │
  │   'model': {                                   │
  │       'transformer.wte.weight': tensor(...),   │
  │       'transformer.wpe.weight': tensor(...),   │
  │       'transformer.h.0.ln_1.weight': ...,      │
  │       ... (hundreds of tensors, ~10 M - 1 T) ..│
  │       'lm_head.weight': tensor(...),           │
  │   },                                           │
  │   'optimizer': {                               │
  │       'state': {                               │
  │           0: {'step': 50000,                   │
  │                'exp_avg': tensor(...),         │ ← Adam m
  │                'exp_avg_sq': tensor(...)},     │ ← Adam v
  │           1: {...},                            │
  │           ...                                  │
  │       },                                       │
  │       'param_groups': [...]                    │
  │   },                                           │
  │   'model_args': {'n_layer': 12, ...},           │
  │   'iter_num': 50000,                           │
  │   'best_val_loss': 2.84,                       │
  │   'config': {...},                             │
  │ }                                              │
  └───────────────────────────────────────────────┘

  File size ≈ 3 × model_params × 4 bytes (model + Adam m + Adam v in fp32)
```

For GPT-2 (124M params): ~1.5 GB per checkpoint. For 70B: ~850 GB.

**Atomic write pattern**:

```
  Naive (dangerous):
    torch.save(ckpt, "ckpt.pt")
         ▲
         │ Crash HERE → file half-written, old ckpt already overwritten.
         │ Training run dead.

  Atomic:
    torch.save(ckpt, "ckpt.pt.tmp")     ← writes to temp
    os.replace("ckpt.pt.tmp", "ckpt.pt") ← atomic rename (OS guarantees)
         ▲
         │ Crash anywhere before replace: ckpt.pt unchanged.
         │ Crash during replace: impossible on POSIX.
```

Always use atomic writes for long runs.

**Checkpoint frequency trade-off**:

```
  Too frequent (every 10 steps):
    Save overhead > compute time. Training crawls.
    1.5GB × 6000 saves/min = 9 TB of writes per minute = NVMe death.

  Too rare (every 100k steps):
    Crash loses 100k steps = 4 hours of compute.

  Sweet spot (every N steps where save_time ≈ 1% of step time × N):
    E.g. step = 140 ms, save = 5 s → save every ~3500 steps = ~8 min.
```

nanoGPT default: 2000 steps.

**The "best vs latest" checkpoint**:

```
  Train checkpoints over 10000 steps:

  loss  │
        │●
        │ ●
        │  ●
        │   ●    ←── best_val_loss snapshot (step 2500, val=1.87)
        │    ●
        │     ● ●
        │        ●●
        │          ●●●●●●    ←── latest (step 10000, val=1.93 — overfit)
        │
        └──────────────────────────── step

  Keep both:
    ckpt_best.pt    ← load for inference / eval
    ckpt_latest.pt  ← load for resuming training
```

**Production checkpoint structure** (nanochat-style):

```
  ~/.cache/nanochat/checkpoints/
  ├── base/
  │   ├── step_2000/
  │   │   ├── model.safetensors
  │   │   ├── optimizer.pt
  │   │   └── config.json
  │   ├── step_4000/...
  │   ├── latest.json  (points to newest)
  │   └── best.json    (points to best val)
  ├── sft/
  │   └── step_1500/...
  └── rl/
      └── step_500/...
```

Separate by training phase, versioned by step, metadata tracked.

## Exercises

1. Train Shakespeare to step 2000. Kill the process. Resume with `init_from='resume'`. Verify loss picks up where it left off.

2. Save a checkpoint, then manually edit the `.pt` file's byte 100 (corrupt it). Try to load - see the failure mode. Now implement `atomic_save` and see why it protects you.

3. Read `nanochat/checkpoint_manager.py`. Identify: how it lists checkpoints, picks the latest, handles failures.

4. Load a nanoGPT checkpoint with `torch.load`, inspect its keys, print the sizes of different tensors. Understand what's in there.

## Next

`08_evaluation_val_loss_perplexity.md` - the other half of training: knowing if it's working.
