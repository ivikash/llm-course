# 6.2 - GPU Memory and VRAM

"CUDA out of memory" will be your most common error. Understanding the memory budget saves hours.

## What fills VRAM during training

For one model step, VRAM holds:

1. **Parameters** (weights): the model itself. fp32 = 4 bytes each.
2. **Gradients**: same shape as parameters. Same dtype.
3. **Optimizer state**: AdamW needs 2 extra tensors per parameter (m, v). Usually fp32.
4. **Activations**: intermediate tensors saved for backprop. Scales with batch × seq_len × layers × hidden.
5. **KV cache** (inference only).
6. **Workspace**: CUDA's scratch memory for kernels. Small but non-zero.

## The per-parameter training formula

With bf16 mixed precision + AdamW:
- fp32 master weights: 4 bytes
- bf16 working weights: 2 bytes
- bf16 gradients: 2 bytes
- fp32 Adam m: 4 bytes
- fp32 Adam v: 4 bytes
- **Total: ~16 bytes per parameter**

For a 1B parameter model: 16 GB just for this. Before activations. Before batch data.

For a 7B parameter model: 112 GB. Won't fit on one H100 (80GB). Enter FSDP (Lesson 6.4).

## Activation memory

Every layer's intermediate tensors have to be stored for backprop. Roughly:

```
activation_memory ≈ B × T × C × L × constant
```

- B = batch
- T = sequence length
- C = hidden dim
- L = number of layers
- constant = 10-40, depending on architecture and implementation

For GPT-2 small (L=12, C=768) with B=12, T=1024: ~1-2 GB of activations. Small.

For Llama-7B (L=32, C=4096) with B=1, T=4096: ~8-15 GB of activations. Not small.

**Gradient checkpointing** trades compute for this: don't save activations, recompute them during backward. 30% slower, ~4x less activation memory.

## Inference memory

Serving is different:
- No gradients, no optimizer state.
- Just weights + KV cache + activations.
- Weights can be in bf16 or int8/int4 (quantization).

Rough formula:
```
memory = (bytes_per_weight × N) + (2 × L × head_dim × n_kv_head × context × concurrent_users × bytes_per_float)
```

A 7B model in bf16 serving 10 concurrent users with 4K context:
- Weights: 14 GB
- KV cache per user: ~1-2 GB
- Total: ~30 GB. Fits on A100-40GB.

## How to read nvidia-smi

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05   Driver Version: 535.104.05   CUDA Version: 12.2   |
+-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA H100 80GB   Off   | 00000000:0E:00.0 Off |                    0 |
| N/A   42C    P0  230W / 700W  |  72543MiB / 81559MiB |     95%      Default |
+-------------------------------+----------------------+----------------------+
```

Key fields:
- `72543MiB / 81559MiB`: using 72 GB of 81 GB.
- `95%` GPU-Util: 95% of the time, GPU is busy (not sitting idle). Good.
- `230W / 700W`: drawing 230W of 700W budget.

Low GPU-Util (< 50%) usually means you're bottlenecked by data loading or CPU.

## When OOM happens: fixes in order

1. **Lower micro-batch size**: half it. Effective batch stays same with gradient accumulation.
2. **Lower seq_len** (if applicable).
3. **Enable bf16** (if using fp32).
4. **Enable gradient checkpointing**.
5. **Use FSDP** (sharded param + optim across GPUs).
6. **Use fp8** (H100+ only).
7. **Smaller model** (last resort).

## Tools for profiling

- `torch.cuda.memory_allocated()` / `torch.cuda.max_memory_allocated()` - programmatic.
- `nvidia-smi -l 1` - live GPU stats.
- `nvtop` - htop for GPUs.
- `torch.cuda.memory_snapshot()` + `visualize_tensor_sizes` - detailed breakdown.
- **PyTorch Profiler**: `torch.profiler.profile(...)` for operation-level timing.

## In nanoGPT

nanoGPT doesn't do anything fancy. The `--batch_size` and `--block_size` flags are your OOM levers.

## In nanochat

`--device-batch-size` is the main lever. The training script reports MFU - if it's low, you're memory-bound or data-bound.

## Visualize this

**Memory budget for training (per-parameter accounting)**:

```
  bf16 mixed precision + AdamW:

  1 parameter uses (in VRAM):
  ┌─────────────────────────────┐
  │ fp32 master weight   4 bytes│
  │ bf16 working weight   2 bytes│
  │ bf16 gradient        2 bytes│
  │ fp32 AdamW m         4 bytes│
  │ fp32 AdamW v         4 bytes│
  └─────────────────────────────┘
   16 bytes per param

  Model size   →   Param memory
  ──────────────   ─────────────
  125M            2.0 GB
  1B              16 GB
  7B              112 GB   ← doesn't fit on 1× H100
  13B             208 GB   ← needs FSDP across 2+ GPUs
  70B             1.12 TB   ← needs 16 GPUs minimum
  175B (GPT-3)    2.8 TB
  1T              16 TB     ← requires clusters of many nodes

  Plus activations (scales with batch × seq_len × layers).
```

**Why bf16 helps (but not as much as you'd think)**:

```
  Activations (the variable part):
  ─────────────────────────────────
  fp32:  4 bytes × B × T × C × L × const
  bf16:  2 bytes × B × T × C × L × const   ← half

  Net training memory typically drops 30-40% going fp32 → bf16.
```

**How to read nvidia-smi**:

```
  $ nvidia-smi

  +-----------------------------------------------------------+
  | NVIDIA-SMI 535.xx     Driver Version: 535.xx   CUDA 12.2   |
  +----------------+------------------+------------------------+
  | GPU  Name      | Memory-Usage     | GPU-Util  Temp  Power  |
  +================+==================+========================+
  | 0  H100 80GB   | 72543/81559 MiB  | 95%  42C   230W/700W   |
  |                                   ↑                        |
  |                    KEY: this is the "MFU ≈ 95%" target    |
  +----------------+------------------+------------------------+
  | 1  H100 80GB   | 72541/81559 MiB  | 96%  41C   228W/700W   |
  | 2  H100 80GB   | 72540/81559 MiB  | 94%  43C   231W/700W   |
  +----------------+------------------+------------------------+

  Healthy training:
  - GPU-Util: 60-95% sustained.
  - Memory: high (not OOMing, but using what's there).
  - Power: near max (GPUs working hard, not idle).

  Unhealthy:
  - GPU-Util < 30%: bottlenecked elsewhere (data loading? CPU?).
  - Memory: very low → underutilizing GPU, increase batch.
  - Power: low → idle, check if training actually started.
```

**Live monitoring command**:

```bash
watch -n 1 nvidia-smi
# or prettier:
pip install nvitop
nvitop
```

`nvitop` gives an htop-like colorful TUI. Highly recommend.

**OOM debugging flowchart**:

```viz
{"viz": "memory_calculator"}
```

Slide model size, GPU, precision. See where memory goes (model weights + gradients + AdamW state + activations) and whether it fits. Try: 7B model on A100-40GB with fp32+AdamW → OOM. Switch to bf16+8-bit-adam → fits.

```
  "torch.cuda.OutOfMemoryError"
           │
           ▼
  1. halve micro batch size    ──→ fixed? DONE
           │
           │ no
           ▼
  2. shorten block_size         ──→ fixed? DONE (if context can be shorter)
           │
           │ no
           ▼
  3. switch fp32 → bf16          ──→ fixed? DONE
           │
           │ no
           ▼
  4. enable grad checkpointing   ──→ fixed? DONE (slower, less memory)
           │
           │ no
           ▼
  5. use FSDP (shard model)      ──→ fixed? DONE
           │
           │ no
           ▼
  6. use fp8 (if H100+)          ──→ fixed? DONE
           │
           │ no
           ▼
  7. smaller model. last resort.
```

Work through them in order. Usually #1 or #3 fixes it.

## Exercises

1. In Python, before creating a model:
   ```python
   import torch
   print(torch.cuda.get_device_properties(0).total_memory / 1e9, "GB VRAM")
   ```

2. Train the Shakespeare model with increasing batch_size until OOM. Note the breaking point.

3. Enable gradient checkpointing on a medium model, observe how batch size can increase.

4. Compute for Llama-7B in bf16 training: total memory needed for weights + grads + AdamW state. Does it fit on a single A100-40GB? (No. 112 GB required.)

## Next

`03_distributed_training_ddp.md` - how to use many GPUs in parallel.
