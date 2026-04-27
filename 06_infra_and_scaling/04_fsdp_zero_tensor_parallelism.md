# 6.4 - FSDP, ZeRO, and Tensor Parallelism

DDP requires the whole model to fit on one GPU. For 7B+ models with optimizer state, that's 100+ GB - doesn't fit on an 80GB H100. Several "sharding" strategies solve this.

## The strategies

Four levels of splitting a big model across GPUs:

### 1. Data Parallel (DDP)
- Each GPU has a full copy of weights, grads, optimizer state.
- Each GPU sees different data.
- Works up to model_size ≤ GPU_memory/4 (weights + grads + AdamW state).

### 2. Data + Param sharding (FSDP, ZeRO-3)
- Each GPU has a **shard** (1/N) of weights, grads, optimizer state.
- Weights are **gathered** just-in-time before each layer's forward/backward.
- Supports much bigger models on the same hardware.
- PyTorch native: `FullyShardedDataParallel` (FSDP).

### 3. Tensor Parallelism
- Split matmuls themselves across GPUs: half of the weight matrix per GPU, each GPU computes half the output.
- Low latency (GPUs work simultaneously).
- High communication within a layer.
- Best for within-node (NVLink). Used in Megatron-LM.

### 4. Pipeline Parallelism
- Split layers across GPUs: GPU 0 has layers 1-8, GPU 1 has layers 9-16, etc.
- "Pipeline" micro-batches through so GPUs don't idle.
- Inter-layer communication only.
- Often combined with TP and DP → "3D parallelism."

## ZeRO stages (DeepSpeed's naming)

ZeRO = "Zero Redundancy Optimizer". Developed by Microsoft Research.

- **ZeRO-1**: shard optimizer state only. Saves most memory for AdamW (which is 2/3 of parameter memory).
- **ZeRO-2**: shard optimizer state + gradients.
- **ZeRO-3**: shard optimizer + grads + parameters. Equivalent to FSDP.

PyTorch's FSDP is essentially ZeRO-3, integrated natively.

## FSDP mental model

Forward pass, layer by layer:
1. Rank 0 has shard 0 of layer 1's weights, rank 1 has shard 1, etc.
2. Before layer 1 forward: all-gather the full layer 1 weights to all ranks.
3. Each rank computes layer 1 forward on its own data.
4. Free the gathered weights (keep only the shard).
5. Repeat for layer 2, 3, ...

Same during backward, with reduce-scatter of gradients.

Trade-off: more communication (gather before every layer) for much less memory.

## FSDP in PyTorch

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision

mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
)

model = FSDP(
    model,
    mixed_precision=mp_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,   # this is ZeRO-3
    device_id=local_rank,
)

# then training loop is unchanged
```

Some code tweaks: optimizer creation now happens after FSDP wrap, not before. Flat parameter view changes behavior of some weight-access patterns.

## When each strategy makes sense

| Model size | Strategy |
|-----------|----------|
| ≤ 2B | DDP is fine on single GPU |
| 2B-30B | FSDP (or DeepSpeed ZeRO-3) |
| 30B-200B | FSDP + tensor parallelism within node |
| 200B+ | Full 3D parallelism (DP + TP + PP), custom frameworks |

Frontier labs (Meta, OpenAI, Anthropic) build custom training stacks for their largest models.

## Frameworks

- **PyTorch native**: FSDP (Module 6 default).
- **DeepSpeed**: ZeRO, plus more features (CPU offloading, pipeline). Microsoft's stack.
- **Megatron-LM**: tensor + pipeline parallelism. NVIDIA's stack.
- **Megatron-DeepSpeed**: hybrid. Used for a lot of published work.
- **ColossalAI, TorchTitan**: newer PyTorch-first options.
- **Apex** (deprecated): older NVIDIA library.

For learning, stick with PyTorch FSDP. It's clean and sufficient up to ~70B models.

## Does nanoGPT / nanochat use FSDP?

No. Both use DDP. They target model sizes that fit on one GPU:
- nanoGPT: up to ~1.5B (GPT-2 XL).
- nanochat: up to depth 40 or so (~2B), small enough for DDP with bf16 + fp8.

For bigger, you'd port to FSDP. PRs welcome on nanochat.

## Activation checkpointing + FSDP

Big models also benefit from **activation checkpointing**: don't save intermediate activations, recompute them on backward.

```python
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    for layer in self.layers:
        x = checkpoint(layer, x, use_reentrant=False)
    return x
```

Combined with FSDP, you can fit much bigger models at the cost of ~30% more compute time.

## CPU offloading

When even FSDP doesn't fit: offload shards to CPU RAM, bring to GPU when needed.

```python
FSDP(model, cpu_offload=CPUOffload(offload_params=True))
```

Slow (CPU-GPU transfer is slow) but can train models that otherwise wouldn't fit at all. Used for extreme cases.

## Visualize this

**DDP vs ZeRO-1/2/3, per-GPU memory:**

```viz
{"viz": "fsdp_shard"}
```

Switch strategies. Watch each GPU's memory bar shrink as more gets sharded. DDP = full copy on every GPU (wasteful). ZeRO-1 shards just optimizer state (~half the waste). ZeRO-2 also shards gradients. FSDP/ZeRO-3 shards weights too — lowest per-GPU memory, but most network traffic.

**DDP vs FSDP (ZeRO-3) vs Tensor Parallelism**:

```
  DDP (data parallel):
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │ GPU 0         │   │ GPU 1         │   │ GPU 2         │   │ GPU 3         │
  │ full model   │   │ full model   │   │ full model   │   │ full model   │
  │ full opt state│   │ full opt state│   │ full opt state│   │ full opt state│
  │ batch 0       │   │ batch 1       │   │ batch 2       │   │ batch 3       │
  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
  Each GPU: has the full model. All-reduces gradients per step.

  FSDP / ZeRO-3 (sharded params + grads + opt state):
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │ GPU 0         │   │ GPU 1         │   │ GPU 2         │   │ GPU 3         │
  │ 1/4 weights   │   │ 1/4 weights   │   │ 1/4 weights   │   │ 1/4 weights   │
  │ 1/4 grads     │   │ 1/4 grads     │   │ 1/4 grads     │   │ 1/4 grads     │
  │ 1/4 opt state │   │ 1/4 opt state │   │ 1/4 opt state │   │ 1/4 opt state │
  │ batch 0       │   │ batch 1       │   │ batch 2       │   │ batch 3       │
  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
  Before each layer: all-gather the 1/4 shards into full layer weights.
  After backward:    reduce-scatter grads back to 1/4 shards.
  4× less memory per GPU. More communication.

  Tensor Parallelism (split matmul across GPUs):
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │ GPU 0         │   │ GPU 1         │   │ GPU 2         │   │ GPU 3         │
  │ half of W₁   │   │ other half W₁│   │ half of W₁   │   │ other half W₁│
  │ attention:    │   │ attention:   │   │              │   │              │
  │ heads 0-3     │   │ heads 4-7    │   │ heads 8-11   │   │ heads 12-15  │
  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
  Each matmul split across GPUs; all-reduce after each.
  Very high communication; best within a node (NVLink).

  Pipeline Parallelism (split layers across GPUs):
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │ GPU 0         │   │ GPU 1         │   │ GPU 2         │   │ GPU 3         │
  │ layers 1-8   │───▶│ layers 9-16  │───▶│ layers 17-24 │───▶│ layers 25-32 │
  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
  Each GPU holds a different "slice" of layers.
  Activations flow forward, gradients flow backward.
  Bubble of idle time; micro-batching reduces this.
```

**Which strategy for which model size**:

```
  Model fits in 1 GPU (+ optimizer state)?
           │
           ▼
     YES → DDP (simple, fast)    e.g. GPT-2-small, Mistral-7B bf16 on A100-80GB

           │
           ▼
     NO  → Does it fit across multiple GPUs?
                │
                ▼
          FSDP / ZeRO-3 (shard param/grad/opt)    7B-30B typical
                │
                ▼
          Still OOM? Model too big for sharding alone?
                │
                ▼
          Add tensor parallelism (within a node)
                │
                ▼
          + pipeline parallelism (across nodes)     70B+

  Frontier: 3D parallelism = DP × TP × PP together.
  E.g. Llama-3 405B uses 8-way TP × 16-way PP × DP across 1000s of GPUs.
```

**FSDP memory savings (dramatic for AdamW)**:

```
  1B param model, bf16 mixed precision + AdamW:
  ──────────────────────────────────────────────
                      DDP per GPU    FSDP (4 GPUs) per GPU
  weights (fp32)       4 GB            1 GB (shard)
  weights (bf16)       2 GB            0.5 GB (shard)
  gradients            2 GB            0.5 GB (shard)
  AdamW m (fp32)       4 GB            1 GB (shard)
  AdamW v (fp32)       4 GB            1 GB (shard)
  ──────────────────── ────────────   ─────────────────
  Total:               16 GB           4 GB  ← 4× less!

  → can train models 4× bigger on the same hardware.
```

**ZeRO stages in one picture**:

```
  ZeRO-1: shard optimizer state only
    per GPU: full weights + full grads + 1/N opt state

  ZeRO-2: shard opt state + grads
    per GPU: full weights + 1/N grads + 1/N opt state

  ZeRO-3 (= FSDP): shard all of it
    per GPU: 1/N weights + 1/N grads + 1/N opt state
    (weights gathered just-in-time for each layer)
```

Bigger ZeRO stage = less memory but more communication.

## Exercises

1. Compute: for Llama-2-70B with AdamW in bf16 mixed precision, total training memory? (70 × 16 = 1.12 TB. On 8 H100s (640 GB total), needs FSDP ZeRO-3.)

2. Read PyTorch FSDP docs' "Getting Started." Understand the `auto_wrap_policy` - how you tell FSDP where to shard.

3. Study Megatron-LM's tensor parallelism section in their README. Understand "Row parallel" vs "Column parallel" linear layers.

4. If you're curious, read the [ZeRO paper](https://arxiv.org/abs/1910.02054). 15 pages, very readable.

## Next

`05_flash_attention_fp8_performance.md` - the kernels that make big training fast.
