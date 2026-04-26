# 5.3 - Pretraining with base_train.py

This is the longest script and the one that consumes most compute in a nanochat run. It's also the most similar to what frontier labs do.

## The pitch

`base_train.py` takes a fresh tokenizer and a big corpus of tokenized text, and produces a **base model** - a transformer that can complete text. After a 3-hour speedrun, you have something comparable to GPT-2 (124M params, 2019) in a few of hours instead of days.

## Launch

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 \
    --run=my_first_run
```

What each flag does:
- `torchrun --nproc_per_node=8`: launch 8 parallel processes, one per GPU.
- `--depth=24`: 24-layer transformer (≈300M params, GPT-2 large size).
- `--target-param-data-ratio=8`: train on 8 tokens per param (GPT-2 is ~8; Chinchilla says 20; speedrun undertrains for speed).
- `--device-batch-size=16`: each GPU processes 16 sequences per micro-batch.
- `--fp8`: enable fp8 matmul (H100+).
- `--run=my_first_run`: wandb run name.

## The model: `nanochat/gpt.py`

Open it. 700 lines, vs nanoGPT's 400. Let's tour the differences.

### GPTConfig

```python
@dataclass
class GPTConfig:
    depth: int = 24          # number of layers
    vocab_size: int = 2**15  # 32768
    seq_len: int = 2048      # context length
    # computed:
    n_embd: int = None
    n_head: int = None
    n_kv_head: int = None
    def __post_init__(self):
        # scale width with depth (compute-optimal)
        self.n_embd = 64 * self.depth
        self.n_head = ...
        self.n_kv_head = ...        # GQA
```

Single dial philosophy: set depth, everything else is derived.

### The architecture

`class GPT(nn.Module)` is similar to nanoGPT but with Llama-style choices:

1. **RMSNorm** instead of LayerNorm:
   ```python
   def rmsnorm(x):
       return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
   ```
   No mean subtraction, no learnable scale (some implementations include one).

2. **RoPE** for position encoding: applied to Q and K inside attention, not added to input embeddings.

3. **SwiGLU** MLP: two input projections, SiLU gate, one output projection.

4. **GQA (Grouped-Query Attention)**: fewer K/V heads than Q heads.

5. **No bias** on linear layers: `bias=False` everywhere. Common in modern LLMs.

6. **Shared embedding + unembedding** (weight tying), as nanoGPT also does.

7. **(Optional) fp8 matmuls** in some linears, via `nanochat/fp8.py`.

### Attention with Flash + GQA

```python
# simplified
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        # Q uses n_head heads, K/V use n_kv_head
        self.wq = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, rope_cache, kv_cache=None):
        ...
        # apply RoPE to q and k
        q, k = apply_rope(q, k, rope_cache)
        # repeat K/V to match Q heads (GQA)
        k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        v = v.repeat_interleave(..., dim=1)
        # flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(kv_cache is None))
        ...
```

Note the RoPE + GQA integration. Read the full file once to see it cleanly.

## The training script

Open `scripts/base_train.py`. ~800 lines; most is argument parsing and logging. The core loop is very similar to nanoGPT's `train.py`:

1. Parse args (dataclasses).
2. Init DDP.
3. Load tokenizer.
4. Set up distributed data loader (streaming shards).
5. Build model (`GPT(config)`), move to GPU, wrap in DDP.
6. Build optimizer(s) - AdamW + Muon, with different LRs for different param groups.
7. Training loop:
   - Cosine LR schedule (depth-derived max LR).
   - Forward pass with bf16 (and fp8 on eligible matmuls).
   - Backward with gradient accumulation.
   - Gradient clipping.
   - Optimizer step.
   - Log to wandb (train loss, val bpb, grad norm, MFU, throughput).
8. Periodic eval:
   - Val bpb (via `nanochat/loss_eval.py`).
   - CORE metric (via `nanochat/core_eval.py`) — only periodically, it's expensive.
   - Sampling from the model, to see qualitative progress.
9. Periodic checkpointing (via `nanochat/checkpoint_manager.py`).
10. At end, write `report/base_train.md`.

## Interesting details

### The data: streaming shards

`nanochat/dataloader.py` manages:
- Enumerating shards on disk.
- Distributing shards across ranks.
- Opening each shard with np.memmap.
- Randomly sampling from a rolling pool.
- Automatically downloading more shards if you run out.

This is the production-grade version of nanoGPT's 10-line `get_batch`.

### Muon + AdamW

nanochat uses two optimizers:
- **Muon** for 2D matrices (most weights).
- **AdamW** for embeddings and the final projection.

Muon is a 2024 optimizer that orthogonalizes momentum; see `nanochat/optim.py`. It trains slightly faster than pure AdamW.

Setup:
```python
muon_params = [p for n, p in ... if p.dim() == 2 and 'embed' not in n]
adamw_params = [p for n, p in ... if p not in muon_params]
opt_muon = Muon(muon_params, lr=muon_lr)
opt_adamw = AdamW(adamw_params, lr=adamw_lr, weight_decay=0.0)
# step both each iteration
```

### fp8 matmul

If `--fp8` is set, certain `nn.Linear` layers are replaced with a custom fp8 version from `nanochat/fp8.py`. Requires H100 or newer. About 25% speedup.

### MFU tracking

Model FLOPs Utilization = actual FLOPs / theoretical peak. For nanochat on H100, good MFU is 40-55%.

Computed in-line during training based on model size and token throughput.

## Output

After the run:
- Base model checkpoint: `~/.cache/nanochat/checkpoints/base/...`
- wandb dashboard with curves.
- `~/.cache/nanochat/report/base_train.md` with a summary.

This base model can complete text but doesn't follow instructions. Next stages (SFT, RL) fix that.

## Visualize this

**nanochat vs nanoGPT architecture side-by-side**:

```
  nanoGPT (GPT-2)            nanochat (Llama-style)
  ────────────────           ──────────────────────

  Input                      Input
    │                         │
    ▼                         ▼
  tok_emb + pos_emb          tok_emb  (position via RoPE inside attn)
    │                         │
    ▼                         ▼
  ┌─────────┐               ┌─────────┐
  │LayerNorm│               │ RMSNorm │
  └────┬────┘               └────┬────┘
       ▼                         ▼
  MHA (MHA, bias)            GQA (no bias, RoPE-rotated Q,K)
       │                         │
       + (residual)              + (residual)
       ▼                         ▼
  ┌─────────┐               ┌─────────┐
  │LayerNorm│               │ RMSNorm │
  └────┬────┘               └────┬────┘
       ▼                         ▼
  MLP: GELU                  MLP: SwiGLU (no bias)
       │                         │
       + (residual)              + (residual)
       ▼                         ▼
  × N layers                 × depth layers
  (final ln_f)               (final RMSNorm)
  lm_head (with bias)         lm_head (no bias, tied weights)
```

Same skeleton, 5-6 targeted upgrades. That's the Llama recipe.

**Training run structure**:

```
  base_train.py execution (~3 hours on 8xH100 for GPT-2 grade):

  minute 0:      Parse args, init DDP, load tokenizer
  minute 0-1:    Build model (1.5B params for depth 24)
  minute 1-2:    Setup optimizer (Muon + AdamW), LR schedule
  minute 2:      First batch → first step. Loss ≈ log(32768) ≈ 10.4.
                 You've seen this in nanoGPT too.
  minute 3:      Warmup complete, LR at peak.
                 Loss dropping fast (~5.0 by step 1000).
  minute 5-120:  Main training. Loss trajectory:
                   step 1k:  5.0
                   step 5k:  3.2
                   step 10k: 2.8
                   step 20k: 2.5
                   step 50k: 2.3
                   step 100k:2.1  ← GPT-2 level, speedrun completes
  minute 120:    Final eval. CORE benchmark runs.
  minute 130:    Save final checkpoint.
```

**The metrics that matter** (live on wandb):

```
                   what to watch
  ────────────────────────────────
  train/loss       → monotonically decreasing (mostly)
  val/bpb          → decreasing; this is the money metric
  train/lr         → warmup then cosine decay
  train/grad_norm  → should stay near 1.0
  train/mfu        → model FLOPs utilization; 40-55% is healthy on H100
  train/tok_per_sec → throughput; stable = good
  gpu_memory       → high but not OOMing

  Every ~5000 steps:
  eval/core        → overall capability metric (DCLM composite)
  eval/val_loss    → validation loss
  sample_text      → "Quick brown fox ..." completions
```

**The Muon + AdamW split**:

```
  Parameter          Optimizer    Reason
  ────────────────── ──────────── ─────────────────────────
  2-D weight matrices Muon         orthogonalized momentum,
                                    trains well at high LR
  Embeddings          AdamW        high-dim sparse, AdamW handles
  Final projection    AdamW        sharp distribution, AdamW more stable
  Biases, LayerNorm   AdamW (WD=0) small tensors, no decay

  Both optimizers step every iteration; each only touches its params.
```

**Depth dial in action** (what you'd set for various goals):

```
  Goal                              depth  wall time on 8xH100
  ──────────────────────────────    ─────  ──────────────────
  Quick sanity check (CPU)            4     30 min CPU
  Laptop experiments (1x4090)         8     2 hours
  Research toy runs                  12     5 min
  Speedrun (beat GPT-2)              24     ~2 hours
  Practical GPT-2 grade              26     ~2.5 hours
  Approaching GPT-2 Large            36     ~12 hours
  Small Llama-1 territory            40     ~20 hours, big memory
```

## Exercises

1. Read the argparse section at the top of `scripts/base_train.py`. Understand every flag.

2. Open `nanochat/gpt.py`'s `GPT.forward` method. Compare to nanoGPT's `GPT.forward`. Note:
   - Both take `(idx, targets)` input.
   - Both output `(logits, loss)`.
   - nanochat's uses RoPE, RMSNorm, SwiGLU.
   - Otherwise the structure is identical.

3. Find how Muon and AdamW are set up in base_train.py. Which parameters go to which optimizer?

4. Find the MFU calculation. What FLOPs does it use for the numerator? (Hint: it's roughly `6 * num_params * tokens_per_sec`.)

5. Read `nanochat/dataloader.py`. ~250 lines. Understand the shard-rotation loop.

## Next

`04_midtraining_and_chat_formatting.md` - the chat formatting that bridges pretraining to chat.
