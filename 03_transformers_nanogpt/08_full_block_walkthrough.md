# 3.8 - Full Block Walkthrough

Time to pull it all together. A full transformer block, end to end, one shape at a time.

## Setup

- `B = 2` (batch size)
- `T = 5` (sequence length)
- `C = 8` (embedding dim)
- `n_head = 2`
- `head_dim = 4`

Input: `x` of shape `(2, 5, 8)`.

## Step-by-step

```python
# Input x: (B=2, T=5, C=8)
x = torch.randn(B, T, C)

# ---- Attention sub-layer ----

# LayerNorm (pre-norm)
x_ln = layernorm_1(x)   # (2, 5, 8)

# Q, K, V projection (packed then split)
qkv = c_attn(x_ln)       # (2, 5, 24)  -- 3C output
q, k, v = qkv.split(C, dim=-1)  # each (2, 5, 8)

# Reshape to multi-head
q = q.view(B, T, n_head, head_dim).transpose(1, 2)   # (2, 2, 5, 4)
k = k.view(B, T, n_head, head_dim).transpose(1, 2)   # (2, 2, 5, 4)
v = v.view(B, T, n_head, head_dim).transpose(1, 2)   # (2, 2, 5, 4)

# Attention (per head, parallel)
scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)   # (2, 2, 5, 5)
scores = scores.masked_fill(causal_mask[:T, :T] == 0, float('-inf'))
att = F.softmax(scores, dim=-1)   # (2, 2, 5, 5)
out = att @ v                     # (2, 2, 5, 4)

# Merge heads
out = out.transpose(1, 2).contiguous().view(B, T, C)   # (2, 5, 8)

# Output projection
attn_out = c_proj(out)   # (2, 5, 8)

# Residual add
x = x + attn_out   # (2, 5, 8)

# ---- MLP sub-layer ----

x_ln = layernorm_2(x)    # (2, 5, 8)

h = c_fc(x_ln)           # (2, 5, 32)  (4x C)
h = F.gelu(h)
mlp_out = c_proj_mlp(h)  # (2, 5, 8)

# Residual add
x = x + mlp_out   # (2, 5, 8)

# ----- end of block. Output shape same as input: (B, T, C). -----
```

Read that top to bottom. Every shape is written. Every operation is in your vocabulary by now. If this reads smoothly, you understand a transformer block.

## The whole model

A GPT is just:

```python
# embeddings
x = token_embeddings[idx] + position_embeddings[positions]   # (B, T, C)
x = dropout(x)

# N blocks
for block in blocks:
    x = block(x)    # each call preserves shape (B, T, C)

# final LN
x = layernorm_final(x)

# project to vocab
logits = lm_head(x)   # (B, T, vocab_size)
```

That's the whole transformer. Everything else is just scale - pick C = 8192 instead of 8, N = 80 instead of 1, vocab_size = 128000 instead of 65, and you have a 70B-parameter state-of-the-art model. The architecture literally does not change.

## Parameter counting

Total parameters of a GPT with:
- `n_layer = L` blocks
- `n_embd = C`
- `vocab_size = V`
- `block_size = T_max` (max context)

- Token embeddings: `V * C`
- Position embeddings (learned): `T_max * C`
- Each block:
  - Attention: `c_attn` (C * 3C) + `c_proj` (C * C) = `4 * C^2`
  - MLP: `c_fc` (C * 4C) + `c_proj` (4C * C) = `8 * C^2`
  - Total per block: `12 * C^2` (ignoring biases and LN, which are tiny)
- Final LN, LM head: mostly covered by tied embedding
- **Total ≈ V*C + T_max*C + L * 12 * C^2**

For GPT-2 small (V=50257, C=768, L=12, T=1024):
- Embeddings: 50257 * 768 ≈ 38.6M
- Position: 1024 * 768 ≈ 0.8M
- Blocks: 12 * 12 * 768^2 ≈ 85M
- Total ≈ 124M. ✓ (matches the famous 124M GPT-2 size)

## The "12 C^2" rule of thumb

For transformer models, **each layer has roughly 12 * n_embd^2 parameters**. Keep this in your head. It lets you estimate model size from a few numbers quickly.

## Visualize this

**Watch data flow through a transformer block, shape by shape:**

```viz
{"viz": "transformer_block_flow"}
```

Press **▶ Play flow**. Each row lights up in sequence: LN → Q/K/V → reshape → attention → un-reshape → residual → LN → MLP → residual. Slide B, T, C, heads to see element counts scale. **Shape in = shape out** (`(B, T, C)`), which is why you can stack these blocks without any glue code.

**One transformer block, the big picture**:

```
         input x  (B, T, C)
              │
              ├────────────────── skip ──────────┐
              │                                  │
              ▼                                  │
         ┌───────┐                               │
         │ ln_1  │  (layer norm)                 │
         └───┬───┘                               │
             │                                   │
             ▼                                   │
      ┌─────────────┐                            │
      │  attention  │  (multi-head)              │
      └─────┬───────┘                            │
            │                                    │
            └─────────── + ──────────────────────┘
                         │  ← first residual
                         │
                         ├────────── skip ────────┐
                         │                         │
                         ▼                         │
                    ┌───────┐                      │
                    │ ln_2  │                      │
                    └───┬───┘                      │
                        │                          │
                        ▼                          │
                 ┌──────────┐                      │
                 │   MLP    │  (GELU-based)        │
                 └─────┬────┘                      │
                       │                           │
                       └────────── + ──────────────┘
                                  │  ← second residual
                                  ▼
                            output (B, T, C)
```

Shape never changes. Input in, same-shape output out.

**The full GPT, all blocks stacked**:

```
  tokens [B, T] integer IDs
       │
       ▼
    wte (token emb)  +  wpe (pos emb)
       │
       ▼
   (B, T, C)
       │
       ▼
    Block 0   ───────┐
       │              │  each block:
       ▼              │  - attention (cross-token)
    Block 1           │  - MLP (per-token)
       │              │  - pre-norm + residuals
       ▼              │  - same shape out as in
      ...             │
       │              │
       ▼              │
    Block N-1  ───────┘
       │
       ▼
   ln_f (final layer norm)
       │
       ▼
    lm_head  (linear → vocab_size)
       │
       ▼
  logits [B, T, vocab_size]
       │
       ▼
  cross_entropy loss (against true next tokens)
```

That's the entire GPT architecture in one picture. Read `nanoGPT/model.py` with this in mind - you'll recognize every component.

**bbycroft.net/llm**: open it, click each stage, watch the data flow. Three-dimensional render of this exact picture.

**Scale intuition**: GPT-3 is this same picture with `N=96` blocks, `C=12288`, `vocab_size=50257`. That's it. No new architectural ideas. Scaling the existing recipe is how GPT-3 happened.

## Exercises

1. Do the parameter count for GPT-2 medium (C=1024, L=24, V=50257). Should come out to ~350M.
2. Do it for a nanochat-style d26 model (check the README). Compare.
3. Open `nanoGPT/model.py`, class `GPT`, method `get_num_params`. See how Karpathy computes it. Call it on a default config.

## Next

`09_positional_encoding.md` - how the model knows token order at all (attention by itself is permutation-invariant!).
