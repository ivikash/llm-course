# 3.9 - Positional Encoding

A surprising fact: **pure attention is permutation-invariant**. If you shuffle the input tokens, attention gives you shuffled output. The model has no sense of order.

That's a problem. "The dog bit the man" is *very* different from "The man bit the dog", but identical token bags.

## The fix: add position information

We encode each position (0, 1, 2, ...) as a vector of length C, and **add** it to the token embedding:

```python
final_input[i] = token_embedding[token_i] + position_embedding[i]
```

This is done once, at the very start of the model. The rest of the model proceeds unaware that attention is "about" position - it just sees richer input vectors.

## Two approaches

### 1. Learned position embeddings (GPT-2, nanoGPT)

A learnable embedding table, one vector per position. Shape `(block_size, C)`.

```python
self.wpe = nn.Embedding(config.block_size, config.n_embd)
...
pos = torch.arange(0, T)
pos_emb = self.wpe(pos)   # (T, C)
x = tok_emb + pos_emb     # (B, T, C)
```

**Limitation**: the model can only handle sequences up to `block_size` (e.g. 1024). Longer, and you haven't learned positions that far out.

### 2. Rotary Position Embeddings (RoPE) - nanochat and most modern LLMs

Instead of adding a position vector to the embedding, RoPE **rotates** the Q and K vectors inside attention based on their position. Each position rotates by a different angle.

The clever bit: when you compute `q_i . k_j`, the result depends on `i - j` (the relative position) rather than the absolute position. This generalizes better to long contexts and doesn't have a hard block_size cap.

```python
# in nanochat/nanochat/gpt.py, simplified
def apply_rope(q, k, cos, sin):
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k
```

Math-heavy, but conceptually: rotate Q and K by angles that encode position. Then dot products automatically capture relative positions.

Used in Llama, GPT-NeoX, most models from 2022 onward. nanochat uses RoPE.

### 3. Other notable variants (for awareness)

- **Sinusoidal** (original "Attention Is All You Need"): fixed (not learned) sin/cos functions of position. Similar spirit to learned embeddings, no training needed.
- **ALiBi**: adds a per-head linear bias to attention scores based on distance. Used in some models (MPT, BLOOM).
- **Absolute vs relative**: RoPE and ALiBi are "relative" (they care about `i - j`); original and learned are "absolute" (care about `i` and `j` individually).

## Why RoPE won

- No block_size cap (theoretically extrapolates, though quality degrades outside trained range).
- Fewer params (no separate position table).
- Better length generalization.
- Nice mathematical properties.

Nearly all post-2022 open-source models use RoPE. You should be able to recognize it when you see it.

## In nanoGPT

Learned position embeddings, because it's a GPT-2 clone:
```python
self.wpe = nn.Embedding(config.block_size, config.n_embd)
```

That's why nanoGPT has `block_size = 1024` as a hard limit.

## In nanochat

RoPE. Find in `nanochat/nanochat/gpt.py`:
- `precompute_rotary_emb` or similar function computing cos/sin per position.
- Usage inside attention: `q, k = apply_rotary_emb(q, k, ...)`.

Skim it. You don't need to follow the math on first read - just know that it's "how position info gets injected into attention."

## Exercises

1. In nanoGPT, print the position embeddings after training and plot the first 2 PCA components. They'll form a smooth curve along the position index - positions 0..1023 become geometrically organized automatically.
2. Read the RoPE formula on Wikipedia or the [blog post by Eleuther](https://blog.eleuther.ai/rotary-embeddings/). Accept the math on first reading; it'll click after a few exposures.
3. Find where RoPE is applied in `nanochat/nanochat/gpt.py`. Note that it's applied *inside* `CausalSelfAttention`, to Q and K, before the dot product.

## Next

`10_lm_head_and_sampling.md` - turn final vectors back into words.
