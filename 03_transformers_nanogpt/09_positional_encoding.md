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

## Visualize this

**Why attention is permutation-invariant without positions**:

```
  Input: "the cat sat"     positions: [0, 1, 2]
  Attention processes: {"the", "cat", "sat"}  (as a SET, not a sequence)

  Now shuffle: "sat the cat"  same set!
  Pure attention (no pos encoding): produces the same output.

  → Without position info, word order is invisible to the model.
```

**Absolute learned positions (GPT-2, nanoGPT)**:

```
                        n_embd dimensions
                ┌───┬───┬───┬───┬───┬───┐
  position 0    │   │   │   │   │   │   │  ← learned vector for "position 0"
  position 1    │   │   │   │   │   │   │  ← learned vector for "position 1"
  position 2    │   │   │   │   │   │   │
  ...
  position 1023 │   │   │   │   │   │   │  ← last position we trained on
                └───┴───┴───┴───┴───┴───┘

  Token at position i gets: token_emb + pos_emb[i]
  Limitation: doesn't work for positions > 1023 (never saw them).
```

**Sinusoidal (original transformer, 2017)**:

```
  Position vectors are fixed, not learned. Shape (position, n_embd/2, 2):

     dim 0     dim 2     dim 4     dim 6     ...
  sin(pos/1)  sin(pos/100)  sin(pos/10000)  sin(pos/1000000)
  cos(pos/1)  cos(pos/100)  cos(pos/10000)  cos(pos/1000000)
   (fast)                                       (slow)

  Each dim oscillates at different frequency. Like a binary counter
  where each dim handles a different "place value" of position.
```

Visualize with:
```python
import numpy as np
import matplotlib.pyplot as plt

d = 64; pos = np.arange(100)
pe = np.zeros((100, d))
for i in range(0, d, 2):
    pe[:, i] = np.sin(pos / 10000**(i/d))
    pe[:, i+1] = np.cos(pos / 10000**(i/d))
plt.imshow(pe.T, cmap='viridis', aspect='auto')
plt.xlabel("position"); plt.ylabel("embedding dim")
plt.savefig("sinusoidal_pos.png")
```

You'll see stripes at different frequencies across the dimensions. Beautiful.

**RoPE (nanochat, Llama, modern LLMs)**:

```
  Instead of ADDING to embeddings, RoPE ROTATES Q and K inside attention.

   Position 0:    Q = [a, b, c, d]       (no rotation)
   Position 1:    Q' = rotate(Q by 1°)    (rotated by angle proportional to 1)
   Position 2:    Q' = rotate(Q by 2°)
   ...

  When you compute Q_i · K_j (dot product in attention):
    The result depends on (i - j), the relative distance.
    → "how far apart are these tokens?" is built into the dot product.

  Key property: relative positions encoded naturally.
                Generalizes to longer sequences than trained on.
```

**Interactive RoPE**: https://blog.eleuther.ai/rotary-embeddings/ - animations showing how rotation encodes position.

**Position encoding comparison**:

| Approach | Used by | Pros | Cons |
|----------|---------|------|------|
| Sinusoidal | Original transformer (2017) | Fixed, extrapolates somewhat | Mediocre vs RoPE |
| Learned absolute | GPT-2, nanoGPT | Simple, works | Hard cap at block_size |
| RoPE | Llama, nanochat, most 2023+ | Relative, extrapolates well | Slightly more complex |
| ALiBi | MPT, BLOOM | Very simple, extrapolates | Less expressive |

## Exercises

1. In nanoGPT, print the position embeddings after training and plot the first 2 PCA components. They'll form a smooth curve along the position index - positions 0..1023 become geometrically organized automatically.
2. Read the RoPE formula on Wikipedia or the [blog post by Eleuther](https://blog.eleuther.ai/rotary-embeddings/). Accept the math on first reading; it'll click after a few exposures.
3. Find where RoPE is applied in `nanochat/nanochat/gpt.py`. Note that it's applied *inside* `CausalSelfAttention`, to Q and K, before the dot product.

## Next

`10_lm_head_and_sampling.md` - turn final vectors back into words.
