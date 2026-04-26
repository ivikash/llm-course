# 3.4 - Attention, Mathematically

Now we make the previous lesson precise. By the end you'll read `CausalSelfAttention.forward` like English.

## Input setup

We have an input tensor `x` of shape `(B, T, C)`:
- B = batch size (e.g. 32 sequences at once)
- T = sequence length in tokens (e.g. 1024)
- C = embedding dim, `n_embd` (e.g. 768)

Every token is represented by a vector of length C. We want to update each token's vector using information from other tokens.

## Step 1: Linear projections to Q, K, V

For each token, we produce three new vectors: a query, a key, and a value. These come from three learned linear projections of the input:

```
q_i = W_Q x_i   (query for token i)
k_i = W_K x_i   (key for token i)
v_i = W_V x_i   (value for token i)
```

Where `W_Q`, `W_K`, `W_V` are learned matrices, each of shape `(C, head_dim)` in a single-head model, or `(C, C)` if we pack all heads together and split later (nanoGPT's approach).

nanoGPT packs them into one matrix `c_attn` of shape `(C, 3*C)` and splits the output:

```python
self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
...
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
# each is (B, T, C)
```

## Step 2: Compute attention scores (Q dot K)

For each query-key pair, compute the dot product:

```
score_ij = q_i . k_j
```

Stack into a matrix `S` of shape `(T, T)`. `S[i, j]` is how much token i should attend to token j.

In batched form: `S = Q @ K^T`, where Q and K are `(T, C)`.

There's one more detail: we **scale** the dot product by `1/sqrt(head_dim)`. Without this, for large C, dot products have large variance, pushing softmax to extreme values (saturated gradients, bad training).

```
S = (Q @ K^T) / sqrt(head_dim)
```

In code:
```python
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
```

## Step 3: Apply the causal mask

We only want each token to look at previous (and itself) tokens. Positions `j > i` should have zero weight.

We add `-inf` to the illegal positions before softmax, so after softmax they become zero:

```python
att = att.masked_fill(bias[:,:,:T,:T] == 0, float('-inf'))
```

Where `bias` is a precomputed lower-triangular matrix of 1s (allowed) and 0s (blocked). After this:

```
S[i, j] = -inf   for j > i
S[i, j] = (q_i . k_j)/sqrt(d)  for j <= i
```

## Step 4: Softmax (normalize to weights)

```python
att = F.softmax(att, dim=-1)
```

Softmax is applied row-wise. Each row sums to 1. The `-inf` positions become 0. We now have valid attention weights: for each token i, a distribution over which earlier tokens to attend to.

## Step 5: Weighted sum of values

```python
y = att @ v
```

Shape: `(T, T) @ (T, C) -> (T, C)`. Each output token is a weighted combination of all (allowed) value vectors. High-attention = that value gets a big contribution.

## Put it together

```python
# x: (B, T, C)
q, k, v = self.c_attn(x).split(C, dim=2)   # each (B, T, C)

# score
att = (q @ k.transpose(-2, -1)) / math.sqrt(C)   # (B, T, T)

# mask
att = att.masked_fill(mask[:T, :T] == 0, float('-inf'))

# softmax
att = F.softmax(att, dim=-1)

# aggregate values
y = att @ v   # (B, T, C)

# final output projection (let the heads communicate; see next lesson)
y = self.c_proj(y)
```

The output `y` has the same shape as input `x`, but each position's vector has absorbed context from other positions. **This is attention.**

## Why the `1/sqrt(d)` scale

Skip if you want - but it's a clean intuition. If q and k are vectors with components roughly N(0, 1), then q.k is a sum of d products, each N(0, 1). The sum has variance d, so standard deviation sqrt(d). As d grows, dot products get larger, softmax becomes peakier, gradients vanish. Dividing by sqrt(d) keeps the scale constant.

## Flash Attention

For long sequences, the T×T attention matrix is huge (16M entries at T=4K). **Flash Attention** is a clever algorithm that computes softmax(QK^T)V *without* materializing the full T×T matrix - it tiles the computation to fit in GPU cache. Same math, much faster and less memory.

In PyTorch, `F.scaled_dot_product_attention(q, k, v, is_causal=True)` uses Flash Attention under the hood when your hardware supports it. That's the branch nanoGPT takes:

```python
if self.flash:
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, ..., is_causal=True)
```

## Verify it yourself

Write a tiny by-hand attention:

```python
import torch
import torch.nn.functional as F
import math

torch.manual_seed(0)

B, T, C = 1, 4, 8
x = torch.randn(B, T, C)

# projections
Wq = torch.randn(C, C)
Wk = torch.randn(C, C)
Wv = torch.randn(C, C)

q = x @ Wq
k = x @ Wk
v = x @ Wv

# score
att = (q @ k.transpose(-2, -1)) / math.sqrt(C)

# causal mask
mask = torch.tril(torch.ones(T, T))
att = att.masked_fill(mask == 0, float('-inf'))

# softmax
att = F.softmax(att, dim=-1)
print("attention weights:")
print(att[0])

# aggregate
y = att @ v
print("output shape:", y.shape)  # (1, 4, 8)
```

Print the attention weights. See the triangular pattern. Each row sums to 1.

## Visualize this

**The full attention pipeline, one picture**:

```
  x  (B, T, C=768)
  │
  ├──── @ W_Q ───────► Q  (B, T, 768)    ──┐
  │                                        │
  ├──── @ W_K ───────► K  (B, T, 768)    ──┤  reshape to
  │                                        │  multi-head:
  └──── @ W_V ───────► V  (B, T, 768)    ──┘  (B, n_head, T, head_dim)

                       Q @ K^T
                          │
                   ┌──────┴──────┐
                   │             │
                   ▼             │
              ┌────────┐         │
              │  T × T │         │   / sqrt(head_dim)
              │ scores │         │
              └────┬───┘         │
                   │             │
              mask (causal)      │
                   │             │
              softmax (row-wise) │
                   │             │
              attention weights  │
                   │             │
                   └──── @ V ────┤
                                 │
                                 ▼
                        (B, n_head, T, head_dim)
                                 │
                            reshape to
                        (B, T, C=768)
                                 │
                           @ W_O (output projection)
                                 │
                                 ▼
                         same shape as input
```

**Run this to see attention happen step by step**:

```python
import torch, torch.nn.functional as F, math
import matplotlib.pyplot as plt

B, T, C = 1, 6, 8
torch.manual_seed(0)
x = torch.randn(B, T, C)
Wq, Wk, Wv = torch.randn(C, C), torch.randn(C, C), torch.randn(C, C)

q = x @ Wq
k = x @ Wk
v = x @ Wv

# scores
att = (q @ k.transpose(-2, -1)) / math.sqrt(C)
print("raw scores:"); print(att[0].round(decimals=2))

# mask
mask = torch.tril(torch.ones(T, T))
att = att.masked_fill(mask == 0, float('-inf'))
print("after causal mask:"); print(att[0].round(decimals=2))

# softmax
att = F.softmax(att, dim=-1)
print("attention weights (rows sum to 1):"); print(att[0].round(decimals=3))

# heatmap
plt.imshow(att[0].numpy(), cmap='viridis')
plt.xlabel("key"); plt.ylabel("query"); plt.colorbar()
plt.savefig("attn_pattern.png")
```

Run it, print the tensors, look at the heatmap. Every LLM, every transformer, does exactly this.

**3Blue1Brown's attention video**: https://www.youtube.com/watch?v=eMlx5fFNoYc - 26 min, animated attention. Watch after reading.

**bbycroft.net/llm's attention section**: click into one token, literally watch Q vectors light up, dot-product with all K vectors, softmax to weights, sum V vectors. Interactive. Takes 2 min.

## Exercises

1. Run the snippet above. Verify the mask looks right - row `i` has non-zero entries only in columns `0..i`.
2. Change the input `x` - make token 3 be a copy of token 1 (`x[0, 3] = x[0, 1]`). What happens to the attention weight `att[0, 3, 1]`? Higher? Why?
3. Remove the `1/sqrt(C)` factor. Do the attention weights become more peaky or less?
4. Set one of the logit scores in `att` before softmax to +100. See the resulting row become nearly one-hot.

## Next

`05_multi_head_attention.md` - instead of one attention, do 12 in parallel. Why and how.
