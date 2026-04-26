# 3.2 - Embeddings

Tokens are integers. But integers aren't useful for neural math - the distance between token 57 and token 58 has no meaning. We need to convert each token into a **vector** that we can do math on. This is an **embedding**.

## The definition

An **embedding** layer is a lookup table: a matrix `E` of shape `(vocab_size, n_embd)`. Token ID `i` maps to row `E[i]`, a vector of length `n_embd`.

```python
import torch
import torch.nn as nn

vocab_size = 50257
n_embd = 768   # GPT-2 small
emb = nn.Embedding(vocab_size, n_embd)   # E has shape (50257, 768)

token_ids = torch.tensor([1000, 2000, 42])   # a sequence of 3 token IDs
vectors = emb(token_ids)
print(vectors.shape)   # (3, 768)
```

That's it. Internally, `E` is just a tensor, and `emb(token_ids)` is equivalent to `E[token_ids]` - a fancy lookup. The weights are *trainable*, so the embedding values are learned during training.

## Why this matters

After training, similar tokens end up with similar embedding vectors. "King" and "queen" will be close. "Paris" and "London" will be close. This is called a **semantic embedding space**, and it's where the model *reasons* about meaning.

Whether you choose characters, BPE subwords, or words doesn't change the mechanism. Each token becomes a vector. The rest of the model (attention, MLPs) operates on those vectors.

## In nanoGPT

Open `model.py`, class `GPT`, `__init__`:

```python
self.transformer = nn.ModuleDict(dict(
    wte = nn.Embedding(config.vocab_size, config.n_embd),   # token embeddings
    wpe = nn.Embedding(config.block_size, config.n_embd),   # position embeddings
    ...
))
```

Two embeddings:
- `wte`: word/token embeddings - one vector per token ID.
- `wpe`: position embeddings - one vector per position (0, 1, 2, ..., block_size-1).

In the forward pass:

```python
tok_emb = self.transformer.wte(idx)   # (B, T, n_embd)
pos = torch.arange(0, t, dtype=torch.long, device=device)
pos_emb = self.transformer.wpe(pos)   # (T, n_embd)
x = self.transformer.drop(tok_emb + pos_emb)   # (B, T, n_embd)
```

The token embedding and position embedding are **added**. Simple but effective. Each input token is now a vector that encodes *both* what it is and where it is. We'll unpack position embeddings in Lesson 9.

## Weight tying (a common optimization)

At the end of the model, we have an `lm_head` (another linear layer that projects from `n_embd` back to `vocab_size` to produce logits). It's a weight matrix of the *same shape* as the token embedding matrix: `(n_embd, vocab_size)` vs `(vocab_size, n_embd)`.

A clever trick: make them share weights (the input embedding matrix transposed IS the output projection matrix). Saves ~50M parameters in GPT-2 and often works slightly better. Called **weight tying**.

In `nanoGPT/model.py`:
```python
self.transformer.wte.weight = self.lm_head.weight   # weight tying
```

Subtle but standard.

## Embedding dimension as a hyperparameter

`n_embd` (or `d_model` in papers) controls how much information each vector can carry.
- GPT-2 small: 768
- GPT-2 medium: 1024
- GPT-2 large: 1280
- GPT-2 XL: 1600
- GPT-3: 12288 (!)
- Llama-2 70B: 8192

Bigger `n_embd` means more expressive per-token representations, but more compute and memory everywhere downstream (every matmul involves `n_embd`).

In nanochat, `n_embd` is computed from `depth` via a scaling rule (width grows with depth in a compute-optimal way). See `nanochat/nanochat/gpt.py` `GPTConfig.__post_init__`.

## Exercise

1. In PyTorch: make an `nn.Embedding(10, 4)`. Print `emb.weight`. Embed the tokens `[1, 2, 3]` - you should see rows 1, 2, 3 of the weight matrix.

2. Verify that `emb(torch.tensor([1,2,3]))` equals `emb.weight[[1,2,3]]`. Yes - an embedding really is just a fancy lookup.

3. Find the embedding creation in `model.py` (search `wte`). Find where it's used in `forward`. Note: shape goes from `(B, T)` integers to `(B, T, n_embd)` floats. Remember this pattern.

4. Count parameters: for GPT-2 small (vocab 50257, n_embd 768), how many parameters are in the token embedding alone? `50257 * 768 = ~38.5M`. That's nearly a third of GPT-2's 124M total!

## Next

`03_attention_intuition.md` - the key operation, first pass, zero math.
