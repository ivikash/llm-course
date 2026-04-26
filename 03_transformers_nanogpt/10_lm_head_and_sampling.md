# 3.10 - LM Head and Sampling

After all the blocks, each token has a rich vector. How do we turn it into a word?

## The LM head

A single linear layer that projects from `(B, T, C)` to `(B, T, vocab_size)`. The output is a distribution over the vocabulary at each position - what token should come next?

```python
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
...
logits = self.lm_head(x)   # (B, T, V)
```

This matrix has `C * V` parameters. For GPT-2: `768 * 50257 = 38.6M` - the same size as the token embedding, which is why weight tying saves so much.

## Computing loss during training

During training, we have targets (the actual next tokens):

```python
loss = F.cross_entropy(logits.view(-1, V), targets.view(-1), ignore_index=-1)
```

That's it. Combined with `loss.backward()` and optimizer step, the whole model trains.

## Generation / sampling

To actually *generate* text, we feed in a prompt and sample the next token, then append and repeat.

```python
@torch.no_grad()
def generate(idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]   # crop to context length
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature   # take last position's logits
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

That's `nanoGPT/model.py`'s `generate` method. Read it. It's the whole generation story.

## Sampling strategies

Picking the next token from a distribution is not trivial. Options:

- **Greedy** (temperature → 0, or just argmax): always pick the most likely. Produces the most coherent but most repetitive text.
- **Random sampling**: pick proportional to probability. Diverse but can be incoherent.
- **Temperature**: scales logits before softmax. Higher = more random. Usually 0.6-0.9 for chat.
- **Top-k**: keep only the top-k most likely tokens, sample among those. Prevents the model from picking an absurdly unlikely token.
- **Top-p (nucleus)**: keep smallest set of tokens whose probabilities sum to p (e.g. 0.9). Adaptive - more tokens for uncertain positions.
- **Repetition penalty**: penalize recently-used tokens. Helps with repetition issues.

In `nanoGPT/sample.py`:
```python
logits = logits[:, -1, :] / temperature
if top_k is not None:
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = -float('inf')
probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)
```

Temperature + top-k. Simple, effective.

## KV cache (for efficiency)

During generation, we call the model repeatedly with ever-longer sequences. Naive implementation: recompute attention for all previous tokens every step. Quadratic work.

**KV cache**: save the K and V tensors from previous positions. For each new token, only compute the new Q/K/V, then attention between the new Q and the cached Ks/Vs.

- nanoGPT does NOT implement KV cache - it's meant for teaching.
- nanochat DOES - see `nanochat/nanochat/engine.py`. We'll walk through it in Module 5.

KV cache makes generation roughly 10-100x faster for long outputs.

## Exercises

1. Run sampling with the Shakespeare-trained nanoGPT model (after capstone). Try temperatures 0.1, 0.8, 1.5. Note how output changes.
2. Implement a simple top-p (nucleus) sampling function in PyTorch:
```python
def top_p_sample(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    mask = cumulative_probs > p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_logits[mask] = -float('inf')
    # scatter back
    logits = torch.full_like(logits, -float('inf'))
    logits.scatter_(1, sorted_indices, sorted_logits)
    return logits
```

3. Read `model.py`'s generate() function in full. Understand the `idx_cond` cropping - it's because the model can only process `block_size` tokens at a time.

## Next

`11_train_py_walkthrough.md` - every line of `nanoGPT/train.py`, now that you understand the model.
