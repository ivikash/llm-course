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

## Visualize this

**Sampling strategies side-by-side**:

```
  Logits for next token: [1.2, 0.8, 3.5, -0.1, 2.1, ...]  (vocab size 50257)

                              Softmax probabilities

  Greedy (temp=0):
                              ▮                              all mass on argmax
  ─────────────────────────── 3.5 ──────────────────────────  (deterministic)

  Temperature = 1.0:
                          ▮   ▮▮    ▮         (natural distribution)
                       ▮▮▮▮▮▮▮▮▮▮▮▮▮

  Temperature = 0.3:
                              ▮▮▮                          (sharper, biased to top)
                            ▮▮▮▮▮▮

  Temperature = 2.0:
                       ▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮  (flatter, more random)

  Top-k (k=5):
                                ▮                         (only top 5 kept)
                            ▮▮  ▮
                          ▮ ▮  ▮

  Top-p (p=0.9):
                             Adaptive: keep tokens until probability mass reaches 0.9
```

ChatGPT's "temperature" slider literally does this.

**Try all three samplers live:**

```viz
{"viz": "sampling_strategies"}
```

Same logits (12 candidate next tokens). Slide temperature, top-k, top-p. Blue bars = probabilities after filtering. Press "Sample 20×" — red marks show the empirical distribution after 20 samples.

Try:
- Temperature 0.1, top-k 12, top-p 1 → nearly deterministic, always picks "cat"
- Temperature 1, top-k 3 → random among top 3
- Temperature 1.5, top-p 0.9 → adaptive nucleus; 5-6 candidates
- Temperature 3 → uniform chaos

**Autoregressive generation, step by step**:

```
  step 1:
    prompt: "Once upon a time"
    model input: [token("Once"), token(" upon"), token(" a"), token(" time")]
    model output: probability for next token
    sample: " there"  (let's say)

  step 2:
    prompt: "Once upon a time there"   (append last output)
    model input: [t("Once"), t(" upon"), t(" a"), t(" time"), t(" there")]
    model output: probability for next token
    sample: " lived"

  step 3:
    prompt: "Once upon a time there lived"
    ...and so on, until <|endoftext|> or max length.
```

Each token costs one full forward pass (without KV cache) - that's why generation is slow. KV cache (Module 5) makes each step O(1) past context instead of O(T²).

**Inside the LM head**:

```
  Final hidden state at position T-1:
     h = [0.12, -0.45, 0.8, ..., 0.33]   (shape: n_embd=768)

  Projection to vocab:
     logits = h @ W_lm_head  + b
                 (shape: 768 × 50257)
              →  (shape: 50257,)  ← one score per possible next token

  Softmax:
     probs = softmax(logits / temperature)
              →  (shape: 50257,)  ← probabilities summing to 1

  Sample:
     idx = torch.multinomial(probs, num_samples=1)
              →  one integer: the chosen next token
```

That's it. Everything past the final ln_f is this simple.

**Weight tying in action**:

```
  Input path:
    token_id → embedding matrix (50257 × 768) → vector (768,)
                 ^
                 │  SAME matrix (transposed)
                 │
  Output path:   │
    vector (768,) → projection (768 × 50257) → logits (50257,)

  → ~38M parameters shared between input and output. Saves memory.
```

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
