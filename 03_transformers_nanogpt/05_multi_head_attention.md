# 3.5 - Multi-Head Attention

A single attention operation gives each token one "way of looking at" other tokens. But language has many kinds of relationships - syntactic, semantic, coreferential, positional. It helps to have *multiple parallel attentions*, each free to learn a different specialty.

That's **multi-head attention**.

## The idea

Take the input `x` of shape `(B, T, C)`. Split the C-dim vector into `n_head` pieces, each of size `head_dim = C / n_head`. Run attention independently on each piece. Concatenate the outputs. Apply a final linear projection.

Concrete shapes for nanoGPT-small (C=768, n_head=12, head_dim=64):
- Input: `(B, T, 768)`.
- After Q/K/V projection: each is `(B, T, 768)`.
- Reshape to `(B, n_head, T, head_dim) = (B, 12, T, 64)`.
- Run attention on each head independently - each head sees 64-dim Q, K, V.
- Concatenate back: `(B, T, 768)`.
- Final projection (`c_proj`): another linear `(768 -> 768)`. This lets the heads' outputs mix.

Total compute: same as one big attention, but organized differently. Representationally: richer.

## Why bigger isn't always better

Why not just have one "head" with C=768 full-dim attention? Because you'd be computing one big attention pattern. With 12 heads, the model learns 12 different attention patterns simultaneously. Empirically this works better at the same parameter count. It's a kind of beneficial structural prior.

## In nanoGPT's code

`model.py`, `CausalSelfAttention.forward`:

```python
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
# q, k, v: (B, T, C)

k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
# now each: (B, nh, T, hs)  where nh=n_head, hs=head_dim=C/n_head

if self.flash:
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True, ...)
else:
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
    att = att.masked_fill(...)
    att = F.softmax(att, dim=-1)
    y = att @ v
# y: (B, nh, T, hs)

y = y.transpose(1, 2).contiguous().view(B, T, C)
y = self.resid_dropout(self.c_proj(y))
```

Read this slowly. Identify:
- Where C gets split into (n_head, head_dim).
- Where heads run in parallel (PyTorch broadcasts the matmul over the head dim).
- Where heads are re-stitched together.
- Where `c_proj` mixes them.

## What each head learns (empirically)

Researchers have peered inside trained transformers and found heads that specialize:
- **Syntactic heads**: attend to the previous token, or the token before that, etc.
- **Coreference heads**: pronouns attend to their referents.
- **Delimiter heads**: attend to sentence starts/ends.
- **Copy heads** (in some architectures): attend to where a named entity first appeared.

You can visualize attention patterns with tools like [BertViz](https://github.com/jessevig/bertviz). Fun rabbit hole but not required.

## Variants for awareness

- **MQA (Multi-Query Attention)**: all heads share the same K and V (only Q differs per head). Cheaper at inference. Falcon, PaLM use it.
- **GQA (Grouped-Query Attention)**: compromise - heads are grouped, each group shares K/V. Llama-2 70B, Llama-3, nanochat use this. Sweet spot for inference speed vs quality.

Open `nanochat/nanochat/gpt.py`, class `CausalSelfAttention`. You'll see `n_kv_head` - the number of K/V heads, which can be less than `n_head` (query heads). That's GQA.

## Exercises

1. For `n_embd=768`, `n_head=12`: what's `head_dim`? What if you tried `n_head=11`? (Would fail - must divide evenly.)
2. Calculate the number of parameters in `CausalSelfAttention` for GPT-2 small (C=768, n_head=12):
   - `c_attn` weight: `(768, 3*768)` = 1.77M
   - `c_attn` bias: `3*768` = 2.3K
   - `c_proj` weight: `(768, 768)` = 590K
   - `c_proj` bias: `768`
   - Total ~2.36M per attention layer.
   - 12 layers => ~28M just for attention. (The other ~80M is the MLP blocks.)
3. Compare nanoGPT's `CausalSelfAttention` with nanochat's (in `nanochat/nanochat/gpt.py`). Find the GQA implementation: how are K and V projected vs Q? (Different `out_features`.)

## Next

`06_mlp_block.md` - the other half of a transformer block.
