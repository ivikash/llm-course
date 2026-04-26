# 3.7 - LayerNorm and Residual Connections

Two small tricks that made deep transformers possible.

## Residual connections (skip connections)

Instead of `x -> f(x)`, do `x -> x + f(x)`. Each block adds a "correction" to the running representation, rather than replacing it.

```python
# nanoGPT's Block:
def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x
```

The `+` on each line is the residual. `x` flows through the network *unchanged* except for additions from attention and MLP layers.

### Why it matters

Gradient can now flow backward through the `+` directly, unobstructed. Without residuals, gradients get multiplied through every layer and tend to vanish (or explode) - which made pre-2015 networks hard to train beyond ~20 layers. Residuals introduced in ResNet (2015, for images) made 100+ layers trivial, and transformers took advantage. GPT-3 has 96 layers. You couldn't train that without residuals.

A nice mental model: the model is a base identity function (`x -> x`), and each layer is "what should I change about the current representation?" The default for an unneeded layer is "nothing" - contribute 0, and the skip keeps things flowing.

## LayerNorm

Inside each block, before each operation, we normalize the activations. This keeps them at consistent scale as they flow through dozens of layers.

LayerNorm formula for a vector `x`:

```
mean = x.mean()
std  = x.std()
y = (x - mean) / std       # now mean 0, std 1
y = y * gamma + beta       # learnable scale and shift
```

Applied per-token (across the embedding dimension C).

`gamma` (weight) and `beta` (bias) are learnable parameters of size `(C,)` each.

In PyTorch: `nn.LayerNorm(C)`.

### Pre-norm vs post-norm

The *original* 2017 transformer did: `x = LN(x + Attn(x))`. This is **post-norm**. It's hard to train at scale - gradients are unstable.

Modern models (GPT-2 onward) do: `x = x + Attn(LN(x))`. **Pre-norm**. More stable.

nanoGPT uses pre-norm (look at the Block.forward above - LN is inside the `attn()` and `mlp()` calls, wrapping the input).

### RMSNorm (modern variant)

Many recent models replace LayerNorm with **RMSNorm**: drop the mean subtraction, just divide by root-mean-square.

```
y = x / sqrt(mean(x^2) + eps)
y = y * gamma
```

Simpler, faster, no bias parameter. Llama and nanochat use this.

Open `nanochat/nanochat/gpt.py`, search for `norm`. You'll see `F.rms_norm(...)` or an RMSNorm class.

## The final LayerNorm

There's also a LayerNorm right before the LM head (the projection to vocab):

```python
# nanoGPT
x = self.transformer.ln_f(x)
logits = self.lm_head(x)
```

Just one more normalization on the very final representation.

## A full transformer block in pseudocode

```
def block(x):
    x = x + multi_head_attention(layernorm(x))   # attention sublayer
    x = x + mlp(layernorm(x))                     # MLP sublayer
    return x
```

That's the whole recipe. Stack 12 of these for GPT-2 small. 96 for GPT-3. Each one takes `(B, T, C)` in and outputs `(B, T, C)`. The vector at each position gets progressively richer as it flows through.

## In nanoGPT

`model.py`, class `Block`:

```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

Four components: ln_1, attn, ln_2, mlp. Two residuals. That's a transformer block.

## Visualize this

**Residual connections: the "highway" for gradients**:

```
  Without residuals:                  With residuals:

   input                               input
     │                                   │
     ▼                                   ├───── skip connection
  ┌──────┐                               │           │
  │ attn │                            ┌──▼──┐        │
  └──┬───┘                            │attn │        │
     ▼                                └──┬──┘        │
  ┌──────┐                               │           │
  │ mlp  │                               └─── + ─────┘
  └──┬───┘                                   │
     ▼                                       ├───── skip
  output                                     │           │
                                          ┌──▼──┐        │
  Deep stacks: gradients                  │ mlp │        │
  multiply through many                   └──┬──┘        │
  layers → vanish/explode.                   │           │
                                             └─── + ─────┘
                                                 │
                                             output

  Deep stacks: gradient flows through the + unchanged.
  Each layer is a "correction" to x, not a replacement.
```

Without residuals, training models with 96 layers (GPT-3) wouldn't work. With residuals, it's trivial.

**LayerNorm vs RMSNorm**:

```
  LayerNorm:
  For each token:
    mean(x), std(x)
    x = (x - mean) / std        ← centers and scales
    x = x * gamma + beta        ← learnable scale + shift

  RMSNorm (nanochat, Llama):
  For each token:
    rms(x) = sqrt(mean(x²))
    x = x / rms                 ← just scales, no centering
    x = x * gamma               ← (often no bias)
```

Simpler, faster. Empirically just as good. That's why modern LLMs moved to RMSNorm.

**Pre-norm vs post-norm**:

```
  Post-norm (original transformer, 2017):
    x = LayerNorm(x + Sublayer(x))

    ┌──────┐      ┌────┐      ┌──┐
    │sublay│ ──── │ +  │ ──── │LN│ ──── out
    └──────┘      └────┘      └──┘
      ▲           │
      └───────────┘ (skip)

    Gradient must pass through LN on the skip path. Can be unstable at depth.

  Pre-norm (GPT-2 onward):
    x = x + Sublayer(LayerNorm(x))

    ┌──┐      ┌──────┐      ┌────┐
    │LN│ ──── │sublay│ ──── │ +  │ ──── out
    └──┘      └──────┘      └────┘
                │
      ──────────┴──────── ─────┘  (skip bypasses LN)

    Skip path is identity. Gradient flows freely. Much more stable at depth.
```

Every modern LLM uses pre-norm. Small architectural detail, huge stability impact.

**Run this to see the effect of removing residuals**:

```python
# take the Shakespeare trainer and edit Block.forward:
def forward(self, x):
    # x = x + self.attn(self.ln_1(x))   # original
    # x = x + self.mlp(self.ln_2(x))    # original
    x = self.attn(self.ln_1(x))          # broken: no residual
    x = self.mlp(self.ln_2(x))           # broken: no residual
    return x
```

Train with this. Loss should stagnate or diverge quickly. Revert the change, train again. Loss drops smoothly. You've now *directly felt* why residuals matter.

## Exercises

1. Implement layernorm by hand:
```python
def my_layernorm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / (std + eps) * gamma + beta
```
Verify it matches `nn.LayerNorm` on a random input.

2. Rewrite `Block.forward` without residuals:
```python
x = self.attn(self.ln_1(x))
x = self.mlp(self.ln_2(x))
return x
```
Train a tiny model with this variant. It'll still learn, but slower and less stable. (You'll see this when we train in the capstone.)

3. Look up `RMSNorm` - it's in nanochat. Write down how it differs from LayerNorm. (Answer: no mean subtraction, no bias.)

## Next

`08_full_block_walkthrough.md` - putting all the pieces into one complete, end-to-end pass through a transformer block.
