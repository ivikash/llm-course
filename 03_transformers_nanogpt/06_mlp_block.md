# 3.6 - The MLP Block

Half of every transformer layer is attention. The other half is an **MLP (multi-layer perceptron)**. It's actually the larger half by parameter count.

## Visualize this

**The MLP block, live:**

```viz
{"viz": "mlp_block"}
```

Switch between GELU (nanoGPT) and SwiGLU (Llama, nanochat). Press **▶ Animate** to watch the token vector flow through. See the expand (C → 4C), activate, compress (4C → C) pattern. Parameter count updates with C.

**The structure**:

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
```

Applied **position-wise**: each token's vector goes through the MLP independently. No cross-token interaction here - that's attention's job.

## Why expand to 4x and come back

- `c_fc`: projects `C -> 4C`. Token has more "room to compute" in the wider space.
- `gelu`: introduces non-linearity (covered in Module 2).
- `c_proj`: projects `4C -> C`, back to the shared dimension for the next block.

The 4x ratio is historical (original Transformer paper) and has stuck. Llama uses a slightly different ratio (2.66x with SwiGLU) because the gating unit changes parameter accounting.

## Where most parameters live

For `C = 768`:
- MLP: `c_fc` (768 * 3072) + `c_proj` (3072 * 768) = 4.7M params.
- Attention (prev lesson): 2.36M.

The MLP block is ~2x the params of the attention block. Across all layers, MLPs dominate.

## What is the MLP actually doing?

Recent interpretability research suggests MLPs act as **key-value memories**: they store factual knowledge the model has learned. Attention routes information *across tokens*; MLPs process information *per token*, possibly retrieving stored facts.

This isn't a rigorous truth, but a useful mental model. "Paris is the capital of France" probably lives as an MLP pattern; "Paris" as a word linking to the MLP pattern for "capital of France" lives via attention.

## SwiGLU variant (in nanochat)

As we saw in Lesson 2.2, nanochat's MLP uses SwiGLU - two input projections gated by SiLU. It's what Llama uses, and it's slightly better than plain GELU in practice.

```python
# nanochat/nanochat/gpt.py
def forward(self, x):
    return self.c_proj(F.silu(self.c_fc(x)) * self.c_fc2(x))
```

Notice: two input projections (`c_fc` and `c_fc2`), SiLU on one branch, element-wise multiply, then output projection. More params per layer than plain GELU but often trained with a smaller expansion ratio to compensate.

## Visualize this

**The MLP block as a "wider workspace" for each token**:

```
  Input token vector (768-d)
         │
         ▼
  ┌────────────────┐
  │  c_fc          │  linear: 768 → 3072
  │  (expand 4x)   │
  └────────┬───────┘
           │
           ▼
  ┌────────────────┐
  │  GELU          │  nonlinearity
  └────────┬───────┘
           │  token now lives in a 3072-d "workspace"
           │
           ▼
  ┌────────────────┐
  │  c_proj        │  linear: 3072 → 768
  │  (compress back)│
  └────────┬───────┘
           │
           ▼
  Output token vector (768-d, same shape as input)
```

Each position goes through this independently. No cross-token interaction - that's attention's job. MLPs think "per-token."

**Parameter breakdown in a block**:

```
  Per transformer block:
  ┌──────────────────────────────────────┐
  │ Attention                            │
  │  c_attn: 768 → 2304    ~1.77M params │
  │  c_proj:  768 → 768    ~0.59M params │
  │                                      │
  │ MLP                                  │
  │  c_fc:    768 → 3072   ~2.36M params │  ← the biggest
  │  c_proj: 3072 → 768    ~2.36M params │  ← chunks
  │                                      │
  │ LayerNorm × 2          ~3K params    │
  ├──────────────────────────────────────┤
  │ Total: ~7M params per block          │
  │ MLP is ~68% of the block.            │
  └──────────────────────────────────────┘
```

**The MLP is where knowledge lives** (interpretability research hypothesis): MLPs act as key-value memories storing factual patterns, while attention routes information between tokens. Search "MLP neurons as key-value memories" (Geva 2020).

**SwiGLU vs GELU, visually**:

```
  GELU MLP (nanoGPT):
    x ──→ W_fc ──→ GELU ──→ W_proj ──→ out
               (one path)

  SwiGLU MLP (nanochat, Llama):
         ┌──→ W_fc  ──→ SiLU ──┐
    x ───┤                     ├── multiply ──→ W_proj ──→ out
         └──→ W_fc2 ──────────┘
              (gating path)
```

The gating path (W_fc2) acts like an "attention over feature dimensions" - some features are amplified, some suppressed, per-token. Empirically slightly better than GELU.

## Exercise

1. Sketch the forward of nanoGPT's MLP on paper. Shape check: `(B, T, 768) -> (B, T, 3072) -> gelu -> (B, T, 3072) -> (B, T, 768)`.
2. Count parameters in nanoGPT's MLP for C=768. Compare to the attention block's count.
3. Why no bias on the output projection of nanochat's MLP? (Open the file; note `bias=False` on every Linear. Common in modern LLMs - saves params, slightly better numerics.)

## Next

`07_layernorm_and_residual.md` - the glue that makes deep networks train at all.
