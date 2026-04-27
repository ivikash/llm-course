# 2.2 - Activations and Non-Linearity

## Why we need them

Suppose you stack two linear layers without any non-linearity between:

```
y = (x @ W1 + b1) @ W2 + b2
```

With some algebra, this equals:

```
y = x @ (W1 @ W2) + (b1 @ W2 + b2)
  = x @ W_combined + b_combined
```

...which is **just one linear layer**. Stacking linears without non-linearity gains you nothing.

To build expressive networks, you need to insert a **non-linear function** between linear layers. That's an **activation function**.

## The common activations

| Name | Formula (approx) | Where used |
|------|------------------|------------|
| **ReLU** | `max(0, x)` | Classical. Simple, fast. Used in CNNs, some MLPs. |
| **GELU** | smooth-ReLU-ish: `x * sigmoid(1.702 * x)` approx | GPT-2, BERT, nanoGPT's MLP. |
| **SiLU / Swish** | `x * sigmoid(x)` | Llama, nanochat. |
| **SwiGLU** | gated SiLU, more complex (uses extra linear). | Llama-style models including nanochat. |
| **Tanh** | `tanh(x)`, output in (-1,1) | Older, still occasionally used. |
| **Sigmoid** | `1 / (1 + exp(-x))`, output in (0,1) | For probabilities, gates. Rarely as hidden activation now. |
| **Softmax** | normalizes to probabilities | Last layer in classification, inside attention. Not usually a "hidden" activation. |

For our purposes: modern LLMs use **GELU** (GPT-2, nanoGPT) or **SwiGLU** (Llama, nanochat). ReLU still lives in simpler models.

## Visualize them

```python
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

x = torch.linspace(-5, 5, 200)
plt.plot(x, F.relu(x), label='ReLU')
plt.plot(x, F.gelu(x), label='GELU')
plt.plot(x, F.silu(x), label='SiLU')
plt.plot(x, torch.tanh(x), label='Tanh')
plt.legend(); plt.grid(); plt.show()
```

All are "monotonic-ish curves that are zero or negative on the left, rising on the right." Subtle differences matter at scale but philosophically they're similar.

## Why GELU over ReLU

GELU is smooth. Its derivative is continuous everywhere. ReLU's derivative is exactly 0 for negative inputs, which can cause "dead neurons" - once a neuron's output goes persistently negative, no gradient flows, it stops learning. GELU keeps a small gradient there. Empirically, GELU trains slightly better transformers.

## Where GELU is in nanoGPT

`model.py`, class `MLP`:

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

Linear -> GELU -> Linear -> Dropout. That's the "MLP" inside every transformer block. The expansion-contraction (`n_embd -> 4*n_embd -> n_embd`) is standard - the network gets a wider "work space" in the middle, then compresses back.

## Where SwiGLU is in nanochat

`nanochat/nanochat/gpt.py`, class `MLP`:

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_fc2  = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
    def forward(self, x):
        return self.c_proj(F.silu(self.c_fc(x)) * self.c_fc2(x))
```

Note the two input projections (`c_fc` and `c_fc2`) and the multiply - this is SwiGLU, a "gated" variant. One branch gets a SiLU activation, the other stays linear, and they multiply element-wise before projecting back. Used in Llama and most modern LLMs.

## Visualize this

**The activation function zoo**:

```
  ReLU              GELU              SiLU/Swish           Tanh

     │    ╱             │    ╱             │    ╱              │   ╱─────
     │   ╱              │   ╱              │   ╱               │  ╱
     │  ╱               │  ╱               │  ╱                │ ╱
  ───┼──────         ───┼──▬▬              ───┼──▬▬           ───┼─
      │                 │                     │              ╱  │
      │                 │                     │             ╱   │
      │               ▬▬│                   ▬▬│          ──────────
```

All monotonic-ish, all nonlinear, all have derivatives that flow nicely.

**Run this to see them yourself**:

```viz
{"viz": "activation_zoo"}
```

Left panel: each activation function. Right panel: its derivative. Notice ReLU's derivative is exactly 0 for x&lt;0 — that's the "dying neuron" problem. GELU and SiLU stay smooth. Why modern LLMs use GELU/SwiGLU, not ReLU.

**Why stacking linears without nonlinearity is pointless**:

```
  Layer1:  y = W₁x + b₁
  Layer2:  z = W₂y + b₂
  Combined: z = W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂)
         = a single linear layer with W'=W₂W₁, b'=W₂b₁+b₂
```

No matter how many linears you stack, you collapse back to one. The activation function is what breaks this.

## Exercises

1. Run the plotting snippet. Stare at the curves.
2. Implement your own GELU using the approximation: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))`. Compare to `F.gelu(x, approximate='tanh')`.
3. Take the MLP class from nanoGPT and the one from nanochat. Count parameters in each if `n_embd=768`. Which is bigger? Why? (Hint: SwiGLU has two input projections but many models use `2/3 * 4` expansion to compensate.)

## Next

`03_loss_functions.md` - MSE vs cross-entropy.
