# 2.1 - A Neuron (And Why the Word Is Misleading)

## The one-line definition

A "neuron" in deep learning is a function:

```
y = activation(w . x + b)
```

where:
- `x` is the input vector (shape `(d,)`)
- `w` is the weight vector (same shape `(d,)`)
- `b` is a scalar bias
- `activation` is a non-linear function like ReLU (covered next lesson)
- `y` is a single number output

That's all. One weighted sum, one bias, one non-linearity. Nothing biological about it beyond the name.

## Why the brain analogy hurts more than it helps

You'll hear that artificial neurons "mimic brain neurons." They don't, really.
They were loosely *inspired* by 1940s models of biological neurons, but the
connection is so loose it's misleading. Biological neurons are continuous
electrochemical systems; artificial neurons are matmul + ReLU. Stop thinking
about brains. Think about matrices.

## A layer of neurons = one matrix

If you have `n` neurons in parallel, each taking the same input `x` (shape `(d,)`),
their outputs stack into a vector `y` of shape `(n,)`. Each neuron has its own
weight vector, so the weights stack into a matrix `W` of shape `(d, n)`. Biases
stack into vector `b` of shape `(n,)`. Then:

```
y = activation(x @ W + b)    # shape (n,)
```

A layer of neurons = one matmul + bias + activation. This is exactly `nn.Linear(d, n)` followed by an activation in PyTorch.

## Code it

```python
import torch
import torch.nn as nn

layer = nn.Linear(in_features=4, out_features=3)
print(layer.weight.shape)  # torch.Size([3, 4])
print(layer.bias.shape)    # torch.Size([3])

x = torch.randn(2, 4)   # batch of 2 vectors of size 4
y = layer(x)
print(y.shape)         # torch.Size([2, 3])
# internally this is: x @ layer.weight.T + layer.bias
```

Notice PyTorch stores `W` transposed to `(out, in)` internally. Why? Historical. Doesn't matter; `nn.Linear` handles it.

## Where this appears in nanoGPT

Every `nn.Linear` in `model.py` is a layer of neurons. Count them in one GPT block:

- `self.c_attn = nn.Linear(n_embd, 3*n_embd)` - 3x a linear (packed Q, K, V).
- `self.c_proj = nn.Linear(n_embd, n_embd)` - the attention output projection.
- `self.c_fc = nn.Linear(n_embd, 4*n_embd)` - first linear in the MLP.
- `self.c_proj = nn.Linear(4*n_embd, n_embd)` - second linear in the MLP.

Four matmuls per block. 12 blocks in GPT-2-small. 48 matmuls per forward pass, roughly. Each is a bunch of neurons in parallel.

## Visualize this

**A neuron, in one picture**:

```
        w₁
  x₁ ────●────┐
        w₂   │
  x₂ ────●───┤
        w₃   ├── Σ ──┬──── activation ──── y
  x₃ ────●───┤       │
        w₄   │     + b
  x₄ ────●────┘
```

Weighted sum of inputs, plus bias, passed through an activation function. That's it. Everything beyond this - attention, MLPs, multi-head - is *many of these in parallel, stacked*.

**A layer = neurons in parallel**:

```
  Input (d=4)          Output (n=3 neurons)

    x₁ ─┬─┬─┐          ● y₁
    x₂ ─┼─┼─┤     →    ● y₂
    x₃ ─┼─┼─┤          ● y₃
    x₄ ─┴─┴─┘

  Weight matrix W: shape (4, 3) - 12 numbers learned.
  Bias b: shape (3,) - 3 numbers learned.
```

Three neurons, each doing its own weighted sum in parallel - that's `nn.Linear(4, 3)`.

**Interactive playground**: https://playground.tensorflow.org/ - literally watch neurons learn to classify data. Change activations, layer widths, see effects.

**Or play with a single neuron right here:**

```viz
{"viz": "neuron"}
```

Slide w and b. Change the activation. See how one neuron maps input to output. Compare ReLU vs GELU vs tanh visually.

## Exercise

1. Implement `nn.Linear` by hand:

```python
class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_features, out_features) * 0.01)
        self.b = nn.Parameter(torch.zeros(out_features))
    def forward(self, x):
        return x @ self.W + self.b
```

Use it, verify it produces the same shape as `nn.Linear`. Then delete it and
go back to using `nn.Linear` - that's the whole point of the library.

## What's next

`02_activations_nonlinearity.md` - why we need the `activation` function at all. (Skip it, and the whole network collapses to one linear layer.)
