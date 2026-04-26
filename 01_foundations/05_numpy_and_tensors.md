# 1.5 - NumPy and PyTorch Tensors: Hands-on

Time to work with tensors for real. This lesson is mostly code you run. Don't skim it - type it.

## numpy first (the ancestor)

```python
import numpy as np

a = np.array([1, 2, 3])              # 1-D array, shape (3,)
b = np.array([[1, 2], [3, 4]])       # 2-D array, shape (2, 2)
print(a.shape, b.shape, b.dtype)

c = np.zeros((3, 4))                 # 3x4 of zeros
d = np.ones((2, 2))                  # 2x2 of ones
e = np.random.randn(5, 5)            # 5x5 random (normal distribution)

print(a + 10)                        # [11,12,13]
print(b * 2)                         # element-wise
print(b @ b)                         # matrix multiply: [[7,10],[15,22]]
print(e.mean(), e.std())             # summary stats
print(e[:2, :])                      # slicing - first 2 rows
```

Run this. Play. NumPy has been the scientific-Python foundation for 25+ years; PyTorch copies its API on purpose.

## PyTorch tensors (the ML-grade version)

```python
import torch

a = torch.tensor([1., 2., 3.])       # 1-D
b = torch.tensor([[1., 2.], [3., 4.]])   # 2-D
c = torch.zeros(3, 4)
d = torch.ones(2, 2)
e = torch.randn(5, 5)
print(a.shape, a.dtype, a.device)
```

Tensor has three key properties: **shape**, **dtype**, **device**.

- `shape`: we covered this. `(5, 5)` etc.
- `dtype`: `torch.float32` is default for computation. `torch.float16` and `torch.bfloat16` for mixed-precision (Module 4). `torch.long` for integer indices like token IDs.
- `device`: `cpu` or `cuda:0` or `mps` (Apple Silicon). Tensors on different devices cannot be combined. Moving happens with `.to('cuda')`.

## The three killer features vs NumPy

**1) GPU support.** If you have a GPU:

```python
x = torch.randn(1000, 1000).to('cuda')
y = torch.randn(1000, 1000).to('cuda')
z = x @ y   # happens on GPU, fast
```

Move something back with `.cpu()`:

```python
z_cpu = z.cpu().numpy()   # standard pattern to hand off to matplotlib etc.
```

**2) Autograd (automatic differentiation).**

```python
w = torch.tensor([3.0], requires_grad=True)
loss = (w**2 - 4)**2
loss.backward()
print(w.grad)    # derivative computed for you
```

Every operation on a `requires_grad=True` tensor is recorded into a "computation graph", and `.backward()` walks it to fill in gradients. This is how neural networks train without you ever doing calculus.

**3) Neural net layers.** `torch.nn` provides pre-built components:

```python
import torch.nn as nn

layer = nn.Linear(10, 5)     # 10-in, 5-out. Holds a (10,5) weight + (5,) bias.
x = torch.randn(32, 10)      # batch of 32 inputs of size 10
y = layer(x)                 # (32, 5)
```

Internally, `layer(x) == x @ layer.weight.T + layer.bias`. Just a matmul with stored parameters. `requires_grad=True` is set on those parameters by default.

## Patterns you'll see constantly

```python
# reshape a tensor (must keep total elements the same)
x = torch.randn(2, 3, 4)
y = x.view(6, 4)         # (6, 4) - shares memory with x. fast.
y = x.reshape(6, 4)      # slightly safer, may copy.

# transpose
a = torch.randn(3, 4)
b = a.T                  # (4, 3)
c = a.transpose(0, 1)    # same

# squeeze / unsqueeze: add or remove a dimension of size 1
x = torch.randn(5)                 # shape (5,)
x.unsqueeze(0).shape               # (1, 5)
x.unsqueeze(-1).shape              # (5, 1)
torch.zeros(1, 5, 1).squeeze().shape  # (5,)

# indexing and slicing (same as numpy)
x = torch.arange(12).view(3, 4)
x[1]            # row 1  -> tensor([4,5,6,7])
x[:, 2]         # column 2 -> tensor([2,6,10])
x[1, 2]         # scalar -> 6

# masked fill (used in attention!)
x = torch.randn(3, 3)
mask = torch.tensor([[False, True, True],
                     [False, False, True],
                     [False, False, False]])
x.masked_fill(mask, float('-inf'))   # sets upper-triangle to -inf

# concatenation and stacking
a = torch.randn(2, 3)
b = torch.randn(2, 3)
torch.cat([a, b], dim=0).shape   # (4, 3) - along rows
torch.cat([a, b], dim=1).shape   # (2, 6) - along cols
torch.stack([a, b], dim=0).shape # (2, 2, 3) - NEW dim

# element-wise math
torch.relu(x)
torch.tanh(x)
torch.sigmoid(x)
torch.exp(x)
torch.log(x)
```

## Important gotchas

- **In-place vs out-of-place.** `x.add_(1)` modifies `x`. `x.add(1)` returns a new tensor. The underscore convention is pervasive. In-place can break autograd - avoid during training.
- **.item() for Python scalars.** `loss.item()` extracts the Python float out of a 0-D tensor (for printing).
- **.detach() to stop gradient flow.** `x.detach()` returns a tensor with the same data but no autograd tracking. Used all over: logging, inference, freezing parts.
- **.to(device)** returns a *new* tensor on the device; doesn't modify in place. Pattern: `x = x.to('cuda')`.

## Mini-exercise: a hand-rolled linear layer

```python
import torch
torch.manual_seed(0)

# implement y = x @ W + b
W = torch.randn(3, 2, requires_grad=True)
b = torch.randn(2, requires_grad=True)
x = torch.randn(5, 3)

y = x @ W + b
print(y.shape)     # (5, 2)

target = torch.zeros(5, 2)
loss = ((y - target) ** 2).mean()
loss.backward()

print(W.grad.shape)   # (3, 2) - matches W's shape
print(b.grad.shape)   # (2,)
```

You just wrote a linear regression training step manually. Congrats.

## Where this shows up in nanoGPT

Open `~/workspace/nanoGPT/model.py`. Spend 10 minutes skimming. You won't understand everything. Try to notice:

- Everywhere `.view(...)`, `.transpose(...)`, `.contiguous()` - those are tensor reshaping.
- The `forward` methods take an `x` tensor in, produce tensors out.
- `F.softmax`, `F.cross_entropy`, `F.scaled_dot_product_attention` - we've met some of these.
- `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.Dropout` - standard layers.

You're starting to read this code!

## Exercises

1. Create a random tensor of shape `(4, 5)`, compute its mean along each column (dim=0), then along each row (dim=1). Check the output shapes.
2. Use `.view()` to reshape a (6,) tensor into (2, 3). Then into (3, 2).
3. Make a tensor of shape `(2, 3, 4, 5)`. Transpose dims 1 and 3 (hint: `permute`). What shape?
4. Using `masked_fill`, build a causal (lower-triangular) mask for a 5x5 attention matrix. This is exactly what a transformer does to prevent a token from seeing the future.
5. Write 3 lines of code that represent "for a single linear layer with input 10, output 5, apply it to a batch of 32 inputs, compute MSE loss against random targets, and call backward". Print `.grad` of the weight.

## Next

`capstone_bigram.md` - Module 1 capstone: build a bigram language model with PyTorch tensors, trained by gradient descent.
