# 2.4 - Backpropagation and SGD

## Why we need backprop

You know from Lesson 1.3: to train a neural net, you need the gradient of loss with respect to every weight. If the net has 1 billion weights, you need 1 billion derivatives.

Computing each derivative naively - one at a time, re-running the network from scratch - would take ~1 billion forward passes per training step. Unusable.

**Backpropagation** is the algorithm that computes **all** the gradients in ONE backward pass through the network. Same asymptotic cost as a forward pass. That's the magic.

## The intuition: chain rule, applied systematically

Recall the chain rule: for a composition `y = f(g(x))`, `dy/dx = f'(g(x)) * g'(x)`.

A neural network is `loss = L(layer_N(...layer_2(layer_1(input))))`. Deep composition.

Backprop walks **backwards** from `loss`:
1. Start at loss. Gradient of loss w.r.t. itself is 1.
2. For each layer, in reverse order: given gradient of loss w.r.t. its output, compute gradient of loss w.r.t. (a) its input and (b) its parameters.
3. Pass (a) back to the previous layer. Accumulate (b) into that layer's weight gradients.
4. Continue until reaching the input.

One backward sweep = one gradient for every weight.

## PyTorch does this for you

In PyTorch, every operation on a `requires_grad=True` tensor is recorded into a computation graph. `loss.backward()` walks this graph backwards and fills `.grad` on every parameter.

```python
import torch
x = torch.tensor([2.0], requires_grad=True)
y = x * 3 + 1
z = y ** 2
z.backward()
print(x.grad)  # dz/dx = 2*y*3 = 42. Check.
```

You will never write backprop by hand in this course. But it's worth doing it *once* to lose the mystery. See exercise below.

## SGD and mini-batches

After gradients are computed, you update weights:

```
w <- w - lr * grad
```

This is **gradient descent**. With 1B examples, computing the exact gradient over all of them is wasteful. Instead:

**Stochastic gradient descent (SGD)** estimates the gradient from a small random subset (a **mini-batch**). Less exact per step, but many more steps per unit time. Noisier gradients also help escape shallow bad spots.

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        logits = model(batch.inputs)
        loss = F.cross_entropy(logits, batch.targets)
        loss.backward()
        optimizer.step()       # update weights using gradients
        optimizer.zero_grad()  # clear gradients for next step
```

Five lines. That's the entire training loop, in essence.

## Exercise: backprop by hand on a 2-layer net

Do this once to demystify it. You can skip the math if you're stuck - just understanding the pattern is enough.

```python
# y = (x @ W1) * relu * W2
# loss = (y - target)**2

import torch
torch.manual_seed(0)

x = torch.tensor([[1.0, 2.0, 3.0]])       # shape (1, 3)
target = torch.tensor([[0.0]])
W1 = torch.randn(3, 4)
W2 = torch.randn(4, 1)

# forward
h = x @ W1                    # (1, 4)
h_act = torch.relu(h)         # (1, 4)
y = h_act @ W2                # (1, 1)
loss = ((y - target)**2).sum()

# backward, by hand
dL_dy = 2 * (y - target)                      # (1,1)
dL_dW2 = h_act.T @ dL_dy                      # (4,1)
dL_dh_act = dL_dy @ W2.T                      # (1,4)
dL_dh = dL_dh_act * (h > 0).float()           # relu's derivative
dL_dW1 = x.T @ dL_dh                          # (3,4)

print("Manual W1 grad:", dL_dW1)
print("Manual W2 grad:", dL_dW2)

# verify with autograd
W1.requires_grad_(True)
W2.requires_grad_(True)
h = x @ W1
y = torch.relu(h) @ W2
loss = ((y - target)**2).sum()
loss.backward()

print("Auto W1 grad:", W1.grad)
print("Auto W2 grad:", W2.grad)
```

They match. You just did backprop. Now never do it again.

## Visualize this

**Forward vs backward pass**:

```
                Forward:  data flows →
  input ──→ layer1 ──→ layer2 ──→ layer3 ──→ output ──→ loss
   x        W₁,b₁       W₂,b₂     W₃,b₃              ●

                Backward:  gradients flow ←
   ←──grad──── ←──grad──── ←──grad──── ←──grad──── ←──starts at 1

  Each layer receives dL/d(its output) from the next layer.
  It outputs:
    - dL/d(its input)    ──→ passed to the previous layer
    - dL/d(its weights)   ──→ stored in .grad for the optimizer
```

Backprop is the chain rule, applied right-to-left, automatically by PyTorch.

**The computation graph, pictorially**:

```
     x                 W              b
     │                 │              │
     └──── @ ──────────┤              │
                       │              │
                     (matmul)          │
                       │              │
                       └──── + ───────┘
                              │
                           (add bias)
                              │
                           relu()
                              │
                           (nonlinearity)
                              │
                            loss()
                              │
                              ●  (scalar)
                              │
                          .backward()
                              │
                              ▼
          PyTorch walks backward, filling in .grad on x, W, b
```

**Watch an autograd graph with 3 nodes**:

```python
import torch

x = torch.tensor([3.0], requires_grad=True)
y = x * 2       # PyTorch records: y = MulBackward(x, 2)
z = y ** 3      # PyTorch records: z = PowBackward(y, 3)
z.backward()
# chain rule: dz/dx = dz/dy * dy/dx = (3y²)(2) = 6y² = 6×36 = 216
print(x.grad)   # tensor([216.])
```

**Karpathy's Micrograd video**: https://www.youtube.com/watch?v=VMj-3S1tku0 - 2.5 hours, builds autograd from scratch in 200 lines. Essential watch once in your career. Shows this picture cleanly.

**Visualize a real autograd graph**:
```python
# after building any model and computing loss:
from torchviz import make_dot   # pip install torchviz
dot = make_dot(loss, params=dict(model.named_parameters()))
dot.render("computation_graph", format="png")
```

You get a real PNG showing the actual DAG PyTorch built. Worth doing once on a toy model - demystifies autograd permanently.

## Where to find this in nanoGPT

`train.py`:

```python
optimizer.zero_grad(set_to_none=True)
# ...
logits, loss = model(X, Y)
scaler.scale(loss).backward()   # backprop
scaler.step(optimizer)          # update weights
scaler.update()                 # update the gradient scaler (for mixed precision)
```

Those lines are executed thousands of times during training. Everything else in `train.py` is orchestration - data loading, logging, checkpointing, distributed coordination.

## Next

`05_optimizers_adam_adamw.md` - better weight-update rules than plain SGD.
