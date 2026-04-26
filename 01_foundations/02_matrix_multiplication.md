# 1.2 - Matrix Multiplication

Every LLM, every neural network, is dominated by one operation: matrix multiplication, or "matmul" for short. On a GPU, >95% of the time in training is spent on matmuls. Understanding this one operation is non-negotiable.

Good news: it's simple. Bad news: most textbooks explain it badly. Let's do it right.

## The one-line definition

To multiply matrix `A` (shape `(m, k)`) by matrix `B` (shape `(k, n)`), you get a new matrix `C` (shape `(m, n)`) where:

```
C[i, j] = dot product of row i of A with column j of B
```

That's it.

## Visualizing it

Let `A` be 2x3 and `B` be 3x2:

```
A =  [[1, 2, 3],         B =  [[7,  8],
      [4, 5, 6]]               [9,  10],
                               [11, 12]]
```

Then `C = A @ B` is 2x2:

```
C[0,0] = row 0 of A . col 0 of B = 1*7 + 2*9 + 3*11  = 58
C[0,1] = row 0 of A . col 1 of B = 1*8 + 2*10 + 3*12 = 64
C[1,0] = row 1 of A . col 0 of B = 4*7 + 5*9 + 6*11  = 139
C[1,1] = row 1 of A . col 1 of B = 4*8 + 5*10 + 6*12 = 154

C = [[58,  64],
     [139, 154]]
```

Run it:

```python
import torch
A = torch.tensor([[1.,2.,3.], [4.,5.,6.]])
B = torch.tensor([[7.,8.], [9.,10.], [11.,12.]])
print(A @ B)
```

## The shape rule to burn in

`(m, k) @ (k, n) -> (m, n)`

The inner dimensions (`k`) must match. They cancel. The outer dimensions stay.

If you can mentally track shapes through a pipeline like `(32, 1024) @ (1024, 768) @ (768, 50000) -> (32, 50000)`, you can follow any neural network code. Most shape errors you'll hit are "the inner dimensions don't match."

## Why is this the workhorse operation?

Two reasons:

**1) A linear transformation.** If `x` is a vector of shape `(k,)` and `W` is a matrix of shape `(k, n)`, then `x @ W` is a vector of shape `(n,)`. You've "transformed" `x` into a different vector space. A big chunk of what a neural network does is a sequence of these transformations.

One "layer" of a neural net is usually:
```python
y = x @ W + b   # linear transformation
y = activation(y)  # a non-linear function (e.g. ReLU) - covered in Module 2
```

**2) Batched math, one call.** You almost always process many examples at once. If you have a batch of 32 vectors each of length 768 (shape `(32, 768)`) and a transformation matrix of shape `(768, 256)`, then `X @ W` transforms the entire batch in one operation, shape `(32, 256)`. GPUs are specifically designed to make this one call extremely fast.

## Find matmul in nanoGPT

Open `~/workspace/nanoGPT/model.py`. Search for `nn.Linear`. You'll find lines like:

```python
self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
```

`nn.Linear(in_features, out_features)` is *literally* a wrapper around a matmul. Internally, it stores a weight matrix `W` of shape `(in_features, out_features)` and a bias `b` of shape `(out_features,)`. When you call `self.c_attn(x)`, it does `x @ W + b`.

So the "384 feature channels, 6-layer transformer" in the Shakespeare config is: lots of these matmuls, stacked.

## The compute cost

Matmul `(m, k) @ (k, n)` takes roughly `2 * m * k * n` floating-point operations (the 2 counts a multiply + add together as one "FLOP" by convention in some formulations, or separately in others - you'll see both). So a single forward pass through a transformer is mostly: add up all the FLOPs from all the matmuls.

For GPT-3 (175B parameters, 300B training tokens), total FLOPs ~ 3.14e23. That's the kind of number that gets cloud bills into the millions.

## Attention is just matmuls

Spoiler for Module 3: "attention" - the thing that makes transformers work - is fundamentally three matmuls in a row (with a softmax in the middle). You already have the machinery in your head to understand it. We just need to give the matmuls meaningful names.

## Broadcasting and batched matmul

PyTorch lets you matmul tensors with more than 2 dimensions. The rule: the last two dims are the matmul, everything else is "broadcasting" / "batch" dimensions.

```python
A = torch.randn(32, 10, 20)   # batch of 32 matrices of size (10, 20)
B = torch.randn(32, 20, 5)    # batch of 32 matrices of size (20, 5)
C = A @ B                     # shape (32, 10, 5)
```

Each of the 32 items in the batch is matmul'd independently. This is how attention gets computed per-head.

## Visualize this

**Matmul as "each row dotted with each column"**:

```
   A  (m × k)                B  (k × n)                  C  (m × n)
                            ┌─┬─┬─┐
                            │ │ │ │
                            │ │ │ │
   ┌─────┐                  │ │ │ │                    ┌─┬─┬─┐
   │ ━━━━┣━━━━━━━━━━━━━━━━▶ │ │ │ │     produces       │●│ │ │  <- C[0,0]
   │     │                  │ │ │ │                    │ │ │ │     = row 0 of A
   │     │                  │ │ │ │                    │ │ │ │       · col 0 of B
   └─────┘                  └─┴─┴─┘                    └─┴─┴─┘
                             ↑ ↑ ↑
                             │ │ │
                             column 0 of B
```

Every cell in C is one dot product. An (m,k)×(k,n) matmul is `m × n` dot products.

**Interactive**: http://matrixmultiplication.xyz - paste two matrices, watch them multiply step by step.

**Why GPUs love matmul**:

```
  CPU (8 cores):                     GPU (10,000+ cores):
  ┌───┬───┬───┬───┐                  ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
  │ 1 │ 2 │ 3 │ 4 │                  │1│2│3│4│5│6│7│8│9│0│1│ ...
  ├───┼───┼───┼───┤                  ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
  │ 5 │ 6 │ 7 │ 8 │                  │2│3│4│5│6│7│8│9│0│1│2│ ...
  └───┴───┴───┴───┘                  └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘

  8 multiplications at once           10,000+ at once
```

GPU ≈ 1000× faster than CPU for matmul. That's why every LLM lives on GPUs.

**Watch a real matmul**: in Module 2 you'll run `nn.Linear(10, 5)` on real tensors. Under the hood: one matmul. Literally the entire GPT-2 is ~48 of these matmuls per forward pass.

## Exercises

1. Compute by hand: if `A.shape == (4, 6)` and `B.shape == (6, 3)`, what's the shape of `A @ B`? How many multiplications total?
2. In PyTorch, create `A = torch.randn(2, 3)` and `B = torch.randn(3, 4)`. Verify `A @ B` has shape `(2, 4)`.
3. Try to multiply `A = torch.randn(2, 3)` and `B = torch.randn(4, 2)`. See the error message. Understand why.
4. Find one more matmul in `nanoGPT/model.py`, different from `c_attn` and `c_proj`. Write down what it does (just guess at this point, we'll verify in Module 3).
5. Look at `nn.Embedding`. Is it really a matmul under the hood? (Yes - with a one-hot vector. Think about why.)

## Next

`03_calculus_for_learning.md` - we need one calculus idea (gradients), painlessly.
