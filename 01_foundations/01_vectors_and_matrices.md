# 1.1 - Vectors and Matrices (The Intuitive Version)

You need very little math to understand deep learning. You need *specific* math, well. This lesson gives you the vocabulary of shape, which is 80% of what matters.

## Scalars, vectors, matrices, tensors

**Scalar** = a single number. `5`, `-3.14`, `0.02`.

**Vector** = an ordered list of numbers. `[1, 2, 3]` or `[0.5, -0.2, 1.7, 0.0]`. In neural nets, a vector usually represents *one thing* - one word, one image, one example.

**Matrix** = a grid of numbers, rows and columns. Like:

```
[[1, 2, 3],
 [4, 5, 6]]
```

This is a 2x3 matrix (2 rows, 3 columns). In neural nets, a matrix often represents *a batch of vectors* (each row is one example) or *a linear transformation* (turning one vector into another, see 1.2).

**Tensor** = the generalization. A scalar is a 0-D tensor, a vector is 1-D, a matrix is 2-D, and you can keep going. Inside a transformer you'll see 3-D and 4-D tensors constantly. Example: a batch of 32 sentences, each with 1024 tokens, each token represented as a 768-number vector, is a (32, 1024, 768) 3-D tensor.

In PyTorch, everything is a tensor. `torch.tensor(5)` is a scalar, `torch.tensor([1,2,3])` is a vector, and so on.

## Shape is the most important concept

When people argue about deep learning code, 90% of the time they're arguing about **shapes**. Get comfortable with shape-talk.

The **shape** of a tensor is the tuple of its sizes along each dimension.

- `[1, 2, 3]` has shape `(3,)` - a 1-D tensor of length 3.
- `[[1,2,3],[4,5,6]]` has shape `(2, 3)` - 2 rows, 3 cols.
- A cube of numbers 4x5x6 has shape `(4, 5, 6)`.

Run this in Python:

```python
import torch
x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(x.shape)   # torch.Size([2, 3])
print(x.ndim)    # 2  (number of dimensions)
print(x.numel()) # 6  (total number of elements)
```

## Basic operations

**Element-wise operations**: add/multiply tensors of the same shape element by element.

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([10, 20, 30])
print(a + b)  # tensor([11, 22, 33])
print(a * b)  # tensor([10, 40, 90])   # element-wise, NOT a matrix multiply!
```

**Broadcasting**: if shapes *almost* match, PyTorch stretches the smaller one.

```python
M = torch.tensor([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)
v = torch.tensor([10, 20, 30])            # shape (3,)
print(M + v)  # adds v to each row
# [[11, 22, 33],
#  [14, 25, 36]]
```

This is used constantly - e.g. adding a "bias" vector to each row of a batch of activations.

**Dot product** of two vectors: multiply element-wise, sum. A single number out.

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
print(a @ b)     # 1*4 + 2*5 + 3*6 = 32
print((a*b).sum())  # same thing
```

The dot product measures how aligned two vectors are. Big positive = pointing similar direction. Zero = perpendicular. Negative = opposite. This is the basis of the **attention** mechanism - attention literally computes dot products between "query" vectors and "key" vectors. We'll get there.

## What vectors actually represent

Here's the payoff concept: in neural nets, we represent *anything* - a word, an image patch, a user profile - as a vector of numbers. Then similar things end up having similar vectors (after training).

A famous classic example (from `word2vec`, 2013):

```
vector("king") - vector("man") + vector("woman")  ≈ vector("queen")
```

That kind of relationship *emerges* from training. You don't program it. This is why people get so excited about neural networks: the vector space "learns meaning" in a geometric way. In a transformer, every token ends up as a vector. Attention lets those vectors refine each other. The final vector of the last token is basically "the compressed meaning of everything the model read."

## A taste of what's coming

In `nanoGPT/model.py`, when we say "the embedding dimension `n_embd` is 768", we mean every token is a 768-length vector inside the model. When we say "4 attention heads, head_dim 64", we mean each head works with 64-length vector pieces (4 x 64 = 256... wait, that's not 768 - let me redo: typical setup is n_embd=768, n_head=12, head_dim=64, since 12 x 64 = 768). Shapes, shapes, shapes.

## Exercises

1. Create a (4, 5) tensor filled with zeros, then fill the 3rd row with 1s. Print its shape.
2. Make two random vectors of length 10 and compute their dot product.
3. Using broadcasting, subtract the per-column mean from a (3, 4) matrix. (Hint: `M.mean(dim=0)` gives a (4,) vector.)
4. Predict the shape of `a @ b` where `a.shape == (3, 4)` and `b.shape == (4, 2)`. Verify. (This is matrix multiplication - lesson 1.2.)

## Next

`02_matrix_multiplication.md` - the single operation that runs the entire AI industry.
