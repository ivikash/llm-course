# 2.3 - Loss Functions

"Loss" is a single scalar that measures how wrong your model currently is.
Training minimizes loss. Every model has a loss. Choosing the right loss is
often 50% of the engineering problem.

## The two losses you must know

### Cross-entropy (for classification and language modeling)

Covered in Lesson 1.4. The LLM loss. Keep this one in your bones.

```python
import torch.nn.functional as F
loss = F.cross_entropy(logits, targets)
# logits: (N, C) - model outputs for N examples, C classes
# targets: (N,) - integer labels
# returns a single scalar, averaged over N.
```

Internally: `softmax(logits) -> pick the true class's probability -> take negative log -> average`. PyTorch does this in a numerically-stable way; don't compute softmax yourself and then log.

### Mean Squared Error (MSE, for regression)

When your target is a continuous number (like "predict tomorrow's temperature"):

```python
loss = F.mse_loss(predictions, targets)
# same shape on both
# returns (predictions - targets)**2 averaged
```

Used in: regression problems, auto-encoders, some RL value functions. **Rarely used for LLMs** - they're always classification over the vocabulary.

## A few other losses you'll hear about

- **Binary Cross-Entropy (BCE)** - cross-entropy for exactly 2 classes. `F.binary_cross_entropy_with_logits`.
- **Hinge loss** - margin-based, used in classic SVMs. Mostly historical.
- **Contrastive losses** (InfoNCE, NT-Xent) - for representation learning, used in CLIP and embedding models.
- **KL divergence** - measures difference between two probability distributions. Used in RLHF (keep fine-tuned model close to base) and distillation.
- **RL rewards** - in PPO/GRPO, the "loss" is a combination of policy gradient + value loss + KL penalty. Module 5 covers this.

You should know the *names* now. You'll understand them in context later.

## Why cross-entropy specifically for LLMs

- LLM predicts a distribution over vocabulary. That's a classification problem with a huge number of classes (e.g. 50,000+).
- Cross-entropy is the "right" loss for probabilistic classification (it corresponds to maximum-likelihood training).
- It has well-behaved gradients: the gradient of cross-entropy w.r.t. logits is simply `softmax(logits) - one_hot(target)`. Beautiful.

## Numerically-stable tip

Never do this:
```python
probs = F.softmax(logits, dim=-1)
loss = -torch.log(probs[target]).mean()
```

The `log` can blow up if `probs[target]` is tiny. Use `F.cross_entropy` directly; it combines the steps in a stable way (the `logsumexp` trick).

## In nanoGPT

`model.py`:

```python
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
```

The `.view(-1, ...)` flattens batch and sequence dims together. `ignore_index=-1` lets us skip "padding" positions marked with -1.

In nanochat, SFT training adds masking so that the loss only applies to the assistant's tokens, not the user's. You'll see this in `scripts/chat_sft.py`. Same cross-entropy, just with a mask.

## Visualize this

**MSE vs cross-entropy intuition**:

```
  MSE (for continuous targets)       Cross-entropy (for class labels)

  L = (pred - target)²                L = -log(p_correct_class)

       │                                   │
       │ pred far                          │ tiny prob on truth =
       │ from target                       │ huge loss
       │                                   │
       │      ●                            │●
       │    ╱                              │ ╲
       │  ╱                                │  ╲
       │╱  target                          │    ╲──────
       ●─── at here                        │         ●
    ───┴─────── pred                       │      probability of
         loss = squared distance           │      correct class
                                          ───┴──── 0 ──────── 1
```

MSE: symmetric, smooth. Cross-entropy: asymmetric, explodes when the model is confidently wrong.

**See softmax + cross-entropy together** (this is what every LLM uses):

```
  Correct class: "mat" (token ID 47)

  Model outputs logits for all 50257 possible next tokens:
  [12.1, 3.2, -1.4, ..., 8.7, ..., 4.2]     (indices 0, 1, 2, ..., 47, ..., 50256)
                          ↑
                       "mat" logit = 8.7

  Softmax makes them probabilities summing to 1.
  Probability of "mat" = 0.12 (let's say).
  Cross-entropy loss = -log(0.12) = 2.12

  We want this loss low.
  → Model should assign high probability to the true next token.
  → Training nudges weights to make "mat" score higher when context suggests it.
```

**Why cross-entropy has nice gradients** (for the curious): the gradient of cross-entropy w.r.t. logits is simply `softmax(logits) - one_hot(target)`. Clean, no tricky derivatives. That's why every classification problem uses it.

## Exercise

1. Build a tiny MSE training example:
```python
import torch
w = torch.tensor([0.0], requires_grad=True)
x = torch.tensor([2.0])
target = torch.tensor([10.0])
for _ in range(100):
    pred = w * x
    loss = F.mse_loss(pred, target)
    loss.backward()
    with torch.no_grad():
        w -= 0.01 * w.grad
        w.grad = None
print(w)  # should be ~5
```

2. Derive cross-entropy for a single example: 3-class logits `[1.0, 2.0, 3.0]`, target = class 2. Compute by hand:
   - `softmax = exp(logits) / sum(exp)` = `[e, e^2, e^3]/(e + e^2 + e^3)` ≈ `[0.09, 0.245, 0.665]`.
   - `loss = -log(0.665)` ≈ 0.408.
   Verify with `F.cross_entropy(torch.tensor([[1.,2.,3.]]), torch.tensor([2]))`.

## Next

`04_backprop_and_sgd.md` - the algorithm that makes gradient computation efficient.
