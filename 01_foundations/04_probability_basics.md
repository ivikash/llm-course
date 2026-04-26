# 1.4 - Probability Basics (Softmax and Cross-Entropy)

Two concepts. If you understand these, you understand how LLMs are trained and evaluated.

## Probability distributions

A **probability distribution** over a set of choices is just a list of non-negative numbers that sum to 1.

If there are 4 choices (A, B, C, D) with probabilities `[0.5, 0.2, 0.2, 0.1]`, that means:
- A is the most likely (50%)
- B and C are equally likely (20% each)
- D is the least likely (10%)
- The 4 probabilities sum to 1.0. (They must, to be a valid distribution.)

An LLM outputs a probability distribution over the vocabulary. If vocab size is 50257, the model outputs 50257 numbers that sum to 1, giving the probability of each possible next token.

## The problem: raw model outputs aren't probabilities

A neural network's final layer produces just... numbers. Arbitrary. Could be `[-3.2, 7.1, 0.5, -0.4]`. They're not probabilities - they could be negative, they don't sum to 1.

These raw numbers are called **logits**.

We need a way to turn logits into a valid probability distribution. The standard answer is **softmax**.

## Softmax

Softmax takes any list of numbers and produces a valid probability distribution.

Formula:

```
softmax(x_i) = exp(x_i) / sum_j exp(x_j)
```

In English: exponentiate each number (`exp` = `e^x`, the exponential function), then divide each by the total sum. Done.

Why it works:
- `exp(x)` is always positive. So all outputs are positive.
- Dividing by the sum forces them to total 1.
- Bigger `x` -> *much* bigger `exp(x)` (exponential growth). So the largest logit wins big, smaller ones fade fast. The "soft" in softmax means it's a smooth version of "pick the max".

Example:

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([1.0, 2.0, 3.0])
probs = F.softmax(logits, dim=-1)
print(probs)      # tensor([0.0900, 0.2447, 0.6652])
print(probs.sum()) # 1.0
```

Notice: logit 3 got 67% probability, logit 2 got 24%, logit 1 got 9%. A linear difference in logits becomes a big exponential difference in probabilities.

### Temperature

You can **divide the logits by a "temperature" `T`** before softmax to control sharpness:

- `T = 1` : normal softmax.
- `T < 1` (e.g. 0.5) : logits get magnified, probabilities get peakier - model becomes more confident/deterministic.
- `T > 1` (e.g. 2.0) : logits get squashed, probabilities become more uniform - more random output.
- `T -> 0` : picks the max almost always (argmax).
- `T -> infinity` : uniform random.

In LLM chat UIs, "temperature" is literally this. OpenAI's default is often 0.7.

In `nanoGPT/sample.py`, you'll find:

```python
logits = logits[:, -1, :] / temperature
probs = F.softmax(logits, dim=-1)
```

## Cross-entropy: measuring how wrong the distribution is

Suppose the correct next word is "mat" (call it token index 47). The model outputs a probability distribution over 50257 tokens. How do we score it?

**Idea:** look at the probability the model assigned to the true answer (token 47). If that probability is close to 1, great. If close to 0, bad.

To turn "probability" into a "loss" (where 0 is good, high is bad), we take the **negative log** of it.

```
loss = -log(probability_assigned_to_correct_token)
```

- If the model was 100% confident in the right answer: `-log(1.0) = 0`. Zero loss. Perfect.
- If the model gave 50% to the right answer: `-log(0.5) ≈ 0.69`.
- If the model gave 1% to the right answer: `-log(0.01) ≈ 4.6`. Bad.
- If the model gave ~0% to the right answer: `-log(0.0001) ≈ 9.2`. Really bad.

This is **cross-entropy loss**.

In practice, PyTorch provides `F.cross_entropy(logits, targets)` which does softmax + negative log in one numerically-stable call. It's the standard loss for classification and language modeling.

## Where this is in nanoGPT

Open `~/workspace/nanoGPT/model.py`, find the `forward` method of the `GPT` class (around line 180). You'll see:

```python
logits = self.lm_head(x)
...
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
```

That's the entire language-modeling objective:

1. Compute logits (shape: `(batch, seq_len, vocab_size)`).
2. Compare each position's logits to the true next token at that position.
3. Cross-entropy averages across all positions.

When training output says `loss 1.47`, that's the average cross-entropy across the batch. A totally uninformed model (uniform distribution) would give `-log(1/50257) ≈ 10.8` as baseline. GPT-2 gets down to ~3.1 on OpenWebText. Every step down is hard-won.

## Perplexity (bonus concept)

**Perplexity** = `exp(cross_entropy_loss)`. Just a reformulation. If cross-entropy loss = 1.47, perplexity = `exp(1.47)` ≈ 4.35. Intuition: "the model is effectively choosing uniformly between 4.35 possibilities at each step."

Sometimes people report perplexity, sometimes cross-entropy. They're the same info.

nanochat uses a related metric called **bpb (bits per byte)** which is cross-entropy normalized differently so it's comparable across different tokenizer vocab sizes.

## Visualize this

**Softmax turns anything into a valid probability distribution**:

```python
# run this and look
import torch, torch.nn.functional as F
import matplotlib.pyplot as plt

logits = torch.tensor([1.0, 2.0, 3.0])
for T in [0.1, 0.5, 1.0, 2.0, 5.0]:
    probs = F.softmax(logits / T, dim=-1)
    plt.bar([f"T={T}\n{p:.2f}" for p in probs.tolist()], probs.tolist())
    plt.title(f"Temperature {T}")
    plt.savefig(f"softmax_T{T}.png")
    plt.clf()
```

What you'll see:
- `T=0.1`: almost all probability on the highest logit (deterministic).
- `T=1`: smooth distribution - one answer favored but others still likely.
- `T=5`: nearly uniform - outputs become random.

This is literally the "temperature" slider in ChatGPT and every LLM. Now you know what it does.

**Cross-entropy as "surprise"**:

```
  Probability model assigned to correct answer:
  1.0 ────────────────── 0 loss (not surprised at all, perfect)
  0.5 ────────────────── 0.69 loss
  0.1 ────────────────── 2.30 loss
  0.01 ───────────────── 4.60 loss
  0.001 ──────────────── 6.91 loss (very surprised, very wrong)
```

Cross-entropy = `-log(probability)`. Low probability on truth = high surprise = high loss.

Training minimizes surprise. Every training step, the model becomes *slightly less surprised by reality*.

**Interactive**: https://www.desmos.com/calculator/dzvxqldizw - play with softmax temperature live.

**3Blue1Brown's chapter on softmax**: https://www.youtube.com/watch?v=eMlx5fFNoYc#t=22m - watch that specific timestamp (22 min) for a beautiful 5-minute explanation.

## Exercises

1. By hand (or Python), compute the softmax of `[0, 0, 0]`. (Should be `[1/3, 1/3, 1/3]`, uniform.)
2. Compute softmax of `[10, 10, 10]`. Same answer as above, right? Softmax is shift-invariant.
3. Compute softmax of `[10, 0, 0]`. Very peaky. The "10" gets almost all the probability.
4. For `logits = [1.0, 2.0, 3.0]` and the correct class being index 2, compute cross-entropy by hand: softmax then `-log(probs[2])`.
5. Do the same with index 0 as correct. Notice how much larger the loss is.

## What to remember

- Logits: raw model outputs. Arbitrary real numbers.
- Softmax: turns logits into a valid probability distribution.
- Temperature: scales logits to control sharpness.
- Cross-entropy: `-log(prob_of_correct_answer)`. The loss we minimize for language modeling.
- Lower cross-entropy = better model at predicting the right next token.

## Next

`05_numpy_and_tensors.md` - a hands-on tour of PyTorch tensors, then the Module 1 capstone.
