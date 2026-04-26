# 2.6 - Regularization: Dropout and Weight Decay

Neural networks can memorize training data instead of learning generalizable patterns. This is **overfitting**. Regularization = techniques that discourage memorization.

For transformers at scale, overfitting is *less* of a problem (you have so much data, one epoch is as much as you can afford). But regularization still helps, and the techniques apply everywhere.

## Dropout

During training, randomly set a fraction of activations to 0. Forces the model to be robust - it can't rely on any single neuron. At inference time, dropout is turned off.

```python
nn.Dropout(p=0.1)      # zero out 10% of activations randomly each forward pass
```

In nanoGPT, look at the `CausalSelfAttention` class - there are two dropouts:
```python
self.attn_dropout = nn.Dropout(config.dropout)
self.resid_dropout = nn.Dropout(config.dropout)
```

The Shakespeare config uses `dropout = 0.2` (small model can overfit). The full GPT-2 config uses `dropout = 0.0` (large model on large data doesn't need it).

nanochat mostly uses `dropout=0.0` because it operates at a compute-optimal scale where the model is undertrained, not overtrained.

## Weight Decay

A penalty that shrinks weights toward zero each step. In AdamW:

```
w <- w - lr * weight_decay * w
```

Typical values: 0.01 - 0.1. Keeps weights small, which tends to generalize better.

Covered in the previous lesson - review which parameters get it and which don't (2D weights yes, biases and LayerNorms no).

## Other regularization techniques (for awareness)

- **Early stopping**: save the model with the best validation loss, not the last one.
- **Data augmentation**: show slightly-modified training examples. Huge in vision. Less common for LLMs (paraphrasing is an emerging analogue).
- **Label smoothing**: instead of 1-hot targets, use 0.9 for the true class and 0.1/N for others. Sometimes used.
- **Mixup / CutMix**: mostly vision techniques.

## Under vs overfitting: the training-vs-validation gap

You split data into training (the model sees it) and validation (the model doesn't). You watch both losses:
- Both high: **underfit**. Model is too small, or not trained enough, or learning rate wrong.
- Train low, val high: **overfit**. Model memorized training, doesn't generalize.
- Both low, val slightly higher than train: **healthy**.
- Val lower than train: suspicious - data leak or bug.

In nanoGPT/nanochat, you see this in logs:
```
iter 1000: train loss 3.2, val loss 3.3      # healthy
iter 5000: train loss 1.1, val loss 2.8      # overfitting
```

## Visualize this

**Overfitting, in one picture**:

```
  loss
    │
    │──  train loss
    │────
    │    ─────
    │         ─────
    │              ────────── (training fits data more and more)
    │
    │  val loss
    │─────              ●←── "best" checkpoint (lowest val loss)
    │      ────    ●●●
    │          ●●●       ●●
    │                      ●●●●●  (val loss rises: overfitting)
    │
    └──────────────────────────── step
         early               late
```

Watch train-val gap during training. If val starts climbing while train keeps dropping, you're overfitting. Stop training, or add regularization.

**Dropout, visually**:

```
  Without dropout:         With dropout (p=0.5):
                           (random neurons zeroed each pass)

  input                    input
    │                        │
    ● ● ● ● ●                ● ●   ● ●    (2 dropped)
    │╲│╳│╳│╱│                │       │
    ● ● ● ● ●                ●   ● ●     (1 dropped)
    │                        │
  output                   output
```

Forces the network to be robust - can't rely on any single neuron. Typically p=0.1 to 0.5.

**Weight decay, visually**:

```
  Without weight decay,       With weight decay,
  weights grow freely:        weights stay small:

  weight
    │                          │
    │      ●                   │
    │   ●                      │●────●─────●────●
    │●                         │
    └────────── time           └────────── time
```

Smaller weights = smoother function = often better generalization.

**In practice for LLMs**: dropout is usually 0.0 during pretraining (plenty of data, hard to overfit). Set to 0.1-0.2 during fine-tuning on small datasets (Shakespeare toy, your personal corpus).

## Exercise

1. Open `nanoGPT/config/train_shakespeare_char.py`. Find `dropout = 0.2`. Imagine what happens if you set it to 0.0 - would the baby GPT still learn? Probably, but might overfit (memorize Shakespeare).
2. Train the Shakespeare baby GPT twice (once we get to Module 3), once with `dropout=0.2` and once with `dropout=0.0`. Compare val loss.

## Next

`07_training_loop_anatomy.md` - putting forward/loss/backward/step into a real loop.
