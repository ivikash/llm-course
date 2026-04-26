# 4.8 - Evaluation: Val Loss, Perplexity, BPB

Training cross-entropy goes down. How do you know the model is actually learning (not memorizing)? How do you compare different runs?

## Train loss vs val loss

**Train loss**: averaged cross-entropy on batches the model was trained on.
**Val loss**: averaged cross-entropy on held-out data the model has never seen.

If they're close, the model is generalizing. If train is much lower than val, the model is overfitting (memorizing training examples).

For LLMs pretrained on huge data, overfitting is unusual - you rarely do even one full epoch. But val loss is still your primary "am I improving?" signal.

## In nanoGPT

`train.py`:

```python
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# called periodically:
if iter_num % eval_interval == 0 and master_process:
    losses = estimate_loss()
    print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
```

Two things to note:
- `model.eval()` disables dropout for evaluation; `model.train()` re-enables afterward.
- `@torch.no_grad()`: no gradient tracking during eval. Saves memory.
- It averages over `eval_iters` batches (default 200) for a stable estimate.

## Perplexity

**Perplexity = exp(cross_entropy)**. Just a reparameterization.

Intuition: perplexity P means "the model is as uncertain as if it had to choose uniformly between P possibilities at each step."

- Random guessing on vocab of 50K: perplexity ≈ 50000.
- GPT-2 on WikiText-103: perplexity ~29.
- GPT-3 on same: perplexity ~20.
- GPT-4: probably ~15 (not publicly reported).

When papers report perplexity instead of loss, they're showing the same info in a more intuitive unit.

## Bits per byte (BPB) - nanochat's metric

Cross-entropy is reported in **nats** (natural log). Divide by `ln(2)` to get **bits**.

But cross-entropy depends on the tokenizer. A larger vocabulary encodes more information per token, so cross-entropy goes up for the same "amount of text modeled." This makes cross-entropy incomparable across tokenizers.

**Bits per byte** fixes this: normalize by the number of raw bytes encoded.

```
BPB = (cross_entropy_nats / ln(2)) * (total_tokens / total_bytes)
```

A BPB of 1.0 means "one bit of uncertainty per byte of text." A perfect English model has BPB ~0.7 (Shannon's estimate). Current frontier models are below 1.0 on high-quality English corpora.

nanochat's `scripts/base_eval.py`:

```python
bpb = cross_entropy_nats / math.log(2) / bytes_per_token
```

Used throughout nanochat, so different tokenizer experiments can be compared fairly.

## Benchmark scores: beyond val loss

Val loss measures "can the model predict the next token". But we often care about specific capabilities:

- **MMLU** (Massive Multitask Language Understanding): 57 subjects, multiple-choice. Tests general knowledge.
- **ARC**: grade-school science reasoning.
- **HumanEval**: Python coding problems.
- **GSM8K**: grade-school math word problems.
- **BBH** (Big Bench Hard): diverse reasoning tasks.
- **CORE** (DCLM paper): an aggregate of many tasks. What nanochat primarily tracks.

These measure whether the model can actually do things, not just predict tokens. Lower val loss correlates with higher benchmark scores, but the relationship is noisy. We cover benchmarks fully in Module 5 Lesson 7.

## When to stop training

Two signals:
1. **Val loss plateaus**: no more learning happening.
2. **Compute budget exhausted**: you planned to train for X hours, done.

Pretraining is usually bounded by (2), not (1). Fine-tuning is usually bounded by (1) - you stop when val loss stops improving for K eval intervals ("early stopping").

## Estimating loss from scaling laws

Before training, you can *predict* what loss you should expect:

Chinchilla-approximated:
```
loss ≈ 1.69 + (406.4 / N^0.34) + (410.7 / D^0.28)
```

Where N = params, D = training tokens.

If your actual loss is much higher than predicted, you have a bug. If much lower, you're either overfitting or the prediction formula doesn't apply (different dataset, architecture).

nanochat uses scaling-law fits to predict `val_bpb` before a run - see `dev/scaling_analysis.ipynb`.

## Other metrics worth logging

During training, log to wandb:
- `train/loss` every step
- `val/loss` every eval
- `lr` every step
- `grad_norm` every step
- `tokens_per_second` (throughput)
- `mfu` (Model FLOPs Utilization)
- `gpu_memory_used`

Any deviation from usual patterns signals something to investigate.

## Visualize this

**Train loss vs val loss patterns**:

```
  Healthy (val slightly above train, both dropping):
    loss │●
         │ ●●  train
         │   ●●
         │     ●●●
         │        ●●●●●●●●
         │           ●●●●●●●●●  (train continues)
         │
         │   ●   val
         │     ●
         │       ●●
         │          ●●●
         │              ●●●●  (val plateaus, ~0.2 above train, normal)
         └────────────────────── step

  Underfitting (both high, not dropping):
    loss │●●●●●●●●●●●●●●●●●
         │●●●●●●●●●●●●●●●●●
         └────────────────────── step
    Fix: bigger model, more training, fix bug.

  Overfitting (train drops, val rises):
    loss │●
         │ ●●  train
         │   ●●●●●●●●●●●●●●●
         │                  (train keeps dropping)
         │   ●●
         │     ●●●
         │        ●●      ←── val bottoms out
         │          ●●
         │            ●●  ←── val starts rising: overfit!
         │              ●●
         └────────────────────── step
    Fix: regularization, early stop, more data.

  Divergence (spikes):
    loss │
         │●
         │ ●
         │  ●  ●●●●●●● NaN
         │    ●●      │
         │   ● ●      ▼
         └────────────────────── step
    Fix: lower LR, check data, grad clip.
```

**Perplexity scale intuition**:

```
  Model                  Perplexity (WikiText-103)
  random (uniform 50K)   50,000
  n-gram (bigram)        ~300
  LSTM                   ~50
  GPT-2 small            ~29
  GPT-2 large            ~18
  GPT-3 175B             ~20
  GPT-4                  ~13 (estimated)
  human lower bound      ~10 (estimated)

  Halving perplexity ≈ doubling quality (subjectively).
  Going from 30 → 15 is a big deal, 15 → 14 less so.
```

**Bits per byte (BPB) - why nanochat uses it**:

```
  Two models, different tokenizers:
    Model A: tokenizer vocab 50K, val cross-entropy = 3.1 nats
    Model B: tokenizer vocab 32K, val cross-entropy = 3.4 nats

  Which is "better"? Can't tell from cross-entropy alone - different vocabs!

  Fix: normalize by bytes of text, not tokens:
    BPB = (cross_entropy / ln(2)) × (tokens / bytes)
        = bits of uncertainty per byte of original text

  Now they're directly comparable:
    Model A: BPB = 0.90
    Model B: BPB = 0.87  ← actually better

  nanochat's speedrun reports val_bpb. Lower = better.
```

**GPT-2 level = BPB ~0.95 on validation**. Modern frontier models ~0.7.

**Visualize your evaluation**:

```python
# after training, plot everything:
import wandb
wandb.init(project="my-run")
wandb.log({
    "train/loss": train_loss,
    "val/loss": val_loss,
    "val/perplexity": math.exp(val_loss),
    "val/bpb": bpb,
    "step": step,
})

# wandb automatically produces:
#   - Line plots
#   - Multiple-run comparisons
#   - Hyperparameter sweeps
```

## Exercises

1. Read `nanoGPT/train.py`'s `estimate_loss`. Understand: why `eval_iters=200`, why `@torch.no_grad()`, why `model.eval()`.

2. Train Shakespeare. Plot train loss and val loss vs step. Is val consistently a bit higher than train?

3. Compute perplexity from your loss: `math.exp(1.47)`. What is it?

4. Read `nanochat/scripts/base_eval.py`. Find the BPB computation. Find the CORE metric computation (uses `nanochat/core_eval.py`). You don't need to understand every line; just know it's there.

## Next

`capstone_custom_dataset.md` - Module 4 capstone: train on your own text.
