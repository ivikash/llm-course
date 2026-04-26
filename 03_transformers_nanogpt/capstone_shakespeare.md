# Module 3 Capstone - Train Your Own Shakespeare GPT

You now understand everything in `nanoGPT/model.py` and `train.py`. Time to run the real thing.

## Goal

Train nanoGPT on Shakespeare's complete works at character level. After ~3 minutes on a GPU (~20-60 minutes on a CPU), you'll have a small GPT that produces Shakespeare-like text.

## Step 1: Prepare data

```bash
cd ~/workspace/nanoGPT
python data/shakespeare_char/prepare.py
```

This creates `data/shakespeare_char/train.bin` and `val.bin` - the tokenized text. You already have these files (I see them in the directory).

## Step 2: Train

If you have a GPU:

```bash
python train.py config/train_shakespeare_char.py
```

If CPU only:

```bash
python train.py config/train_shakespeare_char.py \
    --device=cpu --compile=False --eval_iters=20 --log_interval=1 \
    --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 \
    --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

Watch the terminal. You'll see lines like:

```
iter 0: loss 4.2814, time 543.21ms
iter 100: loss 2.5689, time 140.22ms
iter 200: loss 2.1045, time 140.11ms
...
iter 5000: loss 1.4697, time 140.01ms
```

Notice: loss starts around 4.2 (which is roughly `-log(1/65) = 4.17`, i.e. random guess over the 65 chars). Drops to ~1.47 - dramatic learning.

## Step 3: Sample

```bash
python sample.py --out_dir=out-shakespeare-char
```

You'll see generated Shakespeare-like dialogue. Likely nonsense semantically, but syntactically Shakespeare-flavored - character names, colons, blank-verse layout, even some real English fragments.

## What just happened

You trained:
- A 6-layer transformer (or 4-layer in the CPU config)
- 384 embedding dim (or 128)
- 6 heads (or 4)
- Context 256 (or 64)
- Character-level, vocab ~65
- AdamW optimizer
- ~10.7M parameters (the 384-dim version)
- ~5000 iterations on Shakespeare

The model has observed patterns in 1MB of Shakespeare text. It has learned:
- Which characters usually follow others (like our bigram).
- Longer-range patterns like line structure, punctuation, name capitalization.
- Vague word-like chunks ("the", "and", etc.).

But with only ~11M params trained for 3 minutes, it doesn't really "understand" anything. Add more data, more compute, more params, and you get GPT-2. Billions more, and you get GPT-4. The *architecture* is what you just trained.

## Inspecting what you trained

The checkpoint is at `out-shakespeare-char/ckpt.pt`. You can load it:

```python
import torch
ckpt = torch.load('out-shakespeare-char/ckpt.pt', map_location='cpu')
print(ckpt.keys())  # ['model', 'optimizer', 'model_args', 'iter_num', 'best_val_loss', 'config']
print(ckpt['model_args'])
print("val loss:", ckpt['best_val_loss'])
```

The `model` key holds all the weight tensors. You could reload them and play.

## Visualize this

**What you're about to actually see**:

```
  Terminal after running `python train.py config/train_shakespeare_char.py`:

  Overriding config with config/train_shakespeare_char.py:
  tokens per iteration will be: 16,384
  found vocab_size = 65
  number of parameters: 10.65M
  num decayed parameter tensors: 26, with 10,738,688 parameters
  num non-decayed parameter tensors: 13, with 4,992 parameters
  using fused AdamW: True

  step 0: train loss 4.2858, val loss 4.2888
  saving checkpoint to out-shakespeare-char
  iter 0: loss 4.2870, time 543.21ms
  iter 1: loss 4.2651, time 141.23ms
  iter 10: loss 3.2156, time 141.02ms      ← watch the number fall
  iter 100: loss 2.5689, time 140.89ms
  iter 250: evaluating...
  step 250: train loss 1.9452, val loss 2.0621
  saving checkpoint to out-shakespeare-char
  ...
  iter 5000: loss 1.4697, time 140.55ms     ← final, near-optimal for this model
```

This exact run trains a functioning Shakespeare-style language model in ~3 minutes on a GPU.

**What the generated samples look like over training**:

```
After iter 100:
  QqZvvxMmRnF h,jWwaLoq e!h?kpkmqjlK ue,uNDvmJbQl!z    (nonsense)

After iter 500:
  WAS:
  Thne now, my lord, I and wind to me.
  ....                                                   (word-like, but broken)

After iter 2500:
  ROMEO:
  And I will not be a trater of the world,
  When thou art in the bravie tham the heart:
  What say you, sir?                                    (Shakespeare-ish)

After iter 5000:
  ROMEO:
  Now in my name, I have deserved her so,
  Whom I have taught her from the throne of death,
  And sit thee in her time.                              (solid fake-Shakespeare)
```

You'll see the progression. It's the most satisfying visualization in the course.

**The loss curve you should produce**:

```python
# after training, plot from the checkpoint's stored losses
# or log to wandb during training:
import wandb
wandb.init(project="nanoGPT-shakespeare")
wandb.log({"train/loss": loss.item(), "step": iter_num})

# then on wandb.ai you get:
#
#   loss
#     │ 4.3
#     │ 4.0  ●
#     │ 3.5  ●
#     │ 3.0    ●
#     │ 2.5     ●●
#     │ 2.0       ●●●
#     │ 1.5          ●●●●●●●●●●●●●●●●
#     │
#     └──────────────────────────────── steps
#       0  500  1000  2500  5000
```

**Make your own report**:

After training, write up what you saw. Include:
- Final loss.
- A few generated samples.
- Time per iteration.
- Peak GPU memory (if applicable).

This is the template for every experiment you'll run in your career.

## Reflection questions

After running:

1. What was your final val loss? How does it compare to the README claim of 1.4697?
2. How did the loss evolve over training? Rapid drop, then slow? Plateau?
3. What kind of mistakes does the model make in generation? Repeated letters? Nonsense words? Real words in wrong order?
4. If you trained longer (say, 10000 iters instead of 5000), would val loss keep going down? (Try it!)
5. What if you made the model half the size? Double? (Try it!)

## Extensions

1. **Your own dataset**: Take any text file (a book, your journal, code), tokenize it character-level, train. The `data/bhagavad_gita` directory already exists in your nanoGPT - someone has tried this with the Bhagavad Gita!
2. **BPE tokenization**: Switch to tiktoken BPE. You'd need to modify prepare.py. Check `data/shakespeare/prepare.py` for the BPE version. Different data, similar result.
3. **Bigger model**: Try `n_layer=12, n_embd=768, n_head=12` (GPT-2 small) with the Shakespeare data. You'll overfit wildly (tiny data, big model). Observe the gap between train and val loss.

## Celebrate

You trained a transformer. From scratch. On a laptop, possibly. This is no small thing - you are now in a very different category than "person who has heard of LLMs." You've *built* one.

Modules 4 and 5 take you from "toy model works" to "modern production pipeline." Same skills, bigger scale.

## Next

Module 4. Open `04_training_pipeline/01_datasets_and_dataloaders.md`.
