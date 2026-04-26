# Module 4 Capstone - Train on Your Own Dataset

Apply everything from Module 4 to a dataset you actually care about.

## Goal

Pick any text source, turn it into a nanoGPT-compatible dataset, train a small model on it, and sample from it.

## Pick your data

Any of these work:
- Your email archive (exported as text).
- Your journal entries.
- A book you love (copyright-permitting).
- Your work's internal documentation.
- A codebase you know well.
- Reddit comments from r/askhistorians.
- Song lyrics from your favorite band.

Rule of thumb: you want at least 100 KB of text, ideally 1-10 MB. Less, and the model overfits to nothing; more, and training takes longer than needed for a toy.

## Steps

### 1. Create a dataset directory

```bash
mkdir -p ~/workspace/nanoGPT/data/mydata
cd ~/workspace/nanoGPT/data/mydata
# put your text in input.txt
```

### 2. Write prepare.py

Model on `~/workspace/nanoGPT/data/shakespeare_char/prepare.py`. Minimal version:

```python
# ~/workspace/nanoGPT/data/mydata/prepare.py
import os, pickle, numpy as np

input_file = os.path.join(os.path.dirname(__file__), 'input.txt')
with open(input_file, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

chars = sorted(list(set(data)))
vocab_size = len(chars)
print("vocab size:", vocab_size)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s): return [stoi[c] for c in s]

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

train_ids = np.array(encode(train_data), dtype=np.uint16)
val_ids = np.array(encode(val_data), dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

meta = {'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
```

Run it:

```bash
python ~/workspace/nanoGPT/data/mydata/prepare.py
```

You should see `train.bin`, `val.bin`, `meta.pkl` created.

### 3. Create a training config

`~/workspace/nanoGPT/config/train_mydata.py`:

```python
# copy/tweak from train_shakespeare_char.py
out_dir = 'out-mydata'
eval_interval = 250
eval_iters = 20
log_interval = 10

always_save_checkpoint = True
wandb_log = False
wandb_project = 'mydata'

dataset = 'mydata'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
warmup_iters = 100

beta2 = 0.99
```

### 4. Train

```bash
cd ~/workspace/nanoGPT
python train.py config/train_mydata.py
```

Or on CPU:

```bash
python train.py config/train_mydata.py --device=cpu --compile=False \
    --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 \
    --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0 --eval_iters=20
```

### 5. Sample

```bash
python sample.py --out_dir=out-mydata --num_samples=5 --max_new_tokens=500
```

Or with a seed prompt:

```bash
python sample.py --out_dir=out-mydata --start="Dear diary," --num_samples=3
```

## Deliverables

Save to `~/workspace/llm-course/exercises/` or your personal notes:

- `mydata_sample.txt` - representative samples from your trained model.
- `mydata_report.md`:
  - What data did you use? How much?
  - Final train loss, val loss?
  - Overfitting observed?
  - Qualitative impression: does the output capture the style of your data?
  - What would you change next time?

## Extensions (pick one or two)

### A. Use BPE tokenization instead of character-level

Copy `~/workspace/nanoGPT/data/shakespeare/prepare.py` - it uses tiktoken. Apply the same pattern to your data. Compare: does BPE give better results for your domain? (Usually yes, unless your vocab is very small like character-level music.)

### B. Experiment with dropout

Train twice: once with `dropout=0.2`, once with `dropout=0.0`. Compare train-val gap. Which generalizes better for your dataset size?

### C. Experiment with model size

Train at 3 sizes (small, medium, large - vary `n_layer`, `n_embd`). Plot val loss vs model size. You're doing your own tiny scaling law.

### D. Learning rate sweep

Try LR: 3e-4, 1e-3, 3e-3, 1e-2. Which works best? Any diverge?

## Why this matters

You've now done the full pipeline on your own data: prepare → train → eval → sample. This is 90% of what an ML engineer does day-to-day: take a dataset, design a model, train, evaluate. The only difference at scale is: more data, bigger model, distributed training, wandb integration, papers read.

If you can iterate on this loop confidently, you're past the "can I do it" phase and into the "how do I do it well" phase. That's where the rest of the course takes you.

## Next

Module 5 starts. Open `05_nanochat_full_stack/01_tour_of_nanochat.md`.
