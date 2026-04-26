# 3.11 - train.py Walkthrough (Every Line)

You now understand the model. Time to understand the training script - `nanoGPT/train.py` - line by line. Have it open in a split window.

## Structure at 30,000 feet

The file is ~300 lines of Python. Top to bottom:

1. Config (lines ~1-100) - all the knobs.
2. Distributed setup (~100-120) - for multi-GPU runs.
3. Data loading helper (~120-150) - `get_batch` function.
4. Model initialization (~150-200) - three paths: from scratch, resume, or from pretrained GPT-2.
5. Optimizer and learning rate schedule (~200-220).
6. The main training loop (~220-end).
7. Validation / checkpointing / logging peppered throughout.

Let's walk.

## 1. The config

```python
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'          # 'scratch' | 'resume' | 'gpt2*'
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
# adamw optimizer
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# learning rate decay
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
# DDP
backend = 'nccl'
device = 'cuda'
dtype = 'bfloat16'
compile = True
```

Each of these is a Python variable at module level. The `configurator.py` script at the bottom (`exec(open('configurator.py').read())`) allows overrides from config files or command line. Look at `config/train_shakespeare_char.py` to see a set of overrides for the Shakespeare toy.

## 2. Distributed setup

```python
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
```

If the script is launched via `torchrun`, the `RANK` env var is set. Each process then knows its rank (0, 1, ...), local rank (which GPU to use), and world size (total processes). Covered in Module 6.

For single-GPU or CPU, `ddp=False`, and we run normally. nanoGPT handles both.

## 3. get_batch

```python
def get_batch(split):
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
```

- `np.memmap`: open the binary file of tokens **without** loading it all into RAM. OS maps it to virtual memory. Trained on hundreds of GB with the same 2 lines.
- `torch.randint(...)`: pick `batch_size` random starting positions.
- `x` is `block_size` tokens from each position; `y` is shifted by 1 (the next-token targets).
- `.pin_memory()` + `non_blocking=True`: lets CPU and GPU work in parallel during the transfer. Performance detail.

**This is how large-scale LLM data loading works.** No fancy dataloader, no torch.utils.data. Just a memmapped binary of token IDs and random indexing.

## 4. Model init

```python
if init_from == 'scratch':
    model_args = dict(n_layer=n_layer, n_head=n_head, ...)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    checkpoint = torch.load(ckpt_path, map_location=device)
    # ...reconstruct model from saved config + load state dict
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, override_args)
```

`from_pretrained` (in `model.py`) loads OpenAI's GPT-2 weights from HuggingFace and re-keys them to match nanoGPT's naming. Nice detail to study once - it proves nanoGPT is weight-compatible with the real GPT-2.

## 5. Optimizer & LR schedule

```python
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)
```

- `configure_optimizers`: builds AdamW with weight decay applied only to 2D params (covered in Lesson 2.5).
- `get_lr`: warmup linearly from 0, then cosine decay from max to min. This is the standard LR schedule.

## 6. The main loop

```python
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0
while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # eval periodically
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        # save checkpoint if best val loss so far
        ...

    # gradient accumulation micro-steps
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:  # autocast bf16
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')  # prefetch next batch
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # logging
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, ...")

    iter_num += 1
    local_iter_num += 1
    if iter_num > max_iters:
        break
```

Every piece is something we've covered:
- Set LR for this step.
- Periodically run validation.
- Forward pass (inside autocast for mixed precision), compute loss.
- Backward. Accumulate gradients over `gradient_accumulation_steps` micro-batches to simulate a larger effective batch. (Module 4 Lesson 4 expands on this.)
- Clip gradient norm so explosions don't wreck training.
- `optimizer.step()` to update weights.
- `optimizer.zero_grad()` to clear for next step.
- Log.
- Repeat.

**That's it. That's all that happens when GPT is trained.** Wrap a loop that does this in cloud infrastructure for a few weeks with 1000 GPUs and you have GPT-3.

## Visualize this

**train.py's structure, as a map**:

```
  ┌─────────────────────────────────────────────────┐
  │               nanoGPT/train.py                   │
  ├─────────────────────────────────────────────────┤
  │                                                  │
  │  [lines 1-80]     Config (knobs)                 │
  │                                                  │
  │  [lines 100-120]  Distributed setup (torchrun)   │
  │                                                  │
  │  [lines 120-160]  get_batch (data loading)       │
  │                                                  │
  │  [lines 160-200]  Model init (3 paths)           │
  │                       • from scratch              │
  │                       • resume from ckpt          │
  │                       • from GPT-2 pretrained     │
  │                                                  │
  │  [lines 200-220]  Optimizer (AdamW + groups)     │
  │                                                  │
  │  [lines 220-240]  Learning rate schedule         │
  │                                                  │
  │  ┌───────── MAIN LOOP [lines 240-end] ────────┐  │
  │  │  set LR for this step                       │  │
  │  │  ┌── eval periodically ──┐                  │  │
  │  │  │  compute val loss     │                  │  │
  │  │  │  save checkpoint      │                  │  │
  │  │  └───────────────────────┘                  │  │
  │  │                                              │  │
  │  │  for micro_step in grad_accum_steps:        │  │
  │  │    forward pass (under autocast)            │  │
  │  │    scaled_loss.backward()                   │  │
  │  │    prefetch next batch                      │  │
  │  │                                              │  │
  │  │  clip_grad_norm                              │  │
  │  │  optimizer.step()                            │  │
  │  │  optimizer.zero_grad()                       │  │
  │  │                                              │  │
  │  │  log to wandb/stdout                         │  │
  │  │                                              │  │
  │  │  iter_num += 1                               │  │
  │  └───────────────────────────────────────────────┘
  │                                                  │
  └─────────────────────────────────────────────────┘
```

300 lines. Every line fits in your head. This is the training script that reproduces GPT-2.

**Training rhythm visualization** (what you'll see):

```
  wall clock         console output
  ─────────────     ────────────────────────────────────
  0 sec             iter 0: loss 10.93, time 543.21ms
  0.14s             iter 1: loss 10.85, time 141.23ms
  ...
  14.1s             iter 100: loss 5.32, time 141.01ms
  ...
  141s              iter 1000: loss 3.51, time 141.02ms
  250s              step 1762: evaluating...
  250.5s            step 1762: train loss 3.43, val loss 3.52
  250.5s            saving checkpoint...
  ...
  14,100s (4h)      iter 100000: loss 2.85, time 141.01ms    ← GPT-2 level
  ...
```

**Run nanoGPT and watch this actually happen** - it's the most educational experience in the course.

**Live training dashboard** (with wandb):

```
  wandb dashboard (in browser):
  ┌──────────────────────────┬──────────────────────────┐
  │  train/loss              │  val/loss                │
  │       ●                  │                          │
  │        ●                 │        ●                 │
  │         ●●               │         ●●               │
  │           ●●             │           ●●             │
  │             ●●●          │             ●●●          │
  │                ●●●●      │                ●●●●      │
  └──────────────────────────┴──────────────────────────┘
  ┌──────────────────────────┬──────────────────────────┐
  │  lr                      │  grad_norm               │
  │     ─────                │       ╱╲                 │
  │    ╱                     │      ╱  ╲╱╲              │
  │   ╱    (warmup+cosine)   │     ╱      ╲╱╲           │
  │  ╱          ╲            │   ╱           ╲          │
  └──────────────────────────┴──────────────────────────┘
```

Module 6.7 covers setting up wandb. Worth doing now so you have dashboards for the capstone.

## What you should do now

Read the actual file, top to bottom. You'll encounter `scaler` (for fp16 mixed precision - ignore for now, covered in Module 4), `torch.compile` (makes things faster via JIT), and `mfu` (Model FLOPs Utilization, a performance metric). If a piece isn't obvious, flag it and we discuss.

## Next

`capstone_shakespeare.md` - you train nanoGPT on Shakespeare on your laptop. The moment of truth.
