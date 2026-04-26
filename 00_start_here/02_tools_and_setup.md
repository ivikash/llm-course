# 02 - Tools and Setup

Before we touch any math or any model, let's set up your environment and
understand the tools you'll be using. This lesson is mostly "context" - no
math, no code yet.

## What hardware do I need?

For Modules 0-3: your laptop. Seriously. The Shakespeare toy GPT in nanoGPT
runs on a CPU in a few minutes. It will be slow, but it will run.

For Modules 4+: you'll benefit from a GPU. Options:
- You already have an NVIDIA GPU in your laptop/desktop (e.g. RTX 3060+). Perfect.
- You have an Apple Silicon Mac (M1/M2/M3/M4). PyTorch can use the Mac's GPU via "mps" backend. Workable.
- You have neither. Rent one. Covered in Module 6. ~$1-3/hour for a usable GPU.

## Software you need

1. **Python 3.10 or newer.** Check with `python3 --version`.
2. **pip** (comes with Python) or **uv** (a modern, fast replacement). nanochat uses `uv`. Either works.
3. **Git** - for downloading repos. `git --version`.
4. **A terminal you're comfortable with.** Linux bash, macOS zsh, WSL, all fine.
5. **An editor.** VSCode, Cursor, vim, whatever you like. Cursor has nice AI-assist if you want it.

## The two repos (already on your machine)

```
~/workspace/nanoGPT/    # Module 3 centers on this
~/workspace/nanochat/   # Module 5 centers on this
```

Let's install their dependencies now so they're ready when we need them.

### nanoGPT setup

```bash
cd ~/workspace/nanoGPT
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

That's it. If you're on a laptop without a GPU, the default CPU version of PyTorch is fine.

### nanochat setup

nanochat uses `uv` for faster dependency management.

```bash
cd ~/workspace/nanochat
# install uv if not already there
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv                       # creates .venv/
uv sync --extra cpu           # install deps for CPU-only. Use --extra gpu on a GPU machine.
source .venv/bin/activate     # activate the environment
```

You only need to do this setup when we get to Module 5. But you can do it
now if you want.

## Understanding the landscape

A 30-second model of what the tools do:

**PyTorch** - The library. It gives you:
- `tensors`: like numpy arrays, but they can live on a GPU and they remember how they were computed (for gradient calculation).
- Layers: `nn.Linear`, `nn.Embedding`, etc. - the building blocks of neural networks.
- Autograd: if you do math with tensors, PyTorch secretly records the operations so it can compute gradients when you call `.backward()`. This is magic.
- Optimizers: `torch.optim.AdamW` etc. - they update weights based on gradients.

**numpy** - The older, CPU-only numerical library. PyTorch tensors are modeled after it. We use numpy a little, PyTorch a lot.

**tiktoken** - OpenAI's fast tokenizer, pip-installable. nanoGPT uses it.

**transformers** (HuggingFace) - An enormous library that implements hundreds of model architectures. We use it minimally at first, more later.

**datasets** (HuggingFace) - For downloading training data.

**wandb (Weights & Biases)** - An experiment tracker. When you train a model you pipe metrics (loss, learning rate, etc.) to wandb.com and you get live dashboards. Free for individuals.

**tqdm** - A library that gives you progress bars. Trivial but universally used.

## The terminal: quick refresher

You'll live in the terminal. The commands you'll use constantly:

```bash
cd ~/workspace/nanoGPT             # change directory
ls                                 # list files
python train.py                    # run a Python script
python -m scripts.tok_train        # run a module (used a lot in nanochat)
nvidia-smi                         # show your NVIDIA GPU stats (if you have one)
top / htop                         # show CPU/RAM usage
Ctrl-C                             # kill a running command
```

If you're on a machine with NVIDIA GPUs, run `nvidia-smi` now. You should see each GPU, its memory usage, and which processes are using them.

## GPU, CPU, VRAM - the quickest mental model

- Your **CPU** has ~8-32 cores. Each is fast and smart. Good at "do 10 different complicated things."
- A modern **GPU** has ~10,000+ small cores. Each is simpler. Amazing at "do the same simple math on 10,000 numbers at once."
- Neural network math (matrix multiplication) is exactly "do the same simple math on a huge pile of numbers at once." So GPUs crush CPUs for this. Often 50x-200x faster.
- **RAM** is your main computer memory. GPUs have their own separate memory called **VRAM**. To use a GPU, your tensors must live in VRAM. Copying between CPU RAM and GPU VRAM is slow, so we try to keep data on the GPU.
- When people say "H100 80GB", 80GB is the VRAM.

We'll go deeper in Module 6. For now, know: matrix math big, GPU fast, VRAM is the scarce resource.

## What is "training a model", operationally?

A concrete mental picture of what the next few months of your work will look like:

1. You have a dataset of text (e.g. Shakespeare, or 300GB of web data).
2. You have a script like `train.py`.
3. You run `python train.py`.
4. Your terminal starts printing lines like:
   ```
   iter 0: loss 10.2, time 543.21ms
   iter 10: loss 8.7, time 141.02ms
   iter 20: loss 7.1, time 140.55ms
   ...
   iter 10000: loss 1.47, time 140.11ms
   ```
5. The "loss" number going down = the model is learning.
6. Periodically, a file like `ckpt.pt` (checkpoint) is saved. That's your model.
7. When done, you run `python sample.py` to make the model generate text.

That's the whole rhythm. You'll do it hundreds of times.

## Visualize this

**What happens inside a GPU during training** (watch once, understand forever):

```
  Training step (simplified)

  ┌─────────────┐
  │ Input batch │  shape (batch_size, sequence_length)   e.g. (32, 1024)
  └──────┬──────┘       a tensor of token IDs on the GPU
         ▼
  ┌─────────────┐
  │  Forward    │  tensors flow through model
  │  pass       │  embedding → N transformer blocks → LM head
  └──────┬──────┘  output: (32, 1024, vocab_size) logits
         ▼
  ┌─────────────┐
  │  Loss       │  compare logits to true next tokens
  └──────┬──────┘  scalar number
         ▼
  ┌─────────────┐
  │  Backward   │  compute gradients for every weight
  │  pass       │  (one for each of ~100M-1T numbers)
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │  Optimizer  │  subtract lr × gradient from each weight
  │  step       │
  └──────┬──────┘
         ▼
     (repeat)
```

You'll see this loop every day of your LLM career. In Module 2 Lesson 7 we walk through the exact ~10 lines of Python that implement it.

**Watch `nvidia-smi` while training** (if you have a GPU): `watch -n 1 nvidia-smi` shows live GPU utilization. Healthy training hits 60-90% sustained.

## Check your setup

Run these commands. If any fail, fix before proceeding.

```bash
python3 --version                                   # 3.10+
python3 -c "import torch; print(torch.__version__)" # should print something like 2.2.0
python3 -c "import torch; print(torch.cuda.is_available())"  # True if you have an NVIDIA GPU, False otherwise - either is fine
python3 -c "import numpy; print(numpy.__version__)"
cd ~/workspace/nanoGPT && ls                        # should list model.py, train.py, etc.
cd ~/workspace/nanochat && ls                       # should list README.md, nanochat/, scripts/, etc.
```

## Exercises

1. Run the setup commands above. Fix anything that breaks. (Often it's just `pip install X`.)
2. If you have an NVIDIA GPU, run `nvidia-smi` and read its output. Find: GPU model, total VRAM, free VRAM.
3. Open `~/workspace/nanoGPT/model.py` in your editor. You won't understand it yet, but scroll through. Notice: it's ~400 lines. Smaller than most apps you've worked on. This entire file implements a GPT. That should make you slightly less scared.
4. Open `~/workspace/nanochat/nanochat/gpt.py`. Notice it's ~700 lines, similar idea, more modern. We'll compare them directly in Module 5.

## Next

Open `03_what_is_an_llm.md`. We'll build a 10-line "language model" so you can feel what "predicting the next word" means before we add any complexity.
