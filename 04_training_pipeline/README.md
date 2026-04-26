# Module 4 - Training Pipeline: Data, Loops, Scale

nanoGPT is a single-file trainer. In the real world there are data engineering, mixed precision, gradient accumulation, learning-rate schedules, gradient clipping, checkpointing, evaluation, wandb. This module covers all of it using the same two codebases.

## Lessons (each ~30-60 min)

### `01_datasets_and_dataloaders.md`
- Where does the text come from? Common Crawl, The Pile, RedPajama, SlimPajama, FineWeb, DCLM.
- Data cleaning, deduplication, quality filters.
- Why "one epoch" is usually too much data to see even once at LLM scale.
- memmap'd binary files as the storage format (nanoGPT pattern).
- Walkthrough: `nanoGPT/data/openwebtext/prepare.py`.
- Walkthrough: `nanochat/nanochat/dataset.py` and `dataloader.py` (distributed, sharded).

### `02_bpe_tokenizer_deep.md`
- Deep dive into BPE algorithm.
- tiktoken internals.
- Walkthrough: `nanochat/nanochat/tokenizer.py` + `scripts/tok_train.py` + `scripts/tok_eval.py`.
- Why nanochat trained a fresh tokenizer. Measuring compression.
- Special tokens for chat; the `tokenizer.apply_chat_template` concept.

### `03_mixed_precision_bf16_fp16.md`
- Why computing in fp16/bf16 is faster (2x on modern GPUs).
- The loss scaler: keeping fp16 gradients from underflowing to zero.
- bf16 vs fp16: tradeoffs.
- `torch.amp.autocast`, `torch.cuda.amp.GradScaler`.
- nanoGPT's usage: `dtype='bfloat16'` default.
- fp8: nanochat's new trick. `nanochat/nanochat/fp8.py` - custom low-level kernel code.

### `04_gradient_accumulation.md`
- What if the batch you want doesn't fit in memory? Accumulate gradients over multiple forward/backwards before stepping.
- In nanoGPT: `gradient_accumulation_steps = 5 * 8` means the effective batch is 40x the micro-batch.
- Under DDP: which micro-step syncs gradients.
- How `total_batch_size = micro_batch_size * grad_accum * num_gpus` relates to the "effective batch" you'd compare across runs.

### `05_learning_rate_schedules.md`
- Warmup: why start from 0.
- Cosine decay: the standard.
- Step decay, linear decay, WSD (warmup-stable-decay): newer variants.
- Chinchilla's rule of thumb for max LR vs model size.
- Where nanoGPT's `get_lr(it)` lives and what each branch does.
- How nanochat computes LR from depth automatically.

### `06_gradient_clipping_nan_debugging.md`
- "Loss is NaN" - the nightmare.
- Gradient clipping by norm: `clip_grad_norm_`. Standard value 1.0.
- How to debug NaNs: check weights, check LR, check mixed-precision ordering, check data.
- nanoGPT default: `grad_clip = 1.0`.

### `07_checkpointing_and_resume.md`
- What to save: model state_dict, optimizer state_dict, step number, config, RNG state.
- Atomic writes (save to temp, rename).
- Resuming: load everything, continue from the same step.
- Long-running experiments: save every N iterations.
- nanoGPT: simple `ckpt.pt`. nanochat: `nanochat/checkpoint_manager.py` - more structured.

### `08_evaluation_val_loss_perplexity.md`
- Val loss: the primary signal during training.
- Perplexity vs bits-per-byte (BPB) - why BPB is vocab-invariant.
- Calibrating: what's a "good" loss for a given model size?
- When val loss stops dropping - is the model done, or under-parameterized?
- nanochat's `nanochat/loss_eval.py`.

### `capstone_custom_dataset.md`
- Take your own text data (code, journal, chat logs, any corpus).
- Build a prepare.py for it.
- Train nanoGPT on it for a few hundred iterations.
- Sample outputs. Evaluate qualitatively.
- Measure val loss and compare to Shakespeare baseline.
- Optional: train a BPE tokenizer on your data before training.

## Why this module

This module is the difference between "I understand transformers" and "I can actually train one on real data and know why it works or doesn't." Most research bugs and most career-making engineering wins happen here, not in architecture innovation.
