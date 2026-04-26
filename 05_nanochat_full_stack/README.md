# Module 5 - nanochat: The Full Modern Stack

This is where the course becomes a tour of a real, modern LLM pipeline. nanochat has everything nanoGPT doesn't: its own tokenizer, chat formatting, SFT, RL, a proper evaluation suite, an inference engine with KV cache, and a web UI. By the end of this module you'll have trained an end-to-end chatbot and talked to it.

## Lessons (each ~1-2 hours)

### `01_tour_of_nanochat.md`
- Walk the repo directory in 30 minutes.
- Who is who: `nanochat/` (library), `scripts/` (entry points), `tasks/` (evaluations), `runs/` (bash scripts wiring it all), `dev/` (scratch notes).
- `runs/speedrun.sh` as the "table of contents" for the system.
- Compare key files side-by-side with nanoGPT:
  - `nanochat/nanochat/gpt.py` vs `nanoGPT/model.py`
  - `nanochat/scripts/base_train.py` vs `nanoGPT/train.py`
  - `nanochat/nanochat/dataloader.py` vs nanoGPT's inline `get_batch`

### `02_tokenizer_training.md`
- Walkthrough: `scripts/tok_train.py`.
- The BPE algorithm in practice with Rust backend for speed.
- What's in `tokenizer.py`: encoding, decoding, special tokens, chat templates.
- `scripts/tok_eval.py` - measure compression ratio, compare to GPT-2 tokenizer.

### `03_pretraining_base_train.md`
- `scripts/base_train.py` - probably the most important file in the repo.
- Differences from nanoGPT's train.py: auto-tuned hyperparams from depth, distributed data loading, fp8 path, wandb integration.
- The "depth dial" philosophy - one knob for model size, everything else derived.
- Rigorous eval loop: CORE metric, val_bpb.
- Output: a `base model` that can complete text but doesn't follow instructions.

### `04_midtraining_and_chat_formatting.md`
- Introducing the chat special tokens: `<|im_start|>user`, `<|im_end|>`, etc.
- Walkthrough of chat message formatting.
- `tasks/common.py` - the common task infrastructure (preparing examples).
- Why midtraining helps bridge pretrain to SFT.

### `05_sft_supervised_fine_tuning.md`
- `scripts/chat_sft.py` - SFT on curated (user, assistant) pairs.
- Loss masking: only compute loss on assistant tokens, not user tokens.
- The SmolTalk dataset (`tasks/smoltalk.py`).
- Identity conversations - how you give the model a personality.
- `scripts/chat_eval.py` - evaluate after SFT.

### `06_rl_and_preferences.md`
- The RL landscape: PPO (original RLHF), DPO (direct preference), GRPO (used in nanochat).
- Why RL on verifiable rewards works for math/code (GSM8K, HumanEval).
- Walkthrough of `scripts/chat_rl.py`.
- `tasks/gsm8k.py` as the reward environment.
- Rollouts, advantage, policy loss, KL penalty.

### `07_evaluation_core_mmlu_humaneval_gsm8k.md`
- Benchmarks matter more than architecture in practice.
- CORE metric (DCLM paper).
- MMLU (multiple-choice knowledge, `tasks/mmlu.py`).
- ARC (reasoning, `tasks/arc.py`).
- HumanEval (code, `tasks/humaneval.py`).
- GSM8K (grade-school math, `tasks/gsm8k.py`).
- Spelling Bee and CustomJSON (smaller/custom, `tasks/spellingbee.py`, `tasks/customjson.py`).
- How to build your own benchmark.

### `08_inference_engine_kv_cache.md`
- `nanochat/nanochat/engine.py` - the inference-time code path.
- KV cache: reuse past computations for generation. ~10-100x speedup.
- Batched generation across many conversations.
- Streaming tokens out.
- `nanochat/nanochat/flash_attention.py` - what Flash Attention is and isn't, and the fallback.
- `nanochat/nanochat/execution.py` - sandbox for executing model-produced code.

### `09_serving_chat_web.md`
- `scripts/chat_web.py` - an aiohttp server serving `ui.html`.
- Walkthrough of the chat UI (an HTML + JS single file).
- From "works on my laptop" to "users can talk to it" - the small bridge.
- What would you need to go further: HTTPS, auth, rate limiting, multi-user sessions (not in scope for nanochat but worth discussing).

### `10_speedrun_end_to_end.md`
- `runs/speedrun.sh` - read it line by line.
- What every command produces.
- The concept of "time-to-GPT-2" as a leaderboard metric.
- `runs/scaling_laws.sh` and `runs/miniseries.sh` - other reference runs.
- `runs/runcpu.sh` - the hack for running a tiny version on CPU.

### `capstone_train_and_talk.md`
- Run a minimal end-to-end training on the smallest configuration you can afford (local GPU or $10 of cloud credit).
- Train tokenizer.
- Pretrain a d8 model.
- SFT.
- Launch the web UI and have an actual conversation with your model.
- Write a short postmortem: what worked, what surprised you, what you'd improve.

## The payoff

At the end of Module 5, you will have done something 99% of people who "know AI" haven't: trained your own chatbot end-to-end, bytes-to-dialogue. That is an enormous credibility moment - for yourself and for the people you talk to about this.
