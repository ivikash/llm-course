# 5.1 - Tour of nanochat

Before we dive into specific scripts, let's map the whole nanochat repo. This 45-minute tour will orient you for the rest of Module 5.

## The directory layout

```
nanochat/
├── README.md                 # project overview
├── pyproject.toml            # dependencies (managed by uv)
├── runs/                     # bash scripts that wire everything together
│   ├── speedrun.sh           # the reference end-to-end pipeline
│   ├── miniseries.sh         # compute-optimal sweep of model sizes
│   ├── scaling_laws.sh       # scaling-laws research runs
│   └── runcpu.sh             # minimal version that runs on CPU
├── nanochat/                 # the library code
│   ├── gpt.py                # the model (analogous to nanoGPT/model.py)
│   ├── tokenizer.py          # BPE tokenizer + chat templates
│   ├── dataset.py            # download / shard management
│   ├── dataloader.py         # streaming distributed data loading
│   ├── engine.py             # inference engine with KV cache
│   ├── flash_attention.py    # attention implementation
│   ├── fp8.py                # custom fp8 matmul
│   ├── optim.py              # AdamW, Muon optimizers
│   ├── checkpoint_manager.py # save/load
│   ├── loss_eval.py          # val loss + bpb
│   ├── core_eval.py          # CORE benchmark
│   ├── execution.py          # sandbox for executing model-produced code
│   ├── report.py             # generate report.md at the end of runs
│   └── common.py             # shared utilities
├── scripts/                  # entry points (python -m scripts.X)
│   ├── tok_train.py          # train a BPE tokenizer
│   ├── tok_eval.py           # evaluate the tokenizer
│   ├── base_train.py         # pretrain the base model
│   ├── base_eval.py          # eval base model (bpb, CORE)
│   ├── chat_sft.py           # supervised fine-tuning
│   ├── chat_rl.py            # reinforcement learning (GRPO-style)
│   ├── chat_eval.py          # eval chat model (MMLU, ARC, GSM8K, HumanEval)
│   ├── chat_cli.py           # talk to your model in a terminal
│   └── chat_web.py           # talk to your model in a web UI
├── tasks/                    # evaluation suites
│   ├── common.py             # shared task infrastructure
│   ├── smoltalk.py           # SmolTalk SFT dataset
│   ├── mmlu.py               # MMLU benchmark
│   ├── arc.py                # ARC benchmark
│   ├── humaneval.py          # HumanEval coding
│   ├── gsm8k.py              # GSM8K math
│   ├── spellingbee.py        # custom task
│   └── customjson.py         # custom JSON task
├── tests/                    # pytest tests
├── dev/                      # scratch / research notebooks
│   ├── LOG.md                # Karpathy's running research log
│   ├── LEADERBOARD.md        # the speedrun leaderboard
│   ├── scaling_analysis.ipynb
│   ├── estimate_gpt3_core.ipynb
│   └── gen_synthetic_data.py
└── runs/                     # bash scripts (see above)
```

## The mental model: three axes

### Axis 1: pipeline stages

nanochat implements 5 training stages:

```
 1. Tokenizer training   (scripts/tok_train.py)
 2. Pretraining          (scripts/base_train.py)
 3. [Midtraining]        (part of chat_sft.py in current version)
 4. SFT (Fine-tuning)    (scripts/chat_sft.py)
 5. RL (Reinforcement)   (scripts/chat_rl.py)
```

Each stage takes the output of the previous and improves it. You can stop at any stage and have a useful model:
- After step 2: base model, can complete text.
- After step 4: chatbot, follows instructions.
- After step 5: better at verifiable tasks (math, code).

### Axis 2: evaluation

Every stage has a corresponding eval:
- `tok_eval.py`: tokenizer quality (compression).
- `base_eval.py`: base model quality (bpb, CORE).
- `chat_eval.py`: chat model quality (MMLU, ARC, GSM8K, HumanEval).

And `tasks/` is the registry of individual benchmark tasks.

### Axis 3: deployment

Once you have a trained model:
- `chat_cli.py`: terminal chat.
- `chat_web.py`: web UI via aiohttp + `nanochat/ui.html`.
- `nanochat/engine.py`: the fast inference engine used by both.

## runs/speedrun.sh: the spine

Open `~/workspace/nanochat/runs/speedrun.sh`. It's a ~150-line bash script. It's the single reference document that says "this is how you train a GPT-2 capability chatbot in ~3 hours on 8xH100."

It calls, in order:
1. `python -m nanochat.dataset -n 8` (download 8 data shards).
2. `python -m scripts.tok_train` (train tokenizer).
3. `python -m scripts.tok_eval` (eval tokenizer).
4. `torchrun ... -m scripts.base_train` (pretrain).
5. `torchrun ... -m scripts.base_eval` (eval base).
6. `torchrun ... -m scripts.chat_sft` (SFT).
7. `torchrun ... -m scripts.chat_eval` (eval SFT).
8. `python -m nanochat.report generate` (write final report).

This is the "entry point" to nanochat. Understanding speedrun.sh = understanding the system.

## The "depth dial" philosophy

nanochat has a single configuration knob: `--depth`, the number of transformer layers. Everything else (n_embd, n_head, lr, training tokens, batch size) is computed from it.

```python
# simplified from nanochat/gpt.py
@dataclass
class GPTConfig:
    depth: int = 24
    def __post_init__(self):
        self.n_embd = 64 * self.depth     # width scales with depth
        self.n_head = self.depth // 2      # heads scale too
        # etc.
```

Why? So researchers can sweep one knob to study scaling, and so users can say "I want a bigger model" without needing to understand hyperparameters. `--depth=12` is GPT-1 sized. `--depth=20-26` is GPT-2 sized. `--depth=40+` is GPT-3 sized.

## Key differences from nanoGPT

| Aspect | nanoGPT | nanochat |
|--------|---------|----------|
| Scope | Pretraining only | Full pipeline (tok, pretrain, sft, rl, eval, serve) |
| Tokenizer | Uses GPT-2 / char-level | Trains its own BPE |
| Model | GPT-2 architecture (learned pos, LN, GELU) | Llama-style (RoPE, RMSNorm, SwiGLU, GQA) |
| Config | Many knobs (hand-tuned) | One knob (--depth) |
| Precision | bf16 | bf16 + fp8 option |
| Attention | PyTorch SDPA | Flash attention + custom fallback |
| Distributed | Basic DDP | DDP + fault tolerance, async data |
| Dependencies | Minimal | More (uv, rustbpe, wandb, etc.) |
| Philosophy | Simplest possible | Simplest comprehensive |

Both are excellent. nanoGPT is your first read. nanochat is your second.

## Reading order for Module 5

1. This tour.
2. `02_tokenizer_training.md` → `scripts/tok_train.py`.
3. `03_pretraining_base_train.md` → `scripts/base_train.py` and `nanochat/gpt.py`.
4. `04_midtraining_and_chat_formatting.md` → chat templates.
5. `05_sft_supervised_fine_tuning.md` → `scripts/chat_sft.py`.
6. `06_rl_and_preferences.md` → `scripts/chat_rl.py`.
7. `07_evaluation_...md` → the benchmark suite.
8. `08_inference_engine_kv_cache.md` → `engine.py`.
9. `09_serving_chat_web.md` → `scripts/chat_web.py`.
10. `10_speedrun_end_to_end.md` → running the whole thing.
11. `capstone_train_and_talk.md` → you train your own chatbot.

## Exercises

1. `tree ~/workspace/nanochat` (or `ls -R`). Match the output to my layout above. Note what's extra or different.

2. Open `runs/speedrun.sh` and read it top to bottom. Don't try to understand every command yet; just map them to the pipeline stages above.

3. Open `dev/LOG.md`. Read the first page. This is Karpathy's public running research log - a rare artifact, worth studying as a model for how senior researchers think.

4. Open `dev/LEADERBOARD.md`. Read the entries. Each one is a reproducible "speedrun" improvement. Note how improvements compound (dataset → batch size → fp8 → better data → auto-research).

5. Look at `nanochat/gpt.py`'s `GPTConfig` class. Find the depth-scaling equations.

## Next

`02_tokenizer_training.md` - the first script in the pipeline.
