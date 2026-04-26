# 5.10 - Speedrun End-to-End

Time to walk through `runs/speedrun.sh` in full. If you've been following Module 5, everything here should now be familiar. This lesson is the integration: one script, one flow, soup to nuts.

## Setting

`speedrun.sh` is the reference end-to-end pipeline. It's designed to run unattended on a blank 8xH100 box and produce a chat-capable GPT-2-grade model in ~3 hours. Everything is checkpointed and logged, so you can look back at every decision.

Open `~/workspace/nanochat/runs/speedrun.sh` side-by-side.

## Top-of-file setup

```bash
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
```

- `OMP_NUM_THREADS=1`: prevents PyTorch from spawning too many OpenMP threads, which wastes CPU when you have many DDP ranks.
- `NANOCHAT_BASE_DIR`: where all artifacts (data, tokenizer, checkpoints, reports) live.

## Env install

```bash
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate
```

Installs `uv`, creates a local venv, installs deps. Standard bookkeeping.

## Wandb setup

```bash
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy    # special value: skip wandb
fi
```

If you set `WANDB_RUN=myrun` in the environment, everything logs to wandb. Else it runs without.

## Report reset

```bash
python -m nanochat.report reset
```

Clears `report/` directory, writes a timestamped header (system info, git hash, etc.). Each stage appends a section; at the end we have a complete `report.md`.

## Stage 1: download data

```bash
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
```

- First command: synchronously downloads 8 shards (~2B chars, enough to train tokenizer).
- Second command: asynchronously downloads 170 shards in the background (enough for full pretraining). The `&` puts it in the background; `DATASET_DOWNLOAD_PID` captures the PID so we can wait on it later.

## Stage 2: tokenizer

```bash
python -m scripts.tok_train
python -m scripts.tok_eval
```

Trains BPE on the 8 shards, evaluates it. Few minutes. Output: `$NANOCHAT_BASE_DIR/tokenizer/`.

## Stage 3: wait for data, then pretrain

```bash
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 --target-param-data-ratio=8 \
    --device-batch-size=16 --fp8 --run=$WANDB_RUN

torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- \
    --device-batch-size=16
```

- `wait`: block until background download finishes.
- `torchrun ... base_train`: pretrain for ~2 hours on 8 GPUs. Outputs: base checkpoint, training curves, samples.
- `torchrun ... base_eval`: val BPB + CORE.

Notes:
- `--target-param-data-ratio=8` means "train on 8 tokens per param" - slightly under Chinchilla-optimal (20x), because we want to beat GPT-2's CORE score quickly, not train compute-optimally.
- `--fp8` activates the fp8 matmul path (only runs on H100+).

## Stage 4: SFT

```bash
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
    --device-batch-size=16 --run=$WANDB_RUN

torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
```

- Download 2.3 MB of identity conversations.
- Run SFT on SmolTalk + identity.
- Eval the chat model.

The `-i sft` passes which checkpoint to eval ("sft" tag).

## Stage 5: (commented out: chat CLI/web)

The script's final section is commented out - it's what you'd run *manually* to interact with the model:

```bash
# python -m scripts.chat_cli -p "Why is the sky blue?"
# python -m scripts.chat_web
```

## Stage 6: report

```bash
python -m nanochat.report generate
```

Assembles the final `report.md` by concatenating per-stage sections. This is a beautiful artifact: everything you know about the run in one file. Loss curves, eval scores, sample outputs, hyperparameters, system info.

## Time breakdown on 8xH100

Roughly:
- Env setup + data download: 10 min
- Tokenizer: 5 min
- Pretraining: 140 min (2h20m)
- Base eval: 15 min
- SFT: 20 min
- SFT eval: 30 min
- **Total: ~220 min (~3h40m)**

The "speedrun" leaderboard tracks just the pretraining time (which is the dominant cost). Current record: ~1.65 hours. See `dev/LEADERBOARD.md`.

## On weaker hardware

### 8xA100

Everything runs, just ~2x slower (no fp8). Remove `--fp8` flag. Total ~6-8 hours.

### 1xH100

Remove `torchrun --nproc_per_node=8`, just run `python -m scripts.base_train`. Gradient accumulation kicks in automatically to preserve effective batch. About 8x slower = ~30 hours.

### CPU only

See `runs/runcpu.sh`. Tiny depth, tiny batch. Takes hours for a toy chatbot. Good for code-walkthrough; not for real training.

## Outputs at the end

```
~/.cache/nanochat/
├── data/                    (170 data shards, tokenized)
├── tokenizer/               (BPE vocab and merges)
├── checkpoints/
│   ├── base/                (pretrained base model)
│   └── sft/                 (fine-tuned chat model)
└── report/
    ├── base_train.md
    ├── base_eval.md
    ├── chat_sft.md
    ├── chat_eval.md
    └── report.md            (the composite, final report)
```

Also: wandb dashboard (if enabled) with all training curves.

## The speedrun mentality

The key idea behind nanochat's leaderboard:
- Fix a target capability (beat GPT-2 CORE score).
- Minimize wall-clock time to hit it.
- Publish diffs, every improvement replicable.
- Competition drives optimization at every layer: data, tokenizer, architecture, optimizer, precision.

This is a research methodology worth adopting for your own work: find a fixed benchmark, try to beat it with less compute, publish what worked.

## Exercises

1. Read `runs/speedrun.sh` top to bottom. Every line. Match each to a lesson in Module 5.

2. Read `dev/LEADERBOARD.md`. Identify what changed at each entry.

3. Read `dev/LOG.md`. This is Karpathy's running research log. Study the structure - hypotheses, experiments, results, updated priors. It's a model of how a senior researcher thinks.

4. Mentally plan a speedrun attempt. Pick one hypothesis (e.g. "use Gemma-3 style sliding window attention"). Write a one-page plan: expected speedup, risks, how to measure.

5. (If you have $50 to spare) run the speedrun yourself. See capstone.

## Next

`capstone_train_and_talk.md` - your own end-to-end chatbot.
