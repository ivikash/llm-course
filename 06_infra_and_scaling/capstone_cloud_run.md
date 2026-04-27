# Module 6 Capstone - Your First Cloud Training Run

Rent a real GPU, run a real training job, kill it safely, compute the true cost.

## Goals

1. Provision an 8xH100 or 8xA100 cloud instance.
2. Run `nanochat/runs/speedrun.sh` end-to-end (~3 hours).
3. Log to wandb.
4. Download the trained model.
5. **Terminate the instance.**
6. Compute actual cost.

## Step-by-step

### 1. Pre-flight (do this before spinning anything up)

- [ ] Sign up on [Lambda Labs](https://lambda.ai) (or Runpod, Vast.ai).
- [ ] Add payment. Set a **billing alert** at $100.
- [ ] Set up [wandb](https://wandb.ai): account, API key ready.
- [ ] Ensure you have an SSH key on your local machine.
- [ ] Have a rough 3-hour block. Don't start at 11pm.

### 2. Launch

1. Filter to 8xH100 (or 8xA100 if cheaper/available).
2. Pick a region close to you (latency for SSH).
3. Add your SSH public key (`cat ~/.ssh/id_ed25519.pub`).
4. **Note the instance ID.** Set a phone reminder: "Kill instance XYZ in 4 hours."

Instance provisions in 1-3 minutes.

### 3. SSH in

```bash
ssh ubuntu@<instance-ip>
```

First time: you'll see a host-key prompt. Accept.

### 4. Setup

```bash
# tmux so SSH drops don't kill training
tmux new -s run

# get the code
git clone https://github.com/karpathy/nanochat.git
cd nanochat

# wandb setup
export WANDB_RUN=my-first-run
wandb login    # paste your API key when prompted

# launch the speedrun
bash runs/speedrun.sh
```

The script installs deps, downloads data, trains tokenizer, pretrains, SFTs, evals. ~3 hours on 8xH100.

### 5. Monitor

Open 3 terminals:
1. tmux session with the training (you'll see log lines).
2. Another SSH session: `nvidia-smi -l 5` (live GPU stats every 5s).
3. Your browser: wandb dashboard (live loss curves).

Write down:
- [ ] Time training started.
- [ ] Tokens/sec during pretraining.
- [ ] MFU during pretraining.
- [ ] Val BPB at end of pretraining.
- [ ] CORE score at end.
- [ ] MMLU / ARC / GSM8K after SFT.

### 6. Talk to your model

After speedrun.sh finishes:

```bash
python -m scripts.chat_web
# or
python -m scripts.chat_cli -p "Why is the sky blue?"
```

If web, forward the port:
```bash
# from your local machine:
ssh -L 8000:localhost:8000 ubuntu@<instance-ip>
# then open localhost:8000 in your browser
```

Have a real conversation. Screenshot it.

### 7. Download artifacts

```bash
# from your local machine
scp -r ubuntu@<instance-ip>:~/nanochat/~/.cache/nanochat/report/ ./my-first-run/
# or specific files
scp ubuntu@<instance-ip>:~/.cache/nanochat/report/report.md ./
```

Keep: `report.md`, any final checkpoint, conversation screenshots.

### 8. ⚠️ Terminate the instance ⚠️

Most important step. On Lambda:

1. Web dashboard → Instances → your instance → Terminate.
2. **Verify it says "terminating" or "terminated".**
3. Double-check billing page: current rate should drop to $0.

Do NOT rely on your memory. Set multiple reminders.

### 9. Compute cost

```
Actual cost = (end_time - start_time) × hourly_rate
```

Compare to the estimated $48-75 from nanochat's README. How did you do?

If you used wandb, it also records the GPU hours.

## The deliverable

Write `my-first-cloud-run.md`:

```markdown
# My First Cloud Training Run

## Setup
- Provider: Lambda / Runpod / ...
- Instance: 8xH100 / 8xA100
- Hourly rate: $X
- Start time: ...
- End time: ...
- Total duration: ...
- Total cost: $...

## Run details
- WANDB URL: ...
- Pretraining MFU: ...%
- Pretraining tokens/sec: ...
- Val BPB (end of pretrain): ...
- CORE score: ...
- Final MMLU / ARC / GSM8K / HumanEval: ...

## Conversation with my model
[screenshot or paste]

## Lessons learned
- ...
- ...
- ...

## What I would do differently
- ...
```

## Visualize this

**Live cloud run dashboard**:

```
  Your laptop browser                      Cloud GPU (8×H100)
  ──────────────────                       ────────────────────

  Tab 1: wandb.ai                ◀──────── training curves
  ┌──────────────────┐                      streamed live
  │ train/loss       │
  │  ●                │
  │   ●●              │
  │     ●●●           │
  │        ●●●●●      │
  │             ●●●●  │
  └──────────────────┘

  Tab 2: ssh terminal #1                   (tmux window 1)
  ┌──────────────────┐                     ┌──────────────┐
  │ iter 1000: loss │ ◀────────────────── │ speedrun.sh  │
  │ 3.52, time 141ms│                     │  running...  │
  │ iter 1010: ...  │                     └──────────────┘
  └──────────────────┘

  Tab 3: ssh terminal #2                   (tmux window 2)
  ┌──────────────────┐                     ┌──────────────┐
  │ nvidia-smi       │ ◀────────────────── │ nvidia-smi    │
  │ GPU 0: 94% 72GB  │                     │ live          │
  │ GPU 1: 95% 72GB  │                     │ (watch -n 1)  │
  │ ...               │                     └──────────────┘
  └──────────────────┘

  Tab 4: Lambda dashboard
  ┌──────────────────┐
  │ Instance running  │
  │ Cost: $62 so far   │
  │ [TERMINATE] button │
  └──────────────────┘
```

Four tabs is all you need to monitor a serious run.

**Cost burn chart (what you'll see)**:

```
  $
   │
   │                                        ●
  100│                                 ●
   │                            ●
  80 │                     ●
   │              ●
  60 │        ●               ← pretraining: $60 in 2 hours
   │   ●
  40 │
   │
  20 │
   │ ●
  0  └──────────────────────────────────────── time
     0h   1h   2h   3h   4h

  Each dot: price at that moment.
  Linear: you're burning $25/hr steady.
  Steeper: something went wrong (extra instance, wrong price tier).
```

**Expected wandb metrics at completion**:

```
  For nanochat d24 speedrun:

  train/loss (final):       ~2.11
  val/bpb (final):           ~0.75       ← beats GPT-2 baseline
  eval/core:                  ~0.258     ← beats 2019 OpenAI GPT-2 (0.2565)
  eval/mmlu:                   ~0.33
  eval/gsm8k:                 ~0.10
  eval/humaneval:             ~0.06

  train/mfu:                   45-52%
  train/tok_per_sec:          ~55k-70k tok/s (8×H100 with fp8)
```

**Screenshot checklist for your capstone report**:

```
  [ ] wandb run page with all curves (pin to 3-4 panels)
  [ ] final conversation with chat UI (at least 5 turns)
  [ ] nvidia-smi mid-training (proves you really ran it)
  [ ] provider billing page showing total cost
  [ ] terminal output of final eval (MMLU, ARC, etc.)
  [ ] report.md contents
  [ ] instance-terminated confirmation page
```

All of these together = undeniable proof of execution. Worth keeping.

**Success story format**:

```
  "I trained a GPT-2-grade chatbot from scratch.

   Compute: 8×H100 for 3h 35min = 29 GPU-hours.
   Cost: $89 on Lambda on-demand.
   Architecture: Llama-style d24 (1.5B params, 32768 vocab).
   Data: 41 B tokens from DCLM.
   Metrics: CORE 0.258, MMLU 33%, GSM8K 10%.
   Outcome: works as a chatbot, can answer basic questions.

   Link to report: <wandb URL> / <github repo with writeup>
   Conversation screenshots: attached.
   Total commands run: `bash runs/speedrun.sh` + `python -m scripts.chat_web`.
   Total time spent: 5 hours (3.5h training + 1.5h setup/cleanup)."
```

This is your artifact. Post it. Share it. Put it on your CV.

## Common pitfalls

- **Forgetting to kill the instance.** I cannot say this enough.
- **SSH drops**: solved by `tmux`.
- **Forgetting `WANDB_RUN=`**: speedrun runs without logging, you lose the metrics. Always set it.
- **OOM at step 1**: reduce `--device-batch-size`.
- **No disk space**: delete `~/.cache/nanochat/data/` shards after pretraining, they're huge.
- **SSH too slow to type**: use `tmux`'s `C-b d` to detach, don't SSH in and out.

## Safety reminders

- Never `rm -rf /` or similar on the cloud instance. (It's not YOUR machine, but it's also not free.)
- Don't commit wandb tokens or API keys to git.
- Keep SSH keys private. Don't paste them anywhere.

## What you've now earned

After this capstone you have:
- Personal experience renting, running, and killing GPUs.
- First-hand cost intuition.
- A trained model + report + wandb dashboard + conversation screenshot.
- The right to say "I've trained an LLM on real cloud infrastructure."

Rare credentials. Keep going.

## Next

Module 7. Open `07_research_and_papers/01_how_to_read_a_paper.md`.
