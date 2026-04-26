# Module 5 Capstone - Train Your Own Chatbot

Train a tiny end-to-end chatbot and talk to it in a web browser. This is the moment where you cross from "student of LLMs" to "practitioner of LLMs."

## Two paths

### Path A: Cloud (recommended for realism)

Rent an 8xH100 instance from Lambda (https://lambda.ai) for ~2-3 hours. Cost: ~$50.

```bash
# on the rented instance
cd
git clone https://github.com/karpathy/nanochat.git  # or transfer your copy
cd nanochat
bash runs/speedrun.sh
```

Wait 3 hours. Watch wandb. When done:

```bash
python -m scripts.chat_web
```

Visit the IP in your browser. Talk to your model. Save the report.md.

**Kill the instance when done.** Set a calendar reminder.

### Path B: Local (small scale)

If you have a single GPU (or are willing to wait on CPU), use `runs/runcpu.sh` as your template. Tiny depth, tiny batch, runs for hours on CPU or ~30 min on a consumer GPU.

```bash
cd ~/workspace/nanochat
source .venv/bin/activate    # after uv sync --extra cpu

# Run a minimal variant:
bash runs/runcpu.sh
```

After it's done, run:

```bash
python -m scripts.chat_web
```

Quality will be poor but it'll chat.

## Success criteria

At the end:
1. [ ] Report.md generated with all stages.
2. [ ] You can launch `chat_web` and talk to the model.
3. [ ] You have screenshots of the conversation.
4. [ ] You have the wandb run URL (if used).
5. [ ] You can explain every stage to a non-technical friend.

## What to actually try with your model

Once you're chatting, probe:

1. **Who are you?**
   - Goal: check the identity conversations took. Expect: "I'm nanochat..."
   - Red flag: random LLM names like "I'm an AI developed by OpenAI".

2. **What's 2+2?**
   - Goal: check it follows the chat template correctly, stops at the end.
   - Red flag: keeps generating, or produces `<|user_start|>` tokens.

3. **Tell me a joke.**
   - Goal: creative generation.
   - Expect: often bad, but structurally coherent.

4. **What is the capital of France?**
   - Goal: factual knowledge.
   - For a 300M model: often Paris, occasionally hallucinated.

5. **Write a Python function to reverse a string.**
   - Goal: code. Expect: mostly right syntax, maybe buggy.

6. **Why is the sky blue?**
   - Goal: can it explain. Expect: vaguely correct, short.

7. **Tell me about yourself.**
   - Open-ended, see personality consistency.

## The postmortem

Write `mychatbot_postmortem.md`:

```markdown
# My Chatbot - Postmortem

## Setup
- Hardware: [8xH100 / local GPU / CPU]
- Config: depth=X, device_batch_size=Y
- Cost: $Z (if cloud)
- Wall time: H hours M minutes

## Metrics at end of training
- val_bpb: 
- CORE score: 
- MMLU: 
- ARC-Easy: 
- GSM8K: 
- HumanEval:

## Qualitative
- Best conversation I had: [paste excerpt]
- Worst failure mode observed: [describe]
- Does it follow chat format cleanly? [yes/no]
- Does it stop correctly? [yes/no]
- Factual accuracy impression (1-10):
- Coherence impression (1-10):
- Helpfulness impression (1-10):

## Compared to expectations
- [...]

## What I would change next time
- [...]

## Three things I now understand that I didn't before
- [...]
```

This is the kind of artifact you'd produce after a real research experiment. Keep it. Show it to people. It's evidence of capability.

## Extensions if you have more time/budget

### A. Sweep depth

Train models at depth 8, 12, 16, 20. Plot CORE vs depth. You've just produced a tiny scaling-law plot.

### B. Try RL

After SFT, run `scripts/chat_rl.py` on GSM8K. Compare GSM8K accuracy before/after. You should see a jump.

### C. Custom SFT data

Replace SmolTalk with your own SFT dataset (maybe conversations in your domain, or in a different language). See if you can produce a model with a specific specialty.

### D. Longer pretraining

Change `target-param-data-ratio` to 20 (Chinchilla-optimal) or even 40 (overtrain). Compare val_bpb and CORE. Which gives you the best loss per dollar?

### E. Contribute to the leaderboard

Read `dev/LEADERBOARD.md`. Try to beat the current best speedrun. Submit a PR if you do. This is a legitimate open-source contribution to LLM research.

## What you've learned, specifically

After completing this capstone:

- You have personally executed every stage of a modern LLM pipeline.
- You know what each script does and why.
- You've watched training curves and developed intuitions about what "healthy" looks like.
- You've served a live chat UI.
- You have numbers to quote: "My d24 model got CORE 0.25 after 3 hours on 8xH100 for $50."

This is a rare credential. 99% of people who work with LLMs have never done this. You now have.

## Moving forward

Module 6 deepens your infrastructure knowledge - cloud, distributed, scaling laws, cost engineering. After that, Module 7 teaches you to read papers; you'll be able to decode any new LLM paper in under an hour. Module 8 is about leading the work.

You're past the hardest hill. Keep going.

## Next

Module 6. Open `06_infra_and_scaling/01_cpu_vs_gpu_vs_tpu.md`.
