# 5.5 - SFT: Supervised Fine-Tuning

Time to read `scripts/chat_sft.py` and understand how a base model becomes a chat model.

## Launch

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
    --device-batch-size=16 --run=$WANDB_RUN
```

Differences from `base_train.py`:
- Starts from the base checkpoint (not scratch).
- Uses a much smaller training corpus.
- Much shorter training (1 epoch vs many steps).
- Lower LR (preserve pretrained capabilities).
- Uses masked loss.

## The script structure

Top-level (~300 lines):
1. Parse args.
2. Init DDP.
3. Load tokenizer.
4. Load the base model from the latest base checkpoint.
5. Load SFT data: SmolTalk + identity conversations + (small amounts of) task data.
6. Tokenize all conversations up front (the dataset is small enough).
7. Build batches: pack multiple conversations into each 2048-length sequence to minimize padding waste.
8. Training loop (shorter than pretraining):
   - Forward → compute masked loss → backward → step.
   - Lower LR (e.g. 5e-5 vs 5e-4 for pretraining).
   - Usually 1 epoch over the SFT data.
9. Save SFT checkpoint (separate from base).
10. Run eval (MMLU, ARC, GSM8K, HumanEval) via chat_eval.

## The data loading

SFT data is small enough (~millions of examples, not billions) to fit entirely in RAM. So:

```python
# load all conversations
convs = list(load_smoltalk()) + list(load_identity()) + list(load_task_data())
# tokenize
tokenized = [tokenizer.encode_chat_with_mask(c) for c in convs]
# random shuffle
random.shuffle(tokenized)
# pack into sequences of length seq_len
batches = pack_conversations(tokenized, seq_len=2048)
```

The "pack" step concatenates multiple short conversations into a full-length sequence, separated by `<|bos|>`. This avoids wasting compute on padding tokens.

Attention masking: in a packed sequence, each conversation should NOT attend to other conversations. nanochat handles this via an attention mask that resets at `<|bos|>` boundaries. See `attention_fallback.py` test file.

## The loss with masking

```python
# logits: (B, T, V)
# targets: (B, T)
# loss_mask: (B, T), 1 where we compute loss

# per-position loss
losses = F.cross_entropy(logits.view(-1, V), targets.view(-1), reduction='none')  # (B*T,)
losses = losses.view(B, T) * loss_mask
loss = losses.sum() / loss_mask.sum()
```

Standard pattern.

## LR schedule during SFT

Typically:
- Even lower LR than pretraining (1e-5 to 1e-4).
- Short warmup (few hundred steps).
- Linear or cosine decay.

Why so low? A high LR would overwrite the knowledge from pretraining. The model should gently **adjust** its behavior, not relearn from scratch.

## How long should SFT run?

Usually 1-3 epochs over your SFT data. Watch val loss on a held-out chat set; stop when it starts rising (overfitting is common here - SFT data is small).

For nanochat's speedrun: ~30 minutes of SFT after a 3-hour pretraining. SFT takes about 15% of total compute.

## Sanity checks during SFT

Log every N steps:
- Example assistant completion on a fixed prompt ("Explain why the sky is blue").
- Watch it improve qualitatively over SFT steps.
- Watch for degenerate behaviors: repetition, incoherence, infinite loops, lost stop token.

## After SFT

You have a chat model. Launch `python -m scripts.chat_cli` and talk to it:

```
You> Hello!
Assistant> Hi! How can I help you today?
You> Tell me a joke.
Assistant> Why don't scientists trust atoms? Because they make up everything.
```

It's basic but real. For a ~300M param model trained for 3 hours on 8xH100, that's something.

## Common failure modes

1. **Model drifts off chat format**: it produces `<|user_start|>` tokens itself. Cause: loss mask bug, or too high LR overwriting format knowledge.

2. **Model refuses everything**: trained on data that was too safety-tuned without corresponding positive examples.

3. **Model hallucinates identity**: didn't get enough identity examples, or they were inconsistent.

4. **Model loses general knowledge**: SFT was too long or LR too high - it forgot pretraining.

5. **Model repeats in generation**: something about sampling at inference time, not necessarily SFT's fault. See Lesson 3.10 on sampling.

## Relationship to RLHF

SFT is the first step in the "align a base model" pipeline. The original OpenAI pipeline is:

```
Base (pretrained)
    → SFT (instruction-following)
    → RM (reward model trained on human preferences)
    → RLHF via PPO (policy optimization against RM)
```

nanochat does:
```
Base → SFT → RL via GRPO (optional, on verifiable rewards)
```

Skipping the RM stage because GRPO works on problems where you can automatically score correctness (math, code). We cover this in the next lesson.

## Exercises

1. Read `scripts/chat_sft.py` top to bottom. Compare structure to `base_train.py`. What's different?

2. Find the loss masking code. Verify it only averages loss over assistant tokens.

3. Open `tasks/smoltalk.py`. What's the iteration interface? How big is the dataset?

4. Open `tasks/common.py`. What utilities are shared across tasks?

5. After running an SFT on a smaller model, use `scripts.chat_cli` to talk to it. Rate: does it follow instructions? Does it have the nanochat identity? Does it hallucinate?

## Next

`06_rl_and_preferences.md` - going beyond SFT with reinforcement learning.
