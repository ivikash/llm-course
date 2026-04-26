# 7.6 - RLHF, InstructGPT, and DPO

Going from "base model" to "helpful assistant" is the hardest stretch of the LLM pipeline. This lesson covers three techniques: RLHF (the original), DPO (simpler), GRPO (for verifiable tasks).

## The problem

A base LLM pretrained on web text is a **text completer**, not an assistant.

Prompt it with "How do I write a Python function?" and it might respond:
- "...is a common question. Here we describe three approaches..." (writing an article)
- "...? I've been trying to learn Python and..." (continuing as a forum post)
- "...use def, followed by..." (actually answering)

Random among these. You want option 3 consistently.

## Three stages of alignment (OpenAI's 2022 recipe)

### Stage 1: SFT (Supervised Fine-Tuning)

Collect ~10K prompts. Humans write good responses. Fine-tune the model on these (prompt, response) pairs.

Loss: standard cross-entropy on response tokens. Masked so loss only counts on response tokens (not prompt tokens).

Result: model learns "when prompted like a user, respond like an assistant."

Covered extensively in Module 5 Lesson 5.

### Stage 2: Reward Model

SFT alone is okay but limited. Hard to scale human demonstration. Ratings are easier.

Collect ~50K prompts. For each, sample 2+ responses from the SFT model. Humans **rank** them (which is better: A or B?).

Train a **reward model** (RM): take a prompt + response, output a scalar "quality" score. Loss: prediction should prefer the human-preferred response over the other.

```python
loss = -log(sigmoid(r_chosen - r_rejected))
```

Now the RM is a cheap, fast estimate of human preference.

### Stage 3: RL with PPO against the RM

Use PPO (Proximal Policy Optimization) to train the LM to maximize the RM's score:

```
For each step:
  1. Sample a batch of prompts.
  2. Generate a response from the current model.
  3. Score it with the RM.
  4. Compute policy gradient: make high-reward responses more likely.
  5. Add a KL penalty against the SFT model (prevent drift).
  6. Optimizer step.
```

**Key insight: RLHF lets humans influence the model via preferences (easy) rather than demonstrations (hard).**

### The InstructGPT paper results

SFT alone: beats base model.
SFT + RM + PPO: beats SFT alone by a lot, per human evaluators.

The 1.3B InstructGPT was preferred over the 175B vanilla GPT-3 - a tiny model outperforming a much bigger one on helpfulness, because it was aligned.

This is the algorithm that turned GPT-3 into ChatGPT.

## The problems with classical RLHF

1. **PPO is fiddly**: many hyperparameters, easy to diverge.
2. **Reward model drift**: the LM learns to game the RM (adversarial). KL penalty helps but doesn't fully solve.
3. **Requires three models in memory**: policy, reference (for KL), reward model. Expensive.
4. **Human-label cost**: 50K comparisons is not trivial.

Enter DPO.

## DPO (Direct Preference Optimization, 2023)

Rafailov et al., Stanford. Elegant math.

Core claim: you don't need the RM at all. The loss function for RLHF can be reformulated to train directly on preference pairs.

```
L_DPO = -log sigmoid(β × [log π(y_good|x) - log π_ref(y_good|x)
                        - log π(y_bad|x) + log π_ref(y_bad|x)])
```

Where π is the current model, π_ref is the frozen SFT model.

### How to interpret

- Make `y_good` more likely.
- Make `y_bad` less likely.
- Both relative to the reference model (don't drift wildly).
- β controls how strongly to prefer `y_good` over `y_bad`.

No reward model. No PPO. No rollouts. Just a loss function on (prompt, good, bad) triples. Train like normal SFT.

### Why DPO won in open source

- Vastly simpler to implement.
- Stable.
- Cheaper (one forward pass per update instead of many rollouts).
- Often as good as RLHF+PPO empirically, maybe slightly worse on some tasks.

Used in: Zephyr, Tulu, Starling, most open-source chat models 2023+.

## GRPO (Group Relative Policy Optimization, 2024)

DeepSeek's innovation for RL on verifiable tasks (math, code).

Difference from DPO/RLHF:
- No humans. Use programmatic rewards (correctness checkers).
- No reward model - use 0/1 correctness directly.
- No value network (no "baseline model" to estimate expected reward).
- Compute advantage per rollout as "relative to the group mean."

```
For each prompt:
  Generate K rollouts.
  Reward each (0 or 1 based on correctness).
  Advantage[i] = (reward[i] - mean(rewards)) / std(rewards)
  Policy gradient: ∇ log π(y_i | x) × advantage[i]
```

See Module 5 Lesson 6 for the full walkthrough and nanochat code.

## When to use which

| Technique | Use when |
|-----------|----------|
| **SFT** | Always, as foundation |
| **DPO** | You have preference data (human rankings) |
| **RLHF/PPO** | You have preferences + classical alignment R&D infrastructure |
| **GRPO** | You have **verifiable** reward signals (math, code, structure) |

Most modern frontier models use some combination: SFT + DPO + GRPO.

## The Anthropic variation: Constitutional AI

Claude uses "Constitutional AI" (Bai et al., 2022): the model critiques its own outputs against a set of principles, and RLHF trains on those self-preferences. Variation on RLHF with less human labor.

## Alignment is ongoing research

- **Rejection sampling** as an alternative to RL: sample N, take the best per RM, SFT on that.
- **Iterative DPO**: do DPO, generate new preferences with the new model, repeat.
- **RLAIF** (RL from AI Feedback): use a capable LM to generate preferences.
- **Process rewards**: in multi-step reasoning, reward each intermediate step.

The field is active. Expect this landscape to change every 6 months.

## Exercises

1. Read [InstructGPT paper](https://arxiv.org/abs/2203.02155) abstract + Figure 1. Internalize the three-stage pipeline.

2. Read [DPO paper](https://arxiv.org/abs/2305.18290) intro + equation 7 (the loss). Derive it from RLHF algebraically... or don't, and just use it.

3. In nanochat's `chat_rl.py`, find where the GRPO loss is computed. Map back to the formulas here.

4. Think: for the problem "make a chatbot that responds humorously", which alignment technique is best? (SFT on curated examples; DPO on humor preferences. Verifiable reward is hard since humor is subjective.)

## Next

`07_modern_architectures.md` - MoE, state-space, hybrids.
