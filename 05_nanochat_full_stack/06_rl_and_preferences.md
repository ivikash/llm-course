# 5.6 - RL and Preferences: From RLHF to GRPO

SFT gets you a working chatbot. RL (reinforcement learning) makes it better - specifically on tasks where you can measure quality automatically.

## The landscape

Three major RL-for-LLMs techniques:

### 1. RLHF with PPO (the classical OpenAI pipeline)

1. Collect human preferences: for many prompts, humans rank pairs of model responses.
2. Train a **reward model** (RM): a model that predicts which response a human would prefer.
3. Use **PPO** (Proximal Policy Optimization) to train the base LLM to maximize the RM's score.

This was InstructGPT (2022), which became ChatGPT. Expensive: needs a lot of human labeling.

### 2. DPO (Direct Preference Optimization, 2023)

Skip the RM entirely. Use preference pairs `(x, y_good, y_bad)` directly in a loss:

```
loss = -log(sigmoid(β * (log π(y_good|x) - log π_ref(y_good|x) - log π(y_bad|x) + log π_ref(y_bad|x))))
```

The model learns to make `y_good` more likely and `y_bad` less likely, while staying close to a reference model. Simpler, cheaper, often just as good. Widely adopted in the open-source community (Zephyr, Tulu).

### 3. GRPO (Group Relative Policy Optimization)

Popularized by DeepSeek (2024). Works on **verifiable rewards** - problems where a machine can check if the answer is correct.

Examples of verifiable domains:
- Math: does the answer match the ground truth?
- Code: does the code pass the unit tests?
- Format: did the response follow the required structure?

For each prompt, generate K samples. Check correctness of each. Update the model to make correct samples more likely, incorrect ones less likely, **relative to the group average**. No reward model, no human labels - pure bootstrap from known-answer problems.

This is what nanochat uses, specifically on GSM8K (grade-school math).

## How GRPO works, concretely

For each training step:

```
1. Sample a batch of prompts from GSM8K.
2. For each prompt, generate K=16 different responses.
3. For each response, extract the final numeric answer.
4. Check if it matches the ground truth. reward[i] ∈ {0, 1}.
5. Compute advantage: A[i] = (reward[i] - mean(reward)) / std(reward)
6. Policy loss: maximize A[i] * log π(response[i] | prompt).
7. Add KL penalty to keep model close to pre-RL reference (prevents collapse).
8. Backward, step.
```

The "relative" in GRPO: advantage is computed *within the group* of K samples, not against a baseline value model.

## nanochat's `chat_rl.py`

Open `~/workspace/nanochat/scripts/chat_rl.py`. ~500 lines. Structure:

1. Load SFT checkpoint (starting point for RL).
2. Load GSM8K prompts via `tasks/gsm8k.py`.
3. Training loop:
   - Sample batch of prompts.
   - Generate K rollouts per prompt (using `nanochat/engine.py` for fast generation).
   - Parse answers. Compute 0/1 rewards.
   - Compute advantages per rollout.
   - Forward the policy on each rollout, compute log probs.
   - Compute GRPO loss + KL penalty.
   - Backward, step.
4. Periodically eval on GSM8K test set.

## Key mechanics

### Rollouts

Generation is expensive. Each GRPO step generates K complete responses. Uses `engine.py` which has KV cache for speed.

### Reward parsing

```python
# tasks/gsm8k.py
def extract_answer(text):
    # find "#### <number>" pattern or last number
    m = re.search(r'####\s*([-\d,.]+)', text)
    if m:
        return float(m.group(1).replace(',', ''))
    # fallback: last number in response
    nums = re.findall(r'[-\d,.]+', text)
    if nums:
        return float(nums[-1].replace(',', ''))
    return None

def compute_reward(response, ground_truth):
    predicted = extract_answer(response)
    return 1.0 if predicted == ground_truth else 0.0
```

Simple string-matching reward. Hacky, but works for math.

### Advantage normalization

```python
rewards = torch.tensor([r1, r2, ..., rK])   # K samples per prompt
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
```

This means: if all K samples got the right answer, no gradient signal (they're all "normal"). If 1 of 16 got it right, that one gets a big positive advantage. If 15 of 16 got it right, the wrong one gets a big negative advantage.

### KL penalty against reference

To prevent the model from drifting wildly during RL (which destroys unrelated capabilities):

```
loss = -E[A * log π(y|x)] + β * KL(π || π_ref)
```

Where π_ref is the SFT model frozen. The KL term keeps π close to π_ref in general, but allows drift where there's strong reward signal.

β is a hyperparameter, typically 0.01-0.1.

### Clipping (from PPO, adopted in GRPO)

To prevent big update steps:

```
ratio = π(y|x) / π_old(y|x)
L_clip = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
```

Prevents over-exploitation of the current policy estimate. Standard RL hygiene.

## How much does it help?

For GSM8K specifically: SFT might get 15% accuracy, GRPO-RL on top can get 40-50%. Dramatic gains on the target task, especially for small models where the headroom is large.

Caveat: RL can cause regressions on other tasks (MMLU might drop slightly) if not balanced. Always eval the full benchmark suite after RL.

## Why not RL everything?

- Need verifiable rewards. Not available for creative writing, open-ended conversation, etc.
- RL is unstable; bad tuning collapses the model.
- Computationally expensive (many rollouts per step).
- The last 5% of quality often requires DPO/RLHF on preference data, which is another beast.

Modern frontier models combine: pretrain → SFT → DPO (or RLHF) → GRPO on verifiables.

## `tasks/gsm8k.py` deep dive

Open it. ~150 lines. Defines:
- Dataset loader (from HuggingFace).
- Prompt format: "Question: ... Let's solve step by step."
- Answer extractor.
- Reward function.

This is a **task** in nanochat's vocabulary: a module that provides prompts and a reward function. Tasks are composable - you could add your own custom task (spelling bee, JSON output format, etc.) by imitating this structure.

See also:
- `tasks/humaneval.py` - code generation with test-passing as reward.
- `tasks/spellingbee.py` - a custom letter-based puzzle.
- `tasks/customjson.py` - structured output format compliance.

## Exercises

1. Read `scripts/chat_rl.py` top to bottom. Don't worry about understanding every line - note the overall structure matches what I described.

2. Find where rollouts are generated. Is it a call into `engine.py`? (Yes - see if you can trace it.)

3. Find the advantage normalization code. See it group by prompt.

4. Read `tasks/gsm8k.py`'s `extract_answer` function. Identify edge cases it handles.

5. In `tasks/humaneval.py`, find the reward function. It's: run the code, check if tests pass.

6. Think: What would it take to add a task `tasks/wordle.py` where the model plays Wordle? Describe the reward function.

## Next

`07_evaluation_core_mmlu_humaneval_gsm8k.md` - the benchmark suite and what each benchmark means.
