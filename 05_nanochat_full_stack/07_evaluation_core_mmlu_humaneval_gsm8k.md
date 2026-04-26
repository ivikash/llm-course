# 5.7 - Evaluation: CORE, MMLU, HumanEval, GSM8K

A trained model is useless without evaluations. This lesson explains each benchmark nanochat uses, what it measures, how to interpret the numbers, and where to find the code.

## Why benchmarks matter

During training, val loss drops. That's nice, but it tells you nothing about whether the model can:
- Answer trivia.
- Reason step-by-step.
- Write code.
- Do math.
- Follow instructions.

Benchmarks probe specific capabilities with standardized tests. They're imperfect (every benchmark has flaws) but they're the best communal yardstick we have.

## CORE (the primary nanochat metric)

**Source**: [DCLM paper (Apple, 2024)](https://arxiv.org/abs/2406.11794).

**What it is**: an aggregate score across ~40 diverse tasks - a composite benchmark designed to correlate with downstream usefulness.

**How computed**: for each subtask, compute (raw_accuracy - random_baseline) / (best_known - random_baseline), then average. So 0.0 = random, 1.0 = SOTA. nanochat reports values in the 0.25-0.27 range (roughly GPT-2 level).

**Implementation**: `nanochat/core_eval.py`. Expensive: ~30 min on 8xH100.

**Why nanochat uses it**: balanced, composite, widely reported.

## MMLU

**Source**: [Hendrycks et al, 2020](https://arxiv.org/abs/2009.03300). Massive Multitask Language Understanding.

**What**: 57 subjects (history, physics, law, medicine, ...) with multiple-choice questions.

**How scored**: accuracy. Random baseline = 25% (4 choices).

**Scale**:
- Random: 25%
- GPT-2 small: ~27% (barely above random)
- GPT-3: 44%
- GPT-4: 86%
- Human expert: ~90%

**Implementation**: `tasks/mmlu.py`. Loads from HuggingFace, formats each question as a prompt, asks the model to pick A/B/C/D.

**Interpretation**: measures breadth of world knowledge. Small models are bad at it. Scaling helps more than fine-tuning.

## ARC

**Source**: [Allen AI, 2018](https://arxiv.org/abs/1803.05457). AI2 Reasoning Challenge.

**What**: grade-school science multiple choice. Two splits: Easy (simpler) and Challenge (harder).

**How scored**: accuracy.

**Scale**:
- Random: 25%
- GPT-2: ~30%
- GPT-4: ~95%.

**Implementation**: `tasks/arc.py`.

**Interpretation**: tests basic scientific reasoning. Small models can pass Easy, Challenge is hard.

## HumanEval

**Source**: [OpenAI, 2021](https://arxiv.org/abs/2107.03374).

**What**: 164 Python programming problems. Given a function signature and docstring, complete the function. Evaluated by running unit tests.

**How scored**: pass@k (what fraction of problems get at least one passing solution in k attempts). Reported pass@1 and pass@10.

**Scale**:
- GPT-3: 0% (not good at code)
- Codex: 29% pass@1
- GPT-4: 67% pass@1
- Claude 3.5 Sonnet: ~92% pass@1

**Implementation**: `tasks/humaneval.py`. For each problem, the model generates, then `nanochat/execution.py` runs the tests in a sandbox.

**Interpretation**: measures code generation. Hard for small models because code requires exactly-right syntax.

## GSM8K

**Source**: [OpenAI, 2021](https://arxiv.org/abs/2110.14168).

**What**: 8.5K grade-school math word problems. Multi-step arithmetic reasoning. Open-ended answer (a number).

**How scored**: exact match on the final answer.

**Scale**:
- GPT-3 (without chain of thought): ~15%
- GPT-3 with chain-of-thought prompt: ~57%
- GPT-4: ~92%

**Implementation**: `tasks/gsm8k.py`. Uses chain-of-thought prompting.

**Interpretation**: measures step-by-step reasoning. Dramatic gains from RL (as we saw in Lesson 5.6).

## SpellingBee and CustomJSON

Custom nanochat tasks:

**SpellingBee**: given a set of 7 letters, produce as many valid words as possible. Non-trivial for LLMs since vocab is at BPE token level, not character level.

**CustomJSON**: given a description, produce a valid JSON output matching the schema. Tests format compliance.

Both in `tasks/`. Show you how to write your own.

## The eval scripts

### `scripts/base_eval.py`
- Val BPB on held-out pretraining data.
- CORE score.
- Sample a few completions for qualitative check.

### `scripts/chat_eval.py`
- All the benchmarks above.
- Sweeps: MMLU, ARC easy/challenge, HumanEval, GSM8K, SpellingBee, CustomJSON.
- Takes 1-3 hours on 8xH100 for a typical nanochat model.

### `scripts/tok_eval.py`
- Tokenizer compression and round-trip tests.

## How evaluations actually run

For multiple-choice (MMLU, ARC):

```python
# approach 1: score-based (what nanochat uses)
for q in questions:
    prompt = format_question(q)
    # for each choice A/B/C/D, compute log-probability
    scores = {c: model.log_prob(prompt + " " + c) for c in choices}
    predicted = max(scores, key=scores.get)
    correct = predicted == q.answer
```

This is "likelihood-based" MCQ evaluation. Avoids needing the model to actually produce "A", "B", "C", or "D" as text.

For open-ended (HumanEval, GSM8K):

```python
# sample up to N completions, check each
for q in questions:
    completions = [model.generate(q.prompt, temp=0.2) for _ in range(N)]
    passed = any(check_correct(c, q.answer) for c in completions)
```

## Pitfalls in evaluation

1. **Contamination**: if MMLU is in your training data, your score is invalid. Real worry for frontier models trained on the whole internet.

2. **Prompt format sensitivity**: "A) X" vs "A. X" vs "(A) X" can change MMLU scores by 5%. Most reported scores use one specific format.

3. **Tokenization quirks**: for open-ended, the model might say "4.0" when the answer is "4". String comparison fails. Requires careful normalization.

4. **Sampling sensitivity**: temperature, top_k change pass@k.

5. **Chain-of-thought**: with vs without changes GSM8K dramatically. Scores are only comparable within the same setting.

6. **Reporting games**: "pass@100" sounds better than "pass@1". Always check what's being reported.

Every benchmark paper has a "how to evaluate" section. Read it.

## Beyond standard benchmarks

Large labs run:
- **Chatbot Arena**: human pairwise preferences over 1000s of conversations. Gold standard for chat quality.
- **Private internal evals**: custom, domain-specific, held-out from the world. The real measurement.
- **Long-context tests**: needle-in-haystack, RULER, LV-Eval.
- **Vibe-checks**: senior researchers playing with the model and noting impressions.

Public benchmarks catch gross failures. Real product quality is measured elsewhere.

## Visualize this

**Benchmark scale intuition**:

```
  MMLU (57 subjects, 4-choice):
  ─────────────────────────────
  random              25%
  GPT-2 (124M)         ~27%   ← barely above random
  GPT-3               44%
  GPT-2 XL (1.5B)      ~30%
  Llama-2 7B           46%
  Llama-2 70B          69%
  GPT-4                86%
  human expert         ~90%

  nanochat d24 (1.5B)  ~30-40%

  HumanEval (code, pass@1):
  ─────────────────────────
  random              0%
  GPT-2                ~0%     ← barely any Python in training
  Codex               ~29%
  Llama-2 70B          31%
  GPT-4                67%
  Claude 3.5 Sonnet    ~92%

  nanochat d24         ~5-15%

  GSM8K (grade-school math, chain-of-thought):
  ─────────────────────────────────────────────
  random              0%
  GPT-3                ~15%
  GPT-3 + CoT           ~57%
  GPT-4                92%

  nanochat SFT         ~10-15%
  nanochat post-RL     ~40-50%   ← RL on GSM8K makes a big difference
```

**What your model's CORE number means**:

```
  CORE score (DCLM paper aggregate, 0 to 1 scale):

  0.00 ─ random baseline
  0.15 ─ early language models
  0.22 ─ GPT-2 small territory
  0.25 ─ GPT-2 XL territory  ← speedrun target
  0.30 ─ Mistral-7B level
  0.45 ─ Llama-2 70B level
  0.60 ─ GPT-4 level

  nanochat's goal: beat 0.25 in as little wall time as possible.
```

**How multiple-choice eval actually works**:

```
  Question: "What is the capital of France?"
  Choices: A) Madrid  B) Paris  C) Rome  D) London

  Method 1: ask the model to output the letter
    Prompt: "Question: ... Answer: "
    Model generates: "B"
    Check if "B" is the answer.

    Problem: sometimes model outputs "Paris", or "B)" or "The answer is B".
    Parsing is fragile.

  Method 2: likelihood-based (what nanochat uses)
    For each choice, compute:
      P("Madrid" | prompt), P("Paris" | prompt), P("Rome" | prompt), P("London" | prompt)
    Pick the choice with highest probability.

    More robust. No parsing needed.
```

**The eval suite you'll run after training**:

```
  torchrun ... -m scripts.chat_eval

  Runs through:
    ✓ MMLU (30 min on 8xH100)          → composite accuracy
    ✓ ARC-Easy (5 min)                  → accuracy
    ✓ ARC-Challenge (5 min)             → accuracy
    ✓ HellaSwag (5 min)                 → accuracy
    ✓ GSM8K (20 min, needs generation)  → exact match
    ✓ HumanEval (10 min, needs code exec) → pass@1, pass@10
    ✓ SpellingBee (custom, 5 min)
    ✓ CustomJSON format (5 min)

  Output:
    metric               value
    ──────────────────── ──────
    mmlu                 0.334
    arc_easy             0.512
    arc_challenge        0.287
    hellaswag            0.448
    gsm8k                0.103
    humaneval            0.056
    spellingbee          0.634
    customjson           0.912
    CORE                 0.251  ← the headline number
```

Logged to wandb and report.md.

**Benchmark contamination visualized**:

```
  Training data              Test set
  ──────────────             ─────────────
                               MMLU question:
   "The capital of           "What is the capital of
   France is Paris."   ←──── France?"

   If this appears in         Model saw the answer
   training data,             during training.
   model's score is           Benchmark is corrupted.
   misleadingly high.
```

Frontier labs worry about this constantly. For academic benchmarks, modern LLMs are likely partially contaminated. "Uncontaminated eval" is a hot research area.

## Exercises

1. Open `tasks/mmlu.py`. Read the `iterate` function. See the multiple-choice format.

2. Open `nanochat/core_eval.py`. Skim the list of sub-tasks it runs. ~30 small eval datasets combined.

3. Open `tasks/humaneval.py`. See how it: generates code, extracts it, passes to `execution.py` for sandboxed running.

4. Open `nanochat/execution.py`. Note: it's a sandbox for running model-produced Python safely. Critical for any LLM-does-code evaluation.

5. Read `dev/LEADERBOARD.md`. Note that every line has train time, val_bpb, and CORE score. That's the metric triple to follow.

## Next

`08_inference_engine_kv_cache.md` - making generation fast.
