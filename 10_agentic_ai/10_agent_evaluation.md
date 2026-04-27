# 10.10 - Agent Evaluation

Agents are hard to evaluate. They have branching behavior, multi-step trajectories, partial successes, and non-deterministic outputs. This lesson covers how to measure agent quality rigorously.

## Why evaluation is hard

An LLM evaluation is: prompt → output → score.

An agent evaluation is: task → (agent loops through many states, actions, tool calls) → final outcome + trajectory.

Complications:
- **Non-determinism**: same task, different runs, different paths.
- **Partial success**: task half-completed counts how much?
- **Trajectory matters**: getting the right answer via 20 wrong steps is bad.
- **Cost varies**: some solutions use 100 tokens, others 10000.
- **Human judgment often needed**: "is this a good answer?" is subjective.

## What to measure

### 1. Task success rate

The simplest: did the agent complete the task correctly?

```
success_rate = (tasks completed correctly) / (total tasks)
```

Requires a labeled test set (ground-truth answers or verifiers).

For code: did tests pass? For QA: does answer match gold? For browsing: did final page state match?

### 2. Step-level correctness

For multi-step tasks, score each step independently.

```
step_accuracy = (correct steps) / (total steps)
```

Useful for diagnosing: "agent fails at step 3 of 5, because tool X isn't handled."

### 3. Efficiency metrics

- **Steps to solution**: how many tool calls?
- **Tokens consumed**: input + output.
- **Wall-clock time**: from query to done.
- **Dollar cost**: LLM fees + tool fees.

A task solved in 5 steps is better than the same task solved in 50.

### 4. Robustness

Does the agent handle perturbations?
- Input slightly rephrased.
- Tool temporarily unavailable.
- Network slow.

Test by running the same task with variations.

### 5. Faithfulness / groundedness

For RAG agents: does the answer stick to the retrieved docs, or hallucinate?

Tools: RAGAS, TruLens score this automatically by checking whether claims in the answer are supported by the retrieved context.

### 6. Safety

- Does the agent refuse inappropriate requests?
- Does it ask for confirmation on high-stakes actions?
- Does it leak sensitive info?

Red-team with adversarial prompts.

## Benchmarks

**Interactive benchmark scorecard — explore 8 benchmarks × 8 models:**

```viz
{"viz": "agent_eval_scorecard"}
```

Hover a row for the benchmark description. Click a model column to highlight. Slide the year slider to see capability evolution. Note how ARC-AGI stayed near zero until o3 cracked it.


### General

**GAIA** (Mialon 2023): 466 complex real-world questions requiring tool use, reasoning, multi-modal. Gold standard for general assistants.

Leaderboard: https://huggingface.co/spaces/gaia-benchmark/leaderboard

### Software engineering

**SWE-Bench** (Jimenez 2023): 2,294 real GitHub issues from major Python repos. Agent must produce a PR that fixes the issue and passes tests.

- SWE-Bench Lite: 300 easier tasks.
- SWE-Bench Verified: 500 human-verified tasks.

Top 2025 scores: ~60% solve rate (Devin, Claude). Humans: 100%.

### Web

**WebArena** (Zhou 2023): 812 tasks on 5 live websites (Reddit clone, Shopping, Social forum, GitLab, Classifieds).

Top scores: ~40% in 2025. Up from 15% in 2023.

**Mind2Web** (Deng 2023): cross-website tasks.
**WebVoyager**: real websites, multimodal.

### Agentic reasoning

**AgentBench** (Liu 2023): suite of tasks across OS, DB, Web, Games.
**ToolEval**: tool-use correctness.
**WorkBench**: workplace tasks.

### Code-specific

**HumanEval**: 164 coding problems (also used for LLMs).
**MBPP**: simpler.
**LiveCodeBench**: contamination-free, updated.

## Custom eval design

For your specific agent, build a custom eval. Steps:

### Step 1: define success

Write 20-50 tasks. For each, write the success criterion:
- Exact match on final answer.
- Programmatic check (test passes, API returns 200, file exists).
- LLM-as-judge ("Does this answer satisfy the user's request?").

### Step 2: capture trajectories

Log:
- Each LLM input / output.
- Each tool call + args + result.
- Final answer.
- Cost, tokens, time.

Run each task. Store trajectories.

### Step 3: score

Automated scoring where possible. LLM-as-judge for fuzzy cases.

```python
def score_task(task, trajectory):
    return {
        "success": check_success(task, trajectory.final_answer),
        "steps": len(trajectory.tool_calls),
        "tokens": trajectory.total_tokens,
        "cost": trajectory.cost,
    }
```

### Step 4: analyze failures

Don't just get a score. Cluster failures.
- "Agent fails on tasks involving multi-step math (5/50 tasks)."
- "Agent times out on long PDFs (3/50 tasks)."
- "Agent hallucinates sources (7/50 tasks)."

Fix the worst clusters first.

### Step 5: regression-test

Your eval becomes a regression harness. Before any change:
1. Run on your eval.
2. Compare to baseline.
3. Accept changes only if eval improves.

## LLM-as-judge

For subjective tasks, use a strong LLM to score:

```python
judge_prompt = f"""
Task: {task}
Agent's answer: {answer}
Gold answer: {gold}

Is the agent's answer correct? Score 1-5. Explain.
"""
response = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":judge_prompt}])
```

**Caveats**:
- Judge LLM can be biased (favors its own outputs, favors verbose answers).
- Different judges give different scores.
- Calibration required: periodically human-check a sample.

Best practice: multiple judge models, aggregate scores, human spot-check.

## Cost-quality tradeoff

Plot for each variant: accuracy vs. cost.

```
accuracy
  ^
  |    * variant C (smart, expensive)
  |
  |  * variant B
  |
  | * variant A (fast, cheap)
  +-----------------> cost
```

Different variants for different budgets. For production, pick the knee of the curve.

## Observability in production

For running agents (not just evaluation):

- **LangSmith** (LangChain): traces.
- **Arize Phoenix**: open source trace analysis.
- **Braintrust**: evaluation + monitoring.
- **Helicone**: logging + metrics for LLM APIs.
- **Hand-rolled**: log every call to structured JSON, analyze offline.

Trace every production call. You need it for debugging.

## Common eval pitfalls

### 1. Test set too small
10 tasks isn't enough. Aim for 50-200 for reliable signal.

### 2. Test set contaminates over time
Your eval leaks into dev → model starts memorizing. Keep a held-out set.

### 3. Over-optimizing for the metric
Agent starts gaming the eval. Watch for strange "hacks" that score but don't help.

### 4. Not updating the eval
Your agent's capabilities outgrow your eval. Eval stops discriminating. Update periodically.

### 5. Only measuring success
Efficiency and cost matter. A 99% success rate at $10/query is often worse than 95% at $0.10.

## Integration with CI

For production agents:

```yaml
# .github/workflows/agent-eval.yml
on: pull_request
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: pip install -r requirements.txt
      - run: python run_eval.py --output eval.json
      - run: python compare_eval.py eval.json baseline.json
```

Treat eval as tests. PR should improve or match baseline or be explicitly justified.

## Visualize this

**Why agent evaluation is hard**:

```
  LLM evaluation:
    single call → single output → grade

  Agent evaluation:
    task → many steps, tool calls, decisions, retries → outcome
       │
       ├── multi-step trajectory
       ├── partial successes
       ├── non-deterministic paths
       ├── cost varies per run
       └── human judgment often needed
```

**Key agent metrics**:

```
  Success rate:
    Did the agent complete the task?
    Binary per task. Aggregate: % correct.

  Step-level correctness:
    Of N steps, how many correct?
    Granular; shows where agent fails.

  Efficiency:
    - Steps to solve
    - Tokens consumed
    - Wall-clock time
    - $ cost

  Robustness:
    Same task, slight perturbation, does it still work?

  Faithfulness (for RAG):
    Does the answer match the retrieved docs?
    Or hallucinate?

  Safety:
    Does it refuse harmful requests?
    Confirm before destructive actions?
```

**Agent benchmarks (2026)**:

```
  GAIA (general assistant):
    466 real-world tasks.
    Requires tools, reasoning, multi-modal.
    Top scores: ~50% (top agents). Humans: 92%.

  SWE-Bench (software engineering):
    2,294 real GitHub issues.
    Agent must produce PR that passes tests.
    Top scores:
      GPT-4 (2023): ~20%
      Claude 3.5 Sonnet (2024): ~49%
      o1-preview: ~41%
      Devin: ~14% (early); now better
      Amazon Q Developer: ~47%
    Humans: ~100% (with effort).

  WebArena (web navigation):
    812 tasks on 5 websites.
    Top scores: ~40%.

  AgentBench (diverse):
    OS tasks, DB queries, games, tool use.
    Composite score.

  ToolEval:
    Pure tool-use accuracy.

  Custom internal:
    Most production teams build their own.
    Most valuable; hardest to publish.
```

**Building your own agent eval (recommended)**:

```
  Step 1: Define tasks (20-50 is enough to start)
    Each task:
      - Clear description
      - Ground-truth answer (or checker function)
      - Category (for stratified analysis)

  Step 2: Capture trajectories
    Log:
      - Every LLM call (prompt + response)
      - Every tool call (name + args + result)
      - Wall-clock time
      - Token count + cost

  Step 3: Grade
    Automated: exact match, regex, LLM-as-judge.
    Manual: for ambiguous cases.

  Step 4: Analyze
    - Overall success rate
    - Per-category breakdown
    - Failure cluster analysis (what types fail?)
    - Cost/quality Pareto

  Step 5: Regression test
    Run eval on every change.
    Fail the PR if metrics drop.
```

**LLM-as-judge pattern**:

```
  Task: "What's the capital of France?"
  Agent answer: "The capital of France is Paris."
  Ground truth: "Paris"

  Judge prompt to GPT-4:
    "Given the task, the correct answer, and the agent's answer,
     determine if the agent's answer is correct.
     Task: What's the capital of France?
     Correct: Paris
     Agent: The capital of France is Paris.
     Is the agent's answer correct? Reply: YES or NO."

  GPT-4: "YES"

  → Automated grading. Fuzzy matching via LLM.
  Caveats:
    - Judge can be biased (longer = better in its view)
    - Always human-spot-check a sample
```

**Cost-quality Pareto plot**:

```
  quality
   (success %)
       │
       │     ●● Claude 3.5 Sonnet (90%, $0.50)
       │        ●● GPT-4o (88%, $0.10)
       │   ● GPT-4 (85%, $0.30)
       │     ● DeepSeek-V3 (82%, $0.01)
       │   ●  Claude 3.5 Haiku (75%, $0.05)
       │ ●
       │● GPT-4o-mini (70%, $0.005)
       │
       │ ● Llama-3-70B (65%, $0.02)
       │
       └────────────────────────────── cost per task

  "Pareto front": models not dominated by any other on both axes.
  Pick based on YOUR quality / cost constraints.
```

**Production eval + monitoring stack**:

```
  Dev time:
    CI runs eval suite on every PR.
    Block merges that regress.

  Production time:
    Log every agent run to a trace store.
    Sample 1-10% for human review.
    Alert on unusual patterns:
      - latency spikes
      - cost explosions
      - high error rates
      - unusual tool-call patterns

  Tools:
    - Langsmith / Arize Phoenix / Braintrust (hosted)
    - Custom: structured JSON logs + dashboards
```

**Evaluation pitfalls**:

```
  1. Test set too small:
     10 tasks. 1 passes. Success = 10%?
     No. Confidence interval is huge.
     Need 50-200 tasks minimum.

  2. Contamination over time:
     Eval tasks leak into dev. Model "cheats".
     Rotate or hold out.

  3. Goodhart's Law:
     Optimize for eval; benchmark stops predicting real quality.
     Evolve evals quarterly.

  4. Only measuring success:
     Model is accurate but uses 100× compute.
     Add cost/steps/time metrics.

  5. LLM-as-judge bias:
     Judge prefers its own family's answers.
     Use multiple judges; human spot-check.

  6. Static evals:
     Your model improves; eval stays the same.
     Eventually eval stops discriminating.
     Renew every 6-12 months.
```

**Integration with CI**:

```yaml
# .github/workflows/agent-eval.yml
on: pull_request
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: pip install -r requirements.txt
      - run: python run_agent_eval.py --output eval_new.json
      - run: python compare_eval.py eval_new.json main_eval.json
        # fails PR if success rate drops > 2%
```

Treats eval like tests. Ships quality.

**Quick eval setup for your own agent**:

```python
# eval.py
import json
from my_agent import run_agent

test_cases = [
    {"task": "What's 2+2?",                 "expected": "4"},
    {"task": "Capital of France?",          "expected": "paris"},
    {"task": "List 3 Python data types",    "expected_any": ["list", "dict", "set"]},
    # ... 20+ more
]

def grade(actual, expected=None, expected_any=None):
    a = actual.lower()
    if expected:   return expected.lower() in a
    if expected_any: return any(e in a for e in expected_any)
    return False

results = []
for tc in test_cases:
    trajectory = run_agent(tc["task"])
    correct = grade(trajectory["final_answer"], **{k:v for k,v in tc.items() if k != "task"})
    results.append({
        "task": tc["task"],
        "correct": correct,
        "steps": trajectory.get("steps", 0),
        "cost": trajectory.get("cost", 0),
    })

success = sum(r["correct"] for r in results) / len(results)
print(f"Success: {success:.1%}")
print(f"Avg steps: {sum(r['steps'] for r in results) / len(results):.1f}")
print(f"Total cost: ${sum(r['cost'] for r in results):.3f}")
```

Run before any change. Track over time. Improve data-driven.

**When evals are "good enough"**:

```
  For a personal project / capstone:
    20+ cases
    automated grading
    one run = one page of results

  For a startup:
    100+ cases per category
    LLM-as-judge + human sample
    CI integration

  For enterprise:
    1000+ cases
    human evaluation on samples
    production monitoring + alerting
    rotating / evolving eval suites
```

## Exercises

1. Write an eval for your code-execution agent (Lesson 10.4):
   - 20 data analysis tasks.
   - Programmatic success checks.
   - Log trajectories.

2. Add LLM-as-judge scoring for tasks that don't have a clean success criterion.

3. Try running SWE-Bench Lite on Aider or Claude Code (if available). Look at failures.

4. Build a cost-quality plot for 3 variants of one of your agents (e.g., GPT-4o vs GPT-4o-mini).

5. Set up a daily eval run for your agent: scheduled cron, metrics sent to wandb.

## Next

`11_safety_and_constraints.md` - the safety side of agent engineering.
