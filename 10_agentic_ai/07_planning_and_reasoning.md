# 10.7 - Planning and Reasoning

An agent is only as good as its reasoning. This lesson covers techniques to make agents plan, reflect, and self-correct.

## The spectrum

From lowest to highest reasoning:

1. **React** (no reasoning): model just picks the next token.
2. **Chain-of-Thought** (CoT): model reasons step-by-step in prose before answering.
3. **ReAct** (Reason+Act): alternating thoughts and actions.
4. **Reflexion**: agent critiques and revises.
5. **Tree of Thoughts** (ToT): explore multiple reasoning paths.
6. **o1 / R1 style**: RL-trained long reasoning chains.
7. **Graph / program search**: AlphaGo-style.

Each adds compute-time cost for quality.

## 1. Chain-of-Thought (CoT)

The simplest and most impactful. Prompt: "Let's think step by step."

Before CoT (bad):
```
Q: If I have 5 apples and eat 2, how many do I have?
A: 3
```

With CoT (good):
```
Q: If I have 5 apples and eat 2, how many do I have?
A: Let's think step by step. Start with 5 apples. Eat 2. 5 - 2 = 3. So 3 apples remain.
```

**Effect**: dramatic accuracy gains on math, logic, multi-step questions. Wei et al 2022 showed 20+ percentage point jumps on GSM8K.

Modern models have CoT baked in - they naturally produce reasoning traces. But explicitly prompting helps for reluctant models.

Paper: https://arxiv.org/abs/2201.11903

## 2. ReAct (Reasoning + Acting)

Combines CoT with tool use. Pattern:

```
Thought: I need to find the capital of the smallest EU country.
Action: search_web("smallest EU country by area")
Observation: Malta, 316 km².
Thought: Now I need Malta's capital.
Action: search_web("capital of Malta")
Observation: Valletta.
Final Answer: Valletta.
```

The LLM is now planning over actions, observing results, re-planning.

Paper: https://arxiv.org/abs/2210.03629

## 3. Self-consistency

Sample multiple CoT solutions for the same problem. Take the majority vote on the final answer.

```python
answers = []
for _ in range(10):
    response = llm(prompt + "Let's think step by step.", temperature=0.7)
    answers.append(extract_final_answer(response))
final = max(set(answers), key=answers.count)
```

Expensive (10x compute) but often 10-20 points better on math/logic.

Wang et al 2022: https://arxiv.org/abs/2203.11171

## 4. Reflexion

After failure, let the LLM critique its own attempt and retry with the critique in context.

```
Attempt 1: [LLM tries something, fails]
Self-critique: "In my last attempt, I assumed X which is wrong because Y. Let me try differently."
Attempt 2: [LLM tries again, informed by critique]
```

Works well for code (fix the bug) and reasoning (where you went wrong).

Paper (Shinn et al 2023): https://arxiv.org/abs/2303.11366

## 5. Tree of Thoughts (ToT)

Instead of one reasoning chain, explore a tree:

```
Initial state
├── Branch A: "Try approach A"
│   ├── Sub-branch: "Refine A"
│   └── Sub-branch: "Alternative within A"
├── Branch B: "Try approach B"
└── Branch C: "Try approach C"
```

At each node, evaluate (by LLM self-evaluation or scoring) whether to continue. Prune bad branches. Expand promising ones.

Useful for combinatorial problems (puzzles, creative writing where you want to explore).

Paper (Yao et al 2023): https://arxiv.org/abs/2305.10601

**Cost**: much higher. Not widely used in production due to latency.

## 6. Program-of-Thought (PoT)

Instead of reasoning in prose, reason in code. Then execute.

```
Q: What's the sum of the first 100 primes?
LLM generates:
    def is_prime(n): ...
    primes = [p for p in range(2, 1000) if is_prime(p)][:100]
    print(sum(primes))
Code executes, LLM sees result, answers.
```

For mathy problems, this is much more reliable than pure CoT. Combines with the code-execution agent pattern (Lesson 10.4).

## 7. o1 / R1 style: RL-trained reasoning

The big shift of 2024-2025.

**Standard LLM**: produces answer directly. Small chance of multi-step reasoning.

**Reasoning model**: trained to produce long reasoning traces (often 10K+ tokens internally) before the final answer. Via RL with correctness rewards.

Examples:
- **OpenAI o1** (Sept 2024): reasoning model, much better at math/science/code.
- **OpenAI o3** (Dec 2024 / 2025): successor, beats humans on ARC-AGI.
- **DeepSeek R1** (Jan 2025): open-source reasoning model, published the RL recipe.
- **Qwen QwQ-32B**: open reasoning variant.
- **Anthropic Claude 3.7 Sonnet** (2025): "extended thinking" mode.

### How they work (DeepSeek R1 reveal)

1. Start with a strong base model.
2. Generate N reasoning attempts for problems with verifiable answers (math, code).
3. Reward correct final answers (GRPO-style, as in Module 5.6).
4. The model learns: longer reasoning → higher accuracy.
5. No explicit reward for CoT format; it emerges.

After training, the model spontaneously writes 10K+ token reasoning traces before answering.

### Implications

**Inference-time compute becomes a dial.** Want better answers? Let the model think longer. A new economic model for LLMs: pay for tokens of *thinking*, not just final output.

### Practical usage

```python
response = openai.chat.completions.create(
    model="o1-preview",
    messages=[{"role": "user", "content": "Hard math problem"}],
)
# Response includes visible answer. Internal reasoning is hidden (OpenAI).
# For DeepSeek R1: reasoning traces are visible.
```

o1 is slow and expensive. Use for tasks that warrant it (research-grade problems). Overkill for "what's the weather."

## 8. Planner-executor split

Some agents use two LLMs (or two modes):

- **Planner**: high-level plan. "First search for X, then extract Y, then compute Z."
- **Executor**: executes each step with tools.

Separates concerns. Planner can be a bigger/smarter model; executor can be cheaper.

## Reasoning in multi-step agent workflows

Practical advice:

### For simple tasks (1-3 steps)
Standard tool-calling loop. No explicit planning. The LLM plans implicitly.

### For medium tasks (3-10 steps)
Add a "plan first" prompt:
```
Think about what you need to do. Write a plan. Then execute step by step.
```

### For complex tasks (10+ steps)
Planner-executor split. Or use a reasoning model as the core.

### For verifiable tasks (code, math)
Use Reflexion or PoT. Verify each intermediate step.

### For creative tasks
ToT (explore multiple options). Self-consistency (sample and aggregate).

## Failures and anti-patterns

### 1. Too much planning
Agent spends 500 tokens planning for a 2-step task. Wastes time, rarely helps.

### 2. Plans that don't match reality
"Plan: first search X, then click the first result." If the search returns nothing, plan collapses.

Solution: plans should be loose; re-plan after every step.

### 3. Reasoning hallucinations
The model reasons convincingly but incorrectly. Common in math ("therefore 2 + 2 = 5"). Symbolic verification (code execution) helps.

### 4. Getting stuck in loops
Agent keeps trying the same failed action. Solution: detect loops (same action 3x → escalate).

## A "reasoning-augmented" agent

Add explicit reasoning to your agent:

```python
messages = [
    {"role": "system", "content":
     "For each user request:\n"
     "1. Write a PLAN in <plan> tags.\n"
     "2. Execute via tools.\n"
     "3. After each tool result, REFLECT in <reflect> tags on whether to continue or adjust.\n"
     "4. Give FINAL ANSWER when done."
    },
    {"role": "user", "content": user_task},
]
```

Simple prompt change, measurable quality gain.

## Evaluation

How to tell if reasoning helped?

- **Pass@1 on reasoning benchmarks**: GSM8K, MATH, HumanEval.
- **Multi-step task success**: GAIA, SWE-Bench.
- **Trajectory efficiency**: steps-to-solution.
- **Cost-quality Pareto**: plot accuracy vs. tokens used. More tokens should mean better for reasoning techniques to be worth it.

## When NOT to add reasoning

- **Easy tasks**: user just wants a quick answer.
- **Hot path**: response time matters more than quality. Reasoning is slow.
- **Stylistic tasks**: creative writing often doesn't benefit.

Use reasoning when the task is hard AND correctness matters AND latency is tolerable.

## Reading

- **Chain-of-Thought Prompting** (Wei 2022): https://arxiv.org/abs/2201.11903
- **Self-Consistency** (Wang 2022): https://arxiv.org/abs/2203.11171
- **ReAct** (Yao 2022): https://arxiv.org/abs/2210.03629
- **Reflexion** (Shinn 2023): https://arxiv.org/abs/2303.11366
- **Tree of Thoughts** (Yao 2023): https://arxiv.org/abs/2305.10601
- **DeepSeek R1** (2025): https://arxiv.org/abs/2501.12948
- **OpenAI o1 system card**

## Visualize this

**Reasoning techniques on a complexity ladder**:

```
                   complexity/cost
                         │
   tree search RL       ▲
   (AlphaGo, o1/R1)    │
                       │
   tree of thoughts    ▲
                       │
   self-consistency    ▲
   (sample N, vote)    │
                       │
   reflexion            ▲
   (self-critique)     │
                       │
   ReAct               ▲
   (reason + act)      │
                       │
   CoT                 ▲
   (think step-by-step)│
                       │
   direct               ● (no reasoning; just answer)
                       │
                       └────► compute per query

  Higher up: better accuracy on hard problems, more expensive.
  Pick the lowest on this ladder that solves your problem.
```

**CoT in action**:

```
  Without CoT:
  Q: If I have 5 apples and give 2 to Alice and 1 to Bob, then buy 4 more,
     how many do I have?
  A: 6 (correct by luck maybe)

  With CoT:
  Q: If I have 5 apples and give 2 to Alice and 1 to Bob, then buy 4 more,
     how many do I have?
     Let's think step by step.
  A: I start with 5. Give 2 to Alice: 5-2 = 3. Give 1 to Bob: 3-1 = 2.
     Buy 4 more: 2+4 = 6. So I have 6 apples.
  → Correct, and explainable.

  GSM8K accuracy jump:
  Direct:    ~20%
  With CoT:  ~60%
  With CoT + larger model: ~90%+
```

**ReAct: reason AND act**:

```
  Question: "What's the population of the capital of France?"

  Thought: I need to find the capital of France first.
  Action:  search_web(query="capital of France")
  Observation: "Paris is the capital of France."
  Thought: Now I need the population of Paris.
  Action:  search_web(query="population of Paris")
  Observation: "Paris has a population of about 2.1 million."
  Thought: I have the answer.
  Final Answer: The capital of France is Paris, with a population of about 2.1 million.

  The LLM interleaves:
    - Thinking about what to do next
    - Taking actions
    - Processing observations

  Standard agent pattern.
```

**Self-consistency**:

```
  Instead of sampling 1 CoT, sample N.
  Take the majority answer.

  Question: "What's 237 * 491?"

  Sample 1: "237 * 491 = 237 * 500 - 237 * 9 = 118500 - 2133 = 116367"  → 116367
  Sample 2: "237 * 491 ≈ 240 * 490 = 117600, adjust by ... = 116367"    → 116367
  Sample 3: "237 × 491 ... (mistake) ... = 116377"                       → 116377
  Sample 4: "237 * 491 = 237 * 500 - 2133 = 116367"                      → 116367
  Sample 5: "237 * 491 = ..."                                             → 116367

  Majority: 116367  (4 out of 5)

  Cost: 5× more compute than 1 sample.
  Gains: 10-15 percentage points on hard math.
```

**Reflexion (learn from failure)**:

```
  Attempt 1:
    Task: "Write Python function to check if a number is prime."
    Code: def is_prime(n): return n > 1 and all(n % i for i in range(2, n))
    Test: is_prime(2) → True (correct)
          is_prime(4) → False (correct)
          is_prime(9) → True (WRONG)
    Verdict: failed.

  Reflexion (self-critique):
    "My function returns True for 9, which is wrong because 9 = 3 × 3.
     The range should not include 0 or 1 in division, but I do check n > 1.
     Wait, range(2, 9) = [2,3,4,5,6,7,8] and 9%3=0, so the all() should be False.
     Let me recheck... oh, 9 % 2 = 1 (truthy), 9 % 3 = 0 (falsy). So all() = False.
     Return should be False. But my test says True. Bug must be elsewhere.
     ...let me rewrite: `all(n % i for i in range(2, int(n**0.5)+1))`"

  Attempt 2: use the corrected version.
    Test: is_prime(9) → False (correct now).

  Key: error signal fed back into reasoning → iterative improvement.
```

**o1/R1-style reasoning (the 2024-25 unlock)**:

```
  Old models:
    Input: problem
    Output: answer (maybe wrong)
    Compute: fixed per query.

  Reasoning models (o1, o3, DeepSeek-R1):
    Input: problem
    Output: LONG reasoning trace + answer
    Compute: SCALES with problem difficulty

  Example:
    User: "Solve: find x in x^3 - 6x^2 + 11x - 6 = 0"

    Model's internal reasoning (thousands of tokens!):
      "Let me try to factor...
       Rational root theorem: possible roots are ±1, ±2, ±3, ±6.
       Test x=1: 1-6+11-6 = 0. ✓ So (x-1) is a factor.
       Dividing: x^3 - 6x^2 + 11x - 6 = (x-1)(x^2 - 5x + 6)
       Factor quadratic: x^2 - 5x + 6 = (x-2)(x-3)
       So x = 1, 2, 3.
       Wait, let me verify: (1-1)(1-2)(1-3) = 0*-1*-2 = 0. ✓
       (2-1)(2-2)(2-3) = 1*0*-1 = 0. ✓
       (3-1)(3-2)(3-3) = 2*1*0 = 0. ✓"

    Final answer: x = 1, 2, 3.

  The "thinking" is internal. User sees only the final answer.
  Cost: 50-500× more tokens than a non-reasoning model. Worth it for hard problems.
```

**How R1 is trained (from the paper)**:

```
  1. Start with a base LLM (like DeepSeek-V3).
  2. For a problem with a verifiable answer (math, code):
       - Sample K=16 responses
       - Score each 0/1 (correct?)
       - Policy gradient: reward correct, penalize wrong
  3. Model learns: "longer reasoning → higher accuracy"
  4. Emergent behavior: spontaneous self-reflection, backtracking.

  Key insight: no explicit supervision for CoT format.
  The model learns to think step-by-step because it leads to rewards.

  GRPO (Group Relative Policy Optimization):
    Advantage = (reward - group_mean) / group_std
    Updates policy based on relative performance within a batch.
    Simple, effective.
```

**Planner-executor pattern**:

```
  User: "Research the latest open-source LLMs and build a comparison table."

  ┌─────────────────────────────┐
  │   Planner (big smart LLM)    │  (once, up front)
  │   Makes a plan:               │
  │   1. search for open LLMs     │
  │   2. fetch Hugging Face pages  │
  │   3. extract specs             │
  │   4. format as table           │
  └────────────┬─────────────────┘
               │
               ▼
  ┌─────────────────────────────┐
  │   Executor (cheaper LLM)      │  (for each step)
  │   executes the plan            │
  │   step by step                 │
  └─────────────────────────────┘

  Advantages:
  - Smart model for planning (complex reasoning)
  - Cheap model for execution (just follow orders)
  - Clear separation of concerns
```

**When to add reasoning (cost-benefit)**:

```
  Task type              Reasoning worth it?
  ─────────────────────  ──────────────────────
  Chat, chitchat          No (overkill, slow)
  Simple retrieval         No
  Classification           No
  Math, logic              YES (dramatic gains)
  Code with tests          YES (retry loop shines)
  Multi-step planning       YES (agent with reflection)
  Research                 MAYBE (depends on breadth)

  Rule: if your eval shows > 10% gain with CoT,
        consider reasoning models.
        Otherwise, regular models are cheaper.
```

**Reasoning cost comparison (2026 prices)**:

```
                              Cost per 1K tokens
  ────────────────────────    ─────────────────────
  GPT-4o (non-reasoning)       $10 input, $30 output
  GPT-4o-mini                  $0.60 input, $2.40 output
  o1 (reasoning)               $60 input, $240 output (!!)
  o1-mini                      $12 input, $48 output
  Claude 3.7 Sonnet thinking   $15 input, $75 output
  DeepSeek-R1 (open)            $1 input, $2 output (10-100× cheaper)

  Reasoning costs 5-10× more than regular generation.
  Per query cost can be $0.50-$5 for complex problems.
```

## Exercises

1. Take a hard problem (GSM8K-style). Solve it with a vanilla LLM. Then with "think step by step". Then with self-consistency (5 samples). Compare accuracy.

2. Implement Reflexion for code: LLM writes code, executes, sees error, reflects, retries.

3. Try o1 (if subscribed) on a problem where GPT-4 fails. Observe the extended thinking.

4. Read DeepSeek R1's paper (Jan 2025). Understand the RL recipe.

## Next

`08_multi_agent_systems.md` - the next level of complexity.
