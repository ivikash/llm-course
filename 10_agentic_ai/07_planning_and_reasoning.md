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

## Exercises

1. Take a hard problem (GSM8K-style). Solve it with a vanilla LLM. Then with "think step by step". Then with self-consistency (5 samples). Compare accuracy.

2. Implement Reflexion for code: LLM writes code, executes, sees error, reflects, retries.

3. Try o1 (if subscribed) on a problem where GPT-4 fails. Observe the extended thinking.

4. Read DeepSeek R1's paper (Jan 2025). Understand the RL recipe.

## Next

`08_multi_agent_systems.md` - the next level of complexity.
