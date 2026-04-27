# 8.4 - Evaluations as the Product

The most important sentence in ML leadership: **"Your evals are your product."**

If you can't measure a capability, you can't improve it. If the measurement is weak, the improvement is fake. This lesson is about making evals the center of gravity of your team, not an afterthought.

## The thesis

When researchers say "we improved the model", what they usually mean is "we improved one specific number." The number matters. The number is the product.

Teams that ship real progress:
- Spend 30-50% of their total effort on evaluations.
- Build custom evals for their specific use cases.
- Constantly update evals as old ones saturate.
- Don't trust public benchmarks alone.

Teams that plateau:
- Treat evals as a checkbox at the end.
- Only use public benchmarks.
- Don't re-examine what their evals measure.
- Celebrate wins that don't hold up in production.

## What makes an eval good

1. **Discriminative**: good and bad models get noticeably different scores.
2. **Stable**: similar models get similar scores (not noisy).
3. **Correlated with product value**: moving the number matters to users.
4. **Hard to game**: the model can't hack it.
5. **Uncontaminated**: the test data isn't in the training set.
6. **Cheap enough**: you can run it often.

Public benchmarks usually satisfy 1-3. They often fail at 4-5 and 6.

## The public benchmark treadmill

When a benchmark gets hard, it becomes a target. When something becomes a target, it stops being a good measure (Goodhart's Law).

Sequence:
1. New benchmark released. Models get ~20%. (Too hard; not useful yet.)
2. Over 2 years, models get to 50%. (Golden period: it discriminates.)
3. Models reach 85-90%. (Saturating. Small improvements no longer meaningful.)
4. Everyone trains on it implicitly (via Stack Overflow, arxiv, etc). (Contaminated.)
5. Community creates the next benchmark.

Cycle length: 3-5 years per major benchmark.

Examples:
- GLUE (2018) → saturated by 2020 → superseded by SuperGLUE → saturated → superseded by BigBench → ...
- MMLU (2020) → still useful but top models > 85% → newer harder evals (MMLU-Pro, Humanity's Last Exam).

## Internal evals: the real work

Public benchmarks exist for paper-writing. Your team needs **internal evals** for actual product progress:

- Domain-specific test sets.
- Customer-supplied edge cases.
- Hand-written hard examples from your own staff.
- Red-team attempts.
- Capability-specific tests (tool use, multi-turn dialog, safety).

Examples of what frontier labs have internally:
- "Write a convincing phishing email" (measure refusal rate)
- 1000 hand-crafted math problems designed to be hard for current models
- Multi-turn scenarios mimicking real user interactions
- Specific format-compliance tests (valid JSON, Markdown, etc.)

These rarely get published. They're the team's crown jewels.

## Your team's eval stack

A mature eval stack has 3 tiers:

### Tier 1: fast regression checks
- Runs on every PR or checkpoint.
- Takes < 30 minutes.
- Core capabilities: "still basic competent?"
- If this drops, something is broken.

### Tier 2: weekly evaluation
- Runs on major checkpoints.
- Takes 2-10 hours.
- Full suite: public benchmarks, domain tests, red-team.
- Results in a shared weekly report.

### Tier 3: deep evaluation
- Pre-release.
- Human evaluators (internal or paid).
- Chatbot Arena-style pairwise comparisons.
- Specific scenarios that matter to the business.

Most teams only have tier 2, somewhat. Great teams have all three, automated.

## Eval-driven development

A specific methodology:

1. Before proposing any new technique, define an eval that measures it.
2. Ship the eval first.
3. Baseline all current models on it.
4. Only then, work on the technique.
5. Track the eval as the main metric for the technique.

Avoids: "we built something cool but can't measure if it helps."

## Hardest eval categories

Some capabilities resist easy measurement:

- **Helpfulness**: subjective. Requires human preference data.
- **Honesty**: hard to catch confident lies.
- **Long-horizon coherence**: expensive to evaluate at length.
- **Emergent reasoning**: no single test captures it.
- **Tool use**: composable capabilities, many subtasks.
- **Safety**: red-teaming requires creativity.

For these, assume measurement is imperfect. Use multiple proxies. Update frequently.

## Leading an eval effort

If you're the leader:

1. **Build headcount for eval engineers**, not just model trainers. Ratio of 1:2 (eval:training) is healthy.
2. **Publish eval results** internally, weekly. Visible dashboards.
3. **Protect evals** from contamination - don't let training teams see eval data.
4. **Kill stale evals** - if nothing you do changes the score, it's saturated.
5. **Reward eval improvements** - don't just reward model capability wins.

## Key papers and resources

- Liang et al 2022 "HELM" - early comprehensive benchmark.
- Raji et al 2021 "AI and the everything in the whole wide world benchmark" - critical perspective.
- Chatbot Arena (https://lmsys.org/blog/2023-05-03-arena/) - human preference via live voting.
- OpenAI's [evals framework](https://github.com/openai/evals).
- EleutherAI's lm-evaluation-harness - standard tool for public LM evals.

## Visualize this

**The eval-driven research loop**:

```
  Instead of:
    1. Have an idea
    2. Implement it
    3. Measure something
    4. Hope it improved

  Do:
    ┌─────────────────────────────────────────────┐
    │ 1. Build/find eval first                    │
    │     metric, dataset, gold answers            │
    │                                              │
    │ 2. Baseline current model on eval            │
    │     get N, variance, confidence intervals    │
    │                                              │
    │ 3. Pick improvement idea                     │
    │     hypothesis: "this will move the eval"    │
    │                                              │
    │ 4. Implement                                 │
    │                                              │
    │ 5. Re-run eval                               │
    │     did it move beyond noise?                │
    │                                              │
    │ 6. Pass/fail decision                        │
    └─────────────────────────────────────────────┘
```

**Eval stack tiers**:

```
   Tier 1: FAST (every PR)
   ┌──────────────────────────────────────────┐
   │ Small MMLU subset (100 Qs)                │ ~2 min
   │ Sample coherence check                     │ ~1 min
   │ Basic safety check                         │ ~1 min
   │ Total: ~5 min per commit                   │
   └──────────────────────────────────────────┘

   Tier 2: WEEKLY (major checkpoints)
   ┌──────────────────────────────────────────┐
   │ Full MMLU / ARC / HumanEval / GSM8K        │ ~2 hours
   │ CORE metric                                │ ~30 min
   │ Domain-specific internal evals              │ varies
   │ Red team suite                             │ ~1 hour
   │ Total: ~3-5 hours per checkpoint             │
   └──────────────────────────────────────────┘

   Tier 3: DEEP (pre-release)
   ┌──────────────────────────────────────────┐
   │ Human evaluators (paid)                    │ days
   │ Chatbot Arena-style pairwise               │ weeks
   │ Adversarial red-teaming                    │ days
   │ Long-horizon coherence                     │ days
   │ Multilingual quality                       │ varies
   │ Production traffic replay                  │ hours
   └──────────────────────────────────────────┘
```

**The public benchmark treadmill**:

```
  Benchmark lifecycle (typical):

  Year 1: Released. Top models score ~20%.
          "Promising but hard." Useful.

  Year 2-3: Models catch up. Top score ~50%.
            "The gold standard." Very useful.

  Year 4: Top score ~85%. Getting saturated.
          Small gains no longer meaningful.

  Year 5: Top score ~95%. Fully saturated.
          Also, training data contaminated.
          Time for new benchmark.

  Examples:
   SQUAD (QA)         2016 → 2020   saturated
   GLUE               2018 → 2020   saturated → SuperGLUE
   MMLU               2020 → 2024   approaching saturation
   GSM8K              2021 → 2024   top ~95%
   Humanity's Last    2024 → ??     current "hard" benchmark
   Exam (HLE)
   ARC-AGI            2019 → 2025   just solved by o3
```

**Goodhart's Law (critical for ML leaders)**:

```
  "When a measure becomes a target, it ceases to be a good measure."

  Example:
    Start: "MMLU score is a good proxy for knowledge."
    
    Months later: "Let's train specifically to improve MMLU."
    
    Eventually:
    - Model overfits to MMLU-style questions.
    - Training data gets contaminated with MMLU.
    - "Good at MMLU" stops meaning "good at knowledge".

  Defense:
    - Rotate benchmarks regularly.
    - Use internal holdout benchmarks.
    - Check that MMLU gains correlate with downstream wins.
```

**Eval quality checklist**:

```
  For any benchmark you rely on:

  ✓ Discriminative: good models score noticeably higher than bad ones
  ✓ Stable: same model twice → similar score (test-retest reliability)
  ✓ Correlates with real value: moving this metric moves product
  ✓ Uncontaminated: not in training data
  ✓ Diverse: covers many types of inputs
  ✓ Cheap enough to run often
  ✓ Hard to game: model can't trivially get high score via shortcuts

  If a benchmark fails 3+ of these, drop it. Create better.
```

**The "eval is the product" mindset**:

```
  Common mistake:
    Team builds model → "now let's figure out how to measure it"

  Better:
    Team defines eval → "what model would win on this eval?"

  The eval determines:
    - What the model learns
    - What the team optimizes for
    - What the org declares as success
    - What leadership reports up

  → The eval IS the product specification.
  → Design it with the same care you'd design a PRD.
```

**Example: an internal eval suite**:

```
  Company: "Customer support chat assistant"

  Internal eval suite (kept secret, updated quarterly):
  ┌────────────────────────────────────────────┐
  │ 500 real customer questions (labeled)        │
  │ ├── 100 product pricing questions             │
  │ ├── 100 technical troubleshooting             │
  │ ├── 100 billing / refund / policy             │
  │ ├── 100 general chat / small talk             │
  │ └── 100 escalation / complex multi-turn       │
  │                                              │
  │ Metrics per category:                        │
  │ - LLM-as-judge accuracy (vs gold human answer)│
  │ - Hallucination rate (made up facts?)          │
  │ - Policy adherence (didn't give forbidden info)│
  │ - Escalation rate (knew when to hand off)      │
  │                                              │
  │ Overall score: weighted by traffic share      │
  │                                              │
  │ Run on every candidate model.                 │
  │ Ship if beats current by >3% with no regressions│
  └────────────────────────────────────────────┘
```

This eval IS the product. Building it well is the hardest and most valuable work.

## Exercises

1. List the evals you (or your team) currently run. For each:
   - Is it in tier 1, 2, or 3?
   - What's its discrimination? (Biggest - smallest recent score.)
   - When did you last update it?
   
   Most teams discover they have thin tier 2 and no tier 1 or 3.

2. Invent an eval for something you care about but don't measure. 50 examples is enough to start. Build it.

3. Read the [DCLM paper](https://arxiv.org/abs/2406.11794) sections on how they built CORE. Note: they explicitly built the benchmark.

4. If you lead a team: make the eval leaderboard the first thing anyone sees. Not the latest run's loss curve. The eval leaderboard.

## Next

`05_managing_compute_and_budgets.md` - money and clocks.
