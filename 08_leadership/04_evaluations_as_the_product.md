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
