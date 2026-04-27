# 8.2 - The Research Loop

Research is a structured decision-making process, not random tinkering. This lesson gives you the framework.

## The loop

```
  Hypothesis
      ↓
  Experiment design
      ↓
  Run experiment
      ↓
  Analyze result
      ↓
  Update beliefs
      ↓
  (back to Hypothesis)
```

Every experiment closes one cycle. Good researchers do dozens per month. Bad ones do one massive experiment and hope.

## Phase 1: hypothesis

A good hypothesis is:

1. **Specific**: "RMSNorm replaces LayerNorm with similar val loss" (not "norms matter").
2. **Testable**: you can design an experiment to falsify it.
3. **Risky**: if true, changes what you do. If false, also changes what you do.

Bad hypotheses:
- "Our new model will be better." (Better at what? By how much?)
- "Training longer will help." (Obviously. By how much? When do returns saturate?)

Good hypotheses:
- "Removing biases from linear layers reduces loss by < 0.5% without affecting training stability, on Shakespeare data."
- "Using RoPE instead of learned positional embeddings allows the model to generalize 2x beyond its training block_size."

Notice the specificity.

## Phase 2: experiment design

Before running anything, answer:

1. **What am I measuring?** (Val loss? GSM8K accuracy? Inference throughput?)
2. **What am I comparing against?** (A baseline. If you don't have one, your result is meaningless.)
3. **What's the minimum-sufficient experiment?** (Smallest model, smallest data that would give signal.)
4. **What's the compute budget?** (Set before you start.)
5. **What result would refute the hypothesis?** (If you can't answer, the hypothesis is unfalsifiable.)
6. **Controls**: what must stay identical between runs? (Seed, data, all other hyperparameters.)

Write this as a 5-line mini-plan before touching code.

## Phase 3: run

Mechanics:
- Start the smallest run that could give signal. Expected runtime: 15 min - 2 hr.
- Log to wandb.
- While running, draft expected outcomes.

Don't launch bigger runs until the small ones validate the setup.

## Phase 4: analyze

- Look at the wandb dashboard.
- Ask: was my hypothesis right, wrong, or inconclusive?
- Inconclusive is common and fine. Just note it.

Three framing questions:
1. **What surprised me?** (Chase surprises.)
2. **What did I expect?** (Check my calibration.)
3. **What should I do next?** (This is the output of the analysis.)

## Phase 5: update

Write it down. A one-paragraph update to your research log:

```
## 2026-05-03: RMSNorm vs LayerNorm on Shakespeare

Hypothesis: RMSNorm should match LayerNorm within 0.5% val loss.

Setup:
- nanoGPT Shakespeare config
- 3 seeds each for LayerNorm and RMSNorm
- 5000 iterations

Results:
- LayerNorm: val loss 1.470 ± 0.006
- RMSNorm: val loss 1.473 ± 0.004

Conclusion: Effectively identical. RMSNorm is slightly faster to compute.
Matches claim in literature.

Next: try at d16 scale to see if holds. Also check inference latency.
```

Now the next researcher (maybe future you) has the result.

## The log

Keep a **running research log**. A single file, dated entries. Hypotheses, experiments, results, updates.

The canonical example: [Karpathy's nanochat LOG.md](https://github.com/karpathy/nanochat/blob/master/dev/LOG.md). Study it for structure. Notice:
- Dated sections.
- Hypothesis / experiment / result / next.
- Some entries are failures.
- Short and frequent, not long and rare.

Your personal log will:
- Help you remember what you did six months ago.
- Reveal patterns across many experiments.
- Be a gold mine when writing papers or memos.

## Compute-aware research

Each experiment costs money. You have a budget.

Rules:
1. **Before a big run, predict the outcome.** If you're confident of the result, maybe don't run it.
2. **Do the cheap experiments first.** Small scale, short time.
3. **Kill bad runs early.** If loss is diverging at 10% through training, kill it.
4. **Batch similar experiments.** One big sweep > many separate runs.
5. **Don't overbook.** Finish analyzing one experiment before launching three more.

## Seed variance

An overlooked discipline: **same hyperparameters, different seeds, different results.**

For a small effect (< 1% loss difference), you must run 3+ seeds. Otherwise you're looking at noise.

Example fail: run A with seed 42 gets loss 1.47. Run B with seed 42 gets loss 1.44. Declare B better. But seed 47 for A gets 1.43, seed 47 for B gets 1.46. There's no real difference.

Run 3+ seeds per configuration for any claim you plan to publish or action.

## Good experiments produce good code

Experiments in PRs, not in scratch scripts:
- Committable and shareable.
- Tagged with the wandb run.
- Reviewable by peers.

Your PR description should answer:
- What hypothesis did this test?
- What was the result?
- What's next?

When you're senior, you'll spend more time reviewing others' PRs than writing your own. A good PR narrative is worth more than clever code.

## Failure modes

### Trap 1: "Running one big definitive experiment"
You want to know if X works. You design a huge $10K experiment. It takes 3 weeks. It fails. You've learned one bit in 3 weeks.

**Fix**: ladder. $100 toy experiment first. $1000 medium if promising. $10K only when confident.

### Trap 2: "Attached to a hypothesis"
You ran the experiment. Result disagrees with your hypothesis. You add complications to "rescue" it. You run variants. Weeks pass.

**Fix**: treat your hypotheses like candidates, not babies. Let them die gracefully.

### Trap 3: "No controls"
You run experiment A at 10K iters, B at 20K. B wins. But was it because of the idea or the extra iters?

**Fix**: always change exactly one variable at a time. If you must change more, spell it out.

### Trap 4: "No writeup"
You ran 20 experiments. None got written up. 6 months later, you remember nothing.

**Fix**: log entry per experiment, in the moment. 2 minutes of writing saves hours of reconstructing.

## The rhythm for a productive week

- Monday: plan 3-5 experiments. Estimate cost and expected outcome.
- Tuesday-Thursday: run, analyze, update log, iterate.
- Friday: write a one-page summary of the week's findings. Share with team/advisor.

This rhythm produces 3-5 log entries and one weekly memo. After a month: 15-20 experiments and 4 memos. After a year: 150+ experiments and 50 memos. That's a portfolio.

## Visualize this

**The research loop, one diagram**:

```
                 ┌─────────────────┐
                 │  Open question   │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │   Hypothesis     │  "I bet X helps Y because Z"
                 │   (specific,     │
                 │   testable,      │
                 │   falsifiable)   │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Experiment      │  minimum-sufficient design
                 │ design           │  compute estimate
                 │                  │  kill criteria
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Run              │  actually execute
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Analyze          │  what did we learn?
                 │                  │  vs. what we expected?
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Write it down    │  in your research log
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ New hypothesis   │  (or revised)
                 └────────┬────────┘
                          │
                          └──── loop ─────────┐
                                              │
                                              ▼
                                        (this is research)
```

**Good vs bad hypotheses**:

```
  BAD (vague, untestable):
  - "I think attention is important."
  - "Our model will work better."
  - "This should improve performance."
  - "Let me try something."

  GOOD (specific, measurable, falsifiable):
  - "Replacing LayerNorm with RMSNorm in a d12 nanoGPT reduces
     validation BPB by ≥ 3% with no change in wall-clock time."
  - "Setting AdamW β₂ to 0.95 instead of 0.999 reduces loss variance
     across seeds from ±0.02 to ±0.005 on our GSM8K RL task."
  - "Training on FineWeb-only (vs FineWeb+arXiv) costs 2% on GSM8K
     but improves MMLU by ≥ 1 percentage point."

  What makes GOOD hypotheses good:
  - Specific model / dataset / metric
  - Specific quantitative threshold
  - Can be disproven by one experiment
```

**The "pre-experiment" 5-line plan**:

```
  Before every experiment, write this in your log:

  ─── 2026-04-26, experiment XYZ ─────────────────────
  Hypothesis: [one sentence, specific, testable]
  Metric: [what will you look at?]
  Baseline: [what are you comparing against?]
  Budget: [compute and wall-clock estimate]
  Kill criteria: [when do you abandon?]
  Expected outcome: [what will the number be?]
  ─────────────────────────────────────────────────────

  Writing this catches bad experiments BEFORE launching.
  If you can't answer all 6, don't run yet.
```

**Common experiment designs**:

```
  1. A/B comparison (most common):
     Baseline vs your variant. Same everything else.
     Run 3 seeds each. Report mean ± std.

  2. Ablation:
     Full method vs method-with-X-removed.
     Reveals whether X matters.

  3. Scaling study:
     Run at sizes {small, medium, large}.
     Does your improvement hold at all scales?

  4. Hyperparameter sweep:
     Grid or random search over a few HPs.
     Find optimal config, report.

  5. Compute-matched:
     Fix total FLOPs. Vary other things.
     Directly comparable.

  Pick the simplest design that answers your question.
```

**Controls matter more than you think**:

```
  Bad experiment:
    Run A: baseline, d12, 100k steps
    Run B: my idea, d16, 150k steps
    Run B wins!

    But wait: did B win because of the idea? Or because d16 > d12?
    Or because 150k > 100k? Can't tell. Useless result.

  Good experiment:
    Run A: baseline, d12, 100k steps, seed 42
    Run B: baseline, d12, 100k steps, seed 43
    Run C: baseline, d12, 100k steps, seed 44
    Run D: my idea, d12, 100k steps, seed 42
    Run E: my idea, d12, 100k steps, seed 43
    Run F: my idea, d12, 100k steps, seed 44

    Compare means and std devs. Now the comparison is clean.
```

**Seed variance trap**:

```
  Don't be fooled:

  Run 1 of idea X:  val loss = 2.41
  Run 1 of baseline: val loss = 2.45
  "X wins by 0.04!" you declare.

  But the std across seeds of baseline is ±0.06.
  The "improvement" is noise.

  Rule: at least 3 seeds per config for any claim.
  Rule: if the effect size is < 2× std, it's noise.
```

**The weekly research rhythm**:

```
  Monday:     plan 3-5 experiments (5 lines each in log)
  Tue-Thu:    run, analyze, iterate
  Friday:     write 1-page summary of the week

  Ship weekly. Ship rough. Refine over quarters.

  A year = 52 weeks × 3-5 experiments = 150+ experiments.
  A year = 52 weeks × 1 writeup = 52 memos you can read later.

  That's how careers accrete.
```

## Exercises

1. Read Karpathy's LOG.md from nanochat. Pick 3 entries and note their structure.

2. Create your own LOG.md. Start with one entry for *something you ran this week*. Don't be precious; just get the format in place.

3. For your next experiment: write the 5-line plan before you start. Write the expected outcome before you look at results.

4. Review the last 3 things you "tried" technically. How many had clear hypotheses? How many had controls? How many do you remember in detail?

## Next

`03_project_scoping_and_killing_projects.md` - the hardest senior skill.
