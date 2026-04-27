# 8.5 - Managing Compute and Budgets

Compute is the physical substrate of ML work. Managing it is a skill. This lesson covers the principles.

## The scarcity

No matter how big your company, compute is scarce. OpenAI, Anthropic, Meta - everyone is always negotiating GPU allocations internally.

Why? Because:
- Bigger models / longer runs always seem attractive.
- Product teams always need more inference.
- New projects compete with ongoing ones.

Your job as a leader: make scarce compute produce maximum research velocity.

## The budget planning exercise

Quarterly, for your team:

1. **Total compute available** (GPU-hours this quarter).
2. **Unavoidable** (ongoing training, inference for products).
3. **Available for research** (what's left).
4. **Divide** into: exploration / validation / scaled experiments.
5. **Buffer**: 10-20% for unforeseen needs.

Example for a team with 8xH100 node access for 3 months:
- 8 × 24 × 90 = 17,280 GPU-hours available.
- Unavoidable (serving, eval): 3,000 hours.
- Available for research: 14,000 hours.
- Exploration (many small runs): 5,000.
- Validation (medium runs for promising ideas): 5,000.
- Scaled (big bets): 2,000.
- Buffer: 2,000.

Now tell your team: "Each researcher has ~2,000 hours. Plan accordingly."

## Cost-aware research

Every experiment should have an explicit cost estimate before it starts.

```
expected_cost = expected_hours * num_gpus * hourly_rate
```

A d12 nanochat model for 2 hours on 8xH100: ~16 GPU-hours × $3 = $48.

A junior launches runs without thinking about cost. A senior always knows.

## Queuing and fairness

On shared infrastructure, you need a queue system:

- **Slurm**: classical HPC scheduler. Works.
- **Kubernetes + Kueue**: cloud-native.
- **Internal tooling**: most labs build their own.

Principles:
1. **Priority tiers**: critical / research / nice-to-have. Critical skips the line.
2. **Per-user quotas**: prevent one person hogging.
3. **Preemption**: high-priority jobs bump low-priority ones.
4. **Visibility**: everyone can see the queue and utilization.

Bad queuing (first-come-first-serve, no preemption) leads to:
- Users launching many small jobs to hold resources.
- Queue stuffing.
- Resentment.

Good queuing encourages honest resource requests.

## The "idle machines" problem

Machines sitting idle are the #1 waste in ML organizations.

Typical causes:
- Someone launched a job, went on vacation, job crashed, no one noticed.
- A job finished but the allocation didn't release.
- Someone is "warming up" the machine for future use.

Countermeasures:
- Auto-detect idle GPUs with < 20% util for > 30 min, alert owner.
- Reclaim idle machines after 60 min.
- Culture: "report underutilization" is a good team norm.

## Compute-bound vs researcher-bound

Two regimes:

**Compute-bound**: researchers could iterate faster if they had more GPUs. Classic for frontier labs.

Solution: buy more GPUs.

**Researcher-bound**: unused GPU capacity, but researchers can't design enough good experiments. Common for labs transitioning from small to large scale.

Solution: hire/train more researchers. Improve tooling to make each researcher faster.

Diagnose which one you're in:

| Signal | Compute-bound | Researcher-bound |
|--------|--------------|------------------|
| GPU utilization | >90% | <60% |
| Time-in-queue | Hours to days | Minutes |
| Researchers working late | Yes | No |
| Ideas waiting for compute | Long backlog | Few |

Different problems, different solutions.

## Training large models: the planning matrix

For a big pretraining run, answer:

- **Hypothesis**: what do we expect to learn?
- **Baseline**: what previous run do we compare against?
- **Budget**: exact FLOPs or GPU-hours.
- **Predicted result**: loss range, benchmark range.
- **Kill criteria**: at which step do we declare failure.
- **Checkpoint frequency**: how often do we save (trade-off: storage vs safety).
- **Reviewer**: who signs off before launch.

For a $500K+ run, this should be a formal document and go through a review committee.

## Sharing learnings across the org

Sometimes teams duplicate work. To prevent:

- **Shared experiment log**: every team writes up results in a common format.
- **Weekly cross-team syncs**: 30 min of "what did we each learn?"
- **Published internal memos**: so searchable.
- **Shared baselines**: one baseline model/config everyone agrees to, to avoid redundant training.

## Cost per learning

A useful metric: **$ / insight**.

Example:
- Team A ran one $100K experiment. Found one surprising result.
- Team B ran 20 x $5K experiments. Found 3 surprising results.

Same total cost; Team B got 3x more learnings. Generally prefer many small over few large - assuming the small ones can actually surface the kind of insight you need.

Exception: **scaling laws matter at scale.** Some phenomena only show up at >1B params. You can't avoid big runs entirely, but you can delay them until well-prepared.

## Negotiating more compute

When you need more:

1. **Show current utilization** (high = you need more, low = you need to use what you have).
2. **Propose a specific experiment** (not "more resources in general").
3. **Explain the expected outcome** (a measurable win).
4. **Commit to killing if it fails** (so you're not just asking for a blank check).

This kind of request wins. "We need more GPUs" loses.

## Cost engineering for production

If you're also responsible for serving the model:

- **Batching**: biggest lever. 10-50x throughput for multi-user serving.
- **Caching**: prompt caching, KV reuse.
- **Quantization**: 4x smaller in int4 (some quality loss).
- **Distillation**: a smaller model trained to mimic the big one. Production cost drops dramatically.
- **Workload prediction**: auto-scale GPU fleet to demand.

Module 6 Lesson 10 goes deeper.

## A quick framework for when to buy more vs. rent

**Buy (or reserve cloud capacity)**:
- Sustained usage, predictable.
- Regulatory / privacy requirements.
- Cost crossover point: if renting > buying amortized over 2-3 years, buy.

**Rent (on-demand)**:
- Experimentation, bursty.
- New direction, unsure if it'll stick.
- No internal ops team to manage metal.

Most startups rent. Most big labs have a mix.

## Visualize this

**A team's compute budget pie**:

```
  Quarterly compute for a 5-person team (17,280 GPU-hours on 8×H100):

  ┌─────────────────────────────────────────────────┐
  │ Unavoidable (serving, eval):            17%      │  3,000 hr
  │  └─ ████                                         │
  │                                                   │
  │ Exploration (many small runs):           29%     │  5,000 hr
  │  └─ ██████████                                   │
  │                                                   │
  │ Validation (promising ideas):            29%     │  5,000 hr
  │  └─ ██████████                                   │
  │                                                   │
  │ Scaled experiments (big bets):           11%     │  2,000 hr
  │  └─ ████                                         │
  │                                                   │
  │ Buffer (unplanned):                      12%     │  2,000 hr
  │  └─ ████                                         │
  └─────────────────────────────────────────────────┘

  → ~2,800 GPU-hours per researcher per quarter
  → Each researcher should plan 20-30 experiments fitting in that budget
```

**Cost estimation before launching**:

```
  Back-of-envelope formula:

  cost_hours = FLOPs / (peak_FLOPs_per_sec × MFU)

  Example: d24 nanochat for 100k steps
    FLOPs/step ≈ 3 × N × tokens_per_step × 2
                = 3 × 1.5e9 × 500k × 2 ≈ 4.5e15 FLOPs/step

    100k steps = 4.5e20 FLOPs total

    8×H100 bf16 peak: 8 × 989 TFLOPS = 7.9 PFLOPS
    At 45% MFU: 3.6 PFLOPS

    Hours: 4.5e20 / 3.6e15 / 3600 ≈ 35 hours

    Cost at $25/hr (Lambda): ~$875

  Always estimate before running. Catches mistakes.
```

**The idle GPU problem**:

```
  nvidia-smi on a lab cluster (hypothetical):

  GPU 0: 95% util   ← healthy training run
  GPU 1: 94% util   ← healthy
  GPU 2:  3% util   ← hmm
  GPU 3:  0% util   ← !!
  GPU 4: 98% util   ← healthy
  GPU 5: 68% util   ← fine
  GPU 6:  5% util   ← !!
  GPU 7: 92% util   ← healthy

  3 out of 8 GPUs essentially idle.
  At $3/hour per GPU, 3 GPUs idle for a week = $1,500 wasted.
  Multiply by many labs = enormous waste.

  Solutions:
  - Auto-detect idle, alert owner, reclaim if no response in 30 min
  - Quota + fair-share queueing
  - Visible utilization dashboards (social pressure)
```

**Compute-bound vs researcher-bound (which are you?)**:

```
  Compute-bound:                     Researcher-bound:
  ──────────────                     ──────────────────
  ✓ GPU util > 90% sustained          ✗ GPU util < 60%
  ✓ Long queues waiting                ✗ GPUs sit idle
  ✓ Researchers waiting > 24h          ✗ Researchers run few experiments
  ✓ Ideas backlog > compute supply     ✗ Lots of compute unused

  Different fixes:
  → Buy more GPUs                      → Hire more / train more researchers
                                        → Build better tooling
                                        → Smaller scale experimentation
```

**Cost-per-insight metric**:

```
  Compare strategies over a quarter:

  Strategy A (big bets):
    1 run × $100K = $100K
    Got: 1 big result (or 1 null)
    Cost per insight: $100K

  Strategy B (many small):
    20 runs × $5K = $100K
    Got: 6 insights (some positive, some negative, all valuable)
    Cost per insight: $17K

  Same budget, 6× more learnings in Strategy B.
  Unless your question requires scale (then Strategy A).
```

**Negotiating for more compute**:

```
  The wrong pitch:
    "We need more GPUs."

  The right pitch:
    "Here's our current 95% utilization dashboard.
     Here's the backlog of 12 experiments waiting for capacity.
     Each experiment has estimated cost $X.
     Collectively, they'd answer 5 business-critical questions.
     Timeline: 4 weeks if we have 2× current compute, else 10+ weeks.
     Ask: 8 more H100s for 4 weeks ($25K)."

  Show data. Propose specific work. Commit to outcomes.
```

**A compute governance checklist for leads**:

```
  Weekly:
    [ ] Check utilization dashboard
    [ ] Review experiment queue
    [ ] Any idle > 30 min? Investigate
    [ ] Any expired-but-running jobs?

  Monthly:
    [ ] Total spend vs plan
    [ ] Per-person utilization
    [ ] Cost-per-insight metrics
    [ ] Any unused reserved capacity?

  Quarterly:
    [ ] Budget rebalancing
    [ ] Hardware upgrade planning
    [ ] Reserved vs on-demand decisions

  Annually:
    [ ] Vendor negotiation / renewal
    [ ] Long-term capacity planning
```

## Exercises

1. Estimate your team's (or your personal) compute budget for the next month. List all expected usages. Sum.

2. Identify one way you could reduce cost by 30%. (Smaller models? Shorter runs? Better batching?)

3. For your most expensive planned experiment: write the "pre-flight" doc (hypothesis, budget, kill criteria).

4. Audit your GPU utilization over the last week (if you have data). Any idle time? Why?

## Next

`06_team_structure_and_hiring.md` - who's in the team, and how to pick them.
