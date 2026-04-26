# 8.7 - Working with Product and Engineering

Research teams and product teams live in different time zones. Research thinks in quarters; product thinks in sprints. Neither is wrong. Both must coexist.

## The cultural divide

| Dimension | Research | Product/Eng |
|-----------|----------|-------------|
| Time horizon | Quarters-years | Weeks-months |
| Metrics | Loss, benchmarks | Latency, cost, retention |
| Deliverable | Insight, paper, model | Feature, user experience |
| Tolerance for failure | High | Low |
| Language | FLOPs, bpb, MFU | p95, SLOs, cost-per-user |

When teams clash, it's usually because they don't see each other's constraints. Leaders translate.

## The research-to-product pipeline

A model goes through phases:

### Phase 1: research breakthrough
- New technique tested in isolation.
- Works on a benchmark.
- Nobody outside the team has used it.

### Phase 2: productization
- Someone wraps it in an API.
- Integrated into a product.
- Used by a small number of users.

### Phase 3: scale
- Performance engineered for low latency, low cost.
- Robust to failures.
- Used by millions.

Each phase has different failure modes and different heroes.

## What research forgets about product

- **Latency matters more than quality past a threshold.** A 99% accurate model with 5-second response time loses to a 97% model with 500ms.
- **Cost compounds.** At 1M users × 10 queries/day, $0.001/query = $10K/day. Even small cost increases break business models.
- **Edge cases dominate.** A researcher cares about average performance. A product cares about the 0.1% catastrophic failure that becomes a viral screenshot.
- **Reliability is invisible when present.** An outage is immediate; good uptime never gets celebrated.

## What product forgets about research

- **Insights take time.** You can't "just try X" in a week if X requires 3 months of experimentation.
- **Negative results are valuable.** "We tried RLHF and it didn't help for our use case" is worth knowing.
- **Model changes are not feature flags.** Replacing a model affects every user. Blast radius considerations.
- **Reproducibility is hard.** Training runs aren't deterministic the way CI/CD is.

## Translation patterns

### From research to product
- Wrap model in stable API.
- Define SLOs (response time, availability).
- Set up monitoring.
- Write runbooks.
- Plan for rollback.

### From product back to research
- User behavior data.
- Failure modes (what confuses users? what do they abandon?).
- Which capabilities are most-used vs least-used.
- Real-world latency and cost constraints.

The research-product loop should be bidirectional. Research ships to product; product's real-world data feeds back into research priorities.

## Shared artifacts

Good cross-functional teams share:

1. **A single eval suite**. Product teams should be able to run it on candidate models.
2. **A shared dashboard** showing current model performance on shared metrics.
3. **Regular demos** - research shows product what's coming. Product shows research what users actually do.
4. **A model card** per release. Specifications, known limitations, evaluation results.

## The "demo trap"

Common failure: researcher builds a demo showing an impressive capability. Executive / product team gets excited. They want to ship it. But the demo was hand-picked, the capability is inconsistent, and shipping causes user frustration.

Rule: **a demo is not a capability.** Before any productization commitment, show consistent performance across a representative sample of real inputs. Not just the 5 examples that worked.

## The "science team in a product org" tension

If research is in a product-org reporting chain:
- Product cycles pressure research.
- Research goals compress into a single quarter.
- Long-horizon bets get killed.

If research is separate:
- It drifts from product reality.
- Invents solutions to problems users don't have.

Balance: research should be somewhat insulated, but have a "product embedded" channel where user data flows in and prototypes flow out.

## The product-manager relationship

An ML PM is valuable when they:
- Translate business needs into testable metrics.
- Protect research time from constant product firefighting.
- Push back on unrealistic timelines.
- Champion evals as a first-class product.

An ML PM is harmful when they:
- Demand features without understanding ML limits.
- Compress timelines below research feasibility.
- Promise capabilities the model can't deliver.

As a research leader, invest in the PM relationship. Most research failures I've seen were PM-research communication failures, not technical failures.

## Hands-on engineering coordination

Things to agree on explicitly between research and engineering:

### Model packaging
- Format: pickle, safetensors, GGUF?
- Versioning: model name, training run ID, data version.
- Release cadence: how often does production get a new model?

### API contract
- Input/output format.
- Rate limiting.
- Error handling.

### Monitoring
- Latency percentiles.
- Quality metrics (from eval suite, online).
- Cost tracking.

### Rollback
- How fast can we revert to the previous model?
- What's the A/B testing setup?
- What's the canary rollout?

Without these agreements, shipping new models is chaos.

## Communication rhythms

Weekly or bi-weekly joint syncs between research and product:
- Research shares: what we learned this week, upcoming experiments.
- Product shares: user feedback, usage patterns, feature requests.
- Together: prioritization, blockers.

Keep it short (30 min). Not every researcher needs to attend. Lead researcher + lead PM + 1-2 others.

## Shipping a new model

A checklist:

- [ ] Eval suite passed (with prior-model comparisons).
- [ ] Red team suite passed (safety).
- [ ] Latency acceptable (p50, p95).
- [ ] Cost acceptable (per query, per user).
- [ ] Rollback plan documented.
- [ ] Monitoring dashboards updated.
- [ ] Model card published.
- [ ] Customers notified (if external API).

Missing any? Shipping is a risk.

## Exercises

1. For your current or last project: identify the biggest research-product communication failure. What caused it?

2. Sketch a shared dashboard (metrics) that both research and product would look at weekly.

3. If you don't currently have an ML PM, what's the closest equivalent role? What's missing?

4. Write a model card for a hypothetical release. Sections: intended use, training data, evaluation, known issues.

## Next

`08_communicating_results_up_and_out.md` - memos and executive updates.
