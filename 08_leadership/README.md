# Module 8 - Leading Science Teams

You can be the best individual researcher in the world and your team will still fail without good leadership. Most "AI science team" failures come from leadership gaps, not technical ones. This module collects what isn't in textbooks.

## Lessons (each ~30-60 min of reading, much longer to internalize)

### `01_what_ml_scientists_actually_do_day_to_day.md`
- The myth: "coming up with genius ideas."
- The reality: 30% coding, 30% debugging, 20% reading papers, 10% meetings, 10% writing.
- The experiments-per-week metric.
- "Research velocity" > "research quality on any single project".

### `02_the_research_loop.md`
- Hypothesis → design → experiment → analyze → update beliefs → next hypothesis.
- How to form good hypotheses (specific, testable, falsifiable).
- Experiment design: control variables, null hypotheses, compute budget per experiment.
- "Compute-aware" research: set a budget upfront.
- Negative results matter. Document them.

### `03_project_scoping_and_killing_projects.md`
- Scope: "what will we have at the end?" specific deliverable.
- Estimating: multiply by 2x or 3x the time you think.
- The hardest senior skill: killing projects that aren't working.
- "Kill early, kill cheap" - the longer a project runs, the harder to kill.
- Post-mortems: what did we learn? Document for the org.

### `04_evaluations_as_the_product.md`
- Thesis: "Your evals ARE your model." If you can't measure it, you can't improve it.
- Building new benchmarks is often more valuable than building new models.
- Public benchmarks vs internal benchmarks. Why you need both.
- Eval-driven development: no new method without a new eval.
- Goodhart's Law: when a measure becomes a target, it stops being a good measure. So evolve evals.

### `05_managing_compute_and_budgets.md`
- Every LLM team has a finite compute budget.
- Compute planning: how many experiments at what scale fit in a quarter?
- Killing idle machines (culture issue).
- Queuing, fairness, priority. Tools like Slurm, Kubernetes, or internal schedulers.
- The scaling-law cost estimator: before committing, compute the expected cost.

### `06_team_structure_and_hiring.md`
- Team roles: research scientist, research engineer, data engineer, eval engineer, infra engineer, ML product manager.
- How they overlap in small teams (one person often does 3 jobs).
- Hiring bar: pick for curiosity and follow-through, not pedigree.
- Interviews that work: "tell me about an experiment that surprised you" beats Leetcode.
- Avoiding monocultures - different thinking styles.

### `07_working_with_product_and_eng.md`
- Research teams vs product teams: different time horizons, different vocabularies.
- Translating research results into product features.
- Handoff: what does "shippable" mean for your org?
- The "demo trap": demos that impress executives but don't generalize.
- Writing user-facing docs for your model.

### `08_communicating_results_up_and_out.md`
- One-page memos for executives.
- Three-number summary: capability, cost, risk.
- Graphing discipline: one plot, one message.
- Handling the "why aren't we better than OpenAI?" question.
- Public communication: blog posts, papers, tweets - how much is too much?

### `09_research_taste.md`
- What is taste? The ability to pick problems worth solving.
- Signs of good taste: your problems keep "yielding" even after you thought you were done; you find small details that turn into big insights.
- Signs of poor taste: chasing SOTA on old benchmarks, copying fashionable methods without understanding them, mistaking scale for substance.
- How to develop it: read constantly, try your own small experiments often, seek unfiltered feedback from more experienced researchers.
- The "strong opinions, loosely held" mode.

### `10_staying_current_in_a_field_that_moves_weekly.md`
- The firehose problem: you cannot read everything.
- Filters: a few Twitter/X researchers, a few newsletters (Interconnects, Jack Clark, Import AI), arXiv sanity, Papers with Code.
- Deep dives vs broad scans: alternate rhythms.
- Communities: Discord servers (Eleuther, LMSYS), conferences, internal slack.
- Building your own intuition pump: a personal doc of "things I've learned this year."

## Why this module

Most engineers, when they rise, are thrown into management without preparation and founder. This module is the distilled set of lessons that people usually only learn by failing for 2-3 years. Read it once now to know what to watch for. Re-read it when you start actually leading.
