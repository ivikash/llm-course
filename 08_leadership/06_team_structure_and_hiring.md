# 8.6 - Team Structure and Hiring

At some point you'll need more than one person. Building an ML team is an art. Key principles here.

## Roles in a modern ML team

Real teams have diverse skills. You can't hire one "ML engineer" and expect them to cover everything.

### Core technical roles

**Research Scientist (RS)** - the person who generates hypotheses, designs experiments, interprets results. Often PhD-trained, but not always. Measures success in insights and published/shipped techniques.

**Research Engineer (RE)** - the person who builds the training infrastructure, debugs mysterious performance issues, makes sure experiments can actually run. Measures success in throughput, reliability, reproducibility.

**Data Engineer** - the person who builds data pipelines, cleans data, manages large-scale storage and processing. Underrated role. Great data beats great models most of the time.

**Eval Engineer** - the person who builds and maintains the eval suite. Often doesn't exist as a dedicated role in smaller teams, but should. Module 8 Lesson 4.

**Infra Engineer** - the person who manages clusters, scheduling, monitoring. Essential at scale.

### Product / cross-functional roles

**ML Product Manager** - the person who defines "what should the model do" in concrete terms. Interfaces with customers, translates needs into evals.

**UX / HCI Researcher** - for products with chat interfaces, this matters a lot. How should the interface surface uncertainty? Handle errors?

**Safety / Policy Researcher** - for frontier or deployed models, someone has to think about misuse, red-teaming, governance.

## Team sizes and shapes

### Solo / duo (1-2 people)
- Individual contributors doing research.
- Wearing many hats.
- Scope: experiments and writeups, not products.

### Small team (3-8 people)
- Typically one lead + 5-7 ICs covering: 2-3 researchers, 2-3 engineers, one data/eval person.
- Can ship a model and maybe serve it in a product.
- Scope: one research program with a product outcome.

### Medium team (10-30 people)
- Multiple focused sub-teams.
- Dedicated eval and data teams.
- Scope: multiple research directions plus product.

### Large (30+)
- Org structure, VPs of Research, etc.
- Complex coordination.
- Examples: OpenAI, Anthropic research teams.

Most of us work at small-to-medium.

## Hiring principles

### What to filter for

In rough order of importance:

1. **Curiosity**. Can they get obsessed with a problem?
2. **Follow-through**. Can they finish projects? Evidence: shipped code, published papers, merged PRs.
3. **Technical depth** (in something specific, not everything).
4. **Communication**. Can they write a clear memo?
5. **Honesty**. Will they say "I don't know" when they don't?
6. **Taste**. (Harder to test in interviews, but try.)

### What NOT to filter for (as primary signals)

- School pedigree (useful but weak on its own).
- Years of experience (often poorly correlated with quality).
- Leetcode ability (negative correlation for research roles).
- Buzzword familiarity (often fakes understanding).

## Interview formats that work

### 1. Take-home / homework
Give them a small research problem. 4-8 hours of their time. They come back with:
- Code that runs.
- A short writeup.
- Numbers.

Evaluate: Did they scope correctly? Did they write good code? Did they make good tradeoffs? Can they explain their decisions?

This is the single best signal. Insist on it for serious roles.

### 2. Research discussion
60 minutes. Ask them about a recent paper they read, or a project they worked on.

Probe: What went wrong? What did they learn? Why is the work important? How would they extend it?

Red flags: can't name any project's weaknesses, can't articulate trade-offs, can't distinguish signal from hype.

### 3. Live coding
Not Leetcode. Have them implement a specific, ML-relevant thing: a small transformer layer, a custom loss function, a KV cache.

Observe: how do they think about shapes? Do they test their code? Do they notice bugs?

### 4. System design
For senior hires only. "Design a training pipeline for a 70B model." Observe architecture thinking.

## Red flags in interviews

- Dismissive of others' work without specific technical critique.
- Can't explain a failed project.
- Over-claims expertise.
- Talks only about models, never about evaluation or data.
- No opinion on open debates (Mamba vs Transformer, MoE vs dense, etc).

## Green flags

- Strong convictions, flexibly held.
- Mentions failures naturally.
- Asks good questions about the team / role.
- Has a GitHub with real projects.
- Articulates what they're weak at.

## Team composition

Avoid homogeneity. A team of 10 "researchers with PhDs from the same lab" has shockingly little diversity of thought. Recipe for collective blindspots.

Mix:
- Industry and academic backgrounds.
- Research and engineering strengths.
- Age / career stage.
- Domain expertise (NLP, CV, RL, systems).

Diversity in problem-solving styles > diversity in identity alone, but both matter.

## The tech lead role

For a 5-10 person ML team, the tech lead:
- Makes final calls on research direction.
- Runs weekly technical reviews.
- Drives the roadmap.
- Mentors juniors.
- Protects team from org chaos.

Ideally, not the manager. Ideally, spends > 50% of time still in the code.

## The ML engineer paradox

ML moves fast. Someone who's been an ML engineer for 2 years may know more than someone with 10 years of "data science."

Be open to hiring people with 1-2 years of focused LLM experience over people with 10 years of classical ML.

## Growing vs hiring

You can't always hire the senior you need. Sometimes you need to grow them.

Signals someone is ready to grow:
- Consistently delivers on their current scope.
- Seeks feedback actively.
- Mentors juniors informally.
- Has strong opinions backed by experience.

Give them stretch projects. Promote eagerly when they deliver.

## Firing

As painful as it sounds. Do it when:
- Someone is consistently underperforming despite feedback.
- Someone is toxic to the team culture (brilliant or not).
- Someone is hired into the wrong role (move or let go).

Lazy firing is slow and hurts teams. Compassionate firing is fast, generous, and gives everyone clarity.

## Managing without authority

Often you lead without being someone's manager. You have influence, not command.

How to make this work:
- **Be right more often**: earn deference with track record.
- **Be generous with credit**: people follow you if you lift them.
- **Write clearly**: your memos become the consensus.
- **Unblock others**: become someone everyone wants to work with.

Pure positional power rarely makes good ML leaders. Influence does.

## Culture

### Healthy culture signs
- People push back on ideas constructively.
- Failures are discussed openly.
- Killing projects is not career-ending.
- Junior people feel safe asking "stupid" questions.
- Work-life balance is respected (burned-out researchers don't research well).

### Unhealthy signs
- Everyone agrees with the most senior person.
- Nobody mentions failures.
- "Crunching" is normalized.
- Team stratifies into "genius" and "implementers."

## Visualize this

**A healthy ML team org structure (size 10-15)**:

```
                  Research Lead
                       │
       ┌───────────────┼───────────────┬───────────────┐
       │               │               │               │
    Research       Research       Research       Research
    Scientist      Scientist      Engineer       Engineer
    (PhD, ideas)   (PhD, ideas)   (systems)      (infra)
       │               │               │               │
       └───────────────┴───────────────┴───────────────┘
                            │
                  ┌─────────┼─────────┐
                  │         │         │
             Data Eng    Eval Eng   ML PM
             (pipelines) (benchmarks)(product)

  ~1 lead + 2 scientists + 2 engineers + 1 data + 1 eval + 1 PM = 8
  Good ratio: scientists/engineers roughly 1:1 (not 3:1!)
```

**Roles and what they do day-to-day**:

```
  Research Scientist (RS):
    - Generates hypotheses
    - Designs experiments
    - Writes papers / memos
    - Usually PhD background
    - Day: coding + writing + reading

  Research Engineer (RE):
    - Builds training infrastructure
    - Debugs perf issues
    - Writes tooling
    - Strong software engineering
    - Day: coding + debugging + reviewing

  Data Engineer:
    - Builds data pipelines
    - Quality filtering, dedup
    - Often overlooked; huge impact
    - Day: SQL + distributed systems + LLM data quality

  Eval Engineer:
    - Builds and maintains eval suite
    - Runs benchmarks, interprets
    - Red-teams models
    - Day: datasets + metrics + reports

  ML Product Manager:
    - Translates biz needs to evals
    - Manages release coordination
    - Protects research time
    - Day: meetings + docs + prioritization
```

**Interview signals (green vs red flags)**:

```
  Green flags (hire):
    ✓ Can articulate a failed project and what they learned
    ✓ Strong opinion on a debate (MoE vs dense, etc.) backed by experience
    ✓ GitHub with real, shippable projects
    ✓ Asks you thoughtful questions
    ✓ Says "I don't know" comfortably
    ✓ Specific about past work ("I tuned X from 0.42 to 0.51 on Y")
    ✓ Curious, quick to connect ideas

  Red flags (don't hire):
    ✗ Can't name a project weakness
    ✗ Dismisses others' work without technical critique
    ✗ Leetcode-perfect but can't explain their past ML work
    ✗ Buzzword-heavy but details thin
    ✗ No GitHub / no published work at all
    ✗ Hostile to feedback
    ✗ All claims are generic ("built an LLM model")
```

**A good ML take-home test**:

```
  Task (4-8 hours):
    "Given the nanoGPT Shakespeare config, propose one
     architectural change and test whether it improves
     val loss on that dataset. Submit:
     - Your modified code (as a PR-style diff)
     - A 1-page writeup of your hypothesis, setup, and results
     - Commentary on what you'd do next with more time."

  What this tests:
    ✓ Can they code? (diff quality)
    ✓ Do they follow the research loop? (writeup structure)
    ✓ Are they scientific? (seeds, ablations)
    ✓ Communication? (1-page vs 5-page)
    ✓ Taste? (which change did they pick?)
    ✓ Honesty? (do they report failures?)

  Better than any Leetcode interview for ML roles.
```

**Research culture: healthy signs vs unhealthy signs**:

```
  Healthy:
    ✓ Junior researcher pushes back on senior, respectfully. Both update.
    ✓ Someone says "I was wrong about X" in a meeting. No shame.
    ✓ Killed projects discussed openly, post-mortemed.
    ✓ Everyone reads each other's memos.
    ✓ "Show don't tell" - results in plots, not claims.
    ✓ Tight feedback loop on PRs.
    ✓ Work-life boundaries respected.

  Unhealthy:
    ✗ Senior person's ideas always win regardless of evidence.
    ✗ Nobody mentions failures. Shiny demos only.
    ✗ "Crunching" is routine.
    ✗ Dead projects keep getting resources.
    ✗ Juniors never speak up in meetings.
    ✗ Team stratified into "geniuses" and "implementers".
    ✗ High turnover.
```

**Team diversity (more important than it sounds)**:

```
  Good team has diverse:
    Backgrounds:    industry vs academia
    Experience:     senior mentors + junior builders
    Specialties:    systems + science + data + eval
    Problem-solving styles: bottom-up coders + top-down thinkers
    Identity:       gender, race, culture (brings different heuristics)

  Anti-pattern:
    Entire team is "ex-DeepMind PhDs" → echo chamber
    Entire team is "self-taught builders" → no first-principles
    All men or all women → biased product
    Everyone from same lab → same blind spots
```

**Growing vs hiring**:

```
  When to GROW someone internally:
    ✓ Reliably delivers on current scope
    ✓ Seeks feedback, applies it
    ✓ Mentors juniors informally already
    ✓ Has strong opinions from experience

  When to HIRE externally:
    ✓ Need specific expertise team lacks
    ✓ No internal candidate for the role (in 1 year)
    ✓ Adding, not replacing
    ✓ Want cultural infusion

  Rule: grow when you can, hire when you must.
  Promoting someone is often worth 2-3 external hires.
```

**The "firing" reality**:

```
  Hard truth: you will occasionally need to let people go.

  Reasons that DO justify firing:
    - Consistent underperformance after feedback (3+ months)
    - Toxic to team culture (brilliant jerk syndrome)
    - Wrong role fit (offer to move or let go compassionately)
    - Illegal / unethical behavior

  Reasons that DON'T:
    - One bad quarter
    - Disagrees with you on technical direction
    - Is introverted / not charismatic
    - Isn't promoted "fast enough"

  Lazy firing is slow and hurts teams.
  Compassionate, decisive firing is respect.
```

## Exercises

1. Draw your current team's org. Roles, seniorities. Any gaps?

2. Write a job description for your next hire. Include: must-haves, nice-to-haves, day-in-the-life, growth path.

3. Audit your last 5 hiring decisions. Were they good? If they went bad, what were the red flags you missed?

4. Identify someone on your team with growth potential. What's your plan for them?

## Next

`07_working_with_product_and_eng.md` - cross-functional life.
