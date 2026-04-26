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

## Exercises

1. Draw your current team's org. Roles, seniorities. Any gaps?

2. Write a job description for your next hire. Include: must-haves, nice-to-haves, day-in-the-life, growth path.

3. Audit your last 5 hiring decisions. Were they good? If they went bad, what were the red flags you missed?

4. Identify someone on your team with growth potential. What's your plan for them?

## Next

`07_working_with_product_and_eng.md` - cross-functional life.
