# 8.1 - What ML Scientists Actually Do Day-to-Day

You read the headlines about "AI researchers." You should know what the job actually looks like. Spoiler: less dramatic than the press releases, more grinding than the textbooks suggest.

## The breakdown

A typical week for a mid-level ML researcher at a frontier lab or serious startup:

| Activity | % of time |
|----------|-----------|
| Coding, debugging, running experiments | 35% |
| Reading papers, colleagues' code, Slack threads | 20% |
| Meetings, syncs, planning | 15% |
| Writing (docs, memos, papers, PRs) | 15% |
| Thinking/walking (no joke, this is how problems get solved) | 10% |
| Other (hiring, team stuff, admin) | 5% |

Contrast to what people imagine:
- "Coming up with genius ideas": ~5% of the work.
- "Whiteboarding with Nobel laureates": rare.
- "Using a supercomputer": yes, but mostly through a terminal.

## The shape of a project

A typical 3-month research project:

### Week 1-2: scope and baseline
- Understand the problem.
- Find or build a baseline.
- Define the metric you'll track.
- Estimate compute cost.

### Week 3-6: iterate
- Hypothesize.
- Run a small experiment.
- Look at results.
- Update hypothesis.
- Repeat.

Most hypotheses are wrong. 80% of ideas don't work. Senior researchers are comfortable with this; juniors get frustrated.

### Week 7-10: refine the best idea
- Pick the top 1-2 things that worked.
- Run at bigger scale.
- Ablations.
- Negative results documented.

### Week 11-12: write up
- Blog post, memo, or paper.
- Code cleanup for release.
- Share with team.
- Plan next project.

Most projects produce: a 2-4 page writeup + working code + some benchmark numbers. That's the unit of output.

## The daily rhythm

### Morning
- Check yesterday's runs. Did they finish? What did the numbers say?
- Usually 1-3 runs running overnight. Some crashed (~20% rate). Restart.
- Spend 30 min looking at wandb plots, making sense.

### Midday
- Code new experiments.
- PR review.
- Maybe a meeting or two.

### Afternoon
- Push new runs.
- While they run: read one paper, answer Slack.
- Work on writeup for last week's project.

### Evening (often, honestly)
- Think. Walk. Take a shower. Real insights come when you step away.

Good scientists don't grind constantly. The "ideas phase" is not hands-on keyboard. The "execution phase" is.

## Velocity

One metric matters most for individual productivity: **experiments per week**.

- New researchers: 1-3 per week. Often slow debugging cycles.
- Mid-career: 5-15.
- Top performers: 20+.

This doesn't mean "running 20 different projects"; it means "20 gradient updates on understanding." Each experiment tests one small hypothesis, finishes in 4-24 hours, informs the next.

Frameworks to maximize velocity:
- Small models first. Never test ideas at scale.
- Templates for experiments. Avoid re-coding common parts.
- Automated post-processing: wandb plots, generated reports.
- Set a time budget per idea. Kill if over.

## Tooling setup

Every productive researcher has:
- A good compute allocation (laptop + cluster access).
- wandb (or equivalent).
- A git-backed workspace where everything lives.
- Clean keyboard shortcut muscle memory.
- Fast internet and a fast SSH stack (mosh, tmux).
- A reliable way to reproduce any old experiment.

If you're spending > 20% of your time fighting tools, fix tools. Time invested in tooling compounds.

## The underrated skill: spotting the interesting thing

Running experiments is easy. Knowing which result is **surprising and worth investigating** is the real skill.

Example: you train a model, val loss goes down smoothly. Boring, expected. You train another, val loss *spikes at step 500* and then settles. Interesting! Why?

That spike might be a bug. It might also be the seed of a new discovery. The skill is noticing it and chasing it down, not just shrugging and saying "training noise."

"Pattern recognition over many experiments" - that's the skill that develops over years.

## What differentiates seniors from juniors

Seniors:
- Know which experiments **not** to run. Compute is always scarce.
- Have good intuitions from many prior runs.
- Write tight memos - one page over ten.
- Kill projects that aren't working, even their own.
- Mentor and unblock juniors.

Juniors:
- Run lots of experiments, not all valuable.
- Debug longer than they should.
- Get attached to their ideas.
- Write long writeups trying to convince.

This is natural. Gap closes over ~3-5 years if you're paying attention.

## Myths busted

- **Myth**: "AI research is all math." **Reality**: mostly engineering + taste.
- **Myth**: "You need a PhD." **Reality**: helps for some roles, not required for most. Skills > credentials.
- **Myth**: "Researchers are geniuses." **Reality**: disciplined iteration.
- **Myth**: "Big results every month." **Reality**: most weeks are quiet. Big wins come from accumulated small wins.

## If you're aspiring

Do these:
- Build things. Ship them.
- Read papers weekly.
- Reproduce one paper every quarter.
- Contribute to an open-source ML repo.
- Write up what you learn. Even if nobody reads it.
- Find people ahead of you. Ask questions. Do the homework before asking.

Skip these:
- Waiting for permission to start.
- Credentials and resume polishing.
- "Networking events" without substance.
- Reading without doing.

## Visualize this

**A week in the life, by the numbers**:

```
  Hours per week (40-hour baseline):

  ┌─────────────────────────────────────────────────────────────┐
  │ Coding / debugging           ████████████████████████  14h  │
  │ Reading (papers, code, Slack) ████████████████         8h   │
  │ Meetings                      ████████████              6h   │
  │ Writing (docs, memos)          ████████                 4h   │
  │ Thinking / walking             ████                     4h   │
  │ PR reviews / mentoring         ████                     2h   │
  │ Admin (hiring, planning)       ██                       2h   │
  └─────────────────────────────────────────────────────────────┘

  What people imagine:
   "AI researcher" = whiteboarding genius ideas 90% of the time

  Reality:
   Most hours are staring at wandb loss curves and debugging shape errors.
```

**The "research velocity" curve**:

```
  Experiments per week, over career:

         junior                 senior             principal
        ▀                      ▀▀▀▀▀▀▀▀▀▀▀▀      ▀▀▀▀▀
        1-3                      10-20              5-10
                                                   (more mentoring)

         What improves it:
         - Better tooling muscle memory
         - Strong prior intuitions (know what NOT to try)
         - Reusable experiment templates
         - Clearer hypothesis formation
```

**A typical day**:

```
  9:00 am   - Check overnight runs on wandb
  9:15      - One or two crashed, restart
  9:30      - Read arXiv daily, skim 3 new papers
  10:00     - Coding: implement yesterday's debugging insight
  11:00     - 1:1 with teammate
  11:30     - Coding continued
  12:30 pm  - Lunch
  1:30      - Launch new runs
  2:00      - While runs compute: write up last week's finding
  3:30      - Code review for junior teammate
  4:00      - Deep work: tackle hard problem, walking helps
  5:30      - Check runs again, adjust or launch more
  6:00      - Commit, go home

  Many variations. Common thread:
  - Runs are always running in the background.
  - You're not always "typing code".
  - Writing and thinking take real time.
```

**Senior vs junior: what differs**:

```
  Task                  Junior          Senior
  ────────────────────  ────────────    ────────────
  Pick experiment        wanders a bit   laser-focused
  Runtime of experiment  4-8 hours       10 min → 8 hours, scaled
  Interpretation         sometimes wrong  rapid, usually right
  Time debugging         sometimes days  usually hours
  Writeup                long, defensive  short, confident
  Knowing when to stop   rarely          often

  Develops over 2-5 years of deliberate practice.
```

**The "why ML feels slow" curve**:

```
  Tasks per day:

  Junior:
    Day 1: ...debug data loader...
    Day 2: ...still data loader...
    Day 3: launch run!
    Day 4: run crashed, debug
    Day 5: run works but loss is weird
    Day 6: debug weird loss
    Day 7: aha, fixed, relaunch
  End of week: 1 experiment run.

  Senior:
    Day 1: run small baseline (1 hour, done)
    Day 1 cont.: run 5 small variants in parallel
    Day 2: analyze, pick best
    Day 3: scale up winner, run next small variants
    Day 4-5: same pattern
  End of week: 30+ experiments run, 3-5 meaningful insights.

  This is what "research velocity" looks like.
```

**Karpathy's research log** (from `nanochat/dev/LOG.md`):

```
  Real excerpt (paraphrased, showing structure):

  2026-02-05: bumped total batch to 1M tokens
  - hypothesis: larger batch should help MFU
  - observation: tok/sec improved but val BPB slightly higher
  - decision: keep the change, small BPB hit worth the speed
  - next: test at d26 to confirm

  2026-02-08: tried SwiGLU init scaling
  - hypothesis: better init might help early training
  - observation: no measurable difference
  - decision: drop the change
  - next: try attention gating variants
```

Dated entries, structured (hypothesis → observation → decision → next), short.
This is the template to adopt.

**Mistakes juniors make**:

```
  1. Running one big definitive experiment:
     "I'll run GPT-2 for 3 days then see if my idea works!"
     → 3 days gone, 1 bit of info learned.
     Fix: small experiments first. Scale only what works.

  2. Attached to hypotheses:
     "This didn't work, but if I tweak it... and this... and..."
     Fix: give it 2 small runs. If no signal, pivot.

  3. Skipping the writeup:
     "I ran 20 experiments, now I'm moving on."
     6 months later: remember nothing.
     Fix: 5-line log entry per experiment. In the moment.

  4. Reading too much, doing too little:
     Endless arxiv, no code.
     Fix: one paper read, followed by at least 30 min of related coding.

  5. Perfection over velocity:
     Spend 3 days polishing a 10-line change.
     Fix: ship ugly, iterate.
```

## Exercises

1. Check your own time allocation. For the last week: what % was coding, what % reading, what % thinking, what % meetings? Most people are out of balance. Adjust.

2. Pick a small hypothesis you can test this week. Scope it to one experiment. Run it. Write a one-paragraph result.

3. Read Karpathy's [research log in nanochat](https://github.com/karpathy/nanochat/blob/master/dev/LOG.md). Study the structure: hypothesis → experiment → observation → next hypothesis. Adopt this format for your own log.

## Next

`02_the_research_loop.md` - structured decision-making.
