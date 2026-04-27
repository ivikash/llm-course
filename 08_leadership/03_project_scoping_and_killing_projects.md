# 8.3 - Project Scoping and (Especially) Killing Projects

Picking projects to start is easy. Killing them is hard. Senior researchers are distinguished primarily by their ability to kill projects.

## Scoping a project

Before starting, write a one-page project brief:

### 1. Problem
What specifically is the problem? Be concrete.

Bad: "Improve chat quality."
Good: "Reduce our model's refusal rate on legitimate technical questions from 12% to under 4%, without increasing unsafe response rate."

### 2. Metric
How will you measure progress? One or two numbers, not a collection.

### 3. Success criteria
What "done" looks like. "Refusal rate < 4% on our red-team set."

### 4. Kill criteria
What signals would make you stop. "If after 4 weeks we haven't moved the metric at all, we stop."

### 5. Timeline
Realistic estimate. Then double it.

### 6. Resources
Compute, data, headcount.

### 7. Plan
Week-by-week outline of experiments.

### 8. Risks
Where could this fail? How would we know?

This is 8 bullet points, 1 page, 1 hour to write. Refusing to do this is a classic junior mistake.

## The 2-3x rule

Whatever timeline you predict, double or triple it. Research takes longer than you think. Always.

"We'll have results in 2 weeks" → reality is 6 weeks. This is the rule. Accept it.

## The planning fallacy

Specific biases that make plans wrong:
- **Optimism bias**: we imagine the happy path, not failure modes.
- **Success bias**: we remember when things worked, forget when they didn't.
- **Coherence bias**: our plan sounds good in the meeting, so we believe it.

Defenses:
- Ask "what could go wrong?" explicitly.
- Consult skeptics.
- Compare to similar past projects.
- Track your past predictions vs outcomes. Over time you calibrate.

## Killing projects

This is the hardest leadership skill.

Reasons to kill a project:

### 1. Hypothesis has failed
Your main claim turned out to be false. Well-designed experiments proved it. Accept and move on.

### 2. The market changed
You set out to build X. Now a better open-source X is on the Hub. Keep building is wasteful.

### 3. The metric won't budge
Weeks of work, no signal. Sometimes the problem is harder than you thought; sometimes you're attacking it wrong.

### 4. Cost exceeds value
Even if it works, is it worth the cost? Sometimes the answer is no.

### 5. Team consensus eroded
If key people on the project have lost conviction, the project is dead. Acknowledge and move.

## Why killing is hard

Emotional:
- You've invested.
- You don't want to admit failure.
- The team has pride.
- Leadership wants "wins" to announce.

Structural:
- Org incentives often reward "keep going."
- Funding / headcount is tied to the project.
- Killing feels like admitting defeat.

### The countermeasures

1. **Set kill criteria upfront.** When you scope the project, write "if X by Y date, we stop." Then it's easier - you're just following your own rule.

2. **Celebrate killing, not just shipping.** A team that only rewards successes will keep broken projects alive.

3. **Reframe**: "We didn't fail; we de-risked a direction." Useful language for reports.

4. **Separate identity from project**: "The project died, but we're alive."

5. **Set a ladder**: if you have 10 projects, maybe 3 should die this quarter. Make it normal.

## Killing early, killing cheap

The cost of killing a project grows over time:
- Week 1: trivial. One person walks away.
- Month 1: some embarrassment.
- Month 3: team of 5 people, $100K compute, now you have to explain to VPs.
- Year 1: organizational identity tied to it, nearly impossible to kill.

Intervene early. The earlier you kill, the cheaper.

## What "killing" really looks like

Not: "Meeting, declaration, everyone fired."

Usually:
- Weekly review of goals vs progress.
- Early signals of a project not working flagged.
- Discussions with team: "are we still confident in this?"
- Formal pivot or wind-down with a timeline.
- Postmortem: what did we learn?
- People reassigned to viable projects.

Handled well, a kill is just another project milestone.

## The postmortem

Every killed project should produce a postmortem:

- Original hypothesis / goals.
- What we tried.
- What worked, what didn't.
- Why we're stopping.
- Lessons for other teams.
- What we'd do differently.

Share broadly. These are gold for the organization. Don't bury failures.

## The psychological cost

Killing projects is emotionally expensive even when done well. Expect:
- Self-doubt. "Should I have seen this earlier?"
- Team morale dip.
- Awkwardness with leadership.

Recover by:
- Writing the postmortem honestly.
- Taking a brief break.
- Starting the next project with renewed scope discipline.
- Talking to peers who've been through it.

Over a career, you'll kill more projects than you ship. That's normal and correct.

## The worst kind of "kill"

Not actually killing, but pretending to. Signs:
- "We're pausing the project" (it's dead).
- "Moving to back burner" (it's dead).
- "Long-term investigation" (it's dead).

If it's dead, say it's dead. Pretending wastes headcount and clutters planning.

## Your checklist, for any running project

Answer weekly:

- [ ] Is the primary metric moving?
- [ ] Do I still believe the hypothesis?
- [ ] Are we on budget (time, compute, headcount)?
- [ ] If I started over knowing what I know now, would I start this project?

If most answers are "no": kill.

If most are "yes": keep going.

If mixed: decision needed soon.

## Visualize this

**The "cost to kill" over time**:

```
  Cost of killing a project vs when you kill it:

  Effort/emotional/political cost
   │                                                 ●  (huge: whole org)
   │                                              ●
   │                                         ●
   │                                     ●
   │                                 ●
   │                             ●
   │                         ●
   │                     ●
   │                 ●
   │             ●
   │         ●            ← by month 3: reasonable, still painful
   │      ●
   │    ●
   │   ●                  ← by week 4: easy
   │ ●                   ← week 1: trivial
   └─────────────────────────────────────── time

  The longer a project runs, the harder to kill.
  Biggest leadership skill: killing early.
```

**Kill criteria, pre-committed**:

```
  BAD (no kill criteria):
    "We'll just keep iterating until it works."
    → Projects zombie-shuffle for years.

  GOOD (explicit kill criteria):

  ┌──────────────────────────────────────────┐
  │ Project: "Add RL phase to improve chat"  │
  │                                           │
  │ Budget: $20K compute, 6 weeks             │
  │                                           │
  │ Kill criteria (any ONE triggers):          │
  │  1. Week 2 baseline runs show < 1% gain   │
  │  2. Week 4 experiments all regress MMLU    │
  │  3. Compute overrun > 1.5× budget           │
  │  4. Team confidence (2 senior folks) < 30% │
  │                                           │
  │ Escalate-to-kill-review:                   │
  │  - Lead researcher + EM + 1 outside reviewer│
  │  - Monday 9am meeting                       │
  │                                           │
  │ Kill decision is FAST after criteria hit. │
  └──────────────────────────────────────────┘
```

**Signs a project should die**:

```
  Obvious (but often ignored):
    ✗ Main metric hasn't moved in 3+ weeks
    ✗ Compute budget blown past
    ✗ Team has lost conviction
    ✗ Better alternative published externally

  Subtle (easier to rationalize):
    ✗ Each week's findings contradict last week's
    ✗ Scope keeps expanding
    ✗ Nobody can clearly articulate "why this matters"
    ✗ Leadership asks weekly; team answers vaguely

  Warning: if you're saying any of these, it's dead.
    "We just need one more experiment."
    "Let's give it another sprint."
    "Almost there."
    "Pivoting the goal slightly."
```

**The postmortem template (after every killed project)**:

```markdown
  # Project X Postmortem

  ## TL;DR
  What happened: [one sentence]
  What we learned: [one sentence]

  ## Original goal
  [what you set out to do]

  ## Timeline
  Week 1: [what happened]
  Week 2: [what happened]
  ...

  ## What worked
  - ...

  ## What didn't work
  - ...

  ## Why we killed it
  - [specific signal from kill criteria]

  ## What we'd do differently
  - [scoping mistake? bad bet?]

  ## Reusable artifacts
  - [code we salvaged]
  - [data we kept]

  ## Open questions for future work
  - [what's still worth investigating?]
```

A clean postmortem makes the death productive. The org now knows what not to try again.

**Projects that should NOT be killed**:

```
  Sometimes you're tempted to kill but shouldn't:

  ✓ Long-horizon basic research (measured in years, not weeks)
  ✓ Work that's de-risking a future project
  ✓ Projects ~1 week from a big commit
  ✓ Someone's first project (preserve learning)

  Judgment call. Err on side of "letting hot things cool" vs
  "thrashing between projects".
```

## Exercises

1. For a project you're currently working on (at work or personal): write the 1-page brief retroactively. Be honest.

2. Name a project in your past that should have been killed earlier. What were the signals? What did you do? What should you have done?

3. Identify someone who regularly kills their own projects well. What's their method?

4. For your next project, explicitly write the kill criteria upfront. Share with your team.

## Next

`04_evaluations_as_the_product.md` - why this matters even more than killing.
