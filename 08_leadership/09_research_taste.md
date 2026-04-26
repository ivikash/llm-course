# 8.9 - Research Taste

The highest-value skill in ML research. Also the hardest to teach. This lesson is an attempt anyway.

## What is taste?

**Taste** = the ability to pick problems worth solving, and to recognize promising early results.

People with taste:
- Find small details that become big insights.
- Kill their own bad ideas quickly.
- See patterns across experiments that others miss.
- Spot fundamental flaws in otherwise impressive papers.
- Predict which research directions will matter.

People without taste:
- Chase trendy topics without understanding why.
- Get attached to ideas that aren't working.
- Overinvest in incremental improvements.
- Mistake scale for substance.

## Signals of good taste

### In yourself:
- Your proposed experiments keep producing interesting results (positive or negative).
- You repeatedly identify a paper's weakness before reviewers do.
- You consistently pick problems that turn out to matter.
- You have strong, well-founded opinions.

### In others:
- Their "small side project" keeps yielding papers.
- They ask questions that reframe the problem.
- They kill projects courageously.
- Their estimates of a direction's value end up correct.

### Bad signs:
- Flitting between topics without depth.
- Following whatever's trendy this month.
- Can't name a project they killed.
- Predictions are always optimistic and often wrong.

## Where taste comes from

Two sources:

### 1. Volume

You get taste by doing lots of experiments. Many thousands. Each one teaches you something subtle. After enough, patterns form.

There's no shortcut. A senior researcher with 5 years and 5000 experiments has real taste. A junior who memorized all the papers doesn't.

This is why reproductions (Module 7) matter - each one is a whole mini-project worth of experiments.

### 2. Exposure to great thinkers

Surround yourself with people whose taste you respect. Watch them work. Listen to them explain why they're skeptical of an idea. Notice what they ignore.

If you can't physically be around them, read them: Karpathy's blog, Noam Shazeer's retrospectives, Chris Olah's interpretability posts, Francois Chollet's essays, Yann LeCun's papers. Understand their framings.

## Specific moves taste-driven researchers make

### 1. Bet on simple ideas
History: simple, scalable ideas beat complex specialized ones. (The Bitter Lesson.)

Good bet: "scale the standard transformer."
Bad bet: "a novel architecture with 50 specific components."

### 2. Focus on the thing that matters
80% of research outcome is determined by 20% of decisions. Identify the 20%.

For LLMs: data > architecture > optimizer > hyperparameters, roughly.

### 3. Exploit surprise
If a result surprises you, that's gold. Chase it. Most career-making findings started as "huh, that's weird."

### 4. Be skeptical of your own wins
When a result is too good, look for bugs first, celebrate second. Many papers don't replicate because the authors didn't check.

### 5. Read widely
Neuroscience, physics, statistics - fields outside ML often import into ML successfully. The transformer's attention mechanism echoes biological memory. Diffusion came from thermodynamics.

### 6. Ignore most hype
When everyone's talking about the same thing on Twitter, it's usually past its peak value. Work on what's one step ahead, or three steps behind (foundations).

## Taste on specific research questions

Some questions pattern-match to "bad research":

- "Can we beat GPT-4 at X?" - usually a scale issue, not a research issue.
- "We propose a new architecture that combines..." - 99% of these don't scale.
- "Our model achieves state-of-the-art on benchmark Y" - OK if ablations and analysis are solid; suspicious if single number.

Some questions pattern-match to "interesting":

- "Here's something nobody has measured before."
- "We investigated a claim in paper X and found it doesn't hold."
- "We scaled our approach 10x and the behavior qualitatively changed."
- "We found a simple intervention that changes behavior dramatically."

## Your taste tools

### Papers you should re-read
Keep a list of 5-10 "foundational" papers you re-read yearly. Each re-read deepens understanding.

### A log
Keep a research log (text file, notebook, doc). Write:
- Hypotheses before you run experiments.
- Observations after.
- Predictions vs reality. Track your calibration.

Over time: you'll see your taste evolve.

### Heroes
Pick 3-5 researchers you consider exemplars. Read all their work. Watch their talks. Mimic their framings until you have your own.

### Anti-heroes
Equally useful. Pick 2-3 researchers (or papers) you think exemplify bad taste. Articulate why. This sharpens your own standards.

## Developing taste over a career

- **Year 1-2**: copy. Imitate well-known researchers.
- **Year 3-5**: diverge. Start noticing where you disagree.
- **Year 5-10**: synthesize. Develop your own framework.
- **Year 10+**: mentor. Help others develop theirs.

This is not a linear path. You'll regress and advance.

## The dark side of taste

Bad versions:
- **Elitism**: dismissing ideas because of who proposed them (or the venue).
- **Preservation**: refusing to update when new evidence comes.
- **Gatekeeping**: using taste as a weapon against newcomers.

Healthy version:
- **Epistemic humility**: "I could be wrong about this."
- **Openness**: willing to try the opposite of your prior.
- **Charity**: try hard to find what's right about disagreements.

## Exercises

1. Pick a recent paper that impressed you. Write: what's the actual novel contribution? What would I do differently? What's the weakest claim?

2. Pick a recent paper that underwhelmed you. Write: what did the authors get right, even if the overall claim was weak? (Charity exercise.)

3. Make 3 predictions about the next 6 months in LLM research. (E.g., "Model Y will hit benchmark X, lose at Z.") Track accuracy in 6 months.

4. Identify 3 researchers whose taste you admire. What do they have in common? What do you want to emulate?

5. Identify 3 research directions that are currently hyped but you suspect won't pan out. Why? Revisit in 2 years.

## Next

`10_staying_current_in_a_field_that_moves_weekly.md` - the firehose problem.
