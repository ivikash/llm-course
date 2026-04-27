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

## Visualize this

**Taste, as a feedback loop**:

```
  ┌─ observe the field ────┐
  │   read papers, watch   │
  │   other researchers    │
  │   ↓                    │
  │ ─── form priors        │
  │                        │
  │ ─── experiment         │ ← this is the key step
  │   predict outcome      │   where taste is trained
  │   run, observe         │
  │   outcome vs prediction │
  │   ↓                    │
  │ ─── update priors      │
  │   (where were you    │
  │   right? wrong?)       │
  │                        │
  │ ─── over time →        │
  │   accurate predictions  │
  │   = good taste         │
  └───────────────────────┘

  Taste is a calibrated predictive model. Build it by closing
  prediction-observation loops often.
```

**The taste curve over career**:

```
  Prediction accuracy on "will this work?"
   │
   │                                    ●  (senior: 70%+ accurate
   │                                        on experiments in their domain)
   │                              ●
   │                        ●
   │                  ●
   │              ●
   │          ●
   │        ●   (mid: 50% accurate)
   │      ●
   │    ●
   │   ●         ← (junior: 20-30%, barely above noise)
   └───────────────────────────────────── years of experience
     0    2    4    6    8    10
```

**Signs of good taste (vs bad)**:

```
  Good taste shows up as:
    ✓ Predicting which papers will be cited in 2 years
    ✓ Naming what's wrong with a proposed method in 30 seconds
    ✓ Killing your own projects quickly
    ✓ Picking specifically which experiment to run first
    ✓ Discerning 1% improvements that matter from 1% that don't

  Bad taste shows up as:
    ✗ Chasing whatever's on Twitter this week
    ✗ Excited about tiny benchmark gains
    ✗ Mistaking complexity for sophistication
    ✗ Overtrusting your heroes, undertrusting yourself
    ✗ Thinking "more params = better"
```

**Developing taste: the practice**:

```
  Week 1-2:
  ─────────
  Read 5 papers. For each:
    - BEFORE reading methodology, predict from abstract:
      "I think this will work" / "I think this won't"
    - Predict what ablations they'll need.
    - After reading, check your predictions.

  Month 1:
  ─────────
  Pick 10 open questions in your field.
  Write your gut answer for each.
  Revisit in 6 months; score yourself.

  Year 1:
  ───────
  Track your predictions about the field.
  "Model X will saturate benchmark Y by date Z."
  Check accuracy at year end.

  Continuous:
  ───────────
  For every experiment you design, predict outcome BEFORE running.
  Keep a track record. Your calibration is your taste.
```

**Taste ≠ being right always**:

```
  Good-taste researcher is wrong a lot.
  The difference:
    ✓ Knows she's about to be wrong
    ✓ Designs experiments that surface being wrong cheaply
    ✓ Updates quickly when new evidence comes
    ✓ Keeps exploring after being wrong, not wounded

  Bad taste: holds strong views forever regardless of evidence.
  Good taste: strong opinions, loosely held.
```

**Learning taste from exemplars**:

```
  Find 3-5 researchers whose taste you respect.

  For each:
    Study their most famous paper. What framing choice did they make?
    Watch their talks. How do they ask questions?
    Read their blog posts. What do they think is important?
    Follow their predictions on X/Twitter. How often are they right?

  Examples to study:
    Andrej Karpathy (pragmatic, code-first, teaches)
    Yann LeCun (foundational, skeptical of hype)
    Noam Shazeer (architectural innovations)
    Chris Olah (interpretability, visuals)
    Ilya Sutskever (bold bets, often right)
    Francois Chollet (contrarian, careful)

  Over time, form your OWN taste — blend of your heroes + your experience.
```

**Anti-taste examples** (also useful):

```
  Study researchers whose taste you find questionable.
  Articulate WHY.

  Common anti-patterns worth naming:
    - "Benchmark gaming" researchers: +0.3% SOTA with 3× compute
    - "Architecture zoo" researchers: proposes complex design with no ablation
    - "LLM-as-magic" researchers: claims emergence without careful measurement
    - "Trendy topic" researchers: works on whatever's hot, never deep
    - "Pedigree" researchers: cites only senior voices, dismisses juniors

  By articulating what's WRONG, you clarify what's RIGHT.
```

**Applied taste questions**:

```
  For each decision, ask:

  "Is this the ONE thing that matters most?"
     (If not, why am I doing it?)

  "What's the weakest part of my plan?"
     (Attack that first.)

  "What would prove me wrong?"
     (Design that test.)

  "If I were skeptical, what would I ask?"
     (Be your own critic.)

  "Is this just hype?"
     (Would it matter in 2 years?)

  Answering these honestly — daily — is taste in action.
```

**The bitter lesson, again**:

```
  "Bitter" because of what it implies about taste:

  Sutton 2019: "Methods that scale with compute win.
                 Human-engineered shortcuts lose."

  Corollary for taste:
    Bet on methods that get better with more compute.
    Be skeptical of clever tricks that don't scale.
    Be humble about your clever ideas; the compute
    may render them irrelevant.

  A taste filter that compounds: "does this scale?"
```

**Taste is the thing you can't Google**:

```
  Googleable:
    ✓ What does SwiGLU mean?
    ✓ How to implement RMSNorm?
    ✓ What's the Chinchilla rule?

  Not Googleable:
    ✗ Is this idea worth 3 months?
    ✗ Will this paper hold up?
    ✗ What should we work on next?
    ✗ Is scale the answer or is something missing?

  Your taste determines the second list.
  Your career impact scales with it.
```

## Exercises

1. Pick a recent paper that impressed you. Write: what's the actual novel contribution? What would I do differently? What's the weakest claim?

2. Pick a recent paper that underwhelmed you. Write: what did the authors get right, even if the overall claim was weak? (Charity exercise.)

3. Make 3 predictions about the next 6 months in LLM research. (E.g., "Model Y will hit benchmark X, lose at Z.") Track accuracy in 6 months.

4. Identify 3 researchers whose taste you admire. What do they have in common? What do you want to emulate?

5. Identify 3 research directions that are currently hyped but you suspect won't pan out. Why? Revisit in 2 years.

## Next

`10_staying_current_in_a_field_that_moves_weekly.md` - the firehose problem.
