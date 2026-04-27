# 7.9 - How to Write a Paper (If You Want To)

Most people in ML industry never publish. That's fine. But if you want to contribute research formally, here's the playbook.

## When should you write a paper?

You should if:
- You have a genuine novel finding (or strong replication of a surprising one).
- You can defend it against 3 rounds of peer review.
- You want to join academic ML (apply to grad school, get research roles).

You shouldn't if:
- You built a cool tool (write a blog post instead).
- You reproduced something (blog post).
- You have interesting ideas but no experiments (talk to people, blog).
- You're chasing resume-padding (wastes your time, diminishing returns in industry).

Most ML progress happens outside paper pipelines now - in open-source repos, blog posts, tweets. Papers are still the academic coin, but they're not the only valid medium.

## Paper structure

Standard structure, 8 pages for a workshop, up to 25 pages with appendix for a main conference.

### Title
- Should convey the contribution in 8-12 words.
- Bad: "A Novel Approach to Neural Language Modeling"
- Good: "Chinchilla: Training Compute-Optimal Large Language Models"

### Abstract
- 150-250 words.
- Sentence 1-2: the problem.
- Sentence 3: your approach.
- Sentence 4-5: results.
- Sentence 6: implication.

### Introduction
- 3-5 paragraphs.
- Motivation, prior work (briefly), your contribution, paper structure.
- Last paragraph = contributions list: "In this paper, we (1) ... (2) ... (3) ..."

### Related Work
- 1-2 pages.
- Groups papers by theme, cites charitably.
- Not a history lesson - show what's missing that your paper provides.

### Method
- The core idea.
- Equations, diagrams, pseudocode.
- Complete enough that a careful reader can reimplement.

### Experiments
- Setup: datasets, baselines, metrics, hyperparameters.
- Main results: table or figure.
- Ablations: remove each component, show it matters.
- Analysis: why does this work? Failure cases?

### Discussion / Limitations
- Be honest. Every method has limits.
- Reviewers love honest limitations sections. They hate papers that overclaim.

### Conclusion
- 1 paragraph. Restate contribution. Point to future work.

### References
- Cite prior work. Too few citations = amateur. Too many = padded.

### Appendix
- Extended results, hyperparameter details, additional experiments.

## Figure craft

One of the most undervalued skills.

**Rules**:
- One clear message per figure.
- Caption should be self-contained (reader can skim figures only and still get the paper).
- Axes labeled with units.
- Legends readable at the paper's printed size.
- Color-blind friendly palettes (avoid red-green contrasts).
- No 3D pie charts, ever.

**Famous headline figures**:
- Chinchilla's "isoFLOP curves" (Figure 1).
- Scaling laws' "power law on log-log" plots.
- Original transformer's Figure 1 (architecture).

Invest 10% of paper-writing time in the headline figure. It's what 80% of readers see.

## Writing style

- **Short sentences.** Readers skim.
- **Active voice** usually. "We trained a model" > "A model was trained."
- **Define every symbol before use.**
- **No marketing language.** "Revolutionary", "breakthrough", "paradigm shift" - cut.
- **Be specific.** "Our method outperforms baselines" → "Our method improves accuracy by 5.3 percentage points on benchmark X."
- **Include negative results** in a limitations or appendix section. Makes you credible.

## Writing process

### Phase 1: Experiments first (70% of time)

Write the paper **after** you have strong results. Many junior researchers write the paper first and then do experiments to support it. This is backwards and often leads to overclaiming.

### Phase 2: Outline (1 day)

Bullet-point every section. Cite key papers. Identify gaps.

### Phase 3: First draft (1 week)

Write rough prose. Don't polish. Get it done.

### Phase 4: Iterate (1-2 weeks)

- Advisor/colleague feedback.
- Tighten arguments.
- Redo figures.
- Trim wording.

### Phase 5: Polish (a few days)

- Grammar, typos, LaTeX.
- References consistency.
- Final figure tweaks.

## Venues

### Main conferences
- **NeurIPS** (December): general ML, biggest.
- **ICML** (July): general ML.
- **ICLR** (April/May): representation learning, LLMs live here.
- **ACL** / **EMNLP** / **NAACL**: NLP-specific.
- **CVPR** / **ICCV** / **ECCV**: vision.

Acceptance rate: 20-30%. Reviewers are mixed quality. Multiple rounds.

### Workshops
- Attached to main conferences.
- Higher acceptance rate, lower status.
- Good for early-career papers, negative results, opinion pieces.

### arXiv-only
- Just post to arXiv, skip reviewing.
- Many industrial labs do this now (DeepMind, Anthropic, OpenAI often skip peer review).
- Benefit: no 6-month delay. Downside: no "official" peer review.

### Tracks in 2024+
- Many interesting papers appear first on arXiv, later in conferences as formal publications.
- The community is splitting between "arxiv velocity" and "formal credit". Watch for developments.

## Getting accepted

Reviewers ask:
1. Is the contribution novel?
2. Is it significant?
3. Are the experiments sound?
4. Is the paper well-written?

You can control #3 and #4 entirely. #1 and #2 require actual research.

**Rebuttal strategy**: when reviewers raise concerns, don't argue. Address them. Add the experiment they asked for. Change the framing.

## Common rejection reasons

- "Not enough ablations." Fix: do more ablations.
- "Comparison to X is missing." Fix: compare.
- "The paper claims too much." Fix: claim less, honestly.
- "Writing is unclear." Fix: rewrite, get external help.
- "Significance is low." Hardest to fix. Sometimes the paper just doesn't merit the venue.

## The industry alternative

Many great ideas are shared via:
- **Blog posts**: Anthropic's "Circuits" series, LessWrong posts, personal blogs.
- **Technical reports**: OpenAI's GPT-4 technical report, Llama's papers.
- **Open-source releases**: the code is the paper.
- **Threads on X/Twitter**: quick insights, early results.

None of these go through peer review. All of them can be more influential than a NeurIPS paper. Choose your medium.

## Visualize this

**Paper structure mapped to effort**:

```
  Writing a paper takes ~200 hours of pure effort over 2-6 months.
  Distribution:

  ┌────────────────────────────────────────┐
  │ 40%  Running experiments                 │████████████████
  │ 25%  Refining figures                    │██████████
  │ 15%  Writing / rewriting prose            │██████
  │ 10%  Polishing / formatting              │████
  │ 5%   Submitting, rebuttals               │██
  │ 5%   Related work search                 │██
  └────────────────────────────────────────┘

  Surprise: figures take more time than writing.
  First figure draft → final figure = 20+ iterations.
```

**Paper structure template**:

```
  ┌──────────────────────────────────────────────────┐
  │ Title (12 words, specific)                         │
  ├──────────────────────────────────────────────────┤
  │ Abstract (150-250 words)                           │
  │ ├─ Problem statement (1-2 sentences)               │
  │ ├─ Approach (1 sentence)                           │
  │ ├─ Key result (1-2 sentences, quantified)          │
  │ └─ Impact (1 sentence)                             │
  ├──────────────────────────────────────────────────┤
  │ 1. Introduction (1-1.5 pages)                      │
  │ ├─ What problem                                    │
  │ ├─ Why important                                   │
  │ ├─ Why prior approaches insufficient               │
  │ ├─ Our approach, in brief                          │
  │ └─ Contributions (bulleted, specific)              │
  ├──────────────────────────────────────────────────┤
  │ 2. Related Work (0.5-1 page)                       │
  ├──────────────────────────────────────────────────┤
  │ 3. Method (2-4 pages)                              │
  │ └─ complete enough to reproduce                    │
  ├──────────────────────────────────────────────────┤
  │ 4. Experiments (3-5 pages)                         │
  │ ├─ Setup, datasets, baselines                       │
  │ ├─ Main result (headline figure)                   │
  │ ├─ Ablations                                        │
  │ └─ Analysis                                         │
  ├──────────────────────────────────────────────────┤
  │ 5. Discussion & Limitations (0.5 page)             │
  ├──────────────────────────────────────────────────┤
  │ 6. Conclusion (0.5 page)                           │
  ├──────────────────────────────────────────────────┤
  │ References                                          │
  │ Appendix (experiments, extra details)               │
  └──────────────────────────────────────────────────┘

  Standard conference: 8 pages main + unlimited appendix.
  Total ~20-30 pages with appendix.
```

**A great Figure 1 (the "headline figure")**:

```
  Rules:
    - Self-contained (caption explains everything)
    - Single clear message
    - Not cluttered
    - Uses color sparingly, purposefully
    - Legible at printed size

  Examples of iconic headline figures:
    - Transformer paper: the architecture diagram
    - Chinchilla: the isoFLOP U-curve
    - Kaplan: the scaling law straight line on log-log
    - LoRA: the "adapter" parallel to base weight diagram

  Invest 10% of paper time on Figure 1. It's the 80% signal.
```

**The abstract as a recipe**:

```
  ┌─────────────────────────────────────────────────┐
  │ [Problem setup, 2 sentences]                    │
  │                                                  │
  │ Recent advances in X have enabled Y, but are    │
  │ limited by Z.                                   │
  │                                                  │
  │ [Approach, 2 sentences]                         │
  │                                                  │
  │ In this work, we propose METHOD, which          │
  │ addresses Z by KEY_IDEA. METHOD works by ...    │
  │                                                  │
  │ [Results, 2 sentences, with numbers]            │
  │                                                  │
  │ METHOD achieves 15% improvement on BENCHMARK    │
  │ and is 3× faster than the previous SOTA.         │
  │                                                  │
  │ [Implication, 1 sentence]                       │
  │                                                  │
  │ Our findings suggest that FUTURE_IMPLICATION.   │
  └─────────────────────────────────────────────────┘
```

**The rebuttal pattern**:

```
  Reviewer comment:                    Your response:
  ─────────────────                    ──────────────

  "The paper claims X but doesn't     "Thank you for the suggestion. We've
  compare against Y."                   added a comparison in Appendix B,
                                         which shows our method outperforms Y."

  "The writing is unclear in §3."     "We've rewritten §3 to improve
                                         clarity. See new version."

  "Significance is unclear."          "We added §1.4 motivating the impact:
                                         this method enables X application
                                         previously infeasible."

  Pattern: acknowledge, explain, fix. Don't argue.
```

**Where to publish**:

```
  Your research type:                   Best venue:
  ──────────────────                   ───────────────
  Novel theoretical insight            NeurIPS / ICML
  Novel LLM technique                  ICLR / NeurIPS
  Engineering improvement              arXiv, workshop papers
  Reproduction                         blog post, arXiv
  Negative results                     workshop, blog post
  Industry application                 blog post, tech report
  
  For career: conferences = credentials (academia, research roles).
  For impact: blog posts / Twitter = higher reach (industry).
  
  Karpathy: mostly writes blogs/code. No papers in years. Lots of impact.
  Hinton, LeCun: heavy conference publishers.
  Both paths are valid.
```

**The "industrial alternative" to papers**:

```
  Paper                              Alternative artifacts
  ──────                              ─────────────────────
  Published paper                    GitHub repo + README
  Workshop short paper               Blog post
  Tech report                        Twitter thread
  Thesis                              Long-form essay
  
  Often the alternatives have MORE IMPACT than formal papers.
  Karpathy's blog > most NeurIPS papers.
  3Blue1Brown's videos > any paper on visualization.

  The academic hierarchy is 30 years old. The internet is faster.
```

## Exercises

1. Open a paper you admire. Study its figure 1 for 10 minutes. What's the message? How do visual choices support it?

2. Read "The Bitter Lesson" by Sutton. ~600 words. Study how it makes its point.

3. Outline a hypothetical paper: "RMSNorm is as good as LayerNorm for small transformers." What experiments would you need? What would the figures be?

4. Pick one of your reproductions from Lesson 7.8. Outline it as if you were writing it up. Good practice even if you don't publish.

## Next

`capstone_paper_replica.md` - Module 7 capstone.
