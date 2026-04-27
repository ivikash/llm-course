# Module 7 Capstone - Reproduce One Figure

Reproduce one specific figure from a published paper using nanoGPT or nanochat as your engine. This is the final test of your paper-reading and engineering skills.

## Why just one figure

Full paper reproductions take months. One figure is tractable in 1-4 weeks and teaches almost as much.

## Pick one

Easy / moderate:

### Option A: "Chinchilla-style isoFLOP" for your nanochat
Train 4-6 nanochat models at different (N, D) combinations, all with the same total compute (FLOPs). Plot val_bpb vs model size. Observe the U-shape: loss is minimized at some intermediate size.

- Paper: Hoffmann et al 2022, Figure 3.
- Compute: ~$30-50 on Lambda if you use small models.
- Time: 1-2 weeks including analysis.

### Option B: "Attention is All You Need Figure 3" visualization
Train nanoGPT on Shakespeare. Extract attention weights from a trained head. Visualize as a heatmap.

- Paper: Vaswani et al 2017, Figure 3.
- Compute: near-zero.
- Time: a weekend.

### Option C: "RoPE vs learned positions" comparison
Replace nanoGPT's learned position embeddings with RoPE. Train both on Shakespeare. Compare val loss and generalization beyond block_size.

- Paper: Su et al 2021 (RoPE), claims better length generalization.
- Compute: near-zero.
- Time: 1-2 weeks.

Harder:

### Option D: "RMSNorm vs LayerNorm" ablation
Replace nanoGPT's LayerNorm with RMSNorm. Train both. Report loss, training time, any differences.

- Paper: Zhang & Sennrich 2019 / Llama papers.
- Compute: near-zero.
- Time: a weekend.

### Option E: "DPO actually works" on your nanochat
After SFT, collect preference pairs (manually write 100-500 (prompt, good, bad) triples). Train DPO. Compare to SFT-only on qualitative chat.

- Paper: Rafailov et al 2023 (DPO).
- Compute: moderate.
- Time: 2-3 weeks.

### Option F: "Flash Attention is faster" benchmark
Implement a simple tiled attention in Triton or pure PyTorch. Compare its speed to PyTorch's SDPA on varying sequence lengths.

- Paper: Dao et al 2022 (Flash Attention).
- Compute: near-zero.
- Time: 1-2 weeks.

## The deliverable

Write `my_reproduction.md` with sections:

```markdown
# Reproducing [Paper Title], [Figure N]

## Source
- Paper: [link]
- Figure: [what you're reproducing]
- Original result: [what it shows]

## Setup
- Engine: nanoGPT / nanochat
- Hardware: [your setup]
- Code changes: [link to your fork or a diff]
- Data: [what you used]
- Hyperparameters: [table]

## My results
- [Your figure, side-by-side with the paper's figure]
- [Your numbers]

## Differences
- [Any gaps; honest analysis]
- [Likely causes]

## What I learned
- [3-5 specific takeaways]

## Code
- [GitHub link]
```

## The process

### Week 1: Understand

- Pass 1-3 on the paper, focused on your figure.
- Enumerate every hyperparameter the figure depends on.
- Find the authors' code (if any) and note relevant functions.
- Write a plan with dates.

### Week 2: Implement + prototype

- Modify nanoGPT/nanochat minimally.
- First, run at the smallest scale that gives signal. Iterate.
- Run once fully at your target scale.

### Week 3: Analyze + compare

- Plot your figure.
- Compare to theirs.
- Investigate gaps.
- Re-run if needed.

### Week 4: Write up

- Draft the reproduction report.
- Include the figure.
- Share with peers for feedback.
- Post publicly.

## Visualize this

**The reproduction quality ladder**:

```
  Level 1 (beginner):
  Took the paper's released code, ran it, got similar numbers.
  ─ Effort: 1-2 days.
  ─ Learning: low.
  ─ Value: you verified the code works.

  Level 2 (your capstone target):
  Reimplemented key result in nanoGPT/nanochat from scratch,
  got within 2× of paper's number, wrote up differences.
  ─ Effort: 1-4 weeks.
  ─ Learning: high.
  ─ Value: deep understanding of one technique.

  Level 3 (serious):
  Reimplemented, found a bug or extension, contributed back.
  ─ Effort: 1-3 months.
  ─ Learning: deep.
  ─ Value: made the field better.

  Level 4 (research contribution):
  Extended the paper's method into a new one, published yourself.
  ─ Effort: 6+ months.
  ─ Learning: you're a researcher now.
```

**A Gantt chart of a good reproduction**:

```
  Week 1  │ understand ████████████
  Week 2  │ implement           ████████
  Week 3  │ debug                     ██████████
  Week 4  │ experiments                        ████████
  Week 5  │ analyze + writeup                          ██████████
          └──────────────────────────────────────────────────────

  Scoped properly = 5 weeks.
  Over-scoped = 3-6 months (one of those "I'll finish next month" projects).

  Limit scope ruthlessly. One figure. One comparison. One claim.
```

**Expected output artifacts**:

```
  After completing the capstone, you should have:

  1. GitHub repo:
     ├── README.md             (project overview)
     ├── code/                 (your implementation)
     ├── results/
     │   ├── my_figure.png     (your reproduction of paper's figure)
     │   └── original.png       (their figure for comparison)
     └── writeup.md             (the report)

  2. Published blog post or GitHub README (with figures embedded).

  3. Tagged the authors on Twitter/X or email.

  4. Maybe: a reply from the authors acknowledging your work.

  This is a real, portable artifact. Show it in every interview.
```

**What good reproduction writeups look like**:

```
  Good examples to imitate:
  - Dao's reproduction of attention variants: github.com/Dao-AILab
  - Karpathy's "Let's build GPT from scratch" (a reproduction-style video)
  - Hugging Face's reproduction of Chinchilla scaling laws
  - Any of the "Awesome reproductions" lists on GitHub

  They share:
    ✓ Specific numbers (not "close to the paper")
    ✓ Clear diff: what matched, what didn't
    ✓ Honest explanation of gaps
    ✓ Code that actually runs
    ✓ Visual side-by-side with the original
```

**The typical "gap" you'll find (and why)**:

```
  Their paper: "We achieved 74.5% accuracy on MMLU."
  Your run:    "I achieved 68.3% accuracy on MMLU."

  WHY the gap?

  Likely causes (most common first):
  1. Smaller model (you used 1B, they used 7B).        → rescale + document
  2. Different training data volume                    → document + note
  3. Different hyperparameters (unspecified in paper) → guess + document
  4. Different hardware / precision                    → note it
  5. Contamination difference (MMLU in their training) → hard to fix
  6. Evaluation prompt template mismatch              → align templates
  7. Random seed variance                              → average over 3 seeds
  8. Bug in your code                                  → debug!

  A good writeup explains each potential cause.
  Honest gap reporting is more valued than exact matching.
```

**Celebrating what you learn, not what you 'prove'**:

```
  Bad outcome to fear:   "I couldn't reproduce X."
  Better framing:         "I found that reproducing X at 10M scale
                           gives these numbers [X]. Matches/doesn't match
                           at my scale. Here's what I'd need to fully
                           reproduce."

  Bad outcome:            "My method doesn't work."
  Better framing:         "The technique has sensitivity to [specific HP],
                           which wasn't in the paper. Future work..."

  Negative results are valuable. Just articulate them clearly.
```

## Tips for success

1. **Start small.** If the paper's figure uses 8 models, start with 2. If it uses 1B params, start with 10M.
2. **Use wandb.** You'll want to look back at loss curves.
3. **Budget time.** A "one-figure reproduction" can balloon to months. Budget 4 weeks. At 4 weeks, write up what you have, even if incomplete.
4. **Don't trust the paper.** Verify every formula, every claim, every number.
5. **Log everything.** Commit code per experiment. Snapshot data. You *will* want to backtrack.

## Publishing

Once done:
- Post your writeup as a blog post (Medium, Substack, personal site).
- Put code on GitHub with a good README.
- Tag authors on X/Twitter, share link.
- If authors respond (often they do), engage thoughtfully.

A good reproduction is often more shared than the original paper.

## What you've achieved

After this capstone:
- You've closed the loop: paper → code → result.
- You have a concrete artifact to show: "I reproduced X".
- You've felt the texture of real research: hypotheses, experiments, debugging, write-up.
- You're now a contributor to the field, not just a consumer.

Seriously, this capstone is the most valuable deliverable in the course. Take it seriously.

## Next

Module 8. Leadership.
