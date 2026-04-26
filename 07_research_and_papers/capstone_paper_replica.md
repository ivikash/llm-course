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
