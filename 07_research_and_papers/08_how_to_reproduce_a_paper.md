# 7.8 - How to Reproduce a Paper

Reading papers passively teaches you vocabulary. Reproducing papers teaches you the field.

## Why reproduce

- Deepest way to understand a technique: you can't fake your way through implementation.
- Discovers gaps: the paper left out details, you hit them, you realize what's hard.
- Builds credibility: "I reproduced X" is a real signal to employers and collaborators.
- Discovers errors: papers have bugs. Some replications invalidate their original claim.
- Gets you noticed: write up your reproduction as a blog post.

## What to reproduce

### Good candidates

- Small, focused papers with one clear result.
- Papers with released code (makes it 10x easier).
- Recent (within 2 years) - less likely to be obsolete.
- Architectures or training tricks, not massive models.

### Bad candidates

- Frontier models (GPT-4) - private, impossible without their data and infrastructure.
- Huge multi-modal systems - too many moving parts.
- Papers with no code and vague method descriptions - too much to reconstruct.

## A sample list of reproducible papers

1. **FlashAttention**: implement it in Triton, benchmark vs SDPA.
2. **RoPE**: implement on top of nanoGPT, compare to learned pos embs on Shakespeare.
3. **RMSNorm**: replace LayerNorm in nanoGPT, compare.
4. **GQA**: implement on nanoGPT, measure inference speedup.
5. **LoRA**: fine-tune a HuggingFace model with adapters.
6. **DPO**: implement on top of your SFT'd nanochat model.
7. **Chinchilla scaling law**: do a mini version with 3-5 small runs on nanoGPT.
8. **Speculative decoding**: implement with a small draft model + big model.

Each of these is 1-2 weeks of focused work. You'll learn tons.

## The method

### Week 1: understand

- Read the paper three times (passes 1, 2, 3).
- Study any released code.
- Find related blog posts, tweets, lectures.
- Outline the algorithm in pseudocode.
- List every hyperparameter you'll need.

### Week 2: prototype

- Build the smallest version that could work.
- On the smallest data that could show signal.
- Don't worry about performance or correctness of edge cases yet.
- Goal: see the number go in the right direction.

### Week 3: validate

- Compare your results to the paper's.
- Expect a gap. Investigate.
- Is it data? Hyperparameters? Seed variance? Subtle bug?
- Track down one gap at a time.

### Week 4: write up

- Notebook / blog post with results.
- Be honest about gaps.
- Include code on GitHub.
- Tweet / share. Tag authors.

## Common gotchas

1. **Tokenizer**: often not specified. Your tokenizer is different from theirs. Accounts for small loss differences.
2. **Data mix**: the actual dataset may not be public. Use a similar one and note it.
3. **Seed variance**: run each experiment 3 times. Report mean ± std.
4. **Init**: tiny differences in weight init can matter.
5. **Gradient clipping, warmup, LR schedule**: often underspecified.
6. **Hardware differences**: H100 vs A100 can show different numerics.
7. **Library version**: PyTorch 2.0 vs 2.1 kernels may differ.

Every paper has ~3-5 "missing details" that take you a week to recover.

## Replication as a career move

Early in your career, reproducing papers is one of the best ways to:
- Demonstrate skill (stronger than toy projects or interviews).
- Get noticed by researchers (who love seeing their work extended).
- Learn what a "real" experiment looks like.

Examples from careers I know:
- One person reproduced the FlashAttention paper on AMD GPUs, got hired by a lab doing that.
- Another reproduced a Meta RL paper, got invited to their internship.
- A grad student reproduced DPO, started a successful open-source project.

## Starting a reproduction

Workflow:
1. Fork the paper's code repo (if available), or start fresh from nanoGPT/nanochat.
2. Create a `reproduction/` folder with your notes, experiment logs, results.
3. Use wandb for every experiment.
4. Write the blog post as you go, not after.

## Ethics

- Cite the original paper clearly.
- Credit authors for the idea.
- Be honest about differences.
- If you find a bug in their method or an error in their numbers, report it to authors first before going public.

## When to give up

Sometimes a paper just won't reproduce. Could be:
- Their method actually doesn't work (rare but happens).
- Their specific data/compute setup is essential (more common).
- You made a subtle mistake (most common).

Set a time budget (say, 4 weeks). If you haven't matched within a 2x factor by then, write up what you learned and move on.

## Exercises

1. Pick a paper from the list above. Write a one-page reproduction plan: scope, data, hyperparameters, timeline, success criteria.

2. Open nanochat's `dev/LOG.md` — Karpathy's running research log. Note how he iterates: hypothesis, experiment, result, next hypothesis. Model your reproduction log on this.

3. Browse Papers With Code (https://paperswithcode.com/) - find 3 papers with released implementations that interest you.

4. Start small: take the Shakespeare training, add RMSNorm (replacing LayerNorm). Train both. Compare val loss. You've just reproduced one detail from the Llama paper.

## Next

`09_how_to_write_a_paper.md` - the other side.
