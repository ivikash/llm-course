# 6.9 - Scaling Laws

Two papers changed how people train LLMs:
1. **Kaplan et al 2020** - "Scaling Laws for Neural Language Models" - established that loss is a power law in compute/data/params.
2. **Hoffmann et al 2022** - "Training Compute-Optimal LLMs" (Chinchilla) - corrected Kaplan: he overweighted model size, underweighted data.

Understanding these is critical for planning any training run.

## The core observation

Train many models of different sizes, on different amounts of data, for different amounts of compute. Plot loss.

Finding: **loss decreases as a predictable power law in each of {parameters, data, compute}**.

```
L(N) ≈ A * N^(-α)          (for fixed data and compute)
L(D) ≈ B * D^(-β)          (for fixed params and compute)
L(C) ≈ K * C^(-γ)          (for fixed ratio)
```

Where N = params, D = tokens, C = compute (FLOPs). The exponents are small (~0.3 for each).

## Kaplan's claim (2020)

Given a fixed compute budget C, how should you allocate between bigger N or more D?

Kaplan's answer: "go bigger". Allocate most of the compute to increasing N. Consequence: train large under-trained models.

GPT-3 (2020) followed this: 175B params, 300B tokens (ratio ~1.7:1 of tokens per param).

## Chinchilla's correction (2022)

DeepMind trained many models, carefully. Found: Kaplan was wrong. The loss-optimal allocation is **roughly equal**: scale N and D together.

Rule of thumb: **~20 tokens per parameter** for compute-optimal training.

Under this rule:
- GPT-3 (175B params) should have been trained on ~3.5T tokens, not 300B. GPT-3 was **undertrained**.
- Chinchilla (70B params) trained on 1.4T tokens beat GPT-3 (175B on 300B).

## The formulas

Chinchilla's fit:
```
L(N, D) ≈ E + A/N^α + B/D^β
```

With specific constants (approximate):
- E ≈ 1.69 (irreducible loss floor)
- α ≈ 0.34
- β ≈ 0.28

To find compute-optimal (N*, D*) for a given C:
- `N* ≈ G * C^a`
- `D* ≈ H * C^b`
with G, H, a, b empirically fit. Rough approximation: N* ~ sqrt(C/120), D* ~ 20 * N*.

You don't need to memorize the constants. You need to internalize: **when doubling compute, double N and double D, not quadruple N**.

## The "overtrain for inference" caveat (2023+)

Chinchilla is **training-compute-optimal**. But inference is a separate cost. A smaller model (say, 7B) trained on more data than Chinchilla-optimal (say, 2T instead of 140B) will:
- Be slightly suboptimal per training FLOP.
- Be drastically cheaper to serve per token.
- Be dramatically more accessible (fits on smaller hardware).

Llama-2-7B was trained on 2T tokens. Llama-3-8B on 15T tokens. Both dramatically overtrained vs Chinchilla. Reason: they'll be served billions of times; spending more on training pays off.

Practical rule today: for served models, **overtrain** beyond Chinchilla.

## nanochat's use of scaling laws

Open `~/workspace/nanochat/dev/scaling_analysis.ipynb`. Karpathy fits scaling laws to his own nanochat runs - predicts how big a model to train, how many tokens, for a given compute budget.

`runs/scaling_laws.sh` launches the sweep that generates the underlying data.

The `target-param-data-ratio` flag in `base_train.py` controls this: set it to 20 for Chinchilla-optimal, 8 for "undertrain for speed" (the speedrun's default to beat GPT-2 quickly).

## How to use this day-to-day

### Planning a run

Before launching:
1. Decide compute budget C (FLOPs or GPU-hours × TFLOPS).
2. Compute N* and D* from Chinchilla. Use nanochat's scaling fits if you're using it.
3. Predict the loss you should reach. Compare to actual after training.
4. If actual is much worse: bug. Much better: overfitting, or formulas don't apply to your data.

### Comparing runs

Different runs are not directly comparable unless you account for compute. Always plot loss vs compute (or loss vs tokens), not just loss vs step.

### Extrapolating

You can fit a scaling law to your own small runs and extrapolate to bigger. Very useful for "should I spend $10K or $100K on this?"

## Related laws

- **Emergence** (Wei 2022): some capabilities appear suddenly at scale, not gradually. Debated (Schaeffer 2023 argued many emergences are metric artifacts).
- **Inverse scaling** (McKenzie 2022): some behaviors get WORSE with scale. Interesting counterexamples.
- **Data scaling laws by domain**: the "20 tokens / param" is an average; code may want more, math may want different mixes.

## Exercises

1. Open `~/workspace/nanochat/dev/scaling_analysis.ipynb`. Don't try to understand all code; focus on the plots. Note: loss is a straight line on log-log axes.

2. Read the original [Chinchilla paper abstract](https://arxiv.org/abs/2203.15556). Two pages will give you the key result.

3. Plug in: you have $100 on an H100 ($3/hour, so ~33 hours) → approximately 33 × 3600 × 1e15 = 1.2e20 FLOPs. For compute-optimal: N ≈ 40M params, D ≈ 800M tokens. That's a nanochat d8 model for about an hour. Does this match nanochat's scaling behavior?

4. Compute: GPT-3 training = 3.14e23 FLOPs. For Chinchilla-optimal: N* ≈ ? D* ≈ ? (N ≈ 60B, D ≈ 1.3T.)

## Next

`10_cost_engineering.md` - translating scaling laws into dollars.
