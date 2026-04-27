# 7.5 - Scaling Laws Papers: Kaplan and Chinchilla

Two papers shaped how the world trains LLMs. Read both. They're accessible and high-ROI.

## Paper 1: Kaplan et al., 2020 - "Scaling Laws for Neural Language Models"

**Link**: https://arxiv.org/abs/2001.08361
**Authors**: Jared Kaplan et al. (OpenAI).
**Claim**: loss of a language model is a power law in compute, model size, and data.

### Key findings

1. **Power laws are everywhere**:
   ```
   L(N) ∝ N^(-α_N)
   L(D) ∝ D^(-α_D)
   L(C) ∝ C^(-α_C)
   ```
   Where N = params, D = tokens, C = compute. Each exponent is small (0.05 - 0.1 range).

2. **Smooth, predictable curves**. You can extrapolate.

3. **Compute-efficient** training = don't train models to convergence. Stop early when returns diminish.

4. **Bigger model > more data** for a fixed compute budget. Allocate most compute to N, less to D.

### The controversial implication

For fixed C, they recommended:
- Allocate compute: about 75% to increasing N, 25% to increasing D.
- Train large, undertrained models.

This shaped OpenAI's training philosophy for 2020-2022. GPT-3 (175B, 300B tokens) was this paper in action: very big model, relatively little data (~1.7 tokens per param).

## Paper 2: Hoffmann et al., 2022 - "Training Compute-Optimal LLMs" (Chinchilla)

**Link**: https://arxiv.org/abs/2203.15556
**Authors**: DeepMind.
**Claim**: Kaplan was wrong. Optimal is to scale N and D roughly equally.

### Their experimental setup

- Trained 400+ models ranging from 70M to 16B parameters.
- Used 5 different approaches to fitting the loss.
- Triangulated the right answer.

### The result

For a fixed compute budget:
- `N_optimal ≈ sqrt(C × some_constant)`
- `D_optimal ≈ sqrt(C × some_constant)`
- **Ratio: roughly 20 tokens per parameter.**

### The consequence: GPT-3 was undertrained

| Model | Params | Tokens | Tokens/Param | Compute-Optimal? |
|-------|--------|--------|--------------|------------------|
| GPT-3 | 175B | 300B | 1.7 | No, very undertrained |
| Chinchilla | 70B | 1.4T | 20 | Yes |
| Gopher | 280B | 300B | 1.07 | No |

Chinchilla 70B beat Gopher 280B on most benchmarks. Smaller model + more data wins.

### The fit

```
L(N, D) ≈ E + A/N^α + B/D^β
```

Constants (roughly): E ≈ 1.69, A ≈ 406, α ≈ 0.34, B ≈ 411, β ≈ 0.28.

For given compute C, optimization yields specific N*, D*. nanochat's `dev/scaling_analysis.ipynb` does this fit for nanochat's setup.

## The 2023+ correction: "overtrain for inference"

Post-Chinchilla observation (various groups, notably Llama-2 and Llama-3):

Chinchilla is optimal for **training compute**. But your model will be **served** billions of times. Shifting compute from training to serving can matter more.

**Overtrain** (train a smaller model on more data):
- Slightly sub-optimal per training FLOP.
- Dramatically cheaper at inference.
- Often preferred in practice.

Llama-2-7B trained on 2T tokens (286 tokens/param).
Llama-3-8B trained on 15T tokens (1875 tokens/param).

Both drastically overtrained vs Chinchilla-optimal. Deliberately, because inference cost is a huge deal for open models.

## What this means for you

### Planning a run

If you have compute budget C FLOPs:

**Chinchilla**: set N ~ sqrt(C/6), D ~ 20N. You'll get the best training loss per FLOP.

**Overtrain for inference**: set N smaller (say 0.3× Chinchilla-N), D proportionally larger. Slightly worse training loss, much cheaper serving.

### Predicting loss

Given your chosen N and D, you can predict loss within a factor of 2. If your training ends up much worse than predicted, something is broken.

### Estimating experiments

Before launching a 3-day run, fit a quick scaling law on 3 small runs (10 min each) with different N. Extrapolate. If the predicted loss at your target N is disappointing, adjust before committing.

## nanochat's use

Open `~/workspace/nanochat/dev/scaling_analysis.ipynb`. Karpathy fits scaling laws to his own runs. The notebook predicts val_bpb for a proposed run - a useful sanity check before launching.

Also see `runs/scaling_laws.sh` - the sweep that generates the underlying data for the fit.

And `target-param-data-ratio` in base_train.py controls the N:D ratio. Set to 20 (Chinchilla) or 8 (undertrain for speedrun) depending on your goal.

## Limitations of scaling laws

1. **They predict loss, not capability.** Lower loss usually means better benchmarks, but not always.
2. **Data-specific.** The constants depend on the data distribution. Chinchilla's constants were fit on MassiveText; yours will be different on your data.
3. **Architecture-specific.** Fits to transformers with specific recipes. MoE, SSM have different curves.
4. **Don't predict emergent capabilities.** Things like chain-of-thought ability are not a simple function of loss.

## Why read both papers

- Kaplan shows the **empirical method**: run many experiments, fit power laws, extrapolate.
- Chinchilla shows the **self-correction**: ML is empirical, claims are subject to better experiments, progress happens.
- Together they show how scientific a field CAN be when people do experiments carefully.

## Visualize this

**The isoFLOP curves from the Chinchilla paper (description)**:

```
  For each fixed compute budget C, trained many models of different sizes N.
  Each curve is one compute budget. Each dot is one model.

  Loss
    │                                              compute = 1e22 FLOPs
    │       ╲                                                   │
    │        ╲  ●                                               │
    │         ╲ ●                                               │
    │          ╲●                                               │
    │           ● ← lowest loss = Chinchilla-optimal N           │
    │            ●                                              │
    │             ●                                             │
    │              ╱                                            │
    │             ╱                                             │
    │            ╱                                              │
    └────────────────────────────── N (params)    
         small      ↑       big
                   N*
                (optimal)

  Repeat for many different C values. Each gives its own U-curve.
  The minimum of each curve = Chinchilla-optimal (N*, D*) for that C.
  Fit: N* ∝ C^0.5, D* ∝ C^0.5.
  Ratio: D*/N* ≈ 20 tokens per param.
```

**Kaplan (wrong) vs Chinchilla (right)**:

```
  For C = 1e22 FLOPs:

  Kaplan 2020 recommended:
    N = 20B params  (large)
    D = 120B tokens (just a few tokens/param)
    → GPT-3 was built this way (175B, 300B tokens = 1.7 tokens/param)

  Chinchilla 2022 showed:
    N = 3B params    (smaller)
    D = 70B tokens   (more tokens)
    → Gives LOWER loss than Kaplan's recommendation at same compute
    → Chinchilla 70B beat Gopher 280B at 4× less compute
```

**The modern "overtrain for inference" adjustment**:

```
  Llama-3-8B (2024):
    Chinchilla-optimal for 8B would be ~160B tokens.
    Llama-3-8B was trained on 15T tokens = 1875 tokens/param.
    Dramatically overtrained.

    Why? Training happens once. Inference happens billions of times.
    Smaller model + more data → same quality + much cheaper to serve.

   Compute per inference (roughly):
     8B model: $X per 1M tokens
     70B model: $7X per 1M tokens
     At scale, 7× inference cost over a model's lifetime dominates
     training cost.
```

**Predict your loss before training**:

```python
# Chinchilla formula (approximate constants)
import math

def predict_loss(N, D):
    """N = parameters, D = training tokens"""
    E = 1.69
    A = 406.4
    alpha = 0.34
    B = 410.7
    beta = 0.28
    return E + A / N**alpha + B / D**beta

# Your proposed runs:
# 1. Chinchilla-optimal 1B model
print(f"1B×20B:   {predict_loss(1e9, 20e9):.3f}")   # ~2.37

# 2. Overtrain: small model, lots of data (Llama-3 style)
print(f"1B×100B:  {predict_loss(1e9, 100e9):.3f}")  # ~2.17

# 3. Undertrain: GPT-3 style
print(f"175B×300B:{predict_loss(175e9, 300e9):.3f}") # ~2.10

# 4. Compute-optimal for the same compute as (3)
C = 6 * 175e9 * 300e9                                # 3.15e23 FLOPs
N_opt = math.sqrt(C / 120)                            # rough Chinchilla
D_opt = 20 * N_opt
print(f"Optimal for same C: {predict_loss(N_opt, D_opt):.3f}")
```

Run before any big experiment. Catches planning errors.

**Scaling laws in one picture**:

```
  Think of it as 3 power laws:

       │  more model = ● lower loss              ●
  loss │                ●●●  (eventually plateau)
       │                   ●●●
       │                       ●●●●
       │                          ●●●●●●●
       │
       └──────────────────── N (params)

  Same for D (data) and C (compute).

  Joint: L(N, D) ≈ 1.69 + 406/N^0.34 + 411/D^0.28
         Irreducible     Cost of         Cost of
         floor           limited params  limited data
```

**The implications for your career**:

```
  Scaling laws tell you what's possible at scale.
  They don't tell you HOW to get there.

  Research tasks that scaling laws enable:
    ✓ "How much compute will I need for X capability?"
    ✓ "Is my experiment plan cost-justified?"
    ✓ "Has my run diverged from expected trajectory?"

  Research tasks scaling laws don't help with:
    ✗ "Should I use MoE or dense?"
    ✗ "Which data mix is best?"
    ✗ "How do I align this model?"

  Scaling is necessary but not sufficient for good research.
```

## Exercises

1. Read Kaplan 2020's abstract, introduction, and Figure 1. Understand: loss is a straight line on log-log.

2. Read Chinchilla 2022's abstract, and Figure 1-3. Understand: their experiment design + their isoFLOP curves.

3. For Kaplan's formula L(C), derive: if you 10x your compute, how much does loss drop? (Roughly: by a factor of 10^0.05 = 1.12. Small but compounds.)

4. For Chinchilla: GPT-3 was trained with 3.14e23 FLOPs. If instead we had spent it compute-optimally (Chinchilla style), what N and D? (N ≈ 70B, D ≈ 1.4T. Oh look - that's Chinchilla.)

5. Open nanochat's `dev/scaling_analysis.ipynb`. Look at the plots. They're just fit lines.

## Next

`06_rlhf_instruct_dpo.md` - alignment techniques explained.
