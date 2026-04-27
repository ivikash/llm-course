# 7.4 - The GPT Papers: GPT-1, GPT-2, GPT-3, GPT-4

Each was a progression. This lesson covers what each contributed.

## GPT-1 (2018): "Improving Language Understanding by Generative Pre-Training"

**Authors**: Radford, Narasimhan, Salimans, Sutskever (OpenAI).
**Size**: 117M parameters.
**Data**: BooksCorpus, 800M words.

### Contribution

Proposed the **pretrain-then-finetune** paradigm for LLMs:
1. Train a transformer decoder on next-token prediction (pretrain).
2. Add a small task-specific head and fine-tune on labeled data.

Showed that this beat task-specific architectures on many NLP benchmarks (sentiment, entailment, etc.).

### Why it matters

Established that general pretraining is more valuable than task-specific engineering. Before GPT-1, people built custom models per task. After GPT-1, "fine-tune a pretrained LM" became the default.

## GPT-2 (2019): "Language Models are Unsupervised Multitask Learners"

**Authors**: Radford, Wu, Child, Luan, Amodei, Sutskever (OpenAI).
**Size**: 124M / 355M / 774M / 1.5B parameters.
**Data**: WebText, ~40GB.

### Contribution

1. **Scale**. 12x the params of GPT-1.
2. **Zero-shot**. They showed GPT-2 could do some tasks **without any fine-tuning** - just describe the task in a prompt.

Example: to translate, just write "English: Hello. French: " and the model continues.

3. **Release politics**. Initially OpenAI withheld the 1.5B version citing safety concerns. This was controversial and set the template for "responsible release" debates.

### Why it matters

- **Zero-shot works**. Capability emerged from scale, not from supervised data.
- **Scale helps a lot**. Bigger was meaningfully better.
- This paper is what nanoGPT reproduces.

## GPT-3 (2020): "Language Models are Few-Shot Learners"

**Authors**: Brown, Mann, Ryder, Subbiah, Kaplan... (OpenAI; 31 authors).
**Size**: 175B parameters (smallest in family: 125M, biggest: 175B).
**Data**: Common Crawl + books + Wikipedia, 300B tokens.

### Contribution

1. **Massive scale**. 100x GPT-2.
2. **Few-shot prompting** ("in-context learning"). You put 3-5 examples of the task in the prompt, then the new example; the model figures out the pattern without any gradient update.

Example prompt:
```
"Ocean" in Spanish is "océano".
"Mountain" in Spanish is "montaña".
"River" in Spanish is "_____".
```
Model completes: `río`.

3. **Comprehensive evaluation**. Tested on 40+ benchmarks across many domains.

### Why it matters

- In-context learning was genuinely surprising. Shouldn't work by any pre-existing theory. The fact that it does changed what people thought LLMs were.
- Established "just scale up" as a viable research strategy for 2020-2022.
- Kicked off the current LLM arms race. Everyone wanted their own GPT-3.

### What it didn't solve

- Still bad at following instructions directly.
- Often verbose, unhelpful, or offensive.
- That's why ChatGPT needed SFT + RLHF (next paper).

## InstructGPT (2022): "Training language models to follow instructions with human feedback"

**Authors**: Ouyang, Wu, Jiang, ... (OpenAI).
**Contribution**: added SFT + RLHF on top of GPT-3. Result: a model that follows instructions much better. The basis of ChatGPT.

(Covered in Module 5 Lesson 6, Module 7 Lesson 6 - I'll write a dedicated walkthrough later.)

## GPT-4 (2023): No architecture paper, just a system card

**Authors**: OpenAI (the company).
**Size**: Not disclosed. Rumored ~1T params total (MoE), ~200B active per token.
**Data**: Not disclosed.

### "Paper"

The [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) is basically a marketing document with benchmark results. No architecture details, no hyperparameters, no training data.

### What they did disclose
- Dramatic improvement over GPT-3.5 on benchmarks.
- Multimodal (text + images).
- Emphasis on safety (red-teaming, refusals).
- System message architecture.
- 32K context length variant.

### Why that matters

- Frontier labs have stopped publishing. GPT-4, Claude-3, Gemini don't have architecture papers. You can only reason about them from their behavior and from rumors.
- Open models (Llama, Mistral, DeepSeek) remain our best guides to "what frontier architectures look like."
- Reproducibility of frontier work is now essentially impossible without inside access.

## GPT-4o, o1, o3, GPT-5 (2024-2025)

- **GPT-4o** (May 2024): faster, cheaper, multimodal (voice + vision + text).
- **o1** (2024): "reasoning" model, spends compute at inference time via chain-of-thought.
- **o3** (2025): improved o1.
- **GPT-5** (future): not released as of writing.

The "reasoning" turn (o1 onwards) is significant: inference-time compute matters, not just training compute. RL on reasoning traces. Related to DeepSeek-R1, which was open about methodology.

## Takeaway narrative

| Era | Paradigm | Representative model |
|-----|----------|---------------------|
| 2018 | Pretrain + fine-tune | GPT-1, BERT |
| 2019 | Scale + zero-shot | GPT-2 |
| 2020-22 | Scale + few-shot | GPT-3 |
| 2022-23 | Alignment + SFT + RLHF | ChatGPT, Claude |
| 2023-24 | Open weights, Llama recipe | Llama, Mistral |
| 2024 | MoE, long context, multimodal | GPT-4, Mixtral, Gemini |
| 2025 | Inference-time reasoning (RL on CoT) | o1, R1, o3 |

## What the GPT papers DON'T teach

- How to actually build a frontier model (data recipes, infra, secrets).
- Multimodal training details.
- Safety and RLHF specifics at scale (public papers are several years behind what's in production).

For those, watch open-source (Llama, DeepSeek, Qwen papers - they publish more honestly) and inside-baseball blog posts.

## Visualize this

**The GPT family size ladder**:

```
  Params     Model             Year   Key innovation
  ────────── ──────────────── ────── ───────────────────────────────
  117M       GPT-1             2018   pretrain + finetune paradigm
  345M       BERT-base          2018   bidirectional, encoder-only
  1.5B       GPT-2              2019   scale + zero-shot
  175B       GPT-3              2020   scale + few-shot in-context
  ?          InstructGPT        2022   SFT + RLHF → ChatGPT
  ?          GPT-4              2023   multimodal, reasoning, MoE?
  ?          GPT-4o             2024   native voice+vision
  ?          o1 / o3            2024-  inference-time reasoning
```

**Parameters over time (log scale)**:

```
  1e12 │                                                          ●  (GPT-4 ~1T rumored)
       │
  1e11 │                                         ●  GPT-3 (175B)
       │
  1e10 │                               ●  Llama-2 (70B)
       │
  1e9  │                     ●  GPT-2 (1.5B)         ●  Llama-3 (8B)
       │
  1e8  │           ●  GPT-1 (117M)
       │
  1e7  │ ●
       └──────────────────────────────────────────────────────── year
        2017    2018     2019    2020    2021    2022    2023   2024

  Rough trend: 10× params every 2 years (2017-2020).
  Then slowed: Chinchilla said "more data, not more params".
  Now: MoE-style scaling continues capacity growth implicitly.
```

**What each paper DID and DIDN'T do**:

```
  GPT-1 (2018):
    ✓ Showed pretrain + finetune beats task-specific arch
    ✓ 12 transformer decoder layers
    ✗ Too small to be useful zero-shot (117M)

  GPT-2 (2019):
    ✓ Showed scale gives zero-shot ability
    ✓ 1.5B params (10× GPT-1)
    ✗ Initially withheld - release politics
    ✗ Still couldn't follow instructions

  GPT-3 (2020):
    ✓ Showed few-shot in-context learning
    ✓ 175B params (100× GPT-2)
    ✗ Still dumb at following instructions
    ✗ Expensive, undertrained (300B tokens only - Chinchilla error)

  InstructGPT (2022):
    ✓ Showed SFT + RLHF fixes instruction-following
    ✓ Basis for ChatGPT (shipped Nov 2022)
    ✗ Closed: not easy to reproduce

  GPT-4 (2023):
    ✓ Multimodal (text + images)
    ✓ Massively better on benchmarks
    ✗ Architecture undisclosed (rumored MoE, ~1T effective)
    ✗ No paper, just a "system card"

  o1 series (2024-25):
    ✓ Inference-time reasoning via long chain-of-thought
    ✓ Dramatic gains on math, code, science
    ✗ Very slow and expensive per query
    ✗ Architecture undisclosed
```

**The "disclosure cliff"**:

```
  Openness of OpenAI's work:

  2018  GPT-1      ████████████████ detailed paper, code, weights
  2019  GPT-2      ██████████████   paper, delayed release, weights
  2020  GPT-3      ████████████      paper, API only (no weights)
  2022  InstructGPT ██████             paper, no weights
  2023  GPT-4      ██                 "system card", no architecture
  2024  o1         █                   blog post only

  Meanwhile, the open-source world caught up:
  2023  Llama       ████████████████ paper + weights
  2024  Llama-3     ████████████████ detailed paper + weights
  2024  DeepSeek-V3 ████████████████ detailed paper + weights
  2025  DeepSeek-R1 ████████████████ detailed paper + weights

  Today, open-weight models give better transparency
  than frontier closed models. Read Llama/DeepSeek papers for methodology.
```

**Timeline of capability jumps**:

```
  2018   GPT-1: can do sentiment analysis after finetuning
  2019   GPT-2: writes coherent essays, kinda
  2020   GPT-3: few-shot task solving, beats humans on some benchmarks
  2022   ChatGPT: finally feels like talking to a thing
  2023   GPT-4: passes bar exam, writes functional code, beats most humans
  2024   o1: solves PhD-level problems with extended reasoning
  2025   o3: beats humans on ARC-AGI (a holdout test of fluid reasoning)

  Each release feels like sci-fi came slightly more real.
```

**What to read if you want to dig deeper into GPT-4-era**:

```
  Meta (Llama-3 paper, 2024):
    - Comprehensive 92-page technical report
    - Data recipe, architecture, training infra, safety
    - Better than reading GPT-4's "system card" which says nothing

  DeepSeek-V3 (2024):
    - 55-page paper with architecture + training + infra
    - MoE details, fp8 training at scale
    - Best open documentation of a frontier-class run

  DeepSeek-R1 (2025):
    - 25-page paper on GRPO-style reasoning training
    - Actual recipe (unlike o1 which is a black box)
```

## Exercises

1. Read the abstracts and intros of GPT-1, GPT-2, GPT-3 papers in sequence. Notice the "scale works" narrative building.

2. Count citations on Google Scholar for each (estimate). Note GPT-3 > GPT-2 > GPT-1 by orders of magnitude.

3. Read GPT-4 Technical Report's section "Capabilities and Limitations" - understand what they claim and what they carefully don't.

4. Read o1 / R1 release notes or papers. Note the shift toward inference-time compute.

## Next

`05_scaling_laws_papers.md` - Kaplan vs Chinchilla, in detail.
