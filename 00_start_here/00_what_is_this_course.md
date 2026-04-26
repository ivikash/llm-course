# 00 - What This Course Is (and Isn't)

## The promise

At the end of this course, you will understand, from first principles, how
ChatGPT / Claude / Gemini / Llama-style models work, how they are trained,
what it takes to run them in production, how the field does research, and
what it's like to lead a team doing this work.

You will have:

- Trained a small GPT yourself, on your own laptop.
- Trained a small chatbot end-to-end, talked to it in a web UI.
- Read the seminal papers ("Attention Is All You Need", GPT-3, Chinchilla, InstructGPT) and understood them.
- A mental model of the infrastructure: GPUs, distributed training, cloud providers, HuggingFace, wandb, SageMaker.
- A mental model of the research process: forming hypotheses, running experiments, killing ideas, shipping wins.

## The honest reality check

- You cannot learn this in a weekend. Budget ~3-6 months of evenings and weekends.
- You will be confused a lot. That is normal and it is the point. Confusion is where learning happens. If you never feel confused, you're not pushing hard enough.
- "Understanding" is a spectrum. First time through the attention mechanism, you will "get the idea". The tenth time, you'll start to have intuitions. The hundredth time, you can argue about design choices. All are valid, just at different depths.
- The field moves weekly. A technique in Module 7 might be superseded by the time you get there. That's fine. Once you understand the foundations, new techniques become variants you can absorb in an afternoon.

## The method

We work **from the code, outward**.

Most courses start with theory, then math, then "and here's how you'd implement it." That's backwards for a practical engineer. We start with a small, working, readable codebase - `nanoGPT` - and use that as the anchor. Every concept we introduce has a corresponding chunk of code you can point to.

When we say "the attention mechanism lets each token look at others," I will show you the three lines in `nanoGPT/model.py` that implement exactly that. When we say "AdamW is an optimizer," I will point to the line in `train.py` where we create it.

Then, later, we graduate to `nanochat`, which shows what a *real* modern LLM pipeline looks like - tokenizer, pretraining, fine-tuning, reinforcement learning, evaluation, serving. Same philosophy: the code is the spine, the concepts hang off it.

## Who this course is for

- You know Python. You can write functions, use classes, use pip/venv.
- You know basically nothing about ML, AI, neural nets, calculus, matrix math, or any of the jargon.
- You want to be serious about this, not just dabble.

## Who this is **not** for

- People who want a 20-minute "What is ChatGPT" explainer. (Go watch one of Karpathy's YouTube videos - they're excellent - then come back here.)
- People who want to do classical ML (scikit-learn, XGBoost, etc.). That's a different (also valuable) skill set.
- People who want to be LLM "prompt engineers" without understanding the model. This course goes deeper than that role requires.

## Visualize this

Before you read further, spend 10 minutes watching:
- **3Blue1Brown: "But what is a neural network?"** (https://www.youtube.com/watch?v=aircAruvnKk) - the best 18-minute intro to neural nets ever made. Do it now. It will save you hours of confusion later.

If you want a one-image intuition for the whole course:

```
[raw text]  →  [tokens]  →  [embeddings]  →  [attention]  →  [MLP]  →  [logits]  →  [next token]
  strings       integers     vectors         talk to each    think     probability    pick one
                             (numbers)       other           per-token  distribution
                                             (Module 3)      (Module 3)  (softmax)
```

Every arrow becomes a lesson. By the end of Module 5 you will understand each transformation intimately.

---

## The tone

I will not talk down to you. I will also not pretend you know things you don't. If you've just said "I don't know what a matrix is," I will not casually mention eigenvalues three lines later and expect you to fill in the gap. Every term is defined the first time it appears.

When I *do* use a jargon word because it's important and standard (e.g. "tensor", "embedding", "gradient"), I'll bold it on first use and define it right there.

## How to actually do the course

1. **Read** the lesson file. Slowly.
2. **Run** the code snippets, even when you think you understand. Your fingers will notice things your eyes missed.
3. **Do** the exercises. They're small on purpose.
4. **Ask questions.** Talk to me ("explain why X" or "what does line 47 of model.py do"). That's what I'm here for.
5. Don't skip. Each lesson builds on the last. If you skip Module 1 because "math is boring", Module 3 will crush you and you'll quit. Just do them in order.

## Prerequisites, one more time

- A laptop (Linux/Mac/WSL on Windows).
- Python 3.10+.
- The two repos already on your machine (you have them: `~/workspace/nanoGPT` and `~/workspace/nanochat`).
- A terminal you are not afraid of.

That's it. No GPU needed for the first three modules. By Module 4 you'll want one, and I'll show you how to rent one for a few dollars an hour.

## What's next

Open `01_demystified_glossary.md`. Before we do any math or code, we're going to demystify every scary term you've ever heard in AI. You're going to be surprised how many of them are simple ideas hiding behind pretentious names.
