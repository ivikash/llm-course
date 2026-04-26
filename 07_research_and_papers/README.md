# Module 7 - Reading Research Papers

Papers are how knowledge travels in ML. Being fluent in papers is a career-defining skill - you'll be 6 months ahead of people who only use polished products.

## How this module works

We walk through canonical papers, slowly, side-by-side with nanoGPT or nanochat code. The goal isn't to teach each paper perfectly - it's to teach you *how to read*. By Module 7 Lesson 5 you should be able to pick up a random arXiv paper and extract its point in 30 minutes.

## Lessons

### `01_how_to_read_a_paper.md`
- The "three-pass method" (Keshav 2016).
- Pass 1 (5 min): title, abstract, introduction, figures, conclusion. Decide if worth more.
- Pass 2 (1 hr): skim methods, identify key ideas, read the important experiments.
- Pass 3 (4 hr): re-implement mentally or actually. Every claim should be defensible.
- Practical tips: use your glossary from Module 0. Use Explainer LLMs (Claude, ChatGPT) to decode dense sections.
- Red flags: fuzzy evaluation, no ablations, cherry-picked baselines, novelty theater.

### `02_core_papers_you_must_know.md`
An annotated reading list, roughly in order:

1. **"Attention Is All You Need"** (Vaswani 2017). The transformer paper.
2. **"Improving Language Understanding by Generative Pre-Training"** (Radford 2018). GPT-1.
3. **"Language Models are Unsupervised Multitask Learners"** (Radford 2019). GPT-2.
4. **"Language Models are Few-Shot Learners"** (Brown 2020). GPT-3.
5. **"Scaling Laws for Neural LMs"** (Kaplan 2020).
6. **"Training Compute-Optimal LLMs"** (Hoffmann 2022). Chinchilla.
7. **"InstructGPT"** (Ouyang 2022). RLHF for chatbots.
8. **"LLaMA"** (Touvron 2023) and **"LLaMA 2"** (Touvron 2023).
9. **"Mistral 7B"** and **"Mixtral 8x7B"** (2023).
10. **"FlashAttention"** (Dao 2022) and **"FlashAttention-2"** (Dao 2023).
11. **"Direct Preference Optimization"** (Rafailov 2023). DPO.
12. **"DeepSeek-R1"** (2024). GRPO and chain-of-thought RL.
13. **"RoPE"** (Su et al 2021). Rotary embeddings.
14. **"Emergent Abilities of LLMs"** (Wei 2022).
15. **"Chain-of-Thought"** (Wei 2022).
16. **"RAG"** (Lewis 2020). Retrieval-augmented generation.
17. **"Mamba"** (Gu and Dao 2023). State-space alternative to transformers.
18. **"The Bitter Lesson"** (Sutton 2019). Essay, not paper. The most important 600 words in the field.

Each paper gets 1-2 paragraphs in the file: what problem it solves, what its key idea is, why you should care.

### `03_transformer_paper_walkthrough.md`
- Open "Attention Is All You Need" and `nanoGPT/model.py` side by side.
- Map every equation in the paper to lines in the code.
- Notice what's different: pre-norm vs post-norm, dropout placement, learned vs sinusoidal positions.
- Understand that the paper is actually about machine translation (encoder-decoder) and GPT is the decoder-only half.

### `04_gpt_papers_gpt2_gpt3_gpt4.md`
- GPT-1: "unsupervised pre-training helps supervised tasks." Tiny by today's standards.
- GPT-2: "language models can do zero-shot tasks." Release politics.
- GPT-3: "scaling gives you few-shot prompting." The paper that started the arms race.
- GPT-4 system card (not a paper - no architecture details): emphasis on evaluation and safety.
- What each added to the recipe.

### `05_scaling_laws_papers.md`
- Kaplan 2020: power-law loss curves, parameter-data balance (wrongly conservative).
- Chinchilla 2022: corrected, ~20 tokens/param.
- Practical: how to estimate loss for a proposed experiment.
- nanochat's scaling-laws notebook (`dev/scaling_analysis.ipynb`, `dev/estimate_gpt3_core.ipynb`) - read it alongside the papers.

### `06_rlhf_instruct_dpo.md`
- InstructGPT / RLHF pipeline: SFT → reward model → PPO.
- Why this worked: aligning the model's distribution with human preferences.
- DPO: sidestep reward model training, use preference pairs directly.
- GRPO (DeepSeek): RL on verifiable rewards, no reward model or human prefs needed.
- Connection to `nanochat/scripts/chat_sft.py` and `chat_rl.py`.

### `07_modern_architectures.md`
- Llama family (2023-2024): RoPE, SwiGLU, RMSNorm, GQA. The modern recipe.
- Mixture of Experts: sparsely-gated layers, effective for large-capacity cheap-inference models.
- State-space models (Mamba, RWKV): non-attention alternatives. Linear-time in sequence length.
- Longer-context tricks: sliding windows, infinity attention, retrieval augmentation.

### `08_how_to_reproduce_a_paper.md`
- Reproduction is the best-learning exercise in ML.
- Pick a small paper, set a scope ("reproduce figure 2").
- Expect 2-3x the effort of your estimate.
- Use the authors' code if available; don't rewrite from scratch.
- Document your process publicly. This gets you noticed.

### `09_how_to_write_a_paper.md`
- Structure: abstract, intro, related work, method, experiments, discussion, limitations.
- The "stupid experiment" you need: an ablation that shows your key idea is the one that matters.
- Figure craft: one clear message per figure.
- Citations: not theater, they're scholarly courtesy. Cite charitably.
- Workshop vs main track vs arXiv-only.

### `capstone_paper_replica.md`
- Pick a paper, choose one figure or table to reproduce.
- Use nanoGPT or nanochat as your engine.
- Document your work in a Markdown file.
- Compare your numbers to the paper's. Explain any differences.
- Submit as a PR to a relevant community repo, or publish as a blog post.

## Why this module

Research taste - the skill of knowing which papers matter and which don't - is the most valuable and least teachable skill in the field. This module is a start. Real development happens by reading 100+ papers over years.
