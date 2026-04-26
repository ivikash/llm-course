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

## Exercises

1. Read the abstracts and intros of GPT-1, GPT-2, GPT-3 papers in sequence. Notice the "scale works" narrative building.

2. Count citations on Google Scholar for each (estimate). Note GPT-3 > GPT-2 > GPT-1 by orders of magnitude.

3. Read GPT-4 Technical Report's section "Capabilities and Limitations" - understand what they claim and what they carefully don't.

4. Read o1 / R1 release notes or papers. Note the shift toward inference-time compute.

## Next

`05_scaling_laws_papers.md` - Kaplan vs Chinchilla, in detail.
