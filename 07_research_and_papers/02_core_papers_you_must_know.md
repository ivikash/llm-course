# 7.2 - Core Papers You Must Know

An annotated reading list. These are the ~25 papers that form the backbone of modern LLM knowledge. Read in order if new to the field. Skip around if you have context.

For each paper: the problem, the core idea, why you should care, and what the course covers.

---

## Foundations (must-reads)

### 1. Attention Is All You Need (Vaswani et al., 2017)
**Link**: https://arxiv.org/abs/1706.03762
**Problem**: RNNs are slow and can't scale.
**Idea**: An encoder-decoder architecture using only attention, no recurrence. Multi-head self-attention + MLPs + layer norm + residuals.
**Why care**: Every modern LLM is a variant of this. Covered in Module 3, walked in Module 7 Lesson 3.

### 2. Improving Language Understanding by Generative Pre-Training (Radford et al., 2018) - GPT-1
**Link**: [OpenAI blog](https://openai.com/research/language-unsupervised)
**Idea**: Take the transformer decoder. Pretrain on next-token prediction. Fine-tune for downstream tasks.
**Why care**: The first GPT. Introduces the "pretrain + fine-tune" paradigm for LLMs.

### 3. Language Models are Unsupervised Multitask Learners (Radford et al., 2019) - GPT-2
**Link**: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
**Idea**: Just scale. 1.5B params. Show "zero-shot" task performance - tell the model what you want, no fine-tuning.
**Why care**: Validated that scale alone unlocks capabilities.

### 4. Language Models are Few-Shot Learners (Brown et al., 2020) - GPT-3
**Link**: https://arxiv.org/abs/2005.14165
**Idea**: 175B parameters. In-context learning: show a few examples in the prompt, the model infers the task.
**Why care**: Kicked off the current LLM era. Every "give me 3 examples and then answer" prompt uses this paradigm.

---

## Scaling and training

### 5. Scaling Laws for Neural Language Models (Kaplan et al., 2020)
**Link**: https://arxiv.org/abs/2001.08361
**Idea**: Loss is a power law in N, D, C (params, data, compute). Predict loss from these.
**Why care**: Made LLM training planning rigorous. Module 6 Lesson 9.

### 6. Training Compute-Optimal LLMs (Hoffmann et al., 2022) - Chinchilla
**Link**: https://arxiv.org/abs/2203.15556
**Idea**: Corrected Kaplan. Optimal ratio is ~20 tokens per param, not the more-params-than-data approach.
**Why care**: Standard training recipe today. Module 6 Lesson 9.

---

## From base model to chatbot

### 7. Training Language Models to Follow Instructions with Human Feedback (Ouyang et al., 2022) - InstructGPT
**Link**: https://arxiv.org/abs/2203.02155
**Idea**: SFT + Reward Model + PPO = much better instruction-followers. The pipeline behind ChatGPT.
**Why care**: Alignment 101. Module 5 Lesson 6.

### 8. Direct Preference Optimization (Rafailov et al., 2023) - DPO
**Link**: https://arxiv.org/abs/2305.18290
**Idea**: Skip the reward model. Use preference pairs directly in a clever loss function.
**Why care**: Simpler, cheaper, widely adopted. Most open-source chat models use this.

### 9. DeepSeek-V3 / DeepSeek-R1 (DeepSeek-AI, 2024-2025)
**Link**: https://arxiv.org/abs/2412.19437 (V3), https://arxiv.org/abs/2501.12948 (R1)
**Idea**: MoE architecture. R1 uses RL on verifiable rewards (math, code) to teach reasoning.
**Why care**: Cutting-edge open weights. GRPO-style RL. Module 5 Lesson 6.

---

## Open models

### 10. LLaMA (Touvron et al., Feb 2023)
**Link**: https://arxiv.org/abs/2302.13971
**Idea**: Open-weights 7B-65B models. Modern architecture (RoPE, RMSNorm, SwiGLU).
**Why care**: The architecture nanochat uses. Foundation of open-source ecosystem.

### 11. Llama 2 (Touvron et al., July 2023)
**Link**: https://arxiv.org/abs/2307.09288
**Idea**: Open-weights with commercial license. 7B/13B/70B. SFT + RLHF chat models.
**Why care**: The first major open chat model. Started the open-source momentum.

### 12. Llama 3 (Meta, 2024)
**Link**: https://ai.meta.com/research/publications/the-llama-3-herd-of-models/
**Idea**: More data (15T tokens), cleaner pipeline, 8B/70B/405B.
**Why care**: Current open state of the art (405B competes with GPT-4).

### 13. Mistral 7B (Jiang et al., 2023)
**Link**: https://arxiv.org/abs/2310.06825
**Idea**: 7B with sliding-window attention, GQA. Beats larger Llama-2s.
**Why care**: Showed architecture innovations still matter after scale.

### 14. Mixtral 8x7B (Jiang et al., 2024) - MoE
**Link**: https://arxiv.org/abs/2401.04088
**Idea**: Sparse Mixture of Experts. 47B total params, ~13B active per token.
**Why care**: Introduced MoE to open-source mainstream.

---

## Architecture innovations

### 15. RoFormer (Su et al., 2021) - RoPE
**Link**: https://arxiv.org/abs/2104.09864
**Idea**: Rotary Position Embeddings - encode position via rotations of Q, K.
**Why care**: Replaced learned positional embeddings in almost all modern LLMs.

### 16. FlashAttention (Dao et al., 2022) and FlashAttention-2 (Dao, 2023)
**Link**: https://arxiv.org/abs/2205.14135, https://arxiv.org/abs/2307.08691
**Idea**: IO-aware attention implementation. Same math, much less memory traffic.
**Why care**: Default attention kernel today. Module 6 Lesson 5.

### 17. GPT-NeoX / PaLM / GLaM (various, 2022)
**Idea**: Engineering lessons from training very large transformers.
**Why care**: Practical wisdom, not novel ideas.

### 18. Switch Transformer (Fedus et al., 2022) - sparse MoE foundation
**Link**: https://arxiv.org/abs/2101.03961
**Idea**: Replace MLP with "expert" ensemble, route each token to one expert. Cheap inference, huge capacity.
**Why care**: Foundational MoE paper. GPT-4 and others are thought to use MoE.

---

## Reasoning and capability

### 19. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)
**Link**: https://arxiv.org/abs/2201.11903
**Idea**: "Let's think step by step" - ask the model to reason aloud. Accuracy jumps on math and logic.
**Why care**: Possibly the most useful prompting trick ever discovered.

### 20. Emergent Abilities of Large Language Models (Wei et al., 2022)
**Link**: https://arxiv.org/abs/2206.07682
**Idea**: Some capabilities appear suddenly at scale, not gradually.
**Why care**: Foundational claim; debated (Schaeffer 2023 counter-argument), but shaped discourse.

### 21. Self-Consistency / Tree of Thoughts / ReAct (various, 2022-2023)
**Idea**: Make the model reason more by sampling multiple solutions, structure thinking, call external tools.
**Why care**: The foundations of agentic AI.

---

## Retrieval and agents

### 22. RAG: Retrieval-Augmented Generation (Lewis et al., 2020)
**Link**: https://arxiv.org/abs/2005.11401
**Idea**: Retrieve relevant documents before generating. Combines LLM with a search index.
**Why care**: How you make a chatbot that knows up-to-date facts or proprietary docs.

### 23. Toolformer (Schick et al., 2023)
**Link**: https://arxiv.org/abs/2302.04761
**Idea**: Train models to call external tools (calculator, search, translation API).
**Why care**: Foundation of agentic systems.

---

## Alternatives and frontiers

### 24. Mamba (Gu and Dao, 2023)
**Link**: https://arxiv.org/abs/2312.00752
**Idea**: State-space models as a transformer alternative. Linear in sequence length.
**Why care**: One of the only credible alternatives to transformers in 7+ years.

### 25. The Bitter Lesson (Sutton, 2019)
**Link**: http://www.incompleteideas.net/IncIdeas/BitterLesson.html
**Format**: Short essay, not a paper. ~600 words.
**Idea**: Methods that scale with compute win. Human-engineered shortcuts lose.
**Why care**: The single most important piece of AI philosophy in the last 20 years. Read it now.

---

## What about multimodal?

Covered in Module 9:
- **CLIP** (Radford et al., 2021) - image-text contrastive learning.
- **Stable Diffusion** (Rombach et al., 2022) - latent diffusion for image gen.
- **Whisper** (Radford et al., 2022) - speech recognition as a transformer.
- **Flamingo / LLaVA / Idefics** - vision-language models.
- **DALL-E / Imagen / Sora** - image/video generation.

---

## Visualize this

**The 25 papers visualized as a tree of influence**:

```
                    Attention Is All You Need (2017)
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
     GPT-1 (2018)       BERT (2018)      T5 (2019)
            │                 │                 │
     GPT-2 (2019)             │                 │
            │                 │                 │
     GPT-3 (2020)    ──→ Scaling Laws (Kaplan 2020)
            │                 │
            │        ──→ Chinchilla (2022)    ← correction
            │
     InstructGPT (2022) ──→ ChatGPT
            │
            ├── DPO (2023)
            └── RLHF variants
                    │
                    ▼
     [Open-source wave 2023]
            │
            ├── LLaMA (2023)
            ├── Mistral 7B (2023)
            └── Llama-2 / Llama-3 (2023-2024)
                    │
                    ├── Mixtral 8x7B (MoE)
                    └── DeepSeek-V3 (2024)
                                │
                                └── DeepSeek-R1 (2025, reasoning)
                                        │
                                        └── OpenAI o1 / o3 (2024-25)

  Architecture innovations (woven throughout):
      RoPE (2021) → Llama
      FlashAttention (2022) → everyone
      SwiGLU → Llama
      GQA → Llama-2-70B, Llama-3

  Alternative paths:
      Mamba / SSM (2023) - transformer alternatives
      Constitutional AI (Anthropic 2022) - alignment variant
```

**Each paper's "elevator pitch"**:

```
  Paper                     One-line core contribution
  ────────────────────────  ──────────────────────────────────────
  Attention (2017)          Attention + MLP → transformer
  GPT-1 (2018)              Pretrain + finetune paradigm for NLP
  GPT-2 (2019)              Scale → zero-shot task performance
  GPT-3 (2020)              Scale → few-shot in-context learning
  Kaplan (2020)             Loss = power law in N, D, C
  Chinchilla (2022)         Scale N and D equally, ~20 tokens/param
  InstructGPT (2022)        SFT + RLHF = much better chat
  LLaMA (2023)              Modern architecture + open weights
  FlashAttention (2022)     Same math, much faster via tiling
  Mixtral (2024)            MoE: big capacity, cheap inference
  DPO (2023)                Preference tuning without RL
  DeepSeek-R1 (2025)        GRPO: verifiable rewards → reasoning
  Mamba (2023)              State-space alternative to attention
  CoT (Wei 2022)            "Let's think step by step" helps a lot
  The Bitter Lesson (2019)  Scale-with-compute beats human cleverness
```

**The "reading plan" by week**:

```
  Month 1 (foundations):
    Week 1: Attention + GPT-1
    Week 2: GPT-2 + GPT-3
    Week 3: Kaplan + Chinchilla (scaling laws)
    Week 4: The Bitter Lesson (philosophy)

  Month 2 (modern LLMs):
    Week 5: LLaMA 1 + 2
    Week 6: Mistral + Mixtral
    Week 7: RoPE + FlashAttention
    Week 8: GQA + SwiGLU

  Month 3 (alignment):
    Week 9: InstructGPT (RLHF)
    Week 10: DPO
    Week 11: DeepSeek-R1 (GRPO)
    Week 12: Constitutional AI

  Month 4 (frontiers):
    Week 13: Chain-of-Thought
    Week 14: RAG
    Week 15: Mamba
    Week 16: your own interest area

  After 4 months: solid base of ~20 papers read carefully.
  You've got a foundation stronger than most people working in AI.
```

**Where to find curation**:

```
  Daily:    Twitter/X (follow 20 curators)
            @karpathy, @jxmnop, @_jasonwei, @srush_nlp, @jackclarkSF

  Weekly:   - Interconnects (Nathan Lambert's newsletter)
            - Jack Clark's Import AI
            - Latent Space podcast

  Monthly:  - lifearchitect.ai/papers  (curated big-picture)
            - latent.space 2025 papers list

  Per paper:  https://paperswithcode.com  (finds the code)
             https://alphaxiv.org          (community comments)
             https://huggingface.co/papers (daily new, voted)
```

**A paper's half-life**:

```
  "Foundational" papers (indefinite relevance):
    Attention Is All You Need
    Scaling Laws, Chinchilla
    The Bitter Lesson

  "Methodology" papers (~3-5 years):
    LoRA, RoPE, FlashAttention
    InstructGPT, DPO, GRPO

  "Benchmark" papers (~1-2 years):
    MMLU, HumanEval, GSM8K
    Superseded as models improve

  "System" papers (~6 months):
    GPT-4 technical report, Claude system card
    Obsolete as next version ships
```

Invest reading time proportional to half-life. Read foundations; skim systems.

## Your reading plan

### Week 1-2 (foundations)
1 → 2 → 3 → 4 → 5 → 6. That's GPT lineage and scaling laws.

### Week 3-4 (modern LLMs)
10 → 11 → 15 → 16. Llama architecture and innovations.

### Week 5-6 (chat)
7 → 8 → 9. SFT, DPO, GRPO.

### Week 7-8 (reasoning and misc)
19 → 21 → 22 → 25. Chain-of-thought, agents, bitter lesson.

### Ongoing
One new arxiv paper per week, pass 1. Occasional pass 2 on a favorite.

---

## Staying current

Places to find new papers worth reading:

- **arXiv cs.CL** daily listing. Overwhelming - filter.
- **ai.papers** on X/Twitter. Jack Clark's [ImportAI](https://importai.substack.com) newsletter. Nathan Lambert's [Interconnects](https://www.interconnects.ai).
- **Hugging Face Daily Papers** (papers sorted by social vote).
- **Latent Space**'s paper club summaries (like the 2025 Papers list you linked).
- **Lifearchitect.ai/papers/** - Alan Thompson's curated list.

Curation > firehose. Follow 3-5 good curators; skip the rest.

## Next

`03_transformer_paper_walkthrough.md` - deep dive into paper #1 with nanoGPT.
