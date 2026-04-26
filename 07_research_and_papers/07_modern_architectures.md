# 7.7 - Modern Architectures: MoE, State Space, Llama, and Beyond

Four families of modern language-model architectures, each a response to problems with plain decoder transformers.

## 1. The Llama family (still a dense transformer, but modernized)

Starting point: same decoder-only transformer as GPT-2, with these upgrades:

- **RoPE** for position encoding (vs learned embeddings).
- **RMSNorm** for normalization (vs LayerNorm).
- **SwiGLU** for MLP (vs GELU).
- **GQA** (Grouped-Query Attention, vs MHA).
- **No biases** on linear layers.
- **Weight-tied** input-output embeddings.

nanochat uses all of these. This is the "modern default" recipe. Llama 1→2→3 follow it; each iteration is mainly more data, bigger size, and better fine-tuning, not architecture change.

If you grok Module 5's `nanochat/gpt.py`, you understand Llama's architecture.

## 2. Mixture of Experts (MoE)

### The problem

Bigger = better for LLMs, but inference cost scales with model size. How do you get a "big model's knowledge" with a "small model's inference cost"?

### The idea

Replace the MLP in each layer with **many MLPs** (called "experts"), but only use **some** per token. A router decides which.

```
# instead of:
y = MLP(x)

# MoE:
scores = Router(x)                    # (B, T, num_experts)
top_k_experts = top_k(scores, k=2)    # e.g. route each token to 2 experts
y = sum(experts[i](x) * weight[i] for i in top_k_experts)
```

### Concrete example: Mixtral 8x7B

- 8 experts per layer.
- Each token goes to 2 experts (top-2 routing).
- 47B total params in memory.
- Only ~13B params are used per token.
- Inference cost: like a 13B model. Capacity: between 13B and 47B.

### Pros
- More capacity per FLOP.
- Serves larger-capacity models on smaller hardware.
- Training scales further.

### Cons
- Training instability (routing collapse - all tokens go to one expert).
- Memory footprint is still the full 47B.
- Load balancing is tricky (not all experts used equally).
- Harder to fine-tune.

### MoE papers
- **Switch Transformer** (Fedus 2022): foundational.
- **GShard** (Lepikhin 2020): Google's early large-scale MoE.
- **Mixtral** (Jiang 2024): first widely-used open MoE.
- **DeepSeek-V3** (2024): 671B total, 37B active per token. Many routing improvements.
- **Llama 4** (2025, speculative): rumored MoE.

### Is GPT-4 an MoE?

Strong rumors: yes. Likely 8x220B-ish experts, ~2 activated per token. OpenAI has never confirmed.

nanoGPT and nanochat are **dense** (no MoE). Implementing MoE is a good advanced exercise but outside this course.

## 3. State-Space Models (Mamba, RWKV)

### The problem with transformers

Attention is O(T²) in sequence length. For context 100K, that's 10 billion attention entries per layer. Painful.

### The Mamba idea (Gu + Dao, 2023)

Use **linear-time** recurrent-ish structure with a learnable state. Roughly:

```
h[t] = A(t) * h[t-1] + B(t) * x[t]
y[t] = C(t) * h[t]
```

Where `A(t), B(t), C(t)` depend on the input - this is what makes it "selective". Much like LSTMs but with specific parallel-scan tricks and a recency bias that's well-calibrated.

- O(T) time for training (via parallel scan) and O(1) per step at inference.
- No attention matrix - doesn't materialize T×T.
- Can handle infinite context in principle.

### Mamba's status

- **Mamba-1** (2023): first viable competitor to transformers on LM benchmarks.
- **Mamba-2** (2024): better, unified with attention mathematically.
- **Jamba** (2024), **Zamba** (2024): Mamba + transformer hybrids.

Trade-off: Mamba is smaller-loss at same FLOPs, but transformers still win on certain in-context tasks (copying from the prompt). Hybrids seem to be the sweet spot.

### RWKV (Peng 2023+)

Another RNN-transformer hybrid, with a focus on inference efficiency. Mostly open-source community driven.

## 4. Hybrid approaches

Recent papers (2024-2025) combine multiple ideas:

- Attention for a few layers (strong in-context reasoning).
- Mamba/SSM for others (fast long-context).
- MoE for MLPs (capacity).
- Sliding window attention (cheap approximation).

Examples: Jamba, Zamba2, Gemma-3 (uses sliding-window attention).

## What should you focus on?

### For career usefulness:
1. **Llama-style dense transformer**: 95% of what you'll encounter.
2. **MoE**: know conceptually, understand from Mixtral / DeepSeek papers. Bonus for implementation.
3. **Mamba / SSM**: know it exists, read the paper, don't invest heavily unless you're doing research on it.

### For research:
Watch the frontier. The architecture landscape is more active now (2025) than it's been in years. New hybrids emerging constantly.

## Other architectures worth knowing

### Encoder-only (BERT-style)
- Bidirectional attention.
- Trained with masked language modeling.
- Good for classification, retrieval, embeddings.
- Examples: BERT, RoBERTa, DeBERTa.
- Not for text generation.

### Encoder-decoder
- Original transformer design.
- Good for translation, summarization.
- Examples: T5, BART.
- Mostly replaced by decoder-only for general use.

### Diffusion LMs
- Generate text iteratively (denoising), not left-to-right.
- Experimental; not yet mainstream.
- Example: Plaid-LM, Diffusion-LM.

## Exercises

1. Read the Mixtral paper (8 pages) to understand MoE mechanics: https://arxiv.org/abs/2401.04088

2. Read the Mamba paper's introduction and Figure 1 only: https://arxiv.org/abs/2312.00752

3. Find a minimal MoE implementation in code. HuggingFace's `modeling_mixtral.py` is a good reference. Trace the routing + top-k code.

4. Compute: Mixtral 8x7B with top-2 routing. How much VRAM for serving in bf16? (47B × 2 bytes ≈ 94 GB. Needs 2 A100-80GB.)

5. Read "The Bitter Lesson" one more time. Meditate on how MoE/Mamba/etc fit the "bet on compute" philosophy.

## Next

`08_how_to_reproduce_a_paper.md` - putting reading skills into practice.
