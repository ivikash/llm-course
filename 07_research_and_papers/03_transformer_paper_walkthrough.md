# 7.3 - "Attention Is All You Need" Walkthrough

Paper: [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762). 15 pages, June 2017. 8 authors from Google Brain + Google Research. **Among the most-cited papers in ML history.**

Open the PDF and `~/workspace/nanoGPT/model.py` side by side.

## The problem (section 1)

Pre-2017, the state of the art for sequence problems (translation, summarization) was recurrent networks (LSTMs, GRUs) with attention *bolted on* as an auxiliary mechanism.

Problems:
- RNN forces sequential computation. Can't parallelize across time steps.
- Training is slow.
- Long-range dependencies get lost in the hidden state bottleneck.

Their proposal: throw away recurrence. Use only attention. Call it the **Transformer**.

## The architecture (section 3)

Open to **Figure 1**. This is the image.

The paper does **encoder-decoder** (for translation: encode source, decode target). nanoGPT uses only the decoder side.

### Encoder side (left stack)

- Input: tokens of the source sentence.
- Input embedding + positional encoding (section 3.5 - sinusoidal, fixed).
- N=6 layers stacked, each:
  - Multi-head self-attention.
  - Residual connection + LayerNorm.
  - Position-wise feedforward network (same as our MLP).
  - Residual connection + LayerNorm.

Key: the encoder uses **bidirectional** attention - each token attends to all others. No causal mask.

### Decoder side (right stack)

- Input: tokens of the target sentence (during training, shifted one position).
- Input embedding + positional encoding.
- N=6 layers, each:
  - **Masked** (causal) multi-head self-attention.
  - Residual + LayerNorm.
  - **Encoder-decoder attention** (decoder queries + encoder K/V).
  - Residual + LayerNorm.
  - Position-wise FFN.
  - Residual + LayerNorm.

Key: three differences from the encoder - causal masking, an extra attention block that looks at the encoder output, and FFN.

### Final projection

Linear + softmax to produce token probabilities.

## Section 3.2: attention, precisely

This is the math you already know from Module 3.

"Scaled Dot-Product Attention":

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

Match to nanoGPT `CausalSelfAttention.forward`:

```python
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
att = F.softmax(att, dim=-1)
y = att @ v
```

Same three lines. The paper and the code agree.

"Multi-Head Attention":

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

In code: the split into heads and concat at the end (Module 3 Lesson 5 covered this precisely).

## Section 3.3: FFN

"Position-wise Feed-Forward Network":

```
FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
```

That's a linear, ReLU, linear. With `d_ff = 4 * d_model`. In nanoGPT's MLP class, it's the same structure with GELU instead of ReLU.

## Section 3.4: embeddings and softmax

Standard stuff: token embeddings multiplied by `sqrt(d_model)`, tied weights between input embedding and output projection. nanoGPT uses weight tying (`self.transformer.wte.weight = self.lm_head.weight`).

## Section 3.5: positional encoding

Original paper uses sinusoidal encoding:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Fixed (no learnable params), generalizes to arbitrary length.

nanoGPT uses **learned** position embeddings. Works similarly, limited to `block_size`. Modern models use RoPE (Module 3 Lesson 9).

## Section 4: Why self-attention

A table comparing attention vs recurrent vs convolutional on:
- Computation per layer.
- Sequential operations (parallelizability).
- Max path length (how fast info travels across the sequence).

Self-attention: O(T^2) per layer, O(1) sequential ops, O(1) max path. RNNs are O(T) sequential. Attention wins on parallelism.

## Section 5: training

- Dataset: WMT English-German (4.5M sentence pairs), English-French (36M).
- Hardware: 8 P100 GPUs. Training time: ~12 hours for base, ~3.5 days for big.
- Optimizer: Adam with custom schedule (warmup + inverse-sqrt decay).
- Regularization: dropout (P = 0.1), label smoothing.

Consistent with everything in the course. This is our training loop.

## Section 6: results

- Base Transformer: 27.3 BLEU on WMT En-De. SOTA at the time.
- Big Transformer: 28.4 BLEU. Also SOTA.
- Ablations in Table 3: remove components, measure hit. (Study this table - it's how you show your ideas are the ones that matter.)

## What's different now

Things this paper proposed that are **no longer used**:

- **Sinusoidal positional encoding** → replaced by learned (GPT-2) and RoPE (Llama).
- **Post-norm**: LN after residual → replaced by pre-norm (LN inside residual).
- **Separate encoder + decoder**: most LLMs are decoder-only.
- **ReLU** activation → replaced by GELU, then SwiGLU.
- **LayerNorm** → replaced by RMSNorm.
- **Biases** on linear layers → dropped.
- **Dropout** at 0.1 → often 0.0 for large-scale pretraining.

Things this paper got **right** and are still used:

- Attention as the core operation.
- Multi-head attention.
- Residual connections.
- Stack many layers.
- Warmup + LR decay.
- AdamW-family optimizers (they used Adam; we now use AdamW).

## Why this paper matters

1. **Moment of realization**: attention alone is enough. Recurrence isn't needed.
2. **Scalability**: transformers parallelize well, so they scale with compute. RNNs don't.
3. **Generality**: the same architecture works for translation, text generation, vision, audio, code, etc. The transformer is the new default for sequence modeling.
4. **Framework-defining**: every LLM since is incrementally changing pieces of this recipe.

## Exercises

1. Read sections 3.1 through 3.4 carefully with `nanoGPT/model.py` open. Match each equation to lines of code.

2. Look at Figure 1 and identify: where are the 6 encoder layers? The 6 decoder layers? The "add & norm" blocks? The multi-head attention arrows?

3. Reproduce the scaled dot-product formula in Python with `T=4, d_k=8`, random Q, K, V. Check shapes.

4. Read Table 3 (ablations). Which change had the biggest negative impact? (Usually: removing positional encoding catastrophically hurts.)

5. Read the two paragraphs of section 4. Understand why attention is more parallelizable than recurrence.

## Next

`04_gpt_papers_gpt2_gpt3_gpt4.md` - what each OpenAI GPT paper added.
