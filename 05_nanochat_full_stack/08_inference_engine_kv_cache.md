# 5.8 - Inference Engine and KV Cache

Training nanoGPT-style models is slow but steady: forward, backward, repeat. **Generation** has a different shape, and a naive implementation is painfully slow. This lesson covers the tricks that make it fast.

## The problem

To generate 100 tokens with a naive implementation:

```python
tokens = prompt
for _ in range(100):
    logits = model(tokens)            # process ALL tokens
    next_token = sample(logits[-1])    # only use the LAST position's logits
    tokens = tokens + [next_token]
```

Each generation step does a full forward pass over all tokens seen so far. If the prompt is 1000 tokens, step 1 processes 1000, step 2 processes 1001, ..., step 100 processes 1099. Total work: ~O(T^2) where T is total length. Wasteful.

## The fix: KV cache

Key insight: **when generating token `t+1`, only the new token needs Q, K, V computation; K and V from tokens 0..t can be reused.**

During normal forward:
```
q, k, v = compute_qkv(tokens[0:t+1])   # (t+1) × head_dim
```

With KV cache:
```
q_new, k_new, v_new = compute_qkv(tokens[t])   # 1 × head_dim for just the new token
k = concat(cached_k, k_new)                     # (t+1) × head_dim
v = concat(cached_v, v_new)                     # (t+1) × head_dim
cached_k = k                                     # update cache
cached_v = v
```

Attention now computes `q_new` against all cached K and V, producing one output token's worth of work.

Result: each generation step does O(T) work (attention against all past K/V) + O(1) work for the new token's computation. Total O(T^2) but with a much smaller constant. In practice: 10-100x speedup.

## Memory cost

The cache stores K and V for every layer, every head, every past token.

```
cache_size_per_token = 2 * n_layer * n_kv_head * head_dim * bytes_per_float
                     = 2 * 32 * 8 * 128 * 2   (for Llama-7B, bf16)
                     ≈ 131 KB per token per conversation
```

For a 1000-token conversation: ~130 MB per conversation. For 100 concurrent conversations: 13 GB. This is why serving LLMs is memory-bound on KV cache.

## Where it is in nanochat

Open `~/workspace/nanochat/nanochat/engine.py`. This is the inference-time code path, separate from training.

Key class: `Engine`. It:

1. Loads a trained model.
2. Allocates KV cache tensors per layer.
3. Provides methods to "prefill" a prompt (one forward pass over the whole prompt to fill the cache) and then "decode" one token at a time.

```python
class Engine:
    def __init__(self, model, max_seq_len, batch_size):
        # allocate KV cache
        self.kv_cache = [
            (torch.zeros(batch_size, max_seq_len, n_kv_head, head_dim),
             torch.zeros(batch_size, max_seq_len, n_kv_head, head_dim))
            for _ in range(n_layer)
        ]
        self.cache_pos = 0
    
    def prefill(self, tokens):
        # process all prompt tokens at once, fill cache
        ...
    
    def decode_step(self, token):
        # process one new token, extend cache, return next logits
        ...
```

The model's `forward` method has an extra `kv_cache` argument that switches behavior between training (no cache) and inference (uses cache).

Inside `nanochat/gpt.py`'s `CausalSelfAttention`:

```python
def forward(self, x, rope_cache, kv_cache=None):
    q = ...
    k = ...
    v = ...
    if kv_cache is not None:
        cached_k, cached_v, pos = kv_cache
        cached_k[:, pos:pos+T] = k
        cached_v[:, pos:pos+T] = v
        k = cached_k[:, :pos+T]
        v = cached_v[:, :pos+T]
    # then attention as usual
```

## Prefill vs decode

Two phases of generation:

**Prefill**: process the entire prompt. Parallel across positions. Compute-bound (big matmuls). Usually takes ~100ms for a 1000-token prompt.

**Decode**: generate tokens one at a time. Memory-bound (just reading cached K, V, doing small matmuls). Usually ~20-50ms per token on a good GPU.

A typical chat:
- Prompt: 500 tokens. Prefill: ~50ms.
- Response: 200 tokens. Decode: 200 × 30ms = 6 seconds.

Response time is dominated by decode. Fast decode is what makes chatbots feel responsive.

## Batched generation

Serving multiple conversations simultaneously: each gets its own row in the cache, attention happens in parallel across batch. This is **continuous batching** - pick up and drop off requests as they arrive/complete.

Libraries like vLLM, TGI, Triton Inference Server, SGLang, are essentially "batched KV-cache management" with clever scheduling.

nanochat's engine is simpler - batched but no dynamic batching. Fine for single-user chat.

## Speculative decoding (bonus concept)

An even fancier trick: use a small draft model to propose several tokens at once, then the big model verifies them in parallel. Often 2-4x speedup.

Not in nanochat. In production serving (vLLM, tensor-rt) as an option.

## Flash Attention at inference

Lesson 3.4 mentioned Flash Attention for training. At inference it matters too - the Q × K^T attention can still blow up for long contexts. PyTorch's `F.scaled_dot_product_attention` handles this automatically.

For very long contexts (100K+), **Paged KV cache** + Flash Decoding become necessary. vLLM's PagedAttention is the reference implementation.

## Visualize this

**KV cache: why generation is O(T) per token, not O(T²)**:

```
  Without KV cache (naive generation):
  Step 1: input = 1 token, model processes 1 position. Compute: 1 unit.
  Step 2: input = 2 tokens, model processes 2 positions. Compute: 4 units.
  Step 3: input = 3 tokens, model processes 3 positions. Compute: 9 units.
  ...
  Step T: input = T tokens.                             Compute: T² units.

  Total work for T tokens:  O(T³)   ← cubic! generation gets slower per token

  With KV cache:
  Step 1: store K,V for position 0. Compute: 1 unit.
  Step 2: use cached K,V for positions 0..0, compute new K,V for position 1.
          Attention is now (1 query) × (2 key-value pairs).   Compute: 2 units.
  Step 3: Attention is (1 query) × (3 key-value pairs).        Compute: 3 units.
  ...
  Step T: Attention is (1 query) × (T key-value pairs).        Compute: T units.

  Total work for T tokens:  O(T²)   ← still bad but much better per token
  Per-token work:           O(T)    ← instead of O(T²)
```

**KV cache memory layout**:

```
  For each layer, store:
    K: shape (batch, max_seq_len, n_kv_head, head_dim)
    V: shape (batch, max_seq_len, n_kv_head, head_dim)

  Example for Llama-2 7B, batch 1, max_seq_len 4096:
    n_layer = 32
    n_kv_head = 32  (MHA, no GQA)
    head_dim = 128

    Per layer:  2 (K+V) × 1 × 4096 × 32 × 128 × 2 bytes (bf16) = 67 MB
    32 layers:  2.1 GB just for KV cache (one session)

  For Llama-2 7B with GQA (n_kv_head=8):
    Per layer:  2 × 1 × 4096 × 8 × 128 × 2 bytes = 17 MB
    32 layers:  0.5 GB   ← 4× smaller! That's why GQA wins for serving.
```

**Prefill vs decode phases**:

```
  Phase 1: Prefill (process prompt, fill cache)
  ┌──────────────────────────────────────────┐
  │  prompt: "Write a poem about the moon."  │
  │         500 tokens                        │
  │                                           │
  │  model processes all 500 positions         │
  │  in parallel (big matmul, GPU loves this) │
  │                                           │
  │  KV cache fills with K,V for all 500 pos  │
  │                                           │
  │  Time: ~100 ms                            │
  │  Compute-bound (utilizes GPU well)         │
  └──────────────────────────────────────────┘

  Phase 2: Decode (generate tokens one at a time)
  ┌──────────────────────────────────────────┐
  │  token 501: use cached 500, compute 1 new │
  │    ~30 ms                                 │
  │  token 502: use cached 501, compute 1 new │
  │    ~30 ms                                 │
  │  token 503: use cached 502, compute 1 new │
  │    ~30 ms                                 │
  │    ...                                    │
  │                                           │
  │  Memory-bound (reading cache dominates)    │
  │  GPU utilization ~5-10%                   │
  └──────────────────────────────────────────┘

  For a 200-token response: ~100 ms prefill + 6 seconds decode.
  Decode is the bottleneck!
```

**That's why batching helps serving so much**:

```
  Single user:           1 user × 1 token every 30 ms = 33 tok/sec
  Batch 32 users:        32 users × 1 token every 40 ms = 800 tok/sec
                          (shared model read amortized across batch)
```

**nanochat's engine.py flow**:

```
  engine = Engine(model, max_seq_len=2048, batch_size=1)

  engine.prefill(prompt_tokens)    ← fills cache
  for _ in range(max_new_tokens):
      logits = engine.last_logits()
      next_token = sample(logits, temperature)
      if next_token == EOT: break
      engine.decode_step(next_token)  ← one token, uses cache
      yield next_token                ← stream to UI
```

~200 lines of real code. You understand every piece now.

## Exercises

1. Read `~/workspace/nanochat/nanochat/engine.py` top to bottom. Identify: prefill, decode_step, the cache allocation.

2. In `nanochat/gpt.py`, find the attention forward method. Identify the KV cache branch. Understand what's different between training forward and inference forward.

3. Benchmark yourself: if a model generates 50 tokens/sec with KV cache, how many tokens/sec would it generate without? (Hint: with 1000-token context, naive is ~1000x slower. With 100-token context, ~100x. Dramatic either way.)

4. Read `test_engine.py` in `~/workspace/nanochat/tests/`. The tests exercise the engine. Note how they verify correctness (numerical equivalence to non-cached forward).

5. Run `python -m scripts.chat_cli -p "Tell me a joke."` after training. Note the response time.

## Next

`09_serving_chat_web.md` - wrapping the engine in a web UI.
