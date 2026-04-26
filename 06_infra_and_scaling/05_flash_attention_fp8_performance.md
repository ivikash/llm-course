# 6.5 - Flash Attention, fp8, and Performance Kernels

Modern training speed comes from three things: better hardware, better parallelism (last lesson), and better **kernels** - hand-optimized GPU code for specific operations. This lesson demystifies the main ones.

## Flash Attention

### The problem

Vanilla attention:

```
scores = Q @ K.T         # (T, T), can be huge
scores = mask(scores)
weights = softmax(scores)
out = weights @ V
```

For T = 8192 in fp16: the scores matrix alone is 8192 × 8192 × 2 = 128 MB per head per batch item. A model with 32 heads, batch 4: 16 GB just for scores. OOM for anything long-context.

Also: reading/writing this huge tensor to HBM is slow. Attention becomes memory-bandwidth-bound.

### The insight

You don't actually need the full (T, T) score matrix. Compute it in **tiles** that fit in on-chip SRAM, stream through. Use the "online softmax" trick to compute softmax incrementally without seeing all values at once.

Result:
- No (T, T) materialized in HBM.
- Much less memory traffic.
- 2-4x faster for long contexts.
- Exact same math as vanilla attention.

### Versions

- **FlashAttention-1** (2022): the original insight.
- **FlashAttention-2** (2023): better parallelization across heads.
- **FlashAttention-3** (2024): fp8 + Hopper architecture features.

### In PyTorch

`F.scaled_dot_product_attention(q, k, v, is_causal=True)` - PyTorch picks the best available kernel (Flash, mem-efficient, or math). Free speedup, no code change.

nanoGPT uses this:
```python
y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,
    dropout_p=self.dropout if self.training else 0, is_causal=True)
```

nanochat has a custom `nanochat/flash_attention.py` that also handles the packed-sequence case (attention resets at document boundaries during SFT).

## fp8

### Why

H100 has dedicated fp8 tensor cores. Matmul in fp8 is ~2x faster than bf16.

### The challenge

fp8 has only 4 bits of mantissa (or 5, depending on format). Precision is very coarse. Direct fp8 matmul of typical values will produce junk.

### The fix: scaling

Before the matmul, compute the absolute max of the inputs. Scale them into fp8 range. Do the matmul. Scale the output back.

```python
# pseudocode
def fp8_matmul(a, b):
    a_scale = a.abs().amax()  # amax = absolute max
    b_scale = b.abs().amax()
    a_fp8 = (a / a_scale * 448).to(fp8)   # 448 = fp8 E4M3 max
    b_fp8 = (b / b_scale * 448).to(fp8)
    c_fp8 = a_fp8 @ b_fp8
    c = c_fp8 * (a_scale * b_scale / (448 * 448))
    return c
```

The devil is in per-tile scaling (to keep precision per block of the matrix), gradient handling, and choosing which ops are safe in fp8.

### In nanochat

`nanochat/fp8.py` implements a custom fp8 matmul for select `nn.Linear` layers. Only certain ones are safe - usually the big ones in attention and MLP. About 25% end-to-end speedup on H100 when enabled via `--fp8`.

Future: NVIDIA's Transformer Engine library and PyTorch's torchao will bring fp8 to a broader audience.

### int4 / int8 (inference-time)

For deployment, you can **quantize** weights to int8 or int4. Different technique - not about speed (although it helps), it's about fitting bigger models on smaller GPUs.

Tools: bitsandbytes (int8), GPTQ, AWQ, EXL2 (int4 inference), llama.cpp's GGUF format.

A 7B model in int4 fits in 4 GB VRAM, runs fast on a laptop GPU. This is why you can run Llama-2 on a MacBook.

## Other performance tricks

### torch.compile

PyTorch 2.0+ can JIT-compile your model for better kernel fusion:

```python
model = torch.compile(model)
```

Usually 10-30% speedup. Slower first iteration (compilation). Occasionally breaks with edge-case code.

nanoGPT uses it by default: `compile = True`.

### Kernel fusion

Multiple ops in one kernel call = less HBM traffic. Flash Attention is an extreme example. Other fusions:
- LayerNorm + linear
- Activation + linear
- Softmax + dropout

`torch.compile` finds some automatically. Hand-written kernels (Triton, CUDA) for the rest.

### Triton

OpenAI's Python-based GPU kernel language. Many modern kernels (Flash Attention, fp8) are written in Triton rather than raw CUDA.

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)
```

You don't need to write Triton for this course. But you should know it exists and that many modern LLM optimizations are written in it.

## Where to look for these in code

- PyTorch: `F.scaled_dot_product_attention` - Flash.
- nanoGPT: uses it.
- nanochat: `nanochat/flash_attention.py`, `nanochat/fp8.py`.
- HuggingFace Transformers: tools like `attn_implementation='flash_attention_2'`.
- vLLM: heavily uses custom Triton kernels for inference.

## Throughput metrics

- **TFLOPS**: sustained. An H100 does ~900 bf16 TFLOPS, ~3600 fp8 TFLOPS peak.
- **MFU** (Model FLOPs Utilization): achieved / peak. 40-55% is good, 60%+ is world-class.
- **Tokens/sec** during training.

nanochat logs all of these to wandb.

## Exercises

1. Compare speed of `F.scaled_dot_product_attention` vs hand-rolled attention:
   ```python
   import torch, torch.nn.functional as F
   q = torch.randn(1, 8, 2048, 64, device='cuda', dtype=torch.bfloat16)
   k = torch.randn_like(q); v = torch.randn_like(q)
   # SDPA (Flash)
   %timeit y = F.scaled_dot_product_attention(q, k, v, is_causal=True); torch.cuda.synchronize()
   # manual
   def manual(q, k, v):
       s = q @ k.transpose(-2,-1) / 8.0
       s = s + torch.triu(torch.full_like(s, float('-inf')), 1)
       a = F.softmax(s, -1); return a @ v
   %timeit y = manual(q, k, v); torch.cuda.synchronize()
   ```
   Expect 2-5x speedup.

2. Read `nanochat/fp8.py` top to bottom. Don't understand every line; understand: "scale, matmul, unscale".

3. Install `torch.compile` capability, try on the Shakespeare training. Note the first iteration is slow (compiling), subsequent are faster.

## Next

`06_the_cloud_lambda_aws_sagemaker.md` - where do these GPUs actually live?
