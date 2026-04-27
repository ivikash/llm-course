# 4.3 - Mixed Precision: bf16, fp16, fp8

Default PyTorch uses 32-bit floats (fp32). Neural networks don't need that much precision. **Mixed precision** uses 16-bit (or 8-bit) floats for most math, giving 2-4x speedups with minimal quality loss.

## Float formats

| Format | Bits | Range | Precision | Used for |
|--------|------|-------|-----------|----------|
| fp32 | 32 | huge | ~7 decimals | Classic default. Too big for LLMs now. |
| fp16 | 16 | smaller (±65K) | ~3 decimals | Fast. Can underflow/overflow. |
| bf16 | 16 | big (same as fp32) | ~2 decimals | Fast. Big range. **Default for LLMs today.** |
| fp8 (E4M3, E5M2) | 8 | small or big | ~1 decimal | Faster still. H100+ only. Experimental. |
| int8 / int4 | 8/4 | integers | — | Quantization for inference. |

**bf16** (brain-float) was invented by Google for TPUs and is now the LLM standard. It has fp32's **exponent range** but only 7 bits of mantissa, so precision is poor - but in neural nets, range matters more than precision.

## Why speed up?

- Memory: a 16-bit float is half the size of 32-bit. Models that fit twice as big. Batches twice as big.
- Compute: NVIDIA's Tensor Cores do 16-bit matmul ~2-4x faster than fp32, and fp8 ~8x faster on H100+.
- Bandwidth: less data to move between HBM and compute cores.

Net result: bf16 gives ~2x end-to-end training speedup for "free" on A100+ GPUs.

## The catch with fp16: underflow

fp16's range is narrow. Gradients can be very small (1e-10 range). In fp16 they round to 0, training stalls.

Solution: **loss scaling**. Multiply the loss by a big factor (e.g. 2^15) before backward. Gradients get scaled up into the representable range. Divide them back down before the optimizer step.

PyTorch's `GradScaler` automates this.

```python
from torch.cuda.amp import GradScaler

scaler = GradScaler()
with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    logits = model(x)
    loss = F.cross_entropy(logits, y)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

bf16 doesn't need the scaler - its range is already big enough. Just:

```python
with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    logits = model(x)
    loss = F.cross_entropy(logits, y)
loss.backward()
optimizer.step()
```

Simpler. This is why bf16 won: same speed, less footgun.

## In nanoGPT

`train.py`:

```python
dtype = 'bfloat16'   # or 'float16' or 'float32'
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# later, in the training step:
with ctx:
    logits, loss = model(X, Y)
```

The `autocast` context manager wraps any forward pass - ops inside it automatically run in bf16 (or fp16) when safe.

## Master weights

Even with bf16 forward/backward, you usually keep the **master copy** of weights in fp32. AdamW's momentum/variance tensors stay in fp32 too. Why? Weight updates can be tiny relative to weight magnitude - in bf16, `w += small` can round away entirely.

Memory cost per parameter during training (bf16 mixed precision, AdamW):
- fp32 master weights: 4 bytes
- bf16 working weights: 2 bytes
- bf16 gradients: 2 bytes
- fp32 AdamW m (momentum): 4 bytes
- fp32 AdamW v (variance): 4 bytes
- **Total: 16 bytes per parameter**

For a 1B-parameter model: 16GB just for weights+optimizer state. Plus activations. Plus gradients between layers. This is why "1B model needs 24GB VRAM to train" is the rule of thumb.

## fp8: nanochat's trick

H100 and newer have native fp8 tensor cores. 2x faster than bf16. But fp8 has only 4 bits of mantissa - precision is very coarse. Only specific operations (mostly the big matmuls) work well in fp8.

See `~/workspace/nanochat/nanochat/fp8.py`. It's custom code that:
1. Detects when a matmul is "safe" for fp8.
2. Scales inputs before the matmul.
3. Runs the matmul in fp8.
4. Unscales outputs.

About 25% training speedup on H100 over bf16 for nanochat.

fp8 is cutting-edge - expect libraries (Transformer Engine, TorchAO) to take it over in the next year or two. You don't need to write fp8 kernels yourself.

## Inference: int8 / int4 quantization

**Watch quantization degrade a weight matrix across precisions:**

```viz
{"viz": "quantization"}
```

Switch from FP32 down to 1-bit. Left is original, middle is quantized, right is the error. Below ~4 bits the error explodes. For a 70B model: fp16 = 140 GB, int4 = 35 GB (fits single H100), 1-bit = 9 GB (fits RTX 4090).

Post-training, you can quantize weights to int8 or int4 for deployment. Different technique:
- Model runs in fp16 or bf16 activations.
- Weights live as int4, dequantized on-the-fly before each matmul.
- Smaller model, slightly lower quality.
- Tools: bitsandbytes, GPTQ, AWQ, llama.cpp's GGUF.

Relevant for Module 5's serving section.

## Visualize this

**Float formats compared visually:**

```viz
{"viz": "float_formats"}
```

Hover each format. See the sign/exponent/mantissa bit split. Notice bf16 has fp32's exponent width (same range) with only 7 mantissa bits (lower precision) — that's why it's the modern default.

**Float format bits**:

```
  fp32 (32 bits, "single precision"):
  ┌─┬──────────┬───────────────────────────┐
  │S│ exponent │     mantissa (23 bits)     │
  └─┴──────────┴───────────────────────────┘
   1   8 bits          ~7 decimal digits
   sign              ~10^±38 range

  fp16 (16 bits, "half precision"):
  ┌─┬─────┬──────────┐
  │S│ exp │ mantissa │
  └─┴─────┴──────────┘
   1  5       10          ~3 decimals, ~±65000 range
                          (can overflow/underflow in gradients!)

  bf16 (16 bits, "brain float"):
  ┌─┬──────────┬──────────┐
  │S│ exponent │ mantissa │
  └─┴──────────┴──────────┘
   1  8 bits      7 bits     ~2 decimals, ~10^±38 range (same as fp32)

  fp8 (8 bits):
  ┌─┬─────┬────┐
  │S│ exp │mant│
  └─┴─────┴────┘
   1  4-5   2-3   very coarse, needs careful scaling
```

**bf16's key insight: big range, small precision**

```
  fp16:   ±65504 max.       Gradients ~1e-8 underflow to 0 → training dies.
  bf16:   ±3.4e38 max.       Gradients ~1e-8 fine.         → training works.
```

That's why bf16 won. Same memory as fp16, fp32-equivalent range. No loss scaler needed.

**Memory savings per parameter (during training)**:

```
                      fp32            bf16 mixed
  weights (master)    4 bytes         4 bytes
  weights (working)   (same)          2 bytes
  gradients           4 bytes         2 bytes
  AdamW m             4 bytes         4 bytes
  AdamW v             4 bytes         4 bytes
  ─────────────────────────────────────────────
  total per param     16 bytes        16 bytes (same!)

  BUT activations use 2 bytes instead of 4 → halves the big variable cost.
  Net: ~40% memory savings in practice.
```

**Speedup landscape on NVIDIA GPUs**:

```
  A100 TFLOPS (peak):
    fp32:  19.5
    tf32:  156   (8× faster, default matmul on A100+)
    bf16:  312   (16× faster than fp32)
    int8:  624

  H100 TFLOPS (peak):
    fp32:  67
    bf16:  989
    fp8:   1979  (2× faster than bf16, 30× faster than fp32)
    fp8 sparse: 3958
```

Modern LLMs: **bf16 is the default, fp8 where safe, fp32 only for the AdamW master weights**.

**Autocast in action**:

```
  with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
      logits = model(x)           # model weights still fp32 in storage
                                  # but matmuls cast inputs to bf16
                                  # → bf16 output activations
      loss = F.cross_entropy(logits, y)

  loss.backward()                 # bf16 activations, fp32 gradients
  optimizer.step()                # fp32 optimizer updates
```

Safe zones (bf16): matmuls, convs.
Unsafe zones (stay fp32): reductions (sums, norms), loss computation, softmax.
PyTorch's autocast handles it automatically.

## Exercises

1. Run a small model in fp32 and bf16. Measure wall-clock time. You should see ~1.5-2x speedup on A100+.
   ```python
   # fp32
   with torch.no_grad():
       for _ in range(100):
           logits = model(x)
   # bf16
   with torch.amp.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
       for _ in range(100):
           logits = model(x)
   ```

2. Try fp16 without a GradScaler. Does the loss go to NaN after a few steps? (Often yes, depending on the model.)

3. Read `nanoGPT/train.py`'s autocast setup. Understand what happens on CPU (no autocast, falls back to fp32).

4. Find `nanochat/nanochat/fp8.py`. You don't need to understand every line; just note the scaling logic.

## Next

`04_gradient_accumulation.md` - how to simulate big batches on small GPUs.
