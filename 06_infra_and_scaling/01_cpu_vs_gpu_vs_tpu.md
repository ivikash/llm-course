# 6.1 - CPU vs GPU vs TPU

## Why any of this matters

A modern LLM does matrix multiplications. Lots of them. Choice of hardware determines how fast those matmuls happen and therefore how much model you can afford to train.

## CPU

- ~8-64 "cores" in a modern consumer/server CPU.
- Each core is fast, runs arbitrary logic, has big caches.
- Terrible at parallel numeric work: only ~10-100 multiplies in flight at once.
- Matmul throughput on a good CPU: ~1 TFLOPS (trillion floating-point operations/second).

Use CPUs for: preprocessing data, orchestration, small models, debugging.

## GPU (NVIDIA)

- Thousands of small cores (H100: 16,896 CUDA cores + 528 Tensor Cores).
- Each core is simpler, ~1-2 GHz.
- Designed for "same operation, huge data": exactly matmul.
- Matmul throughput on an H100: ~1000 TFLOPS fp16, ~4000 TFLOPS fp8.

A single H100 is **~1000x faster than a CPU** at matmul.

### The NVIDIA lineage (what names mean)

| Generation | Year | Flagship | Highlights |
|------------|------|----------|------------|
| Pascal | 2016 | P100 | First good DL GPU |
| Volta | 2017 | V100 | First Tensor Cores |
| Ampere | 2020 | A100 (40/80GB) | Standard workhorse 2020-23 |
| Hopper | 2022 | H100 (80GB), H200 (141GB) | fp8, massive memory bandwidth |
| Blackwell | 2024 | B100, B200 | Next-gen, fp4 support |

Also relevant: **consumer cards** (RTX 4090, 5090) work for hobby LLM training but lack NVLink (fast GPU-to-GPU) and have less memory.

### Anatomy of a modern GPU

- **Streaming Multiprocessors (SMs)**: the big compute blocks. H100 has 132 SMs.
- **Tensor Cores**: specialized matmul units inside each SM. Where most FLOPs come from.
- **HBM (High-Bandwidth Memory)**: the 80GB of VRAM. ~3 TB/s bandwidth.
- **SRAM (on-chip)**: tiny, fast. Flash Attention uses it.

For LLM training, **memory bandwidth** usually matters more than raw FLOPS. A100 and H100 are both compute-rich; the difference in real training speed comes from HBM bandwidth and interconnect.

### NVLink

GPUs talk to each other via NVLink (fast) or PCIe (slow). On an 8xH100 node, all 8 GPUs have ~900 GB/s NVLink between them. Essential for DDP/FSDP - gradients sync at this bandwidth.

## TPU (Google)

- Google's custom chips for neural networks.
- TPU v4/v5: roughly H100-equivalent. TPU v6e/v5p: next-gen.
- You rent via Google Cloud only (TPUs aren't sold to end users).
- Use JAX (preferred) or PyTorch-XLA.

Why/when use TPUs: you're on GCP, you need massive scale (thousands of chips), or you're doing research that integrates well with JAX.

Why not: PyTorch ecosystem is NVIDIA-first. Most open-source code assumes CUDA.

## Alternatives

- **AWS Trainium / Inferentia** - Amazon's custom chips. Reasonable for serving, less ideal for training.
- **AMD MI300X** - AMD's H100 competitor. Good hardware, weaker software (ROCm vs CUDA).
- **Cerebras WSE** - gigantic single-chip (wafer-scale) accelerator. Good for specific workloads.
- **Groq LPU** - specialized for low-latency inference only.
- **Tenstorrent** - emerging competitor.

Default recommendation for this course and for ~99% of jobs: **NVIDIA GPUs**.

## What "speed" actually means

Three numbers to track:

1. **Tokens/second during training**: how many tokens you can backprop through per second. Directly determines training time.
2. **MFU (Model FLOPs Utilization)**: actual FLOPs / peak FLOPs. Good training: 40-55%. Bad: under 20%.
3. **Tokens/second during inference**: how fast generation happens. Memory-bound, different regime.

Papers and blog posts usually report #1 and #2 for training.

## Picking hardware for this course

- **Modules 0-3**: your laptop.
- **Module 4 capstone**: any GPU (RTX 3060+ works). On a 4090, Shakespeare trains in 2 minutes.
- **Module 5 capstone**: cloud 8xH100 for the speedrun, or run tiny locally for months. Module 6 Lesson 6 covers cloud.

## Exercises

1. If you have a GPU, run `nvidia-smi`. Identify: GPU name, VRAM total/used, CUDA version.

2. Benchmark a matmul on CPU and GPU:
   ```python
   import torch, time
   a = torch.randn(4096, 4096)
   b = torch.randn(4096, 4096)
   t0 = time.time(); c = a @ b; print(f"CPU: {time.time()-t0:.3f}s")
   a, b = a.cuda(), b.cuda()
   torch.cuda.synchronize(); t0 = time.time()
   c = a @ b; torch.cuda.synchronize()
   print(f"GPU: {time.time()-t0:.3f}s")
   ```
   Speedup? (50-500x depending on your hardware.)

3. Look up your GPU's specs: peak fp16 TFLOPS, VRAM, memory bandwidth. Write them in a note - you'll reference them later.

## Next

`02_gpu_memory_and_vram.md` - why "out of memory" keeps happening.
