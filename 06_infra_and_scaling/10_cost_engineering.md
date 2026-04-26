# 6.10 - Cost Engineering

Every decision in LLM work eventually hits a dollar question. This lesson teaches you to think in $ per FLOP, per token, per user.

## Training cost

Rough formula:

```
training_cost = training_FLOPs / effective_FLOPs_per_sec / MFU * gpu_hour_rate
```

Example: Train a 1B-param model on 20B tokens (Chinchilla-optimal).
- FLOPs: 6 × 1B × 20B = 1.2e20.
- On 8xH100 @ peak ~7.2 Pflops bf16, at 45% MFU → 3.24 Pflops real.
- Time: 1.2e20 / 3.24e15 = ~37,000 sec = ~10 hours.
- Cost: 10 × $25/hr = **$250**.

Quick memorize: training FLOPs ≈ 6 × N × D. Good enough for planning.

## Inference cost

The harder cost. Depends on:
- Model size (bigger = slower per token).
- Context length (KV cache memory).
- Concurrent users (batching).
- Hardware.

Rough formula for a single-user, fp16 model:

```
tokens_per_second = (memory_bandwidth) / (model_size_bytes + kv_cache_per_token × context × 2)
```

Memory-bound (not compute-bound). For Llama-7B on H100 fp16:
- Model: 14 GB.
- Memory bandwidth: 3 TB/s.
- Tokens/sec: 3000/14 ≈ 200 tokens/sec at low context.

Cost per million tokens: runtime × $/hr / 3.6B (tokens per million-hour) ≈ cheap at small scale.

With **batching**: serve 64 users concurrently, amortize the model read. Tokens/sec goes up ~30x.

## Reading GPU cloud bills

Typical line items:
- **Compute** (instance time) - dominant.
- **Storage** (EBS, S3) - small, forgettable.
- **Egress** (data leaving the cloud) - can be a trap. Downloading a 50GB checkpoint: ~$5 on AWS.
- **Elastic IPs**, **load balancers**, **managed services**, **network** - can add up for production.

For personal learning: only compute matters. Watch the hourly rate × hours.

## $/token trained

Industry benchmark:
- GPT-4 training: estimated $60-100M. At ~2e25 FLOPs, that's $3-5 per 1e18 FLOPs.
- nanochat speedrun: $48 for 4e19 FLOPs ≈ $1.2 per 1e18 FLOPs.
- Frontier labs with custom hardware: $0.5 per 1e18 FLOPs.

When you optimize pretraining, this is the metric that matters.

## $/token served

Industry benchmark (2024-2026):
- GPT-4 API: $10-60 per 1M output tokens.
- Llama-3-70B on OpenRouter: $1-3 per 1M output tokens.
- Llama-3-8B: $0.05 per 1M output tokens.
- Self-hosted on your own H100: depends, can be $0.01 per 1M for an 8B model at good utilization.

For building products: if your unit economics require low $/token, you pick smaller models, quantize, batch, optimize.

## Cost saving patterns

### For training

- **Spot instances**: 50-70% cheaper, preemptible.
- **bf16 vs fp32**: 2x cheaper per step.
- **fp8**: another 25-50% on H100.
- **Don't overtrain**: if Chinchilla says 140B tokens is optimal, don't train for 500B unless you know why.
- **Cache intermediate state**: if you abort and restart, resume from checkpoint.
- **Batch multiple experiments**: one long run > many short runs (instance spinup costs).

### For serving

- **Quantize**: int8 → 2x, int4 → 4x smaller/cheaper.
- **Batching**: 10-50x throughput.
- **Speculative decoding**: 2-3x faster.
- **KV cache reuse**: for repeated prompts.
- **Cheaper GPUs**: A10 vs A100 for small models.
- **Open-weight models**: no per-token API fees.

### For development

- **Iterate on small models**: prototype at d12, only train d24+ when ready.
- **Early stopping signals**: watch val_bpb, kill bad runs early. One bad run killed at 20% saves 80%.
- **Use public checkpoints**: fine-tune Llama-2 instead of training from scratch.

## The scaling-law-driven decision framework

Before launching any run, answer:

1. **What am I measuring?** (val_bpb, a specific benchmark, qualitative)
2. **What's my compute budget?** (GPU-hours × rate = dollars)
3. **What's the Chinchilla-optimal model size for that budget?**
4. **Do I overtrain for inference, or train compute-optimally?**
5. **What's my sanity-check plan?** (kill criteria for a bad run)
6. **What signals tell me I'm on track?** (loss curve shape, val metrics)
7. **What's the smallest pilot I can run first?** (10% of budget to de-risk)

Seniors run this checklist habitually. Juniors skip it, launch big runs, waste money.

## Opportunity cost

Sometimes the right answer is **don't train**. Options:

- Use an open API (OpenAI, Anthropic) for $10 instead of training for $100K.
- Use a pretrained model (Llama) + LoRA on your data instead of full pretrain.
- Use prompt engineering to solve the problem with no training at all.

Training is expensive in wall time, money, and opportunity. Often the right move for learning, rarely the right move for business unless you have specific needs (domain knowledge, privacy, cost at scale, IP).

## Exercises

1. Compute training cost for:
   - A d24 nanochat model on 8xH100 for 3 hours. (~$75.)
   - A 30B model Chinchilla-trained on 8xH100. (1.8e22 FLOPs; ~250 hours; ~$6,000.)
   - A 1T model on a 1024-GPU H100 cluster. (Multi-millions.)

2. Compute inference cost:
   - A 7B model on a rented A100 serving 10 concurrent users. Tokens/sec? Cost per 1M tokens output?

3. Read nanochat's `dev/LEADERBOARD.md`. Every entry is a cost optimization. Identify the 3 biggest $ savings across the history.

4. Plan a $500 personal experiment budget. What's a meaningful research question you can answer? (E.g., "does Muon beat AdamW at d12?" would take maybe $30-50.)

## Next

`capstone_cloud_run.md` - Module 6 capstone: your first cloud training run.
