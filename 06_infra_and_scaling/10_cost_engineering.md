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

**Calculate throughput + latency for your deployment:**

```viz
{"viz": "throughput_calc"}
```

Pick GPU, model size, batch, precision, context. See: memory fit, decode ms/token, prefill latency, and whether you're memory-bound. The single most useful calculator when sizing production LLM serving.


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

**Prompt caching calculator — see the savings:**

```viz
{"viz": "prompt_caching"}
```

Slide prefix size up (big system prompts + RAG). Watch the cost delta between "no caching" and "with caching" grow. At 20K prefix tokens and 90% hit rate, expect ~80% cost savings and 10× latency drop on hit. This is the single biggest prod optimization in 2025.


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

## Visualize this

**Cost breakdown of training runs**:

```viz
{"viz": "cost_calculator"}
```

Slide model size and training tokens. Table shows real 2026 prices across hardware generations. Try:
- 7B model, 140B tokens → ~$2-5k on H100 8-GPU
- 70B model, 1.4T tokens (Chinchilla-optimal) → ~$500k
- GPT-3 scale (175B, 300B tokens) → ~$2-4M

The gap between V100 era and H100 era is enormous — same model, ~5× cheaper in 2026.

```
  nanochat speedrun (d24, 3.5h on 8xH100):
  ┌────────────────────────────────────┐
  │ Pretraining (2h)    ████████████████│ 75%
  │ SFT (20 min)        ████            │ 10%
  │ Evals (45 min)      ████████         │ 12%
  │ Infrastructure       █              │ 3%
  └────────────────────────────────────┘
  Total: ~$80 (on-demand) / ~$25 (spot)

  GPT-2 reproduction (OpenWebText, 4 days on 8xA100):
  Pretraining: ~$2,500 on-demand
  nanoGPT (baseline): 4 days → $1,500 spot

  GPT-3 (2020, estimated): ~$4-10M on-premise hardware
  GPT-4 (2023, rumored): ~$60-100M
  Llama-3-405B (2024): public estimate ~$100M+
```

**The $/FLOP race**:

```
  Training $/1e18 FLOPs (rough estimates):

  2019 (GPT-2 era, V100):           ~$200
  2020 (GPT-3 era, V100):            ~$50
  2022 (Chinchilla era, A100):        ~$20
  2024 (Llama-3 era, H100 + fp8):    ~$5
  2025+ (Blackwell):                   ~$2

  → Same model 100× cheaper to train than 5 years ago.
  → Why open-source LLMs exploded 2023-2026.
```

**Inference cost landscape (2026)**:

```
  Cost per 1M output tokens (API prices):

  GPT-4 (original)               $60
  GPT-4 Turbo                    $30
  Claude 3.5 Sonnet              $15
  GPT-4o                          $10
  Gemini 1.5 Pro                 $10
  GPT-4o mini                    $0.60
  Claude 3.5 Haiku                $5
  Llama-3.1-70B (via providers)  $0.90
  Llama-3.1-8B                   $0.06

  Self-hosted Llama-3.1-8B on A100 (batched):
    ~$0.02/M (if GPU is 80% utilized)
    ~$0.20/M (at low utilization)

  Conclusion: for high volume, self-host. For low volume, API.
```

**A simple decision tree**:

```
  Do I need per-query < 100 ms latency?
           │
     YES ──▶─── Do I have reliable 60%+ GPU utilization?
                      │
                 YES ──▶─── Self-host (cheapest)
                      │
                 NO  ──▶─── API + caching (simpler)
           │
     NO  ──▶─── API calls (simplest)

  Am I doing more than 1M queries/day?
           │
     YES ──▶─── Seriously consider self-host
           │
     NO  ──▶─── API is fine
```

**The "hidden costs" of cloud you forget**:

```
  1. Storage (EBS): $0.10/GB/month
     Your 100 GB dataset × 12 months = $120

  2. Egress: $0.09/GB typical
     Downloading your 20 GB trained model: $1.80
     Doing that 100 times: $180
     (CRAZY: inbound is usually free, outbound charges kill you)

  3. Elastic IPs: $0.005/hr if unattached
     Forget to release it: $3.60/month forever

  4. Reserved capacity / commitments: "save 30%" if commit 1 year
     If you don't actually need it: just a way to waste money

  5. CloudWatch, Logs, Monitoring:
     Not much, but surprising

  Rule: audit billing monthly. Delete what you don't need.
```

**Scaling laws applied to budget planning**:

```
  Question: "$1000 budget for an LLM experiment. What can I do?"

  At Lambda 8xH100 spot ($15/hr) = 67 hours of compute.
  At ~4e19 FLOPs/hour (fp8), that's ~2.7e21 FLOPs total.

  Chinchilla-optimal: N * D = 2.7e21 / 6 = 4.5e20 token-params.
  Sqrt: N ≈ 21B params, D ≈ 21 × 20 = 420B tokens.

  Reality: that's too many tokens to tokenize/load in 67 hours.
  Practical: d20-d24 nanochat (0.8B-1.5B params), 20-30B tokens.
  Speedrun-style: this is a GPT-2-grade model.

  Translation: $1000 = one GPT-2 reproduction.
```

**Watch your money**:

```bash
# Provider dashboard (check daily):
# - instances running?
# - hourly burn rate
# - monthly projection
#
# Set alerts:
# - $50 / week soft alert
# - $200 / week hard alert
# - $500 / month email
#
# If the number looks wrong, investigate NOW.
# Many a weekend vacation has been spent paying for forgotten GPUs.
```

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
