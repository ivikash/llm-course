# Module 6 - Infrastructure and Scaling

What separates a hobbyist from a professional LLM engineer? The ability to train/serve at scale. This module covers GPUs, distributed training, the cloud, and the ecosystem around models.

## Lessons

### `01_cpu_vs_gpu_vs_tpu.md`
- Why neural nets happen on GPUs: SIMD parallelism.
- Anatomy of a GPU: streaming multiprocessors, tensor cores, memory hierarchy (HBM vs SRAM).
- Specific lineage: Volta (V100) → Ampere (A100) → Hopper (H100) → Blackwell (B200).
- TPUs: Google's alternative. Pros/cons.
- Alternatives in the wings: Cerebras, Groq, Tenstorrent, AWS Trainium.

### `02_gpu_memory_and_vram.md`
- The math of memory: parameters + activations + gradients + optimizer state.
- Example: 1B parameter model in fp32 takes 4GB just for params. AdamW doubles that with (m, v) states. Add activations scaling with batch × seq_len... you see how "small" models explode.
- How to read `nvidia-smi`.
- Why OOMs happen. How to recover: reduce batch, gradient checkpointing, smaller model, fp8.

### `03_distributed_training_ddp.md`
- `torchrun` launches N processes, one per GPU, each with a copy of the model.
- Forward/backward run locally; gradients are all-reduced across GPUs each step.
- NCCL (NVIDIA's collective library) does the all-reduce.
- nanoGPT's DDP code (about 20 lines in train.py).
- `OMP_NUM_THREADS=1` and other gotchas.
- Multi-node: `torchrun --nnodes=2 --node_rank=0 ...`, plus the "infiniband vs TCP" caveat.

### `04_fsdp_zero_tensor_parallelism.md`
- Beyond a single GPU fit: FSDP shards parameters across GPUs.
- ZeRO stages 1, 2, 3: what each shards (optimizer state, gradients, parameters).
- Tensor parallelism (split matmul across GPUs) vs pipeline parallelism (split layers across GPUs).
- When to use which: rule of thumb by model size.
- 3D parallelism: Megatron-LM, DeepSpeed. Not in nanoGPT/nanochat but you should know it exists.

### `05_flash_attention_fp8_performance.md`
- What Flash Attention does under the hood: IO-aware tiling, online softmax, no T×T materialization.
- FlashAttention-1, -2, -3 over time.
- fp8 math: why H100+ have dedicated fp8 tensor cores.
- nanochat's `fp8.py`: custom Triton/CUDA work. You don't need to write this yourself but it's worth reading to see what low-level performance engineering looks like.
- Throughput metrics: tokens/sec, MFU (Model FLOPs Utilization) - the industry standard.

### `06_the_cloud_lambda_aws_sagemaker.md`
- Hobbyist providers: Lambda, Runpod, Vast.ai. By-the-hour GPU. ~$1-3/hour for a single card.
- Enterprise providers: AWS (EC2 p4/p5), GCP (A3/A4), Azure.
- SageMaker specifically: an AWS umbrella of ML services - notebooks, training jobs, endpoints, pipelines. Useful if you're already deep in AWS. Overkill for a personal project. We walk through a "hello world" SageMaker training job.
- Reserved vs on-demand vs spot: cost profiles.
- The hidden costs: storage, egress, networking.

### `07_experiment_tracking_wandb.md`
- `wandb login`, `wandb.init`, `wandb.log`.
- Runs, groups, projects.
- Dashboards, charts, alerts.
- How nanoGPT and nanochat use it.
- Alternatives: tensorboard, Neptune, ClearML, mlflow.

### `08_huggingface_ecosystem.md`
- The HF Hub: models, datasets, spaces. Github-for-ML.
- `transformers`: `AutoModel.from_pretrained("meta-llama/Llama-2-7b")` - three lines to load any model.
- `datasets`: streaming huge datasets without downloading.
- `accelerate`: thin wrapper around DDP/FSDP.
- `peft`: LoRA and friends for cheap fine-tuning.
- `trl`: RL (PPO, DPO, GRPO) implementations for language models.
- How to publish your own models and datasets.

### `09_scaling_laws.md`
- Kaplan et al 2020 ("Scaling Laws for Neural LMs"): loss scales as a power law in parameters, data, compute. Model size should grow faster than data.
- Chinchilla 2022 (Hoffmann et al): Kaplan was wrong about the balance; compute-optimal is roughly 20 tokens per parameter.
- What it means: if you have 1e24 FLOPs to spend, how big a model, and how many tokens?
- The Chinchilla-optimal rule of thumb (and the "overtrain for inference" argument that says: train smaller/longer for better deployment economics).
- nanochat's compute-optimal scaling - see `runs/scaling_laws.sh` and `dev/estimate_gpt3_core.ipynb`.

### `10_cost_engineering.md`
- Reading a GPU cloud bill.
- $/FLOP, $/token trained, $/token served.
- Why reducing `time-to-GPT-2` on the nanochat leaderboard is valuable - it's all cost.
- Inference economics: KV-cache memory, batched serving, speculative decoding.
- When training is the cost, when serving is.
- An illustrative budget for a research project.

### `capstone_cloud_run.md`
- Rent an 8xA100 or 8xH100 for 2 hours (~$40-60).
- Run `bash runs/speedrun.sh` on it (the nanochat full pipeline).
- Watch wandb dashboards in real time.
- Save the final checkpoint to your machine.
- Compute the exact $ cost of what you just did.
- *Kill the machine* immediately after. This is the most-forgotten step.

## Why this matters for you

You can learn a lot on a laptop. But the LLM field lives on GPUs, and serious roles require comfort with distributed training, cloud infra, and cost reasoning. This module fills that gap. No framework or certification necessary - just this practical tour.
