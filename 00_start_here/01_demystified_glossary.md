# 01 - The Demystified Glossary

Read this once, now. Don't try to memorize. The goal is to strip fear from the
words. You'll meet each of these many times through the course - by the fifth
encounter they'll feel natural.

I've tried to give the *shortest honest definition* for each. Some of these I
expand over entire lessons later. For now, the one-liner is enough.

---

## Core concepts

**AI (Artificial Intelligence)** - The umbrella marketing term. In practice today it means "software that does things people used to think only humans could do, using lots of data and statistics."

**ML (Machine Learning)** - A subset of AI. Instead of writing rules by hand, you show the computer examples and it figures out the rules. Spam filter example: you don't write `if "Viagra" in email: spam`; you give it 100,000 emails labeled spam/not-spam and it learns which patterns matter.

**Deep Learning** - A subset of ML that uses **neural networks** with many layers. "Deep" just means "many layers stacked." That's the whole thing. The name is more dramatic than the idea.

**Neural Network** - A math function with millions (or billions) of adjustable numbers ("weights" or "parameters") that you fit to data. Loosely inspired by brains, but it's really just nested matrix multiplication with some non-linear functions between the layers. You'll write one yourself in Module 2.

**Model** - A specific neural network with specific values for its weights. "Training a model" = finding good values for those weights. "Running a model" = doing math with those weights to produce an output.

**Weights / Parameters** - The adjustable numbers inside a model. GPT-2 small has 124 million of them. GPT-4 allegedly has ~1 trillion. They're just floating-point numbers stored in matrices.

**Training** - The process of adjusting the weights so the model's outputs get better at matching the training data.

**Inference** - Running the model after training to produce an output for new input. Also called "generation" or "sampling" for language models.

**Fine-tuning** - Taking a model someone else trained, and training it a bit more on your specific data. Much cheaper than training from scratch.

**LLM (Large Language Model)** - A neural network that produces text, trained on an enormous amount of text. GPT, Claude, Llama, Gemini - all LLMs. "Large" just means it has billions of weights.

**Transformer** - A specific neural network architecture (recipe) invented in 2017. All modern LLMs are transformers. We dedicate Module 3 to this.

**GPT (Generative Pre-trained Transformer)** - A flavor of transformer used for text generation. "Generative" = makes new text, "Pre-trained" = was trained on lots of data before being specialized, "Transformer" = the architecture. nanoGPT trains exactly this kind of model.

**Token** - A small chunk of text. Not always a word. "Hello world" might become 2 tokens (`Hello`, ` world`) or 3, depending on the tokenizer. LLMs don't see letters or words; they see token IDs (just integers).

**Tokenizer** - The component that converts text -> list of integers, and back. Lesson 3.1 covers this.

**Embedding** - A vector (list of numbers) that represents a token. The tokenizer turns "cat" into the integer 14823, and the embedding turns 14823 into, say, a 768-number vector. These vectors are what the model actually does math on.

**Attention** - The key operation inside a transformer. It lets each token "look at" other tokens in the sequence and decide what's relevant. We spend several lessons on this in Module 3. For now: think of it as each word in a sentence getting to ask questions of all the other words.

**Context / Context Length / Block Size** - How many tokens the model can consider at once. GPT-2 had 1024. GPT-4 Turbo had 128,000. Longer context = more memory/compute.

**Vocab / Vocabulary** - The set of all possible tokens the tokenizer knows. A number like 50257 means there are 50257 different token IDs possible.

**Logits** - The model's raw output before turning into probabilities. Just numbers, one per vocab entry.

**Softmax** - The math operation that turns logits into probabilities that sum to 1. Covered in Lesson 1.4.

**Sampling** - Picking the next token based on probabilities the model outputs. Not always the highest-probability one - you often add randomness ("temperature") to make outputs more interesting.

**Temperature** - A knob that controls randomness when sampling. 0 = always pick the most likely token (boring, deterministic). 1 = use the model's distribution as-is. 2 = more random. Usually 0.6-0.9 for chat.

**Loss** - A number measuring how wrong the model currently is. Training = making this number go down.

**Gradient** - For each weight, a number saying "if we nudge this weight up, does the loss go up or down, and how much?". Computed automatically by PyTorch. Covered in Module 2.

**Backpropagation / Backprop** - The algorithm that computes all the gradients efficiently, by walking backwards through the network. PyTorch does this for you with one line (`loss.backward()`).

**Optimizer** - The algorithm that uses the gradients to update the weights. Simple version: Stochastic Gradient Descent (SGD). Better versions: Adam, AdamW. Lesson 2.5.

**Epoch** - One full pass through the training dataset. In LLM training we often don't even do one full epoch because the data is so huge. Instead we count **steps** (number of gradient updates) or tokens (total tokens seen).

**Batch / Batch Size** - How many examples we process in parallel in one step. Larger batch = more stable gradients, more memory used.

**Learning Rate** - How big each weight update is. Too big: training blows up. Too small: training crawls. Scheduling this is an art covered in Lesson 4.5.

---

## Training pipeline vocabulary

**Pretraining** - Training a model from scratch on a huge, generic corpus (e.g. all of Common Crawl). The expensive part. Produces a "base model" that can complete text but doesn't follow instructions well.

**Midtraining** - A newer term for a short training phase between pretraining and SFT, often to adjust data mix, expose the model to chat templates, or teach specific formats. nanochat does this.

**SFT (Supervised Fine-Tuning)** - Fine-tuning on curated (prompt, good-response) pairs to teach the model to behave like a chatbot/assistant. After SFT, "base model" becomes "chat model."

**RLHF (Reinforcement Learning from Human Feedback)** - Using human preferences (A vs B comparisons) to further improve the model. The technique behind ChatGPT's quality. Covered in Module 5.

**DPO (Direct Preference Optimization)** - A simpler, popular alternative to RLHF that skips training a separate reward model.

**GRPO** - A newer RL algorithm used by DeepSeek and nanochat. Works well for verifiable tasks (math, code) where you can check if the answer is right.

**Reward Model** - A separate model trained to score responses. Used in classical RLHF.

---

## Infrastructure vocabulary

**CPU (Central Processing Unit)** - The main chip in your computer. Great at sequential logic. Not great at the massively parallel math LLMs need.

**GPU (Graphics Processing Unit)** - A chip originally for 3D graphics that turned out to be perfect for neural networks, because both workloads are mostly matrix multiplication. Modern LLM training happens on GPUs almost exclusively.

**NVIDIA** - The company that dominates the GPU market. "H100", "A100", "B200", "4090" are NVIDIA GPUs in roughly increasing order of speed/cost.

**CUDA** - NVIDIA's software stack for programming their GPUs. You usually don't touch it directly; PyTorch does.

**TPU (Tensor Processing Unit)** - Google's custom chips for neural networks. You'd use these mostly on Google Cloud.

**VRAM** - Memory *on the GPU*. Separate from your main RAM. H100 has 80GB of VRAM. A big model or big batch might not fit; that's when you see "CUDA out of memory".

**FLOP / FLOPs / FLOPS** - "Floating-Point Operations". A measure of compute. `FLOPs` (lowercase s) usually = total operations. `FLOPS` (uppercase) = per second. Training GPT-3 took ~3e23 FLOPs. An H100 does ~1e15 FLOPS. So GPT-3 training was ~3e8 GPU-seconds.

**Mixed Precision / bf16 / fp16** - Using 16-bit instead of 32-bit numbers to speed up training. Works, mostly. Lesson 4.3.

**fp8** - 8-bit numbers. Even faster, only works on newer GPUs (H100+). nanochat uses this.

**Distributed Training** - Training one model across multiple GPUs or multiple machines. Module 6 covers this.

**DDP (Distributed Data Parallel)** - The simplest form: each GPU has a full copy of the model, processes different data, and they sync gradients each step. What nanoGPT and nanochat use with `torchrun`.

**FSDP (Fully Sharded Data Parallel)** - When the model doesn't fit on one GPU, FSDP splits the model across GPUs as well as the data. Used for big models.

**ZeRO** - Microsoft's earlier sharding approach (stage 1, 2, 3). FSDP is a PyTorch-native version of ZeRO-3.

**Tensor / Model / Pipeline Parallelism** - Different ways to split up a very large model. Needed beyond ~30B parameters or so. Lesson 6.4.

**PyTorch** - The deep learning framework. The library that gives you "tensors" (smart multi-dimensional arrays) and auto-differentiation. Made by Meta, open source. nanoGPT and nanochat are both pure PyTorch.

**TensorFlow / JAX** - Alternatives to PyTorch. Mostly used by Google. You don't need them for this course.

**HuggingFace (HF)** - A company and an open-source ecosystem. Three key pieces:
- **Hub** - a website where people upload trained models (like GitHub for models).
- **transformers** - a Python library that can load almost any model with 3 lines of code.
- **datasets** - a library for downloading training data easily.
Every working ML engineer uses HuggingFace daily.

**wandb (Weights & Biases)** - A website/tool for tracking ML experiments. You log your training metrics to it and it makes pretty charts. Free for individuals. Standard in the industry.

**SageMaker** - Amazon's managed ML service. It's a collection of tools on AWS for training and serving models. Useful in enterprise; overkill for personal projects. Lesson 6.6.

**Cloud GPU Provider** - A company that rents you GPU time by the hour. Lambda, Runpod, Vast.ai for the cheap/hobby end; AWS, GCP, Azure for the enterprise end. An 8xH100 node costs ~$20-30/hr as of 2026.

**Spot Instance** - A cheaper GPU you rent, but the provider can take it back any time. Good for fault-tolerant work.

---

## Architectural & research vocabulary

**Encoder / Decoder** - The two halves of the original 2017 transformer. Encoder reads input, decoder writes output. GPT is decoder-only. BERT is encoder-only. T5 is both. For LLMs, decoder-only has won.

**Causal / Autoregressive** - "The model can only look at past tokens, not future ones." All GPT-style LLMs are causal.

**Positional Encoding / RoPE / ALiBi** - How the model knows token order. Attention alone doesn't care about order, so we inject position info. nanoGPT uses learned position embeddings; nanochat uses RoPE (Rotary Position Embedding). Lesson 3.9.

**LayerNorm / RMSNorm** - Normalization layers inside transformer blocks. Help training stability. Lesson 3.7.

**Residual Connection / Skip Connection** - Adding the input of a layer to its output (`x + f(x)`). Makes very deep networks trainable. Lesson 3.7.

**GELU / SwiGLU** - Activation functions used inside transformer MLPs. Lesson 3.6.

**MoE (Mixture of Experts)** - A model where only some of the weights are used per token, letting you have a huge model for cheap inference. GPT-4 and Mixtral are MoEs. Lesson 7.7.

**Context Length** - How many tokens fit in one input window.

**KV Cache** - A trick that makes generation fast by caching past keys/values so you don't recompute them every new token. Lesson 5.8.

**Flash Attention** - A fused, memory-efficient implementation of attention. Makes training faster without changing the math. Lesson 6.5.

**Quantization** - Storing weights in fewer bits (4-bit, 3-bit, 1-bit) to make inference cheaper. Lossy, but often fine.

**LoRA (Low-Rank Adaptation)** - A fine-tuning technique that trains only a small number of extra parameters. Much cheaper than full fine-tuning.

**Scaling Laws** - Empirical formulas predicting loss as a function of (model size, data size, compute). Kaplan 2020 and Chinchilla 2022. Module 6, Module 7.

**Compute-optimal** - For a given compute budget, the model size and dataset size that give the best loss. Chinchilla's result: roughly 20 tokens per parameter (for that era).

**Eval / Benchmark** - A standardized test set. MMLU (general knowledge), HumanEval (code), GSM8K (grade-school math), ARC (reasoning). Lesson 5.7.

**Perplexity** - A measure of how well the model predicts text. Lower is better. Roughly: "on average, the model is as uncertain as if it had to choose among `perplexity` equally likely tokens."

**BPB (Bits Per Byte)** - A vocab-size-invariant version of perplexity. nanochat uses this as the primary loss metric.

**Hallucination** - When the model confidently produces wrong information. A core unsolved problem.

**Alignment** - Making models behave in a way that is helpful/honest/harmless. SFT + RLHF are alignment techniques.

---

## Research process vocabulary

**Paper** - A document describing new research. Usually on [arXiv.org](https://arxiv.org). Not peer-reviewed until published in a conference, but everyone reads the arXiv version.

**arXiv** - The website where ML papers get posted. Free, no paywall.

**Conference** - Major ML venues: NeurIPS, ICML, ICLR. Less relevant for the LLM industry now because the best work often skips them.

**Benchmark / Leaderboard** - A standard test, plus a public ranking. Chatbot Arena, Open LLM Leaderboard, etc.

**SOTA (State of the Art)** - Best current performance on a benchmark.

**Ablation** - Removing one component of your system to measure its contribution. "We ablated the residual connections and the model didn't train." A key scientific move.

**Reproduce / Replication** - Running someone else's experiment yourself and getting the same numbers. Much harder than it sounds. Crucial for learning.

---

## Confused? Good.

You're not supposed to remember all of this. This is a reference. Come back to it whenever a word in a lesson feels heavy. By the end of the course, most of these will feel obvious. Not because you memorized them - because you built the thing.

## Visualize this

Resources to keep open alongside the glossary:

- **Tiktokenizer** (https://tiktokenizer.vercel.app/) → see "Token" in action.
- **bbycroft.net/llm** (https://bbycroft.net/llm) → see "Transformer", "Attention", "Embedding" work inside a real GPT-2.
- **3Blue1Brown LLM series** (https://www.youtube.com/@3blue1brown/playlists) → visual explanations of most terms here.
- **Hugging Face Inference Playground** (https://huggingface.co/playgrounds) → try "LLM", "Fine-tune", "Temperature" on real models.

A conceptual map of how the terms relate:

```
                   AI
                    │
                    ▼
                   ML
                    │
                    ▼
              Deep Learning
                    │
                    ▼
           Neural Networks
                    │
     ┌──────────────┼─────────────┐
     ▼              ▼             ▼
  CNN (vision)  Transformer    RNN (older)
                    │
     ┌──────────────┼─────────────┐
     ▼              ▼             ▼
  BERT           GPT / LLM     T5
  (encoder)      (decoder)     (enc-dec)
                    │
     ┌──────────────┼─────────────────┐
     ▼              ▼                 ▼
  Pretrain       SFT / Chat         RLHF / DPO
  (lots of data) (instructions)     (preferences)
```

Each of these has its own lesson later. This is the forest. We learn it tree by tree.

Next: `02_tools_and_setup.md`.
