# The LLM Course: From Python to Leading Science Teams

A self-contained, ground-up curriculum to take you from "I only know Python"
to "I can build, train, ship, and lead research on large language models."

Two real codebases anchor every lesson:

- `~/workspace/nanoGPT`   - Karpathy's minimal GPT pretraining repo (~300 lines of model code + ~300 lines of training loop). This is your **microscope**. Every concept is first pinned to a line in this repo.
- `~/workspace/nanochat`  - Karpathy's full modern pipeline: tokenizer training, pretraining, midtraining, supervised fine-tuning (SFT), reinforcement learning (RL), evaluation, inference engine, and a chat web UI. This is your **factory**. It shows how the microscope parts compose into a real system.

You do not need to read the papers first. We read the code, understand what it
does, and then the papers become obvious.

---

## How to use this course

1. Go lesson-by-lesson, in order. Each lesson takes 1-4 hours.
2. Every lesson has three sections:
   - **Concept** - the idea, in plain English, no jargon (and if we use jargon, we define it right there).
   - **In the code** - the specific lines in `nanoGPT` or `nanochat` that implement it.
   - **Exercises** - small things you run or modify. Hands-on, not optional.
3. If you get stuck, ask me ("explain lesson X part Y" or "why does this line do Z"). I have both codebases open.
4. At the end of each **module**, there is a capstone project.

Time commitment estimate: ~80-120 hours of focused work to complete the whole course. Most people should plan 3-6 months at an evening-study pace. You can do the first 3 modules on a laptop CPU. Modules 4+ benefit from a GPU (we cover cloud GPUs when you get there).

---

## Course Map

### Module 0 - Start Here
Set up your tools, set expectations, and demystify the vocabulary.

- `00_start_here/00_what_is_this_course.md`
- `00_start_here/01_demystified_glossary.md` - every scary term you've heard, defined simply.
- `00_start_here/02_tools_and_setup.md` - Python env, PyTorch, GPU concepts, what the terminal does.
- `00_start_here/03_what_is_an_llm.md` - what a language model *is*, physically, with a 10-line toy.

### Module 1 - Math Foundations You Actually Need
Not a math textbook. The minimum math to read `model.py` line by line.

- `01_foundations/01_vectors_and_matrices.md` - numbers, arrays, shapes, dot products.
- `01_foundations/02_matrix_multiplication.md` - the one operation that runs the world.
- `01_foundations/03_calculus_for_learning.md` - derivatives, gradients, chain rule (intuition only).
- `01_foundations/04_probability_basics.md` - what a probability distribution is, softmax, cross-entropy.
- `01_foundations/05_numpy_and_tensors.md` - hands-on, the PyTorch tensor is just a smart array.
- `01_foundations/capstone_bigram.md` - build a "bigram language model" from scratch, 40 lines, no PyTorch.

### Module 2 - Deep Learning from Zero
A neural network is a function with knobs. Training = turning the knobs.

- `02_deep_learning/01_a_neuron.md` - one neuron, one linear layer, why bias exists.
- `02_deep_learning/02_activations_nonlinearity.md` - ReLU, GELU, why you need them.
- `02_deep_learning/03_loss_functions.md` - how we measure "wrong". MSE, cross-entropy.
- `02_deep_learning/04_backprop_and_sgd.md` - how the knobs turn. The heart of all of this.
- `02_deep_learning/05_optimizers_adam_adamw.md` - smarter knob-turning. Why AdamW specifically.
- `02_deep_learning/06_regularization_dropout_weightdecay.md` - keeping the model honest.
- `02_deep_learning/07_training_loop_anatomy.md` - the 10 lines at the center of `nanoGPT/train.py`.
- `02_deep_learning/capstone_mlp.md` - train a tiny MLP on characters. Still CPU-friendly.

### Module 3 - The Transformer via nanoGPT
Read `nanoGPT/model.py` line-by-line with me. By the end, the transformer
stops being magic.

- `03_transformers_nanogpt/01_tokenization.md` - turning text into integers. Characters vs BPE.
- `03_transformers_nanogpt/02_embeddings.md` - integers become vectors.
- `03_transformers_nanogpt/03_attention_intuition.md` - the "talking to each other" operation.
- `03_transformers_nanogpt/04_attention_mathematically.md` - Q, K, V, softmax, masking.
- `03_transformers_nanogpt/05_multi_head_attention.md` - several smaller attentions in parallel.
- `03_transformers_nanogpt/06_mlp_block.md` - the other half of a transformer block.
- `03_transformers_nanogpt/07_layernorm_and_residual.md` - the glue that makes deep nets train.
- `03_transformers_nanogpt/08_full_block_walkthrough.md` - putting one block together.
- `03_transformers_nanogpt/09_positional_encoding.md` - how the model knows word order.
- `03_transformers_nanogpt/10_lm_head_and_sampling.md` - turning vectors back into words.
- `03_transformers_nanogpt/11_train_py_walkthrough.md` - every line of `nanoGPT/train.py`.
- `03_transformers_nanogpt/capstone_shakespeare.md` - train the baby GPT on Shakespeare on your laptop.

### Module 4 - Training Pipeline: Data, Loops, Scale
Moving from "it runs" to "it runs well."

- `04_training_pipeline/01_datasets_and_dataloaders.md` - where does the text come from?
- `04_training_pipeline/02_bpe_tokenizer_deep.md` - how tiktoken / nanochat tokenizer work.
- `04_training_pipeline/03_mixed_precision_bf16_fp16.md` - faster math, why it works.
- `04_training_pipeline/04_gradient_accumulation.md` - training big batches on small GPUs.
- `04_training_pipeline/05_learning_rate_schedules.md` - warmup, cosine decay, and why.
- `04_training_pipeline/06_gradient_clipping_nan_debugging.md` - when training goes bad.
- `04_training_pipeline/07_checkpointing_and_resume.md` - save/restore, the operator's view.
- `04_training_pipeline/08_evaluation_val_loss_perplexity.md` - how do we know it works?
- `04_training_pipeline/capstone_custom_dataset.md` - train a tiny GPT on your own text file.

### Module 5 - nanochat: The Full Modern Stack
nanoGPT stops at pretraining. nanochat is a real chatbot end-to-end.

- `05_nanochat_full_stack/01_tour_of_nanochat.md` - map the whole repo in 1 hour.
- `05_nanochat_full_stack/02_tokenizer_training.md` - `scripts/tok_train.py`.
- `05_nanochat_full_stack/03_pretraining_base_train.md` - `scripts/base_train.py`, compare with nanoGPT.
- `05_nanochat_full_stack/04_midtraining_and_chat_formatting.md` - teaching the model chat structure.
- `05_nanochat_full_stack/05_sft_supervised_fine_tuning.md` - `scripts/chat_sft.py`.
- `05_nanochat_full_stack/06_rl_and_preferences.md` - `scripts/chat_rl.py`, GRPO-style RL on verifiable rewards (gsm8k).
- `05_nanochat_full_stack/07_evaluation_core_mmlu_humaneval_gsm8k.md` - benchmarks, what they mean.
- `05_nanochat_full_stack/08_inference_engine_kv_cache.md` - `nanochat/engine.py`, making generation fast.
- `05_nanochat_full_stack/09_serving_chat_web.md` - `scripts/chat_web.py`, shipping the model.
- `05_nanochat_full_stack/10_speedrun_end_to_end.md` - run `runs/speedrun.sh` and narrate what happens.
- `05_nanochat_full_stack/capstone_train_and_talk.md` - train a tiny chatbot end-to-end and talk to it.

### Module 6 - Infrastructure and Scaling
What makes GPT-4 different from nanoGPT is not just code. It's infrastructure.

- `06_infra_and_scaling/01_cpu_vs_gpu_vs_tpu.md` - hardware 101, what "cores" really means.
- `06_infra_and_scaling/02_gpu_memory_and_vram.md` - why "out of memory" happens.
- `06_infra_and_scaling/03_distributed_training_ddp.md` - `torchrun`, what DDP is doing.
- `06_infra_and_scaling/04_fsdp_zero_tensor_parallelism.md` - training models that don't fit on one GPU.
- `06_infra_and_scaling/05_flash_attention_fp8_performance.md` - why a newer GPU is faster.
- `06_infra_and_scaling/06_the_cloud_lambda_aws_sagemaker.md` - renting GPUs. What SageMaker actually is.
- `06_infra_and_scaling/07_experiment_tracking_wandb.md` - wandb, what it gives you.
- `06_infra_and_scaling/08_huggingface_ecosystem.md` - what HF is, hub vs transformers vs datasets.
- `06_infra_and_scaling/09_scaling_laws.md` - Chinchilla, Kaplan, the param-data tradeoff.
- `06_infra_and_scaling/10_cost_engineering.md` - reading a GPU bill, $/token, compute-optimal training.
- `06_infra_and_scaling/capstone_cloud_run.md` - launch a small cloud run (guided), kill it safely.

### Module 7 - Reading Research Papers
How to go from "scared of PDFs" to actually using papers as a research tool.

- `07_research_and_papers/01_how_to_read_a_paper.md` - the three-pass method, applied to "Attention Is All You Need".
- `07_research_and_papers/02_core_papers_you_must_know.md` - an annotated reading list of ~20 papers.
- `07_research_and_papers/03_transformer_paper_walkthrough.md` - read "Attention Is All You Need" side-by-side with `nanoGPT/model.py`.
- `07_research_and_papers/04_gpt_papers_gpt2_gpt3_gpt4.md` - what each paper added.
- `07_research_and_papers/05_scaling_laws_papers.md` - Kaplan 2020, Chinchilla 2022.
- `07_research_and_papers/06_rlhf_instruct_dpo.md` - InstructGPT, RLHF, DPO.
- `07_research_and_papers/07_modern_architectures.md` - Llama family, Mixture-of-Experts, State-Space, Mamba.
- `07_research_and_papers/08_how_to_reproduce_a_paper.md` - reproduction as a learning technique.
- `07_research_and_papers/09_how_to_write_a_paper.md` - if you want to publish, how to think about it.
- `07_research_and_papers/capstone_paper_replica.md` - reproduce one figure from one paper using nanochat.

### Module 8 - Leading Science Teams
The non-code part of the job, which is 80% of being senior.

- `08_leadership/01_what_ml_scientists_actually_do_day_to_day.md`
- `08_leadership/02_the_research_loop.md` - hypothesis -> experiment -> analyze -> decide.
- `08_leadership/03_project_scoping_and_killing_projects.md`
- `08_leadership/04_evaluations_as_the_product.md` - why evals are the hardest and most important work.
- `08_leadership/05_managing_compute_and_budgets.md`
- `08_leadership/06_team_structure_and_hiring.md`
- `08_leadership/07_working_with_product_and_eng.md`
- `08_leadership/08_communicating_results_up_and_out.md`
- `08_leadership/09_research_taste.md` - the hardest-to-teach skill, demystified.
- `08_leadership/10_staying_current_in_a_field_that_moves_weekly.md`

### Module 9 - Beyond Text: Vision, Audio, Video, Multimodal
The whole non-text AI landscape, in the same style as Modules 0-8.

- ViT and image understanding
- CLIP and multimodal embeddings
- Diffusion models (Stable Diffusion, Flux)
- Text-to-image: DALL-E, Midjourney, SDXL
- Video generation: Sora, Veo
- Whisper and speech-to-text
- Text-to-speech (VITS, VALL-E, XTTS)
- Multimodal LLMs (GPT-4V, Claude, Gemini, LLaVA)
- Audio understanding LLMs (GPT-4o voice, Qwen-Audio)
- World models (Genie, Sora's hypothesis)
- Capstone: hands-on multimodal experiments.

### Module 10 - Agentic AI
The current frontier: LLMs that act.

- What an agent is (vs. a chatbot)
- Tool use and function calling
- RAG: Retrieval-Augmented Generation
- Code execution agents
- Web browsing agents
- Agent frameworks (LangChain, DSPy, MCP)
- Planning: ReAct, ToT, o1/R1-style
- Multi-agent systems
- Memory and statefulness
- Agent evaluation
- Safety and guardrails
- Capstone: build a real agent.

### Resources
- `resources/cheatsheet_math.md`
- `resources/cheatsheet_pytorch.md`
- `resources/visualizations.md` - curated list of interactive visualizations for every concept.
- `resources/faq.md`

---

## Prerequisites

- Intermediate Python (you said you have this - great).
- A computer (Linux/Mac/WSL) and a terminal.
- Willingness to be confused for 30 minutes at a time. This is the learning process. Don't panic.

## What we deliberately don't do

- We don't teach Python itself. You know it.
- We don't teach full classical ML (SVM, random forests, etc.). We're going straight to transformers. That's what you're here for.
- We don't do exhaustive math proofs. You can go deeper later - I'll tell you where.

## Sign of completion

You've finished the course when, given a new LLM paper on arXiv and a repo like
nanochat, you can:

1. Read the paper in an hour and tell someone whether it's worth reproducing.
2. Make a specific change to the nanochat code to test the paper's idea.
3. Run it on a cloud GPU, interpret the metrics, and decide next step.
4. Write a one-page memo explaining your finding to a non-researcher.

That is the real job. Let's get you there.

---

Start now: open `00_start_here/00_what_is_this_course.md`.
