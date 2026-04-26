# 6.8 - The HuggingFace Ecosystem

HuggingFace (HF) is the GitHub of ML. You will touch it every day in this career. This lesson gives you working fluency.

## The three-piece core

### 1. Hub (https://huggingface.co)

A website hosting **models**, **datasets**, and **spaces** (hosted demos). Like GitHub, but:
- Native support for weights (git-lfs for huge files).
- Model cards with usage instructions.
- One-line loading in Python.

Famous models on the Hub:
- `meta-llama/Llama-2-7b`
- `mistralai/Mixtral-8x7B-v0.1`
- `openai/clip-vit-large-patch14`
- `runwayml/stable-diffusion-v1-5`
- `openai/whisper-large-v3`

You can upload your own - free for public, paid for private.

### 2. `transformers` (the library)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

Four lines to load and run a 7B model. That's why everyone uses it.

Supports: hundreds of model architectures (BERT, GPT-2, T5, Llama, Mistral, Whisper, CLIP, ...).

### 3. `datasets` (the library)

```python
from datasets import load_dataset

ds = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
for example in ds.take(10):
    print(example['text'][:200])
```

Streams huge datasets without downloading everything. Built-in splits, processing, memory-mapping.

## The supporting libraries

### `accelerate`

Thin wrapper over DDP/FSDP/DeepSpeed. Write training code once, run single-GPU, multi-GPU, multi-node with no code changes.

```python
from accelerate import Accelerator
accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
for batch in dataloader:
    outputs = model(batch)
    loss = loss_fn(outputs, targets)
    accelerator.backward(loss)
    optimizer.step()
```

Launch with `accelerate launch train.py` - handles torchrun equivalents.

Used when you want distributed training without writing the DDP boilerplate yourself.

### `peft` (Parameter-Efficient Fine-Tuning)

Fine-tuning a 7B model from scratch takes 7B × 16 bytes = 112 GB VRAM. LoRA fine-tuning takes ~1 GB.

**LoRA** (Low-Rank Adaptation): freeze the base model, train small "adapter" matrices that add corrections.

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%
```

LoRA (and QLoRA, its quantized sibling) are how most hobbyist fine-tuning happens.

### `trl` (Transformer Reinforcement Learning)

Implementations of SFT, DPO, PPO, GRPO for language models:

```python
from trl import SFTTrainer, DPOTrainer, PPOTrainer
```

If you want to SFT or RL tune on an HF model without writing the loop from scratch, TRL is the standard.

### `tokenizers`

Rust-based fast BPE and other tokenizers. What nanochat uses underneath. Separate from `transformers`.

### `evaluate`

Standard benchmark evaluations:

```python
import evaluate
bleu = evaluate.load("bleu")
score = bleu.compute(predictions=preds, references=refs)
```

Less used now, as LM evals have specialized harnesses (lm-eval-harness).

## Typical workflows

### Fine-tune an open-source model on your data

```python
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
from datasets import Dataset

ds = Dataset.from_list([{"text": "..." }, ...])

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="out", num_train_epochs=1, per_device_train_batch_size=1),
    train_dataset=ds,
)
trainer.train()
model.save_pretrained("my-fine-tuned-model")
```

Dozens of configuration options, but the skeleton is four objects.

### Run an open model locally for quick prototyping

```python
from transformers import pipeline
gen = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")
print(gen("Why is the sky blue?", max_new_tokens=100)[0]['generated_text'])
```

One line. Works.

### Use datasets for training

Same pattern as nanoGPT's prepare.py, but using HF datasets:

```python
from datasets import load_dataset
ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
for example in ds:
    tokens = tokenizer.encode(example['text'])
    # write to .bin file
```

## When to use HF vs when not

**Use HF for**:
- Trying out new open models quickly.
- Fine-tuning with LoRA.
- Loading a dataset.
- Getting inference working in minutes.

**Don't use HF for**:
- Learning from scratch (you'd skip the fundamentals - that's why nanoGPT exists).
- Cutting-edge performance. HF's `transformers` is general-purpose and has overhead.
- Custom research changes. The abstractions get in the way. Hand-write PyTorch.

Karpathy's stance: nanoGPT/nanochat explicitly avoid HF transformers, to teach you what's underneath. Once you know, HF is a fantastic tool.

## Loading your trained nanoGPT in HF

nanoGPT can interoperate. `model.py`'s `from_pretrained('gpt2')` loads an HF model and converts to nanoGPT format. You could go the other way too: convert your trained nanoGPT to HF format, push to the Hub, share with the world.

## Publishing a model

```bash
huggingface-cli login    # once
huggingface-cli repo create my-model
```

Then add `README.md`, put weights in a `pytorch_model.bin` or `model.safetensors`, add config, push.

People will discover it, cite it, use it. This is how open-source ML spreads.

## Exercises

1. Create a free HF account. Browse the top 5 models by downloads. Read their model cards.

2. In Python:
   ```python
   from transformers import pipeline
   classifier = pipeline("sentiment-analysis")
   print(classifier("I love this course!"))
   ```
   Note it downloads a model and runs inference in ~10 seconds.

3. Use `datasets` to load a small dataset:
   ```python
   from datasets import load_dataset
   ds = load_dataset("yelp_review_full", split="train[:100]")
   print(ds[0])
   ```

4. Browse HuggingFace "Spaces" - live demos of models. Try Llama-2-Chat on a Space. Notice: this is a production-ish chat UI running on HF's infra.

5. Read the PEFT / LoRA docs. Understand: adapter weights vs base weights.

## Next

`09_scaling_laws.md` - the empirical rules that guide every training decision.
