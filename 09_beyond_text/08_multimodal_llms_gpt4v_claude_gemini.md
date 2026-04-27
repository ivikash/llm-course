# 9.8 - Multimodal LLMs (GPT-4V, Claude, Gemini, LLaVA)

A **multimodal LLM** understands text + images (and increasingly video, audio). Given a picture and a question, it can answer. Given a document with diagrams, it can reason about both. This lesson shows how they work.

## The architecture pattern

Almost all multimodal LLMs follow the same 3-piece recipe:

```
Image → vision encoder → image embeddings
                            ↓
                       projector (small MLP)
                            ↓
Text → tokenizer → text embeddings → [combined sequence] → LLM → text output
```

1. A **vision encoder** (usually CLIP's ViT, or SigLIP, or a custom one) turns the image into a sequence of embedding vectors - like "visual tokens".
2. A **projector** (1-2 linear layers + nonlinearity) maps those vision embeddings to the LLM's embedding dimension.
3. These "visual tokens" are inserted into the LLM's input sequence alongside text tokens.
4. The LLM processes them like any other tokens and generates text.

LLaVA exemplifies this pattern cleanly:
```
LLaVA = (CLIP ViT) + (small projector) + (Vicuna/Llama LLM)
```

## Training pipeline

### Step 1: freeze pretrained components
Start with:
- A pretrained vision encoder (e.g., CLIP ViT-Large).
- A pretrained LLM (e.g., Llama-2-7B).

Both are frozen at this point.

### Step 2: train the projector
Only the projector is trainable. Feed image-caption pairs:
- Image → CLIP → projection → prepend as "visual tokens".
- LLM is asked to output the caption.
- Loss: standard cross-entropy on caption tokens.

~500K image-caption pairs, fast to train.

### Step 3: instruction tune
Now train on visual instructions:
- "What is in this image?"
- "How many people?"
- "Describe the scene."

This phase fine-tunes the LLM (and maybe the projector). Visual instruction datasets: LLaVA-Instruct (GPT-4-generated), ShareGPT4V, etc.

### Step 4 (optional): RLHF / DPO
For chat-quality multimodal models. Same as text-only SFT alignment, but with multimodal preferences.

## Key design questions

### How many visual tokens?

A 224×224 image through ViT-Large gives ~256 tokens. But LLM context is expensive - so some models downsample:
- LLaVA-1.5: 576 tokens per image.
- Llama 3.2 Vision: ~64-256 tokens.
- Qwen2-VL: adaptive based on resolution.

Tradeoff: more tokens = better fine detail but more compute.

### Resolution handling

Real images come at various resolutions. Options:
1. Resize to fixed size (simple, loses detail).
2. Crop to grid of fixed-size patches (handles high res, more tokens).
3. Adaptive (compute varies with image).

Modern approaches (LLaVA-NeXT, Qwen2-VL) handle varied resolutions via patching.

### Single image vs multiple

Simple VLMs take one image. Modern ones take many:
- "Compare these two images."
- "Explain this sequence of screenshots."
- "Summarize this PDF (pages as images)."

This is where long-context tricks matter.

## Landscape (2024-2026)

### Open models
- **LLaVA** family (LLaVA-1.5, LLaVA-NeXT, LLaVA-OneVision): academic reference.
- **Idefics 2/3** (HuggingFace): commercial-friendly.
- **Qwen2-VL** (Alibaba): strong, open.
- **InternVL** (Shanghai AI Lab): strong, open.
- **Llama 3.2 Vision** (Meta): 11B and 90B versions.
- **Molmo** (Allen AI): high-quality visual data.

### Closed models
- **GPT-4V / GPT-4o** (OpenAI): frontier, strong integration with chat.
- **Claude 3.5 Sonnet with Vision** (Anthropic): strong on documents, charts.
- **Gemini 1.5/2.0** (Google): native multimodal from scratch, huge context (1M tokens).

All close at key benchmarks; closed typically leads open by ~6-12 months.

## What they're good at

- Image captioning.
- Visual question answering.
- Document understanding (invoices, forms, medical charts).
- Chart and graph interpretation.
- UI understanding (screenshots).
- Code-from-sketch (wireframe → HTML).
- OCR (but text-in-image tokenization limits are real).

## What they struggle with

- Counting (how many birds in this image? Often off).
- Precise spatial reasoning ("is the cat to the left of the dog?").
- Subtle visual details (a small logo, a face in the background).
- Abstract visual puzzles (Raven's matrices, Bongard problems).
- Videos (frame-by-frame is expensive).

Active research areas.

## Using one: LLaVA in code

```python
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from PIL import Image

processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto"
)

image = Image.open("my_image.jpg")
prompt = "USER: <image>\nWhat do you see in this image?\nASSISTANT:"

inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(outputs[0], skip_special_tokens=True))
```

## API access to closed multimodal models

OpenAI:
```python
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ]}]
)
```

Anthropic's Claude, Google's Gemini have similar APIs.

## Video-understanding VLMs

Video as a sequence of frames → same as many-image VLMs, but with more tokens and temporal positional encoding. Challenges:
- Frame sampling strategy (every Nth frame? Keyframes?).
- Compute cost scales with frames.

Models: Video-LLaVA, LLaVA-NeXT-Video, Gemini 1.5 (1M token context = ~1 hour video), GPT-4o video input.

## Key papers

- **LLaVA** (Liu 2023): https://arxiv.org/abs/2304.08485
- **Flamingo** (Alayrac 2022): pioneering multimodal LLM from DeepMind.
- **Qwen-VL** (Bai 2023): strong open VLM.
- **GPT-4V Technical Report** (OpenAI 2023): capabilities, no architecture.

## Visualize this

**VLM architecture at a glance**:

```
  ┌───────────┐      ┌────────────┐
  │   Image    │      │   Text      │
  │  (photo)   │      │ (prompt)    │
  └─────┬─────┘      └─────┬──────┘
        │                   │
        ▼                   ▼
  ┌───────────┐      ┌────────────┐
  │ Vision     │      │ Text        │
  │ Encoder    │      │ Tokenizer   │
  │ (CLIP ViT) │      │             │
  └─────┬─────┘      └─────┬──────┘
        │                   │
        ▼                   ▼
  image embeddings        text tokens
        │                   │
        ▼                   │
  ┌───────────┐             │
  │ Projector  │             │
  │  (MLP)     │             │
  └─────┬─────┘             │
        │                   │
        ▼                   ▼
  "visual tokens" ────────▶ │
  prepended as sequence ───▶│
                            │
                            ▼
                    ┌────────────┐
                    │    LLM      │
                    │ (Llama/etc) │
                    └─────┬──────┘
                            │
                            ▼
                       text output
                   "I see a cat on a couch..."
```

The LLM sees a unified sequence: [visual tokens] + [text tokens] → predicts text.

**Training pipeline (3 phases)**:

```
  Phase 1: Pretrained components
    ├── Text LLM (frozen): Llama-7B, etc.
    └── Vision encoder (frozen): CLIP ViT-L

  Phase 2: Train projector only (~500K image-caption pairs)
    - Freeze everything except the tiny MLP projector
    - Project image embeddings into LLM's vector space
    - Loss: LLM predicts captions from image-via-projector
    - Cheap: ~hours to days on 1 GPU

  Phase 3: Visual instruction tuning (~500K image-prompt-response triples)
    - Unfreeze LLM (and maybe projector)
    - Train on "conversational" data involving images
    - "What is in this image?" "Describe this chart." etc.
    - Result: chat-style VLM like GPT-4V
```

**What's inside the LLM's context**:

```
  For a question "What's in this image?":

  LLM sees (after embedding):
    [visual token 1] [visual token 2] ... [visual token 576]
    [<|user|>] [W] [hat] [ '] [s] [ in] [ this] [ image] [?]
    [<|assistant|>] ← start generating here

  The visual tokens are just floating-point vectors like text tokens.
  LLM attends over all of them (both visual and text).
  It "looks at" the image in the same way it looks at text.
```

**How many visual tokens?**

```
  A 224×224 image through CLIP ViT-L:
    → 14 × 14 = 196 patches
    → 196 visual tokens after projection (with [CLS])

  A 1024×1024 high-res image:
    → 64 × 64 = 4096 patches (huge!)
    → typically split into tiles or downsampled

  Modern VLMs:
    LLaVA-1.5: 576 visual tokens (336×336 image)
    LLaVA-NeXT: up to 2880 tokens (multi-resolution)
    Llama 3.2 Vision: adaptive, ~64-256 tokens
    Qwen2-VL: adaptive based on resolution
```

**Landscape of VLMs (2024-2026)**:

```
  Open:
  ────────────────────────────────────────────
  LLaVA-1.5 / LLaVA-NeXT (academic reference)
  Idefics 2/3 (HuggingFace, commercial-friendly)
  Qwen2-VL (Alibaba, excellent)
  InternVL (Shanghai, strong)
  Llama 3.2 Vision 11B/90B (Meta)
  Molmo (Allen AI, high-quality data)
  NVLM (NVIDIA)

  Closed:
  ────────────────────────────────────────────
  GPT-4V / GPT-4o (OpenAI)
  Claude 3.5 Sonnet (Anthropic) - strong at docs
  Gemini 1.5/2.0 (Google) - very long context
  Pixtral (Mistral, semi-open)

  Specialized:
  ────────────────────────────────────────────
  PaliGemma (Google, OCR-strong)
  Florence-2 (Microsoft, detection-oriented)
```

**Running LLaVA locally**:

```python
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import torch

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
)

image = Image.open("my_photo.jpg")
prompt = "[INST] <image>\nDescribe this image in detail. [/INST]"

inputs = processor(prompt, image, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=300)
print(processor.decode(outputs[0], skip_special_tokens=True))
```

~10 GB VRAM. Runs on laptop GPUs.

**Using GPT-4V via API**:

```python
from openai import OpenAI
import base64

client = OpenAI()
with open("my_photo.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{b64}"
            }},
        ],
    }],
)
print(response.choices[0].message.content)
```

~$0.01-0.05 per image. Good quality.

**What VLMs are good at (2026)**:

```
  ✓ Image description and captioning
  ✓ Visual question answering ("what color?", "how many?")
  ✓ Chart/graph interpretation
  ✓ Document understanding (invoices, forms, PDFs)
  ✓ OCR (reading text in images)
  ✓ UI understanding (screenshots of websites/apps)
  ✓ Code from sketches (wireframe → HTML)
  ✓ Accessibility (describing photos for blind users)
  ✓ Medical imaging (with specialized fine-tuning)
```

**What they struggle with**:

```
  ✗ Precise counting ("How many people?") - often off by 1-2
  ✗ Spatial reasoning ("Is the cat LEFT of the dog?") - ~70% accuracy
  ✗ Subtle visual details (small objects, distant faces)
  ✗ Reading handwriting (improving but not reliable)
  ✗ Identifying specific people (safety measures limit this)
  ✗ Optical illusions, puzzles
  ✗ Mathematical diagrams with complex notation
```

**Comparison across VLMs on common tasks**:

```
  Task: "Describe this chart"
  Benchmark: ChartQA

  Model                  Accuracy
  ─────────────────────  ──────────
  LLaVA-1.5 7B           55%
  Qwen2-VL 7B            75%
  LLaVA-NeXT 34B         80%
  GPT-4o                  85%
  Claude 3.5 Sonnet       88%  ← great at structured data
  Gemini 1.5 Pro         83%

  For charts/docs: Claude or GPT-4o.
  For open-source: Qwen2-VL or InternVL.
```

**The "single-image" vs "multi-image" distinction**:

```
  Single-image VLMs:
    "Describe THIS photo." ← works on one image at a time

  Multi-image VLMs (newer):
    "Compare these two images."
    "Summarize these 10 pages of a PDF."
    "Describe the arc of these 20 movie stills."

  Multi-image requires:
    - Long context (1M+ tokens helps)
    - Efficient visual tokenization (fewer tokens per image)
    - Attention that can associate content across images

  Models: Gemini 1.5+, Qwen2-VL, Llama 3.2 Vision handle this well.
```

## Exercises

1. Run LLaVA on a photo from your phone. Ask: "Describe this image."

2. Ask a multimodal model hard questions: count objects, describe spatial relations, interpret a graph. Note where it struggles.

3. Read the LLaVA paper's Figure 1 (the architecture). Trace the data flow.

4. Compare: GPT-4V via API + LLaVA-7B locally on the same image. Note quality gap, latency gap, cost gap.

## Next

`09_audio_understanding_llms.md` - the same treatment for audio.
