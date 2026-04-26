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

## Exercises

1. Run LLaVA on a photo from your phone. Ask: "Describe this image."

2. Ask a multimodal model hard questions: count objects, describe spatial relations, interpret a graph. Note where it struggles.

3. Read the LLaVA paper's Figure 1 (the architecture). Trace the data flow.

4. Compare: GPT-4V via API + LLaVA-7B locally on the same image. Note quality gap, latency gap, cost gap.

## Next

`09_audio_understanding_llms.md` - the same treatment for audio.
