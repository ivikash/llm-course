# Module 9 Capstone - Multimodal Experiment

Build familiarity with each modality hands-on. One weekend of work.

## Goal

Run 4-5 different modality tools, observe their capabilities and failures, write up what you learned. No new training; just use pre-trained models.

## The tasks

### Task 1: Speech-to-text with Whisper (30 min)

1. Install: `pip install openai-whisper` (or `pip install faster-whisper` for speed).
2. Record a 2-minute voice note of yourself talking about anything.
3. Run:
   ```bash
   whisper my_audio.mp3 --model small
   ```
4. Check output. Count errors.
5. Try translation: non-English audio:
   ```bash
   whisper french.mp3 --task translate --model small
   ```

### Task 2: Text-to-image with Stable Diffusion (30 min)

1. Install: `pip install diffusers transformers accelerate`.
2. Run:
   ```python
   from diffusers import StableDiffusionPipeline
   import torch
   pipe = StableDiffusionPipeline.from_pretrained(
       "stabilityai/stable-diffusion-2-1",
       torch_dtype=torch.float16,
   ).to("cuda")

   for prompt in [
       "a cat wearing a crown, oil painting",
       "a futuristic city, neon lighting, ultrawide",
       "a microscopic view of tardigrades",
       "a tiny robot in a flower pot, depth of field",
       "a watercolor sketch of a mountain lake"
   ]:
       image = pipe(prompt, num_inference_steps=30).images[0]
       image.save(f"{prompt[:20]}.png")
   ```
3. Open the images. Rate each 1-10 for "did it match the prompt?"

### Task 3: Vision-language model with LLaVA or GPT-4V (30 min)

Pick one:

**Local, free (LLaVA)**:
```python
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from PIL import Image

processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto"
)

image = Image.open("my_photo.jpg")
prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"
inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(outputs[0]))
```

**API (GPT-4V via OpenAI)**:
```python
import openai, base64
with open("my_photo.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
    ]}],
)
print(response.choices[0].message.content)
```

Try 5 images:
- A landscape photo.
- A photo of food.
- A screenshot of a chart.
- A selfie.
- A messy bookshelf.

For each: ask 3 questions. Rate accuracy.

### Task 4: Text-to-speech (optional, 20 min)

OpenAI API (simplest):
```python
from openai import OpenAI
client = OpenAI()
response = client.audio.speech.create(
    model="tts-1", voice="shimmer", input="Hello from my AI course project!"
)
response.stream_to_file("hello.mp3")
```

Or locally with XTTS if you want voice cloning.

### Task 5: Pick ONE advanced thing (30-60 min)

Pick one that excites you:
- Train a LoRA on 20 photos of your dog/face/product.
- Use ComfyUI to build a node-based workflow.
- Record with Whisper + auto-translate to 3 languages.
- Generate a 5-second video with CogVideoX or a free Runway/Pika credit.
- Use Whisper + GPT-4 + TTS to build a tiny voice assistant.

## Write-up

`multimodal_report.md`:

```markdown
# My Multimodal AI Experiment

## Setup
Hardware: [your machine]
Time: [hours]
Total API cost: $[amount]

## What I tried and what happened

### Whisper
- [how accurate was transcription?]
- [did translation work?]
- Failures: [e.g., "missed my name's spelling"]

### Stable Diffusion
- Rate out of 10: 5 prompts
- Most surprising generation: [which one]
- Biggest failure mode: [e.g., "hands"]

### Vision-language
- Which VLM: [LLaVA/GPT-4V/etc]
- Task-level accuracy: [count correct answers]
- Qualitative impression: [where did it shine? fail?]

### Text-to-speech
- [quality impression]
- [any eeriness?]

### My advanced task
- What I did: [description]
- Result: [link or description]

## Top 3 things I learned

1. [specific observation]
2. [specific observation]
3. [specific observation]

## What's next I want to explore
- [one thing]
```

This is a real deliverable. Put it in your public portfolio.

## Extension: the voice agent

An ambitious version combining everything:

```python
import whisper, openai, os
from TTS.api import TTS

asr = whisper.load_model("small")
llm_client = openai.OpenAI()
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

while True:
    input("Press Enter, then speak...")
    # record 5 seconds (use sounddevice or similar)
    record_audio("query.wav")
    
    # speech to text
    transcription = asr.transcribe("query.wav")["text"]
    print(f"You: {transcription}")
    
    # LLM response
    reply = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": transcription}],
    ).choices[0].message.content
    print(f"AI: {reply}")
    
    # text to speech
    tts.tts_to_file(text=reply, speaker_wav="reference.wav", language="en", file_path="response.wav")
    os.system("mpg123 response.wav")
```

100 lines for a working voice assistant. This is what Siri was 10 years ago, now runnable by you.

## Costs estimate

- Whisper: free if local; $0.006/min via OpenAI API.
- Stable Diffusion: free if local.
- GPT-4o vision: $0.01-0.05 per image.
- LLaVA: free if local.
- OpenAI TTS: $0.015 per 1000 chars.

Total for this entire capstone: ~$5 if you use APIs, $0 if you have a GPU.

## Why this matters

After this capstone, you've **personally interacted with every major modality of generative AI**. You have direct experience - not secondhand - with the capabilities and limits.

This is rare. Most people who work in AI have only touched one or two modalities. You now have breadth.

## Next

Module 10 - Agentic AI.
