# 9.5 - Video Generation (Sora, Veo, and friends)

Video = image + time. Everything harder. Every second of 24fps HD video is 24 frames × ~2M pixels = ~50M pixels. A 1-minute clip is 3 billion pixels. Compute cost scales accordingly.

This lesson covers where video generation is in 2026: what's possible, what's hard, which models matter.

## The challenges specific to video

1. **Computation**: naïvely, video diffusion is ~60x more expensive than image diffusion for a 1-minute clip at 24fps.
2. **Temporal consistency**: same character from frame to frame. Same lighting. No flicker.
3. **Physical plausibility**: things shouldn't float, change color inexplicably, or violate gravity.
4. **Motion coherence**: believable walking, running, handling.
5. **Long-range consistency**: a scene that spans 30 seconds should stay coherent.

These are **unsolved** in general. Current models are impressive but break on specific prompts.

## Approaches

### Approach 1: frame-by-frame + temporal consistency

Generate individual frames, enforce consistency through:
- Latent-space denoising shared across frames.
- Attention across frames.
- Optical flow constraints.

Examples: AnimateDiff, Stable Video Diffusion (from a single image).

### Approach 2: 3D U-Net (space-time convolutions)

Extend the image U-Net to operate on (time, height, width, channels). Cubes of noise get denoised to cubes of video.

Good quality but doesn't scale to long clips.

### Approach 3: Diffusion Transformers on space-time patches (Sora)

Sora's key insight:
1. Encode a video into **space-time patches** (think 3D cubes of pixels).
2. Flatten to a sequence of tokens.
3. Run a diffusion transformer (DiT) on this sequence.
4. At each denoising step, the transformer attends globally across space AND time.

This scales: longer videos = longer sequences = just compute.

OpenAI's Sora uses this (described in their technical report, no code released).

## Major 2024-2026 models

### Closed / paid

| Model | Company | Notes |
|-------|---------|-------|
| **Sora** | OpenAI | DiT-based. Up to 60s, 1080p. Released in ChatGPT Plus in 2024. |
| **Veo / Veo 2 / Veo 3** | Google DeepMind | Multi-minute, high-resolution. |
| **Kling** | Kuaishou | Strong for the price. |
| **Runway Gen-3 / Gen-4** | Runway | Early commercial leader. |
| **Pika** | Pika | Accessible, short clips. |
| **Luma Dream Machine** | Luma | Photoreal motion. |

### Open source (2024-2026)

| Model | Notes |
|-------|-------|
| **Open-Sora** | Academic reproduction, under-Sora quality but open. |
| **CogVideoX** | Tsinghua / Zhipu AI. Solid. |
| **HunyuanVideo** | Tencent, 2024. Very good for open. |
| **Mochi 1** | Genmo, 2024. Open Apache 2.0. |
| **LTX-Video** | Lightricks, 2024. Fast. |
| **Wan 2.1** | Alibaba, 2025. |

Open-source is ~1 year behind closed, improving fast.

## Architecture: DiT for video in detail

Starting tensor: video of shape (T frames, H, W, C).

### Step 1: encode into a compact latent space

Use a **VAE** trained on video: encodes (T, H, W, C) into (T', H', W', C') where T'/T ~ 1/4 and H'/H = W'/W ~ 1/8. Massive compression.

For a 64-frame 512×512 video: encoded latent is (16, 64, 64, 16) = 1M values instead of 50M pixels.

### Step 2: patchify into tokens

Split latent into patches (e.g., 2×2×2 cubes in time-height-width). Each patch → a token. Linear-project to d_model.

Result: ~10K-100K tokens per video clip.

### Step 3: diffusion on tokens

Standard DiT, with attention over all tokens. Training and inference same as image DiT but with more tokens.

### Step 4: decode

At inference, after denoising, the VAE decoder reconstructs pixels.

## Sora's reported recipe

From OpenAI's February 2024 technical blog post (no formal paper):

- Unified training: images, videos of various lengths/resolutions.
- Re-captioned training data using a strong VLM (GPT-4).
- Scaled transformers.
- Used "patches" as the token unit across modalities.
- DiT-based, no U-Net.

The notable thing: **scaling + diffusion + transformers** - same recipe as LLMs.

## What works

As of 2026:
- **Short clips (5-10s)** at 720p-1080p with good quality.
- **Photorealistic** humans walking, objects moving.
- **Style transfer**: "in the style of Pixar", etc.
- **Image-to-video**: given a static image, generate motion from it.
- **Camera control**: zoom, pan, dolly.

## What doesn't work reliably

- **Long clips** (1+ min) often lose coherence.
- **Complex physics** (pouring liquid, collisions).
- **Text in videos**.
- **Precise character identity** across long spans.
- **Human hands** in rapid motion.
- **Crowds**: each person's motion is individually coherent, but the ensemble falls apart.

## Cost

Per-second of generated video:
- Sora: $0.10-$0.50 (as of ChatGPT Plus pricing).
- Runway / Pika: similar.
- Self-hosted CogVideoX on rented GPU: ~$0.50 per 5-second clip.

Much more expensive than image generation. Will come down.

## World model hypothesis

Some researchers (including some at OpenAI) argue:

> "A sufficiently capable video generator is implicitly learning the structure of the physical world - object permanence, causality, physics. It's a world simulator."

Strong form: "Sora is a path to AGI."
Weaker form: "Video models learn useful representations of the world."

Empirical evidence is mixed. Video models break physics in revealing ways:
- A person walking behind a tree doesn't always emerge on the other side.
- Drinks don't always flow correctly.
- Reflections show wrong things.

So they've learned *some* physics. Not enough to be called simulators.

## Text-to-video vs image-to-video vs video-to-video

- **Text-to-video**: just prompt. Hardest.
- **Image-to-video**: given a starting frame, animate. Much more controllable.
- **Video-to-video**: given a source video, restyle. Easier still.

Most practical products today use image-to-video or video-to-video.

## Running it yourself

### CogVideoX (open source)

```python
from diffusers import CogVideoXPipeline
pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16
).to("cuda")

video = pipe(
    prompt="A panda eating bamboo in a forest",
    num_inference_steps=50,
    num_frames=49,
).frames[0]

# export as mp4
```

Needs ~24GB VRAM. Takes several minutes per 6-second clip.

### Via API

Runway, Luma, Pika have REST APIs. Easier starting point for most people.

## Key resources

- Sora Technical Report (OpenAI 2024): https://openai.com/research/video-generation-models-as-world-simulators
- Open-Sora: https://github.com/hpcaitech/Open-Sora
- HunyuanVideo paper: https://arxiv.org/abs/2412.03603
- Survey paper "Sora: A Review" (2024).

## Exercises

1. Generate a video via Runway, Pika, or OpenAI Sora (if you have access). Note the quality.

2. Download CogVideoX. Generate one 5-second clip. Observe the compute time.

3. Compare outputs: same prompt, three different models. Rank by quality and cost.

4. Read Sora's technical report. Identify: how many patches per clip, how many training video hours (they don't say publicly).

5. Think: where does video generation go next? My bet is 10+ minute coherent clips by 2027.

## Next

`06_whisper_and_speech_to_text.md` - pivot to audio.
