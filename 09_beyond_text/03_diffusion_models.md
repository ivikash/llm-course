# 9.3 - Diffusion Models

Text generation (GPT) and image generation (Stable Diffusion, Midjourney) work very differently. Text is generated left-to-right, one token at a time. Image generation is **iterative denoising**: start with pure noise, gradually remove noise until an image emerges.

This lesson explains diffusion from scratch. It's beautifully simple math once you see it.

## The core idea

### Forward process: gradually add noise

Take a clean image `x_0`. Add a little Gaussian noise. Get `x_1`. Add more. Get `x_2`. ... After T=1000 steps, `x_T` is pure noise, indistinguishable from random.

Mathematically:
```
x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
```

Where `alpha_t` decreases from ~1 (clean) to ~0 (all noise) over T steps. The schedule is fixed, not learned.

### Reverse process: gradually remove noise

Here's the magic. We train a neural network to **predict the noise** at each step:

```
noise_predicted = network(x_t, t)
```

Given a noised image `x_t` and timestep `t`, predict what noise was added. This is the entire training objective! Cross-entropy-like simple.

At inference time:
1. Start with pure noise `x_T`.
2. At each step, ask the network: "what noise is in here?"
3. Subtract (scaled) predicted noise. Get `x_{t-1}` (slightly less noisy).
4. Repeat for T steps.
5. Output is a clean image.

## Why this works

At each denoising step, the network produces a small correction. Over 1000 (or with fancy samplers, 50-100) steps, these small corrections accumulate into a realistic image.

The reason a single small step doesn't overwhelm the model: each step only needs to remove a tiny bit of noise. An easier problem than "generate an image from scratch".

## The network architecture: U-Net (often)

For pixel diffusion, the noise-prediction network is typically a **U-Net** - a CNN with skip connections that downsamples then upsamples the image. Symmetric shape = "U".

Modern approaches replace U-Net with transformers (DiT - Diffusion Transformers). Sora, Stable Diffusion 3 use DiTs.

## Latent diffusion

Pixel diffusion is expensive. A 512×512 RGB image has 786K pixels. Diffusing in pixel space means the U-Net is huge.

**Latent Diffusion** (Rombach 2022, the paper behind Stable Diffusion):

1. Train a **VAE** (Variational Autoencoder) first: maps a 512×512 image to a tiny 64×64×4 **latent** representation, and back.
2. Do diffusion in the latent space.
3. When sampling, run diffusion in latent space, then decode the latent to a pixel image with the VAE.

Result: 64×64 latent instead of 512×512 pixel. 64x fewer elements, orders of magnitude cheaper.

Stable Diffusion 1.x/2.x/XL all work this way.

## Text-to-image: conditioning

To generate "a cat wearing a hat" (text-conditioned), you:

1. Encode the text with a text encoder (CLIP text encoder, or a larger LM).
2. Feed text embeddings into the denoising network via cross-attention.
3. The denoising network now considers both the current noise `x_t` AND the text embedding.

The rest is the same.

## Classifier-free guidance (CFG)

To make the image match the prompt more strongly:

1. At each step, run the denoiser twice: once with text, once without.
2. Compute the difference (direction pointing toward text).
3. Push the sample further in the "with text" direction.

Controlled by `guidance_scale`. Typical values: 5-15. Higher = more faithful to prompt but can look weird.

## Schedulers / samplers

The "remove noise 1000 times" is overkill. Schedulers cleverly jump steps, getting similar quality in 20-50 steps.

Common ones:
- DDPM (original, 1000 steps).
- DDIM (deterministic, ~50 steps).
- PLMS, DPM-Solver, Euler - various trade-offs.

Modern UIs let you pick a scheduler. For most, DPM-Solver++ in 20-30 steps is near-optimal.

## Running Stable Diffusion

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
).to("cuda")

image = pipe("a cat wearing a hat", num_inference_steps=30).images[0]
image.save("cat.png")
```

~4GB VRAM in fp16. Runs on a laptop GPU.

## Landscape of image generation

| Model | Year | Notable |
|-------|------|---------|
| DALL-E | 2021 | OpenAI's first, discrete tokens |
| GLIDE | 2021 | Classifier-free guidance |
| DALL-E 2 | 2022 | CLIP + diffusion |
| Imagen | 2022 | Google, text-conditioned U-Net |
| Stable Diffusion 1.x | 2022 | Open weights, mainstream breakthrough |
| Midjourney | 2022+ | Closed, highly-tuned |
| Stable Diffusion XL | 2023 | Bigger, better |
| DALL-E 3 | 2023 | Integrated with ChatGPT, prompt adherence |
| Stable Diffusion 3 | 2024 | DiT architecture |
| Flux | 2024 | State of the art open, from Stability AI alumni |

Open-source options are competitive with closed in 2025.

## Diffusion for other modalities

Same idea works for:
- **Audio**: Stable Audio, MusicGen.
- **Video**: Sora, Veo - extend diffusion to space-time patches.
- **3D**: DreamFusion, point clouds, meshes.
- **Molecules**: diffusion models for drug discovery.

Diffusion is a general generative modeling technique, not specific to images.

## Diffusion vs other generative approaches

| Approach | Examples | Strengths | Weaknesses |
|----------|----------|-----------|------------|
| **Autoregressive (GPT-style)** | DALL-E 1, Parti | Strong text coherence | Slow, serial sampling |
| **GANs** | StyleGAN | Fast sampling | Mode collapse, training instability |
| **VAEs** | NVAE | Clean latent space | Blurry outputs |
| **Diffusion** | Stable Diffusion | Great quality, stable training | Slow sampling (need 20-50 steps) |
| **Flow matching** | Flux | Conceptually cleaner than diffusion | Newer, less mature |

Diffusion dominates in 2022-2025. Flow matching may surpass it.

## Key papers

- **DDPM** (Ho 2020): the foundational diffusion paper.
- **Improved DDPM** (Nichol 2021): better schedules.
- **Classifier-free guidance** (Ho 2022).
- **Latent Diffusion** (Rombach 2022): Stable Diffusion.
- **DiT** (Peebles 2022): transformer backbone.
- **EDM** (Karras 2022): unified formulation.

## Exercises

1. Read the DDPM abstract + algorithm 1 + 2. Trace the math.

2. Run Stable Diffusion locally. 10 prompts. Vary guidance_scale (3, 7, 12). Compare.

3. In Python:
   ```python
   import torch
   noise = torch.randn(1, 3, 512, 512)
   # that's the starting point of generation. Random noise. Somehow, after denoising, becomes an image.
   ```

4. Read about LoRA for Stable Diffusion. A popular community workflow for fine-tuning SD on your own images (style, character, etc.).

## Next

`04_text_to_image_dalle_midjourney_stable_diffusion.md` - how prompts become images, practically.
