# 9.4 - Text-to-Image in Practice

You understand diffusion (Lesson 9.3). Now use it. This is a hands-on lesson.

## The ecosystem

### Open models you can run locally

| Model | Size | VRAM | License | Notable |
|-------|------|------|---------|---------|
| Stable Diffusion 1.5 | 1B params | 4GB fp16 | CreativeML | The OG. 512×512. Mature. |
| Stable Diffusion 2.1 | 1B | 4GB | CreativeML | Slightly improved 768×768. |
| SDXL 1.0 | 3.5B | 8GB | CreativeML | Big quality jump. 1024×1024. |
| SD3 / SD3.5 | 2-8B | 8-16GB | Stability community | DiT architecture. |
| Flux.1 [dev] | 12B | 24GB | Non-commercial | State of the art open (2024). |
| Flux.1 [schnell] | 12B | 24GB | Apache 2.0 | 4-step fast version. |

### Closed services

| Service | Strengths | Limitations |
|---------|-----------|-------------|
| Midjourney | Artistic, cohesive style | Discord-only UX, no API |
| DALL-E 3 | Prompt adherence, text-in-images | In ChatGPT only |
| Ideogram | Text in images, posters | Smaller community |
| Adobe Firefly | Commercial-safe training data | Less capable |
| Imagen 3 | Google's, strong | Limited access |

## Running Stable Diffusion locally

Simplest path:

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,
).to("cuda")

image = pipe(
    "a cinematic shot of a raccoon at a typewriter, 35mm film",
    num_inference_steps=30,
    guidance_scale=7.5,
).images[0]

image.save("out.png")
```

4GB VRAM. ~3 seconds per image on an RTX 3090.

For SDXL:

```python
from diffusers import StableDiffusionXLPipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")
```

For Flux (needs 24GB or quantization tricks):

```python
from diffusers import FluxPipeline
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
).to("cuda")
image = pipe("a fox in a forest", num_inference_steps=4).images[0]
```

## Prompt engineering for images

Very different from LLM prompting. Rules of thumb:

### 1. Concrete nouns beat abstract concepts
Bad: "something beautiful about nature"
Good: "sunlight filtering through birch trees, morning mist"

### 2. Specify style
- Medium: "oil painting", "photograph", "pencil sketch", "3D render"
- Artist: "in the style of Van Gogh", "by Studio Ghibli"
- Era: "1950s pulp sci-fi cover", "Baroque painting"

### 3. Specify camera / lighting
- "35mm film", "macro lens", "golden hour", "soft rim lighting"
- "low-angle shot", "bokeh background"

### 4. Use quality boosters (model-specific)
SD 1.5: "masterpiece, best quality, highly detailed, 8k, cinematic"
SDXL: quality boosters matter less - prompt naturally.
Flux: prefers natural-language descriptions.

### 5. Negative prompts
Things to avoid: "blurry, distorted, bad anatomy, extra fingers, watermark"
Works in SD / SDXL; Flux uses different mechanism.

## Guidance scale

`guidance_scale=7.5` is default for SD. Effect:
- **3**: loose, interpretive, sometimes weird.
- **7**: balanced.
- **12**: strict adherence, can saturate colors, "fry" images.
- **20**: usually too extreme.

Try: same prompt, scales 3/7/12. Observe the spectrum.

## LoRA fine-tuning

Want the model to generate images **of your dog, your face, your product**? Don't retrain from scratch (impossible). Use LoRA.

1. Collect ~20-50 images of your subject.
2. Caption them ("a photo of sks dog running" where "sks" is a rare token).
3. Train a small LoRA adapter (~100MB) on top of the base model.
4. Use at inference time: base model + LoRA = personalized generation.

Training takes ~30 min to 2 hours on a single GPU.

Tools:
- **Kohya-ss sd-scripts**: community standard for SD/SDXL.
- **diffusers train_text_to_image_lora.py**: HuggingFace's official script.
- **ai-toolkit** (Ostris): modern, supports Flux.

## ControlNet

Problem: diffusion ignores exact composition. You want "a person standing in this exact pose."

**ControlNet** adds a conditioning signal - a sketch, depth map, pose skeleton, edge map, or segmentation map - that the diffusion model conditions on.

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-pose")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")
# pipe takes both a text prompt and a pose-skeleton image
```

Enables: pose transfer, architectural rendering from sketches, consistent-character generation.

## Inpainting

"Mask this region of this image. Regenerate just that region with this prompt."

```python
from diffusers import StableDiffusionInpaintPipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting"
).to("cuda")
image = pipe(prompt="a yellow hat", image=init, mask_image=mask).images[0]
```

Uses: replace objects, extend images (outpainting), remove backgrounds.

## IP-Adapter / ReferenceNet

"Generate an image in the style of this reference image."

Adds visual reference conditioning beyond text. Different from ControlNet - preserves style, not structure.

## Popular UIs

If you want a GUI instead of Python:

- **ComfyUI**: node-based, maximum flexibility. Power-user favorite.
- **Automatic1111 WebUI**: browser UI, rich plugin ecosystem. Older.
- **InvokeAI**: polished, commercial-friendly.
- **Fooocus**: simplest, opinionated defaults. Good for beginners.

ComfyUI is where the community has converged (2024-2025).

## Common failures

- **Hands**: even Flux sometimes gets fingers wrong. Six fingers happens.
- **Text**: models write gibberish text. SDXL is better; Ideogram and DALL-E 3 are best.
- **Reflections / physics**: mirrors show wrong things.
- **Consistent characters**: generating "the same person" across many images is hard. Needs LoRA or IP-Adapter.
- **Compositions with specific counts**: "a room with exactly 5 chairs" often gives 3 or 7.

## Cost

### Self-hosted
- ~$0.0005 per image on an RTX 4090 (electricity cost).
- ~$0.01 per image on cloud A100 (rented).

### API
- DALL-E 3: $0.04-0.08 per image.
- Midjourney: ~$0.01 per image (subscription amortized).
- Stability API: $0.01-0.04 per image.

Self-hosting is cheap if you're generating thousands. APIs win for occasional use.

## Key papers + references

- Stable Diffusion (Rombach 2022): https://arxiv.org/abs/2112.10752
- SDXL: https://arxiv.org/abs/2307.01952
- Flux (blog post, no paper): https://blackforestlabs.ai/announcing-black-forest-labs/
- ControlNet (Zhang 2023): https://arxiv.org/abs/2302.05543
- IP-Adapter (Ye 2023): https://arxiv.org/abs/2308.06721

## Visualize this

**Prompt structure for image generation**:

```
  A good prompt template:

  [subject] + [style] + [composition] + [lighting] + [quality modifiers]

  Example:
    A regal golden retriever wearing a crown ← subject
    oil painting                              ← style
    centered, portrait                         ← composition
    soft warm light                           ← lighting
    masterpiece, highly detailed, 8k          ← quality boost

  → Full prompt:
    "A regal golden retriever wearing a crown, oil painting,
    centered portrait, soft warm light, masterpiece, highly
    detailed, 8k"
```

**The guidance scale axis**:

```
  Same prompt, varying guidance_scale:

  scale 1:   🌫     ←── loose, interpretive, weird
  scale 3:   🐕      ←── somewhat matches
  scale 7:   👑🐕   ←── balanced (most useful)
  scale 12:  👑🐕🎨 ←── strict, can look saturated
  scale 20:  💥    ←── over-fried, artifacts
```

**Negative prompts**:

```
  Problem: model generates weird hands, watermarks, low quality.

  Solution: negative prompt (what NOT to include).

  Negative prompt example:
    "blurry, low quality, distorted, ugly, extra fingers,
     watermark, text, signature, cropped"

  How it works (in SD 1.5 / SDXL):
    The model runs twice per step:
      - with your prompt (positive)
      - with the negative prompt (unconditional replaces this)
    CFG pushes AWAY from negative.
```

**LoRA: fine-tuning on YOUR data**:

```
  Goal: generate images of YOUR face / pet / product / style.

  Step 1: Collect 10-30 training images.
           ┌──────┬──────┬──────┐
           │ face1│ face2│ face3│
           ├──────┼──────┼──────┤
           │ face4│ face5│ face6│
           └──────┴──────┴──────┘
           (varied poses, lighting)

  Step 2: Write captions (use a rare token).
           "a photo of sks person smiling, outdoors"
           "a photo of sks person in profile"
           ...

  Step 3: Train a LoRA adapter (30 min - 2 hr on a GPU).
           Freezes base SD, learns a tiny extra module.

  Step 4: At inference, combine base + your LoRA.
           "sks person riding a dragon, epic fantasy" → your face on a dragon.

  Result: ~100 MB LoRA file you can share, apply to any SD base.
```

**ControlNet: precise composition control**:

```
  Problem: text prompt doesn't specify exact pose / composition.

  Solution: condition on a SECOND input (depth map, pose skeleton, etc.)

  Input 1: text prompt "a dancing figure, oil painting"
  Input 2: pose skeleton:
             ●head
             │
            ●●●   shoulders/arms
           /│\
           │ │
           ● ●    hips/legs
          /│ │\
         ●●  ●●

  → Output: oil painting of a figure IN EXACTLY THAT POSE.

  ControlNet variants:
    - Pose (body skeleton)
    - Canny (edges of reference image)
    - Depth (z-axis map)
    - Scribble (hand-drawn sketch)
    - Segmentation (region labels)
    - Normal map
```

**Running it** (multi-sample exploration):

```python
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

prompt = "a majestic mountain landscape at sunrise, golden hour"
neg = "blurry, low quality, watermark"

# Generate 4 variations
for seed in [42, 7, 123, 999]:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    image = pipe(
        prompt,
        negative_prompt=neg,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=gen,
    ).images[0]
    image.save(f"mountain_{seed}.png")
```

Four images from four seeds. Different compositions. Same theme.

**Cost and time reference**:

```
  Local (your GPU):
    SD 1.5 (512×512):   ~3s per image (4090), $0.001 electricity
    SDXL (1024×1024):    ~6s per image (4090), $0.002
    Flux-dev:            ~15s per image (4090, 24GB VRAM req.)

  API (DALL-E 3):        $0.04-$0.08 per image
  API (Midjourney):      $30/mo subscription, ~200 images/day "allowed"
  API (Replicate):       $0.002-0.01 per image for open models

  Rule: <100 images/month → API.
        >1000 images/month → self-host.
```

**Comparison across prompting style**:

```
  Cheap prompt (SD 1.5):
    "cat"
    → blurry gray cat on nothing specific

  Medium prompt:
    "a fluffy orange cat on a wooden porch"
    → reasonable photo-like image

  Rich prompt:
    "a fluffy orange tabby cat lounging on a sunlit wooden porch,
    shallow depth of field, warm afternoon light, photorealistic,
    national geographic style, Canon 5D, 85mm lens, f/1.8"
    → magazine-quality photograph
```

Your prompt budget is "how many tokens the text encoder will read". For SD: 77 tokens. Be specific but don't waste words.

**Failure modes to expect**:

```
  Hands (still broken in SD 1.5 even in 2026):
    Often 6 fingers, fused fingers, unusual proportions.
    Mitigation: specific hand prompts, ControlNet pose, inpainting.

  Text in image:
    SD 1.5/XL can't write legible text.
    SD 3 and Flux are better.
    Ideogram and DALL-E 3 are best for text-in-images.

  Specific people:
    SD doesn't know your face (unless LoRA'd).
    Celebrities often get "blurred" identity by modern filters.

  Physics:
    "A hand holding a glass of water" often looks wrong.
    Reflections, refractions, shadows often miscomputed.

  Multiple subjects:
    "Two cats" often gives one or three.
    "A cat AND a dog" may fuse them into a chimera.
    Mitigation: regional prompting tools.
```

## Exercises

1. Install diffusers. Run Stable Diffusion 2.1. Generate 10 images with varying guidance_scale. Pick a favorite.

2. Run SDXL if you have 8GB+ VRAM. Compare quality to SD 2.1.

3. Try Flux schnell on Replicate or a hosted service if you don't have 24GB.

4. Pick one LoRA from civitai.com (the community model hub). Load and test it.

5. Set up ComfyUI. Build a simple text-to-image workflow. Observe the node graph.

## Next

`05_video_generation_sora.md` - adding the time dimension.
