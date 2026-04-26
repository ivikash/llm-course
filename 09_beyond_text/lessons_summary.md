# Module 9 - The Remaining Lessons (Short Summaries)

The other Module 9 lessons, each with their key ideas. I'll expand any on demand.

## 9.2 - CLIP and Multimodal Embeddings

**Core idea**: train an image encoder and a text encoder jointly, such that matched image-caption pairs produce similar embeddings and mismatched pairs produce dissimilar ones.

**Loss**: contrastive (InfoNCE / NT-Xent). For a batch of N (image, caption) pairs, positive pair is the match, negative pairs are all other combinations. Maximize similarity of positives, minimize of negatives.

**Outputs**: two aligned embedding spaces.

**Applications**:
- **Zero-shot image classification**: encode the image and encode each class name ("a photo of a cat", ...). Pick the closest.
- **Text-to-image guidance** (diffusion models).
- **Image retrieval** (find similar images given text query).
- **Foundation** for most VLMs (LLaVA uses CLIP ViT as visual encoder).

**Paper**: Radford 2021, 4M (image, text) pairs scraped from the web.

**Key code**:
```python
from transformers import CLIPModel, CLIPProcessor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# encode both modalities; compare with cosine similarity
```

Successors: **SigLIP** (Google, better loss), **OpenCLIP** (open-source CLIP at scale).

---

## 9.4 - Text-to-Image in Practice

**Running Stable Diffusion locally**:
```python
from diffusers import StableDiffusionXLPipeline
pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
image = pipe("a cinematic shot of a raccoon at a typewriter").images[0]
```

**Prompt engineering for images**: very different from LLMs. Use concrete adjectives, artist styles, camera settings ("35mm, bokeh"), negative prompts (things to avoid).

**LoRA fine-tuning**: train a small adapter on ~20 images of you, your dog, your product. Community tools: Kohya-ss, diffusers.

**ControlNet**: condition on sketches, depth maps, poses. Adds fine control.

**Inpainting**: mask a region, regenerate. Lets you edit images with text prompts.

**IP-Adapter / ReferenceNet**: generate "in the style of" a reference image.

**Popular tools/UIs**:
- ComfyUI: node-based, most flexible.
- Automatic1111 WebUI: earlier, still popular.
- Fooocus: simpler, opinionated.
- Commercial: Midjourney (Discord), DALL-E 3 (ChatGPT), Ideogram (text in images).

**Current frontier**: Flux.1 (open), SD3.5 (open), DALL-E 3, Midjourney v6, Imagen 3. Quality is converging.

---

## 9.5 - Video Generation (Sora, Veo)

**The challenge**: video = image + time. Much more compute. Temporal coherence is hard (same character across frames, consistent physics).

**Approach**: diffusion in space-time.

**Sora** (OpenAI, 2024): diffusion transformer on space-time "patches" extracted from video. Can generate up to 1 minute of coherent 1080p video.

**Architectures**:
- Space-time patches: treat a 256x256x16 video as a 4D tensor → flatten to tokens → transformer.
- 3D U-Nets: classical approach, CNN in space + time.
- Cascaded models: generate at low-res, upsample with separate models.

**Open alternatives (2024-2026)**: Open-Sora, CogVideoX, HunyuanVideo, Mochi 1. Quality is catching up.

**Commercial**: Sora, Veo (Google), Kling (China), Runway Gen-3, Pika, Luma Dream Machine.

**Limitations**: compute cost (generating 1 minute of 4k video can cost $5-50), long-term coherence, subject consistency across shots.

**World model hypothesis**: some argue that Sora-like models are implicit simulators of physical reality. Controversial - good models break physics on edge cases.

---

## 9.7 - Text-to-Speech

**Evolution**:
1. **Concatenative** (1990s-2000s): splice recorded phonemes. Limited voices.
2. **Parametric** (HMM-based): generate acoustic params. Better.
3. **Tacotron** (2017): end-to-end neural. Text → mel-spectrogram → (vocoder: mel → audio waveform).
4. **WaveNet** (2016) as vocoder: autoregressive pixel-level waveform. Great quality, slow.
5. **HiFi-GAN** (2020) as vocoder: parallel, fast, near-equal quality.
6. **VITS** (2021): unified end-to-end.
7. **VALL-E** (Microsoft 2023): treat TTS as an LM problem - tokenize audio, predict audio tokens from text + speaker prompt.
8. **XTTS v2** (Coqui): open, multilingual, voice cloning.
9. **ElevenLabs** (commercial): gold standard for voice cloning and prosody.

**Voice cloning**: given 5-30 seconds of target speaker, generate new audio in their voice. Controversial (deepfakes, fraud).

**Open models to try**:
- XTTS v2 (multilingual, voice cloning).
- Fish Speech (2024, very good).
- MetaVoice.
- F5-TTS.

**Commercial APIs**: ElevenLabs, PlayHT, Azure Speech, Google Cloud TTS, OpenAI TTS.

**Realtime + expressive**: The frontier is low-latency streaming with emotion/prosody control. GPT-4o-voice and Gemini Live exemplify.

---

## 9.9 - Audio Understanding LLMs

**Beyond transcription**: models that understand speech prosody, music, environmental sounds - without converting to text first.

**Architectures**:
- **Audio tokenizer** (e.g., EnCodec, SoundStream): audio → sequence of discrete tokens.
- **LLM backbone**: processes audio tokens alongside text tokens.
- **Output head**: depending on task, predicts text or audio tokens.

**Examples**:
- **AudioPaLM**: Google, unified speech + text.
- **Qwen-Audio**: Alibaba, audio understanding.
- **SALMONN**: Tsinghua, open research.
- **GPT-4o voice**: closed, 300ms latency bidirectional voice.

**Applications**:
- Voice assistants (Siri's AI era, Alexa+).
- Emotion detection in calls.
- Podcast understanding / summarization.
- Music understanding (genre, mood, transcription).
- Real-time translation.

**Benchmark**: MMAU (Massive Multi-task Audio Understanding). Still an emerging area.

---

## 9.10 - World Models and Simulators

**Hypothesis**: sufficiently good video/multimodal generators are implicitly learning physics, causality, object permanence. Hence "world models."

**Examples**:
- **Genie** (DeepMind 2024): given an image, generates playable 2D game frames controlled by a latent action. Consistent object dynamics.
- **Sora**: OpenAI described as a "world simulator" hypothesis.
- **Dreamer** family: RL world models used for agent planning.
- **UniSim** (Stanford): attempts to generate physically-consistent videos.

**Why care**:
- Robotics: simulate environments for training.
- Games: infinite procedural content.
- AGI-ish arguments: world-modeling is considered necessary for general intelligence.

**Limitations**: current models break physics on edge cases; long-horizon planning is weak.

**Research frontier**. Not ready for production.

---

## 9.11 (capstone) - Multimodal Experiment

Run these, each in 30 min:

1. **Whisper**: transcribe any audio (phone recording, YouTube download).
2. **Stable Diffusion**: generate 5 images from prompts you care about.
3. **LLaVA or GPT-4V**: show an image, ask detailed questions.
4. **XTTS or ElevenLabs**: generate synthesized speech in your voice (if you can clone).
5. **ComfyUI or Automatic1111**: try inpainting or controlled generation.

Write a short report:
- Cost per output.
- Quality impressions.
- Which capabilities surprised you?
- Which failed?

---

## What this module unlocks

After Module 9 you can read and understand:
- Most vision-LLM papers (LLaVA, Qwen-VL, GPT-4V).
- Diffusion papers (DDPM, Stable Diffusion, Flux).
- Speech papers (Whisper, VALL-E).
- Video generation papers (Sora, DiT).
- World model papers (Genie, Dreamer).

Combined with Modules 0-8, you are fluent across the whole "generative AI" landscape, not just text.

## Next

Module 10 - Agentic AI. Tool use, planning, LLM agents.
