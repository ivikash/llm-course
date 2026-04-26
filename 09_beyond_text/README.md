# Module 9 - Beyond Text: Images, Audio, Video, Multimodal

The original course focused on text-only LLMs (nanoGPT and nanochat). The modern AI field is **multimodal** - models see, hear, speak, generate images, understand video. This module extends your mental model to cover these.

The good news: **the transformer is the workhorse for all of them.** Once you understand transformers (Module 3), picking up images/audio/video is about domain-specific tokenization and sometimes domain-specific heads - not a fundamentally different architecture.

## Lessons

### `01_vision_transformers_vit.md`
- How to turn an image into tokens (patches, projected to vectors).
- Vision Transformer (ViT) architecture.
- Why "images are just sequences of patches".
- Comparison to CNNs: what each is good at.
- Key papers: ViT (Dosovitskiy 2020), DINO, MAE.

### `02_clip_and_embeddings.md`
- CLIP: train image encoder + text encoder so matched pairs have similar embeddings.
- Contrastive learning.
- Applications: zero-shot image classification, image retrieval, text-to-image guidance.
- Key paper: CLIP (Radford 2021).

### `03_diffusion_models.md`
- The forward process: add noise gradually.
- The reverse process: learn to denoise.
- U-Net backbone.
- Latent diffusion (Stable Diffusion): diffuse in compressed latent space, not pixel space.
- Classifier-free guidance.
- Schedulers (DDPM, DDIM, etc.).
- Key papers: DDPM (Ho 2020), Stable Diffusion (Rombach 2022).

### `04_text_to_image_dalle_midjourney_stable_diffusion.md`
- How text prompts condition image generation.
- Cross-attention between text embeddings and image latents.
- Running Stable Diffusion locally (easy, ~4GB VRAM in fp16).
- Where things have gone: DALL-E 3, Flux, Imagen, SDXL.

### `05_video_generation_sora.md`
- Extending diffusion to video: space + time.
- Video diffusion transformers (DiT, Sora-style).
- Temporal consistency tricks.
- Why video is so expensive.
- Key work: Sora (OpenAI), Veo (Google), Kling, Runway, Pika.

### `06_whisper_and_speech_to_text.md`
- Audio as mel-spectrogram "images".
- Whisper: a seq2seq transformer for speech → text.
- Training data (600K hours).
- Multilingual support.
- Real-time variants (distilled Whisper, faster-whisper, Whisper.cpp).
- WisprFlow-style on-device.
- Key paper: Whisper (Radford 2022).

### `07_text_to_speech_tts.md`
- Spectrogram generation (Tacotron-style).
- Vocoder (mel → audio): WaveNet, HiFi-GAN.
- End-to-end text-to-speech: VITS, VALL-E, XTTS, ElevenLabs.
- Voice cloning (few-shot speaker adaptation).

### `08_multimodal_llms_gpt4v_claude_gemini.md`
- Vision-Language Models (VLMs): LLM backbone + vision encoder + a cross-modal projection.
- Training pipeline: pretrained text model + pretrained vision encoder + adapters + fine-tuning.
- Examples: LLaVA (open), Idefics, GPT-4V, Claude 3 vision, Gemini.
- Video understanding: extensions of VLMs.

### `09_audio_understanding_llms.md`
- Audio → text via Whisper, but also direct audio-in models.
- AudioPaLM, GPT-4o voice, Qwen-Audio, SALMONN.
- Emotion, prosody, speaker identification.
- Voice-to-voice conversation architectures.

### `10_world_models_and_simulators.md`
- Genie (DeepMind): "2D game from an image."
- Sora as a "world simulator" hypothesis.
- Robotics-oriented world models.
- Why this matters for the future of AI.

### `capstone_multimodal_experiment.md`
- Run Whisper on an audio file of yours.
- Run Stable Diffusion for 5 prompts.
- Run LLaVA on 5 images.
- Write up: capabilities, failures, latencies, costs.

## Reading list for Module 9

Required:
- Dosovitskiy et al 2020, "An Image is Worth 16x16 Words" (ViT).
- Radford et al 2021, "Learning Transferable Visual Models From Natural Language Supervision" (CLIP).
- Ho et al 2020, "Denoising Diffusion Probabilistic Models" (DDPM).
- Rombach et al 2022, "High-Resolution Image Synthesis with Latent Diffusion Models" (Stable Diffusion).
- Radford et al 2022, "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper).
- Liu et al 2023, "Visual Instruction Tuning" (LLaVA).

Nice-to-have:
- "Scalable Diffusion Models with Transformers" (DiT).
- "Sora: A Review of Background, Technology, Limitations..." (summary papers on Sora-style video).
- "Emu Video" and "Make-A-Video" papers on open video generation.

## Philosophy: transformer everywhere

The high-level story:

1. **Text**: tokens → embeddings → transformer → next token.
2. **Images**: patches → embeddings → transformer → class label / caption / next patch.
3. **Audio**: spectrogram frames → embeddings → transformer → text / audio frames.
4. **Video**: spatiotemporal patches → embeddings → transformer → next frame / caption / action.
5. **Multimodal**: mixed tokens from all above → shared transformer → anything out.

The transformer doesn't care what's in the tokens. That's why it scales to all modalities.

## What this module does NOT cover deeply

- Classical CV (SIFT, HOG, hand-engineered features).
- Signal processing fundamentals (Fourier transforms, mel filters deep theory).
- Robotics stack (ROS, simulators).
- 3D rendering, meshes, point clouds (NeRF, 3D Gaussian Splatting are huge but adjacent).

For these, specialist courses exist. This module gives you LLM-adjacent multimodal competence, which is what 80% of modern AI roles need.

## After Module 9

Combined with Modules 0-8, you now have a working mental model of:
- Text generation (LLMs)
- Image understanding (ViT, CLIP)
- Image generation (diffusion)
- Video generation (video diffusion)
- Speech recognition (Whisper)
- Speech synthesis (TTS)
- Multimodal understanding (VLMs)

Which is basically all of "AI" as the public understands it. You can read any modern paper.

Module 10 extends further: Agentic AI (tool use, planning, agents).
