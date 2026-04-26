# 9.6 - Whisper and Speech-to-Text

Whisper is OpenAI's open-weight speech-recognition model, released September 2022. It quietly revolutionized speech-to-text:
- Near-human English transcription.
- Works on 100+ languages.
- Completely open weights.
- Weighs about 1.5 GB (large-v3).
- Runs on a laptop.

This lesson: how it works, how to use it, what's evolved since.

## Architecture: a transformer encoder-decoder

Whisper is basically a **seq2seq transformer** - similar in structure to the original "Attention Is All You Need" paper.

**Input**: 30 seconds of audio, converted to a mel-spectrogram.
**Output**: text tokens.

```
Audio (raw waveform)
   ↓ (preprocess)
Mel-spectrogram (80 bins × ~3000 time steps for 30 sec)
   ↓ (encoder - transformer)
Audio embeddings (latent)
   ↓ (decoder - transformer with cross-attention to encoder)
Text tokens (autoregressive)
```

No RNN, no CNN (mostly). Just transformers.

## Mel-spectrogram: audio as a "picture"

Audio is hard to process directly. Whisper (and most speech models) converts audio to a 2D spectrogram first:

1. Split audio into tiny windows (~25ms each).
2. For each window, compute the spectrum (frequency content) via Fourier transform.
3. Warp frequencies to the mel scale (perceptually uniform).
4. Log of intensity.

Result: a 2D grid where x = time, y = frequency bin, value = log-intensity.

Now the transformer treats it like an image: patches, embed, feed through attention.

## Sizes

Whisper comes in 5 sizes:

| Model | Params | Approx VRAM | Languages |
|-------|--------|-------------|-----------|
| tiny | 39M | 1 GB | multilingual |
| base | 74M | 1 GB | multilingual |
| small | 244M | 2 GB | multilingual |
| medium | 769M | 5 GB | multilingual |
| large-v3 | 1.55B | 10 GB | multilingual |

For most use, `small` or `medium` is the sweet spot. `large-v3` is best quality.

## Training data

600,000 hours of audio-text pairs, scraped from the internet. Mostly weak supervision (YouTube captions, podcast transcripts, etc.) - hence "Robust Speech Recognition via **Large-Scale Weak Supervision**" in the paper title.

The key insight: scale of weak supervision beats carefully-curated small datasets. Standard story in 2020s ML.

## What Whisper does beyond transcription

- **Language identification** (99 languages).
- **Translation** to English (from any of those languages, via a special token).
- **Voice activity detection** (when there's speech vs silence).
- **Timestamps** (token-level or segment-level).

All from a single model, trained multitask.

## Using Whisper

### Via HuggingFace

```python
from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")
result = asr("audio.mp3")
print(result["text"])
```

### Via the official OpenAI package

```bash
pip install openai-whisper
whisper audio.mp3 --model small
```

### Faster alternatives

- **`faster-whisper`** (CTranslate2 backend): 2-4x faster, same accuracy.
- **`whisper.cpp`**: pure C++, runs on CPU/laptop. Good for edge devices.
- **`insanely-fast-whisper`** (HF): chunked parallelism.

For real-time streaming: use `whisper-live` or similar.

## Limitations

- **Hallucination in silence**: on long silent sections, Whisper sometimes generates text (e.g. "thanks for watching"). Confidence scoring doesn't catch this.
- **Non-standard accents**: performs worse.
- **Overlapping speakers**: single-speaker assumption; doesn't separate.
- **Music**: mostly trained on speech, struggles with sung lyrics.
- **Very long audio**: input window is 30 seconds. For longer, you chunk and stitch.

## What's beyond Whisper

**Whisper is pretty much solved transcription for most use cases.** Newer efforts focus on:

- **Distillation** (smaller, faster versions): distil-whisper.
- **Real-time, low-latency**: streaming ASR with KV cache.
- **Multimodal context**: knowing who's speaking (diarization), combined with visual cues.
- **End-to-end voice assistants**: GPT-4o, Gemini Live don't transcribe then process; they process audio directly.

## Commercial alternatives

- **AssemblyAI**: strong API, speaker diarization.
- **Deepgram**: real-time streaming, enterprise.
- **Google Cloud Speech-to-Text**: widely deployed.
- **Amazon Transcribe**: widely deployed.
- **WisprFlow**: desktop-integrated voice-to-text, leveraging Whisper.

For personal/research: Whisper is free and excellent. For production: consider if you need features Whisper lacks.

## Key paper

Radford et al. 2022: ["Robust Speech Recognition via Large-Scale Weak Supervision"](https://arxiv.org/abs/2212.04356). Read the intro and experimental setup. Scalable-weak-supervision template, applicable broadly.

## Direct-audio-in LLMs (new frontier)

Whisper → text → LLM is a chain. Direct-audio-in models skip the transcription:

- **AudioPaLM** (Google): audio tokens directly to an LLM.
- **Qwen-Audio** (Alibaba): multimodal audio understanding.
- **GPT-4o** (OpenAI): audio in → audio out, ~300ms latency.
- **SALMONN** (Tsinghua): open research variant.

These avoid transcription-loss and can capture prosody, emotion, etc. The future of voice AI.

## Exercises

1. Install Whisper. Transcribe a 2-minute clip of yourself speaking:
   ```bash
   whisper my_audio.mp3 --model small
   ```

2. Try faster-whisper for speed comparison.

3. Try transcribing a video in a non-English language. Whisper auto-detects.

4. Try the `--task translate` flag to translate non-English speech to English text.

5. Read the Whisper paper's Figure 2 (the architecture). Notice: encoder attention is bidirectional on audio, decoder attention is causal on text - just like the original transformer.

## Next

`07_text_to_speech_tts.md` - the reverse direction: text → speech.
