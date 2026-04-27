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

## Visualize this

**Whisper's pipeline**:

```
  audio waveform (16 kHz sampling)        "This is a test."
  ┌────────────────────────┐                      ↑
  │ ╱╲╱╲╱╲╲╱╲╱╲╱╲╲╱╲     │                      │
  │  (just numbers over    │        ┌─────────────┐
  │   time, 16000/sec)     │        │ Decoder      │
  └─────────┬──────────────┘        │ (transformer │
            │                        │  that outputs │
            ▼                        │  text tokens) │
  ┌─────────────────────┐            └─────▲────────┘
  │ Convert to log-mel   │                  │ cross-attention
  │  spectrogram         │                  │
  │  (2D "picture" of    │    ┌─────────────┴──────┐
  │   frequencies over   │    │ Encoder            │
  │   time)              │───▶│ (transformer       │
  └─────────────────────┘    │  reading audio     │
                             │  spectrogram)      │
                             └────────────────────┘
```

Encoder-decoder transformer, same as 2017 paper, applied to audio→text.

**Mel-spectrogram: audio as an "image"**:

```viz
{"viz": "mel_spectrogram"}
```

Switch audio types. See how Whisper turns sound into a 2D picture: horizontal = time, vertical = frequency, color = energy. A chirp becomes a diagonal line. Speech has harmonic structure. Music has persistent bass + melody. Whisper's transformer processes this picture.

```
  Audio (1D over time):
  ╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲

  Process with FFT over sliding windows:

  time steps →
                  t=0   t=1   t=2   t=3   t=4   t=5   ...
                ┌─────┬─────┬─────┬─────┬─────┬─────┐
  low freq     │  .   │  .   │  .   │  ●   │  ●   │  .   │
  mid freq     │  ●   │  ●   │  .   │  .   │  ●   │  ●   │
                │  ●   │  ●   │  ●   │  ●   │  ●   │  ●   │  (each cell:
  mid freq     │  .   │  ●   │  ●   │  ●   │  .   │  .   │   energy at
                │  .   │  ●   │  ●   │  .   │  .   │  .   │   that freq
  high freq    │  .   │  .   │  ●   │  .   │  .   │  .   │   at that time)
                └─────┴─────┴─────┴─────┴─────┴─────┘

  80 frequency bins × ~3000 time steps for 30 seconds of audio
  → a 2D array the transformer can treat like patches
```

**Whisper sizes**:

```
  Model     Params    VRAM      Speed       Quality
  ────────  ────────  ────────  ──────────  ─────────
  tiny       39M      <1GB       very fast   meh
  base       74M      <1GB       fast         okay
  small      244M     2GB         medium       good (sweet spot)
  medium     769M     5GB         slower       very good
  large-v3   1.5B     10GB        slow         excellent

  For most uses: 'small' or 'medium'.
  For production quality: 'large-v3'.
  For real-time on laptop: 'base' or 'small' + faster-whisper lib.
```

**Whisper tasks (one model, many jobs)**:

```
  Special tokens tell Whisper what to do:

  <|transcribe|>   → transcribe speech to text
  <|translate|>    → translate any language to English
  <|en|><|fr|>...  → language tokens (identifies input language)
  <|notimestamps|> → just text, no timestamps
  <|0.00|>...      → include timestamps

  Example calls:
    whisper audio.mp3                           → English transcription
    whisper audio.mp3 --language Chinese         → Chinese transcription
    whisper audio.mp3 --task translate           → English translation
```

**Latency breakdown (real-time use)**:

```
  Streaming transcription target: 300ms latency

  Components:
    Audio capture:          ~50ms (sampling, buffering)
    Preprocessing (mel):    ~30ms
    Encoder pass:           ~80ms (depends on chunk length)
    Decoder pass:           ~50-200ms (autoregressive)
    Postprocessing:          ~20ms
    ─────────────────────── 230-380ms total

  Libraries optimized for speed:
    whisper.cpp             C++ port, runs on CPU, fast on Mac
    faster-whisper          CTranslate2 backend, 2-4× faster
    insanely-fast-whisper    chunked parallelism, HF
    VoxWhisper               real-time streaming variants
```

**The WisprFlow use case (desktop voice input)**:

```
  WisprFlow ≈ Whisper running locally + UX for speech-to-text everywhere.

  Typical flow:
    1. user holds keyboard shortcut
    2. app captures audio
    3. Whisper transcribes offline
    4. transcribed text pasted wherever cursor is

  Why it works:
    - 100% offline → privacy
    - Whisper is free (MIT license)
    - Near-human accuracy
    - Multilingual by default
```

**Running it** (30 seconds):

```bash
pip install openai-whisper
whisper my_audio.mp3 --model small
# outputs: my_audio.txt, .srt, .vtt, .tsv
```

Or Python:
```python
import whisper
model = whisper.load_model("small")
result = model.transcribe("my_audio.mp3")
print(result["text"])
```

That's a production-ready speech-to-text system in 3 lines.

**Faster alternative**:

```python
from faster_whisper import WhisperModel
model = WhisperModel("small", compute_type="int8")
segments, info = model.transcribe("my_audio.mp3", beam_size=5)
for segment in segments:
    print(f"[{segment.start:.2f}s] {segment.text}")
```

2-4× faster than openai-whisper. Runs on CPU decently.

**Typical accuracy**:

```
  Benchmark           large-v3    small     tiny
  ────────────────────  ──────     ─────     ────
  LibriSpeech clean     3.1%       6.8%      12%   (WER, lower better)
  Common Voice English  8.5%       14.0%    22%
  Accented English       6-12%     11-20%    20-35%
  Noisy audio            10-20%    15-30%    30-50%
  Non-English           varies    varies    bad

  Modern phones: Siri/Google have 95%+ accuracy.
  Whisper approaches that at the 'small' model size.
```

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
