# 9.7 - Text-to-Speech (TTS)

Reverse of Whisper: text in, audio out. TTS is another field where transformers now dominate, and the quality has become indistinguishable from human speech for short clips.

## The core problem

Given a string like "Hello, how are you?", produce a waveform of audio that sounds like someone saying it.

Two natural ways to split the problem:

### Option A: two stages (classical neural TTS)

1. **Acoustic model**: text → mel-spectrogram (how loud at each frequency over time).
2. **Vocoder**: mel-spectrogram → audio waveform (~16,000+ samples per second).

Examples:
- Acoustic: Tacotron, FastSpeech, Glow-TTS.
- Vocoder: WaveNet, HiFi-GAN, BigVGAN.

### Option B: end-to-end

Just: text → audio tokens → waveform, one model.

Examples: VITS, VALL-E, XTTS, Style TTS 2.

End-to-end has become the default post-2022.

## Audio representation

Audio is a very long 1D signal. 16 kHz sampling = 16,000 values per second. A 10-second clip is 160,000 floats.

Instead of predicting raw samples, modern TTS predicts intermediate representations:

### Mel-spectrograms
"Audio as a 2D picture." Time × frequency × intensity. ~80 frequency bins, ~80 time steps per second. Much more compact than raw audio.

### Neural audio codecs (EnCodec, SoundStream, DAC)
Compress raw audio into a sequence of discrete tokens (like text tokens). 24 kHz audio → ~75 tokens per second, vocab size 1024. **Treats audio as a language.**

This is the key unlock for modern TTS: if audio is tokens, an LLM can generate them.

## Landmark approaches

### Tacotron 2 (2017)

Text → mel-spectrogram via seq2seq with attention. Vocoder (WaveNet originally, HiFi-GAN later) converts mel to audio.

### WaveNet (2016, vocoder)

Autoregressive: predict each audio sample conditioned on previous samples. High quality, very slow. Mostly obsolete now as a vocoder.

### HiFi-GAN (2020, vocoder)

GAN-based, parallel, fast. Near-human quality. Still widely used.

### VITS (2021)

End-to-end: text → audio in one model, via variational autoencoder + normalizing flow + adversarial loss. First open end-to-end TTS that sounded truly natural.

### VALL-E (Microsoft, 2023)

Reframes TTS as **language modeling**:
1. Encode text to phonemes.
2. Encode audio to discrete tokens via EnCodec.
3. Train an LLM: given phonemes + 3-second speaker prompt, generate audio tokens.
4. Decode audio tokens back to waveform.

Voice cloning from just 3 seconds of reference audio. Revolutionary.

### XTTS v2 (Coqui, 2023)

Open-source voice cloning, multilingual. Similar architecture family. Fine to run locally.

### Style TTS 2 (2023)

Open-source, high quality, style control. Slightly old but still competitive.

### Fish Speech (2024)

Open, large-scale voice cloning. One of the best open models.

### F5-TTS (2024)

Fast, diffusion-based. Non-autoregressive.

### MetaVoice (2024)

Emotion and prosody control. Open.

## Commercial APIs

| Service | Strengths | Weaknesses |
|---------|-----------|------------|
| **ElevenLabs** | Gold standard for voice cloning, emotion, ~30 languages | Expensive |
| **OpenAI TTS** (tts-1, tts-1-hd) | Fast, cheap | Fewer voice options, no cloning |
| **Azure Neural TTS** | Many voices, enterprise-ready | Less natural than ElevenLabs |
| **Google Cloud TTS** | Similar | Similar |
| **PlayHT** | Voice cloning | Like ElevenLabs |
| **Cartesia (Sonic)** | Ultra-low-latency streaming (~75ms) | New |

As of 2026, ElevenLabs is the default for production voice cloning; Cartesia for real-time agents.

## Running TTS yourself

### OpenAI API

```python
from openai import OpenAI
client = OpenAI()
response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Hello, how are you today?"
)
response.stream_to_file("output.mp3")
```

~$0.015 per 1000 characters. Cheap.

### XTTS v2 (local)

```python
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
tts.tts_to_file(
    text="Hello world",
    speaker_wav="voice_sample.wav",
    language="en",
    file_path="output.wav",
)
```

5GB VRAM. Runs in real-time on a consumer GPU. Clone voice from 10-30 seconds of reference audio.

### ElevenLabs API

```python
from elevenlabs import generate, play
audio = generate(text="Hello", voice="Rachel")
play(audio)
```

## Voice cloning ethics

With ~30 seconds of your voice, a modern model can imitate you convincingly.

Risks:
- **Fraud**: fake phone calls from "your boss" / "your mom".
- **Defamation**: fake audio of you saying things.
- **Content saturation**: YouTube / TikTok flooded with AI-voiced content.

Responses:
- Most commercial APIs require consent documents for voice cloning of real people.
- Some add inaudible watermarks to generated audio.
- Detection models exist (Resemble's AntiFake, etc.) - arms race.

Use cloning ethically. Ask permission. Disclose when AI-voiced.

## Streaming and real-time

For voice agents, you need low latency:

- First-token latency: ~100-300ms ideal.
- Streaming: start speaking as soon as first chunk is ready.
- Interruption handling: stop when user interrupts.

Cartesia's Sonic: ~75ms time-to-first-audio. Best-in-class.

OpenAI's real-time API: bidirectional audio streaming for GPT-4o voice.

## Prosody, emotion, style

Current frontier:
- **Prosody control**: "say this loudly / softly / as a question"
- **Emotion**: "sad" / "excited" / "angry"
- **Style transfer**: speak in the style of a reference clip
- **Speaking rate**: adjustable per-phoneme

Emerging via:
- Tags in text ("[angry] I can't believe this!")
- Style reference audio
- Direct emotion labels

ElevenLabs v3 leads commercially; open models catching up.

## End-to-end voice agents

TTS + ASR + LLM + orchestration = voice assistant.

Traditional pipeline:
```
microphone → ASR (Whisper) → LLM (GPT-4) → TTS (ElevenLabs) → speaker
```

Latency: ~1-3 seconds per turn. Disjointed.

Modern (GPT-4o, Gemini Live):
```
audio in → multimodal LLM → audio out
```

One model handles everything. ~300ms latency. Can interrupt, laugh, change tone.

The architectural shift for voice is happening 2024-2026.

## Key papers

- WaveNet (van den Oord 2016): https://arxiv.org/abs/1609.03499
- Tacotron 2 (Shen 2017): https://arxiv.org/abs/1712.05884
- VITS (Kim 2021): https://arxiv.org/abs/2106.06103
- VALL-E (Wang 2023): https://arxiv.org/abs/2301.02111
- NaturalSpeech 3 (Microsoft 2024): https://arxiv.org/abs/2403.03100

## Visualize this

**The two-stage TTS architecture (classical)**:

```
  text: "Hello world"
       │
       ▼
  ┌─────────────────┐
  │ Acoustic model   │  (Tacotron / FastSpeech / VITS)
  │  text → spectrogram│  predicts mel-spectrogram from text
  └─────────┬───────┘
            │
            ▼
  mel-spectrogram (2D picture of sound: freq × time)
            │
            ▼
  ┌─────────────────┐
  │ Vocoder          │  (WaveNet / HiFi-GAN)
  │  spec → waveform │  converts 2D spec to 1D audio
  └─────────┬───────┘
            │
            ▼
  audio waveform (playable sound)
```

**End-to-end TTS (modern)**:

```
  text: "Hello world"
       │
       ▼
  ┌────────────────┐
  │ Unified model   │  (VITS / VALL-E / XTTS / F5-TTS)
  │ text → waveform  │  skip the intermediate spectrogram
  └────────┬────────┘
           │
           ▼
  audio waveform

  Simpler to train. Often higher quality.
```

**VALL-E: TTS as language modeling**:

```
  Key insight: if audio → discrete tokens (via EnCodec),
  then TTS becomes "predict the next audio token" (like GPT!).

  Training:
    (text prompt + 3s speaker sample) → produces audio tokens
                                         matching the speaker's voice

  The model:
    ┌─────────────────────┐
    │ Autoregressive LM    │
    │ input:                │
    │   text tokens         │
    │   + speaker audio     │
    │   tokens              │
    │ output:                │
    │   audio tokens        │
    │   matching speaker    │
    └─────────────────────┘

  At inference: decode audio tokens → waveform via EnCodec decoder.

  Result: voice cloning from 3 seconds of sample audio.
```

**Voice cloning pipeline (XTTS v2)**:

```
  Step 1: speaker reference
  ┌──────────────────┐
  │  reference.wav    │  10-30 seconds of target speaker
  │  "Hi, my name is Bob." │
  └────────┬─────────┘
           │
           ▼
     extract speaker embedding (256-d fingerprint)

  Step 2: TTS with cloned voice
  ┌──────────────────┐
  │  text: "Hello!"  │ + speaker embedding
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  XTTS model       │
  └────────┬─────────┘
           │
           ▼
  audio in Bob's voice saying "Hello!"
```

**The 2026 TTS quality landscape**:

```
  Quality     Cost/char    Cloning?   Runs locally?
  ─────────   ──────────    ─────────  ─────────────
  ElevenLabs   $$$           yes         no (API)
  OpenAI TTS   $              no (preset voices only)  no (API)
  PlayHT       $$            yes         no (API)
  Cartesia     $$$           yes         no (API)
  XTTS v2      free          yes         yes (5GB VRAM)
  Fish Speech  free          yes         yes
  F5-TTS       free          yes         yes (newer, faster)
  MetaVoice    free          yes         yes

  For best quality: ElevenLabs
  For best open: XTTS v2 / Fish Speech / F5-TTS
  For simplest: OpenAI TTS API
```

**Running it** (OpenAI API):

```python
from openai import OpenAI
client = OpenAI()

response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",   # or "echo", "fable", "onyx", "nova", "shimmer"
    input="Hello world! This is AI-generated speech.",
)
response.stream_to_file("hello.mp3")
# cost: $0.015 per 1000 characters
```

Or local XTTS:

```python
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
tts.tts_to_file(
    text="Hello world!",
    speaker_wav="my_voice_sample.wav",   # clone from THIS voice
    language="en",
    file_path="output.wav",
)
```

A voice-cloned TTS running entirely on your laptop.

**Latency comparison (streaming)**:

```
  Task: start speaking as soon as possible after request.

  Service       Time-to-first-audio   Notes
  ───────────  ────────────────────   ─────────────────────
  ElevenLabs    ~300ms                  streaming API, good quality
  OpenAI TTS    ~800ms                  streaming + non-streaming
  Cartesia      ~75ms                   fastest known, "Sonic" model
  XTTS (local)  ~500ms on 4090          decent with optimization

  For real-time voice agents: Cartesia / ElevenLabs.
  For pre-recorded content: any of the above.
```

**End-to-end voice conversation architectures**:

```
  Classic pipeline (3 models, high latency):
     mic → ASR (Whisper) → LLM (GPT-4) → TTS (ElevenLabs) → speaker
             ~500ms           ~1000ms        ~300ms            = ~2 seconds

  Modern unified (GPT-4o, Gemini Live):
     mic → multimodal LLM → speaker
                 ~300ms total

  GPT-4o can laugh, sigh, change emotional tone.
  This is enabled because it processes audio directly, not via text.
```

**Voice cloning ethics matrix**:

```
                     OK to clone          Not OK to clone
                     ──────────           ──────────────
  Your own voice      ✓                   (not an issue)
  Consenting person    ✓                   (not an issue)
  Dead public figure   ?                   (legal gray zone)
  Living public figure ✗ (defamation risk) ✓ (with satire clear)
  Random stranger      ✗                   ✗ (illegal in many jurisdictions)

  Commercial use: always get written consent.
  Research: disclose AI use publicly.
  Fraud: always illegal, often felony.
```

**The biggest 2026 TTS trend: expressive control**:

```
  Old TTS:
    "Hello world." → flat robotic voice

  New TTS (ElevenLabs v3, Cartesia):
    "[excited] Hello world!" → enthusiastic tone
    "[sad] Hello world."      → melancholy tone
    "[whispered] Hello world." → whispered
    "[slow, thoughtful] Hello world." → pacing + mood

  Tags or style embeddings control emotion, pacing, emphasis.
  Enabling: audiobooks, podcasts, games, accessibility.
```

## Exercises

1. Generate speech with OpenAI's TTS API. Try 3 different voices.

2. Clone your own voice with XTTS v2 locally. Record 30 seconds of yourself. Generate new speech.

3. Test the limits: have the model read something emotional, something technical, something with foreign names.

4. Compare: generate the same sentence with XTTS v2, OpenAI TTS, and ElevenLabs (free tier). Rank by quality.

5. Think: what applications are now possible that weren't 3 years ago? (Personalized audiobooks, dubbing, accessibility, voice agents.)

## Next

`08_multimodal_llms_gpt4v_claude_gemini.md` - putting it all together.
