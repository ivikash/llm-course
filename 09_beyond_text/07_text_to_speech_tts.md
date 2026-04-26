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

## Exercises

1. Generate speech with OpenAI's TTS API. Try 3 different voices.

2. Clone your own voice with XTTS v2 locally. Record 30 seconds of yourself. Generate new speech.

3. Test the limits: have the model read something emotional, something technical, something with foreign names.

4. Compare: generate the same sentence with XTTS v2, OpenAI TTS, and ElevenLabs (free tier). Rank by quality.

5. Think: what applications are now possible that weren't 3 years ago? (Personalized audiobooks, dubbing, accessibility, voice agents.)

## Next

`08_multimodal_llms_gpt4v_claude_gemini.md` - putting it all together.
