# 9.9 - Audio Understanding LLMs

Whisper (Lesson 9.6) transcribes audio to text. But lots of information in audio isn't captured by text: speaker identity, emotion, prosody, music structure, environmental sounds.

**Audio-understanding LLMs** work directly with audio, without transcription as a bottleneck.

## Why "audio in, text out" directly?

Transcription-then-LLM loses information:
- "I'm fine." said happily vs sarcastically vs exhaustedly → same text.
- "Yeah sure..." with hesitation vs confidence → same text.
- Music, sound effects, accents → text can't capture.

Audio-in LLMs preserve this. Applications:
- Voice assistants that detect emotion.
- Podcast analysis (who spoke, how passionately).
- Music understanding (genre, structure, transcription).
- Accessibility (describe a scene's audio for deaf users).
- Language learning (pronunciation feedback).

## Architecture pattern

Analogous to Vision-LLMs (Lesson 9.8):

```
Audio (waveform)
   ↓
Audio encoder (often Whisper-encoder or conformer or CLAP-audio)
   ↓
Audio embeddings
   ↓
Projector (map to LLM embedding space)
   ↓
LLM (with audio embeddings as special tokens alongside text)
   ↓
Text output
```

Key architectural choice: does the LLM see:
1. **Continuous audio embeddings** (LLaVA-style continuous projection), or
2. **Discrete audio tokens** (from a neural codec like EnCodec)?

Both work. Tokens are easier to produce audio output too (end-to-end voice).

## Training pipeline

1. Start with pretrained LLM + pretrained audio encoder.
2. Freeze both.
3. Train projector on paired (audio, text) data to align the spaces.
4. Instruction-tune on "audio + question → answer" pairs.

Similar to LLaVA-style training for images.

## Notable models

### Open research models

- **SALMONN** (Tsinghua 2023): Speech, Audio, Language, Music Open-source Neural Network. Understands speech + music + environmental sound.
- **Qwen-Audio / Qwen2-Audio** (Alibaba 2023-2024): strong, open.
- **AudioGPT** (concept from 2023): chains tools (Whisper + LLM + TTS) rather than end-to-end.
- **LLaSM**: Large Language and Speech Model, open.
- **MooER** (Moore Threads 2024): Chinese-focused.

### Closed / commercial

- **AudioPaLM** (Google 2023): unified speech + text model. Research paper only.
- **GPT-4o** (OpenAI 2024): audio input and output, ~300ms latency, the breakthrough commercial product.
- **Gemini Live** (Google 2024-): multimodal with audio streaming.
- **Claude** (Anthropic): has voice input via API as of 2025.

## What these models can do

- Transcribe (same as Whisper).
- **Answer questions about audio**: "What instrument is playing?" "How many speakers?"
- **Detect emotion and tone**: "Does this person sound happy?"
- **Summarize spoken content** with context about delivery.
- **Identify language and accent**.
- **Describe environmental sounds**.
- **Compare audio clips**: "Are these the same song / speaker?"

## Concrete example: GPT-4o voice

Live demo capabilities:
- Real-time bidirectional voice conversation.
- Laughs and responds to emotional cues.
- Adjusts speech style (slower, more excited, singing).
- Translates on the fly.
- Describes images (combined with vision).
- Handles interruptions.

One unified model. The architecture shift is significant: previously you'd pipe Whisper → GPT-4 → TTS, losing information and adding latency. GPT-4o keeps everything in one context.

## The tokenization question

For bidirectional voice (audio in, audio out):
- Need discrete audio tokens (LLM outputs tokens, decoder converts to audio).
- Neural codecs: EnCodec (Meta), SoundStream (Google), DAC (Descript).
- Typical: 24 kHz audio → ~75 tokens/sec × 4 codebooks = 300 tokens/sec.

At 300 audio tokens/sec, a 1-minute clip = 18,000 tokens. Contextual budget is significant.

## Evaluation

Benchmarks emerging:
- **MMAU** (Massive Multi-task Audio Understanding, 2024): diverse audio tasks.
- **SALMONN eval set**: speech + music + sound.
- **AIR-Bench**: audio reasoning.

Still a young area. Most models report results on custom splits; hard to compare.

## Running one

### Qwen2-Audio (open, local)

```python
from transformers import AutoProcessor, AutoModelForCausalLM
import librosa

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct", torch_dtype=torch.bfloat16
).to("cuda")

audio, sr = librosa.load("audio.mp3", sr=16000)

conversation = [
    {"role": "user", "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What do you hear?"},
    ]},
]
inputs = processor(conversation=conversation, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(outputs[0]))
```

### GPT-4o API

```python
import openai
client = openai.OpenAI()
with open("audio.mp3", "rb") as f:
    audio_data = f.read()

# via realtime API for bidirectional
# or via chat.completions with audio input (if available)
```

## Limitations

- **Long audio**: context budget constrains; most models cap at 30 sec to 10 min.
- **Music transcription**: music-specific models (MusicGen, MusicLM descendants) still better.
- **Acoustic fingerprinting** (Shazam-style): not a focus.
- **Real-time on edge**: models are too big for phones currently.
- **Latency**: even GPT-4o voice has 200-500ms; not quite humanlike turn-taking.

## Why this matters

A shift is happening: voice-first AI.

Before 2024: voice assistants were Siri/Alexa-style - pipeline of ASR → NLU → response. Fragile, slow, limited.

After 2024: end-to-end multimodal with native audio. GPT-4o, Gemini Live. Better, faster, more capable.

Expect: voice agents for customer service, education, therapy, companionship, accessibility. The interface of AI is shifting from "type prompts" to "just talk."

## Key papers

- Qwen-Audio (Chu 2023): https://arxiv.org/abs/2311.07919
- SALMONN (Tang 2023): https://arxiv.org/abs/2310.13289
- AudioPaLM (Rubenstein 2023): https://arxiv.org/abs/2306.12925

## Visualize this

**The shift from pipeline to unified audio LLM**:

```
  Pipeline approach (pre-2024):
  ┌─────┐   ┌───────┐    ┌─────┐    ┌───────┐   ┌─────┐
  │ mic │──▶│ ASR    │───▶│ LLM │───▶│ TTS    │──▶│ spk │
  └─────┘   │(Whisper)│    │(GPT)│    │(Eleven)│   └─────┘
             └───────┘    └─────┘    └───────┘
             500ms        1000ms      300ms     = ~2 sec latency

   Loss:
   - Tone, emotion stripped when → text
   - Pauses, hesitations lost
   - Can't laugh, sigh, emphasize

  Unified approach (GPT-4o, 2024+):
  ┌─────┐   ┌─────────────────────┐   ┌─────┐
  │ mic │──▶│ Multimodal LLM        │──▶│ spk │
  └─────┘   │  audio in → audio out │   └─────┘
             └─────────────────────┘
               ~300ms total latency

   Preserves:
   - Prosody (rising/falling intonation)
   - Emotion (excited, sad, calm)
   - Paralinguistics (laughs, sighs)
   - Speaker characteristics
```

**Audio tokens: making audio "language-like"**:

```
  Raw audio:
  [waveform samples at 16000 Hz]
  → 16000 numbers per second
  → way too many to "tokenize" directly

  Solution: neural codec (EnCodec / SoundStream)
  ┌──────────┐                   ┌──────────┐
  │ encoder  │ ─ discrete tokens ─▶│ decoder  │
  │          │   ~75 tokens/sec    │          │
  │  audio   │   4 codebooks       │  audio   │
  │  (wave)  │   each of 1024 vocab │  (wave)  │
  └──────────┘                   └──────────┘

  Effectively: audio → sequence of 75×4 = 300 tokens/sec.
  Now the LLM can generate audio the same way it generates text!
```

**How GPT-4o likely works (not confirmed, educated guess)**:

```
  ┌──────────────────────────────────────┐
  │    Unified Transformer                 │
  │                                        │
  │  Input:                                 │
  │    [audio tokens from user]           │
  │    [image tokens from camera]          │
  │    [text tokens from instructions]    │
  │                                        │
  │  Output:                                │
  │    [audio tokens to speak]            │
  │    [text tokens for logging]           │
  └──────────────────────────────────────┘

  Trained on:
    - Speech pairs (your voice → my voice)
    - Image+speech narration
    - Text-to-audio matching
    - Multi-turn audio conversations
```

**Open research / community options**:

```
  Qwen2-Audio (Alibaba, 2024):
    Input: audio + optional text
    Output: text
    Good at: transcription, speech understanding, music analysis

  SALMONN (Tsinghua, 2023):
    Open audio understanding
    Speech + music + sound events

  AudioPaLM (Google, 2023):
    Research only, not released
    Shows what's possible

  Running Qwen2-Audio:
    from transformers import Qwen2AudioForConditionalGeneration
    # loads like any HF model, audio input supported
```

**Capabilities beyond transcription**:

```
  ✓ "What emotion is in this voice?"          (emotion detection)
  ✓ "How many speakers in this recording?"     (diarization)
  ✓ "What's the genre of this music?"          (music classification)
  ✓ "What's the user's age/accent?"            (speaker profiling)
  ✓ "Describe the audio scene"                 (environmental sound)
  ✓ "Sing this text as a rap"                  (creative TTS)
  ✓ "Continue this audio in the same style"    (audio generation)
```

**Voice agents: the 2026 frontier**:

```
  Traditional phone support agent:
    - Human picks up, 30-60 seconds wait
    - Follows script
    - Transfers to specialist if complex

  Modern voice AI agent (Cartesia + GPT-4o + custom):
    - Always available, 0ms wait
    - Natural conversation
    - Knows user history (RAG over customer data)
    - Can laugh, empathize, show frustration
    - Handles 80% of calls, escalates 20%

  Still struggles with:
    - Very angry customers
    - Rapid speakers
    - Heavy accents
    - Ambiguous intent
```

**Running voice-to-text-to-response (rough prototype)**:

```python
# A minimal voice assistant in ~50 lines
import whisper, sounddevice as sd, openai, numpy as np
from TTS.api import TTS

asr = whisper.load_model("small")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
openai_client = openai.OpenAI()

def listen(duration=5):
    audio = sd.rec(int(16000 * duration), samplerate=16000, channels=1)
    sd.wait()
    return audio.flatten()

def transcribe(audio):
    result = asr.transcribe(audio)
    return result["text"]

def respond(prompt):
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content

def speak(text):
    tts.tts_to_file(
        text=text,
        speaker_wav="reference.wav",
        language="en",
        file_path="response.wav"
    )
    # play response.wav

# Main loop:
while True:
    print("Listening (5s)...")
    audio = listen()
    text = transcribe(audio)
    print(f"You: {text}")
    reply = respond(text)
    print(f"AI: {reply}")
    speak(reply)
```

~50 lines of classical pipeline. Not as good as GPT-4o voice but works.

**Time-to-X latency comparison (2026)**:

```
                       time-to-first-response
  Classical pipeline    2000ms    (Whisper + GPT + TTS)
  GPT-4o voice           300ms    (unified, API)
  Cartesia optimized     150ms    (latest SSM architecture)

  Human-to-human average: ~200ms end-of-turn response.
  Modern voice AI is finally hitting human-like timing.
```

**Emotion-aware generation**:

```
  Old TTS input:
    "I'm so happy to see you!"
    → flat robotic voice
    → user says "sounds fake"

  GPT-4o output:
    "I'm so happy to see you!"
    → model chose enthusiastic tone
    → proper prosody: rising on "happy"
    → slight laugh after
    → user says "felt real"

  This is the unlock that makes voice AI genuinely useful.
```

## Exercises

1. Use OpenAI's voice mode in ChatGPT (if subscribed). Note: latency, interruption handling, emotion.

2. Install Qwen2-Audio locally (or via Transformers). Upload a 30-second voice note you recorded. Ask "what emotion do you hear?"

3. Try audio-in on an API: describe a song's genre, identify an accent, count speakers.

4. Read SALMONN's paper (12 pages). Note the three-stage training.

5. Think: what's the killer app for audio-understanding LLMs that doesn't exist yet?

## Next

`10_world_models_and_simulators.md` - the speculative frontier.
