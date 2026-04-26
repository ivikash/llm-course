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

## Exercises

1. Use OpenAI's voice mode in ChatGPT (if subscribed). Note: latency, interruption handling, emotion.

2. Install Qwen2-Audio locally (or via Transformers). Upload a 30-second voice note you recorded. Ask "what emotion do you hear?"

3. Try audio-in on an API: describe a song's genre, identify an accent, count speakers.

4. Read SALMONN's paper (12 pages). Note the three-stage training.

5. Think: what's the killer app for audio-understanding LLMs that doesn't exist yet?

## Next

`10_world_models_and_simulators.md` - the speculative frontier.
