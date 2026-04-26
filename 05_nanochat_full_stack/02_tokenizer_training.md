# 5.2 - Tokenizer Training in nanochat

We covered BPE in Lesson 4.2. Now we walk through the specific script `scripts/tok_train.py` and understand choices nanochat makes.

## Why train a custom tokenizer?

You could just use GPT-2's tiktoken. But a custom tokenizer:
- **Matches your training data**: better compression on your domain.
- **Uses your vocab size**: 32K tokens for nanochat (smaller = faster embedding, simpler model).
- **Includes your special tokens**: chat markers built in from day one, not bolted on.
- **Is your own**: no external dependency on OpenAI's encoding forever.

This is the first lesson of nanochat: own your stack.

## The script

```bash
python -m scripts.tok_train
```

Open `~/workspace/nanochat/scripts/tok_train.py`. The full script is ~100 lines. Let's walk the key parts.

### 1. Load training text

```python
from nanochat.dataset import list_shards, load_shard
shard_files = list_shards()[:num_shards]
texts = []
for shard in shard_files:
    texts.extend(load_shard(shard))
# concatenate into big string
train_text = "".join(texts)[:max_chars]
```

Loads the first few data shards (already downloaded by `nanochat.dataset -n 8`). Uses the first ~2B characters.

### 2. Train BPE

```python
from nanochat.tokenizer import train_tokenizer
tokenizer = train_tokenizer(
    train_text,
    vocab_size=2**15,     # 32768
    # special tokens reserved for chat:
    special_tokens=[
        "<|bos|>",
        "<|user_start|>", "<|user_end|>",
        "<|assistant_start|>", "<|assistant_end|>",
        "<|python_start|>", "<|python_end|>",
        "<|output_start|>", "<|output_end|>",
        ...
    ],
)
```

Under the hood, `train_tokenizer` uses a Rust BPE implementation (imported as `rustbpe` or similar). Much faster than Python BPE - ~30 seconds instead of hours.

The special tokens are **reserved at the top of the vocab** - they get token IDs that don't conflict with BPE-generated tokens.

### 3. Save

```python
tokenizer.save(os.path.join(NANOCHAT_BASE_DIR, "tokenizer"))
```

Saves:
- `tokenizer.json`: merge rules + vocab.
- Possibly companion files (tokenizer_config.json for HF compatibility).

## What's in `nanochat/tokenizer.py`?

Open it. Key pieces:

### The `Tokenizer` class

```python
class Tokenizer:
    def __init__(self, ...):
        # wraps a rustbpe-or-HF tokenizer
        ...
    def encode(self, text: str, add_special_tokens=False) -> list[int]:
        ...
    def decode(self, ids: list[int]) -> str:
        ...
    def encode_chat(self, messages: list[dict]) -> list[int]:
        # renders messages using chat template
        ...
```

### Chat template rendering

A `messages` list looks like:
```python
[
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"},
]
```

`encode_chat` turns this into:
```
<|user_start|>Hi<|user_end|><|assistant_start|>Hello!<|assistant_end|>
```

...then encodes that string.

The template is defined inside tokenizer.py. Look for something like `_render_message` or `_render_conversation`.

### Loss masking for SFT

For supervised fine-tuning, we only want to compute loss on **assistant tokens**, not on user prompts. The tokenizer also produces a **loss mask**:

```python
ids, mask = tokenizer.encode_chat_with_mask(messages)
# mask[i] = 1 if ids[i] is an assistant token, else 0
```

You pass the mask to `F.cross_entropy(..., reduction='none')` and average only over masked positions. This is standard in SFT.

## Evaluating the tokenizer

`python -m scripts.tok_eval` runs `scripts/tok_eval.py`. It computes:

- **Compression ratio** on held-out text: bytes per token. Higher = better. Target: ~4.5 on English web text.
- **Comparison** against tiktoken's `gpt2` and `cl100k_base` encodings.
- **Round-trip test**: encode → decode → should match original.
- **Special token handling**: all special tokens round-trip correctly.

Output goes to `~/.cache/nanochat/report/tokenizer.md`. At the end of a speedrun, this gets concatenated into the full report.

## Design choices to understand

1. **Why 32K vocab and not 50K or 100K?**
   - Smaller vocab → smaller embedding table (32K × n_embd) → fewer params in the "padding" parts of the model.
   - Slightly worse compression (more tokens for same text).
   - Sweet spot for compute-optimal small models. Bigger models might prefer bigger vocab.

2. **Why train on 2B chars and not more?**
   - BPE converges quickly. Doubling data barely changes the merges.
   - Scales with vocab size, not data size.

3. **Why reserve special tokens upfront?**
   - So token IDs don't shift when you add tokens later.
   - Chat markers are known from day one.

4. **Why Rust?**
   - Pure-Python BPE is painfully slow.
   - Rust (via rustbpe or HF's tokenizers crate) is ~1000x faster.

## In the bigger picture

The tokenizer is a tiny part of the whole system - maybe 10 MB of merge rules + vocab. But choosing it wrong is catastrophic. You'd have to retrain from scratch, and your data prep (turning text into .bin files of token IDs) depends on it.

Lesson: spend the hour on tokenizer choice early. Don't rush it.

## Exercises

1. Run `python -m scripts.tok_train` after downloading shards. Time it. See the vocab file produced.

2. Open the output `tokenizer.json`. Find 10 interesting merges - e.g. `"the "`, `"__init__"`, `"://"`. These show you what the training corpus was dominant in.

3. Run `python -m scripts.tok_eval` and read the output report. Compare compression to GPT-2 tokenizer.

4. In Python, import nanochat's tokenizer and encode a string that uses all special tokens. Verify round-trip: `decode(encode(text)) == text`.

5. Bonus: train a tokenizer on a non-English corpus (your own choice). Compare compression to nanochat's English one. English is dramatically better compressed because the training data was English-dominant.

## Next

`03_pretraining_base_train.md` - the biggest script, where the real money goes.
