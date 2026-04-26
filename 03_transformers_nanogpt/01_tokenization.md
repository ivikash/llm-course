# 3.1 - Tokenization

Before a language model sees text, the text must become a sequence of integers. **Tokenization** is the process.

## Why not just use characters?

We did, in Module 1! And it worked. But character-level has drawbacks for serious LLMs:
- English has ~26 letters + punctuation, a vocab of ~100. That's tiny.
- Text becomes very long. "hello world" = 11 tokens at char level.
- Context window is a scarce resource in transformers (quadratic cost with length). Short sequences = more efficient.

Common word-level also has drawbacks:
- Vocabs of 1M+ are unwieldy.
- Any unseen word becomes "UNK" (unknown token), breaking things.
- Doesn't handle morphology (`run`, `running`, `runner` are unrelated).

Sweet spot: **subword tokenization**. Common words get a single token. Rare words get split into a few subword tokens.

## Byte-Pair Encoding (BPE)

The most common subword algorithm. The idea is embarrassingly simple, but it's a good one:

1. Start with raw bytes (256 possible tokens).
2. Find the most frequent *pair of adjacent tokens* in the training text.
3. Merge that pair into a new token.
4. Repeat until you hit the desired vocab size (e.g. 50,257 for GPT-2).

Example toy corpus: "lowest lower lowly" (spaces matter).

Initial: `l o w e s t   l o w e r   l o w l y`
Most frequent pair: `l o` -> merge into `lo`.
Next pass: `lo w e s t   lo w e r   lo w l y`
Most frequent pair: `lo w` -> merge into `low`.
And so on. Eventually "low" is one token, and rare words still get tokenized into pieces.

Result: a vocabulary where common words/subwords are single tokens, rare words are a few tokens.

## Run it: tiktoken (GPT-2's tokenizer)

tiktoken is OpenAI's fast BPE library. nanoGPT uses it.

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, world! The transformer is neat.")
print(tokens)
# [15496, 11, 995, 0, 383, 47385, 318, 15049, 13]
for t in tokens:
    print(t, repr(enc.decode([t])))
```

You'll see:
```
15496 'Hello'
11    ','
995   ' world'
0     '!'
383   ' The'
47385 ' transformer'
318   ' is'
15049 ' neat'
13    '.'
```

Notice: `' world'` (with leading space) is a single token. "transformer" too. GPT-2's vocab is 50,257 tokens. Common words get dedicated tokens.

Try rare/odd inputs:

```python
print(enc.encode("antidisestablishmentarianism"))
# [415, 312, 29830, 27786, 15344, 1042]   # 6 tokens, split up
print(enc.encode("🦖"))
# [8582, 99, 244]   # multi-byte emoji splits across tokens
```

## Tokenizer design choices

Two levers:
1. **Training corpus**: train BPE on different data, get different merges.
2. **Vocab size**: more tokens = shorter sequences, but bigger embedding table.

GPT-2: vocab 50,257, trained on WebText.
GPT-3: same tokenizer as GPT-2.
GPT-4: vocab ~100K ("cl100k_base"), trained on more recent/multilingual data.
Llama-2: vocab 32K.
nanochat: vocab 32,768 (2^15) trained from scratch on FineWeb / DCLM in `scripts/tok_train.py`.

## In nanoGPT

For character-level Shakespeare, tokens = characters. See `data/shakespeare_char/prepare.py`:
```python
chars = sorted(list(set(data)))
vocab_size = len(chars)  # ~65
stoi = { ch:i for i,ch in enumerate(chars) }
```

For OpenWebText, tokens = BPE tokens via tiktoken. See `data/openwebtext/prepare.py`:
```python
enc = tiktoken.get_encoding("gpt2")
ids = enc.encode_ordinary(text)
```

Either way, by the time `train.py` runs, data is just a giant array of integer token IDs. The model never sees text.

## In nanochat

nanochat trains its *own* tokenizer from scratch using a Rust BPE implementation via a Python wrapper. See `nanochat/nanochat/tokenizer.py` and `scripts/tok_train.py`. After training, it can be used exactly like tiktoken's GPT-2 tokenizer.

## Special tokens

Beyond text tokens, a tokenizer usually has **special tokens**:
- `<|endoftext|>`: marks end of a document.
- `<|im_start|>`, `<|im_end|>`: wrap chat messages (in chat models).
- `<|user|>`, `<|assistant|>`: role markers.
- `<|tool_call|>`, `<|python|>`: for tool/agent models.

nanochat's tokenizer.py and chat_format.py define these. In nanoGPT, only `<|endoftext|>` is used.

## Visualize this

**Tiktokenizer in action**: https://tiktokenizer.vercel.app/

Paste this exactly and watch how it tokenizes:

```
Hello, world! The transformer is neat.
I love Pokémon.
antidisestablishmentarianism
```

You'll see each token in a different color. Notice:
- `" world"` (with leading space) is one token.
- `" Pokémon"` may be 3 tokens (split around the accent).
- `"antidisestablishmentarianism"` is 5-6 tokens (rare words split).
- Single common words like `"the"` are one token each.

**BPE merge, pictorially**:

```
  Start:  l o w   l o w e r   l o w l y   (each char = 1 token)
                     │
     most frequent pair: (l,o)
                     │
                     ▼
  Merge 1: lo w   lo w e r   lo w l y     (~20% fewer tokens)
                     │
     most frequent pair: (lo,w)
                     │
                     ▼
  Merge 2: low     low e r     low l y    (even fewer)
                     │
                     ... repeat ~32,000 times
                     │
                     ▼
  Final vocab: {"the", " and", "ing", "Paris", ..., "低", ...}
  Common subwords get dedicated tokens.
  Rare strings still compose from smaller pieces.
```

**The tokenization penalty for non-English**: tokenize the same sentence in 3 languages and watch token counts:

```
English:  "Hello, how are you doing today?"       → 7 tokens
Spanish:  "Hola, ¿cómo estás hoy?"                 → 10 tokens
Chinese:  "你好，你今天怎么样？"                      → 14 tokens
Hindi:    "नमस्ते, आज आप कैसे हैं?"                  → 18 tokens
```

Same meaning, 2-3x more tokens for non-English. Your "effective context window" shrinks proportionally. A known bias. Modern tokenizers (GPT-4's cl100k, Llama-3) are much better at non-English.

**Chat templates visualized**:

```
Raw conversation:
  user: What's 2+2?
  assistant: 4

What the tokenizer actually sees (nanochat style):
  <|bos|><|user_start|>What's 2+2?<|user_end|><|assistant_start|>4<|assistant_end|>

Token IDs:
  [1, 102, 15, 35, 27, 17, 29, 48, 103, 104, 4, 105]
   │   │                           │   │       │
   BOS user_start                  end start   assistant_end
```

Everything - text, roles, markers - is just token IDs by the time the model sees it.

## Exercises

1. `pip install tiktoken`. Try the above snippets.
2. Tokenize the first paragraph of a Wikipedia article. Count tokens. Now try translating that paragraph to Hindi or Chinese and tokenizing again. You'll notice non-English text gets MANY more tokens (each Chinese char often 2-3 tokens). This is a known bias of Latin-trained BPE; newer tokenizers handle it better.
3. Open `~/workspace/nanochat/nanochat/tokenizer.py`. Skim the class. Note it's building on `tokenizers` library (rustBPE). Try the `encode` on a sample string by importing and using it (after setting up the nanochat env).
4. Open `~/workspace/nanochat/scripts/tok_train.py`. Read the top comments. Understand: it downloads text, trains BPE, saves vocab to disk. ~100 lines.

## Next

`02_embeddings.md` - integers become vectors. The first transformation inside the model.
