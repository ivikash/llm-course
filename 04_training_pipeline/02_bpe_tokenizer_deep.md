# 4.2 - BPE Tokenizer: Deep Dive

We covered tokenization conceptually in Lesson 3.1. Now we go deep: how BPE training actually works, how to train your own, and what's in nanochat's tokenizer.

## The BPE algorithm, precisely

**Input**: a big corpus of text.
**Hyperparameter**: desired vocab size `V`.
**Output**: a list of merge rules and a vocabulary.

```
1. Initialize vocab with all individual bytes (256 tokens: 0-255).
2. Split the text into words (or keep as raw bytes).
3. Represent each word as a sequence of individual bytes/characters.
4. Count all adjacent pairs across the entire corpus.
5. Find the most frequent pair (a, b).
6. Add a new token c = merge(a, b) to the vocab.
7. Replace every occurrence of (a, b) in the corpus with c.
8. Repeat steps 4-7 until vocab reaches V tokens.
```

The **merge list** is the output: an ordered list like `[(e, ' ') -> 256, (t, h) -> 257, ...]`.

To encode new text: apply the merges in order. To decode: look up each token in the vocab.

## Why "byte-pair"?

The original algorithm (Sennrich 2015) worked at the character level. But character sets vary across languages and Unicode is a mess. **Byte-level BPE** (GPT-2) uses raw bytes as the base alphabet - 256 is always enough, no Unicode issues, works for any language.

All modern LLMs use byte-level BPE.

## tiktoken (OpenAI) vs sentencepiece (Google) vs tokenizers (HuggingFace)

Three mainstream implementations:

- **tiktoken**: OpenAI's fast Rust BPE. Used by GPT-2/3/4. Simple API.
- **sentencepiece**: Google's. Used by T5, Llama. Can do BPE or unigram LM tokenization. More features.
- **tokenizers** (HuggingFace): Rust-based, most features, most flexible. Can train, encode, decode, and serve via API.

For training your own tokenizer, `tokenizers` or `sentencepiece` are usual. For using existing ones, `tiktoken` is fastest.

## Training your own: the nanochat way

Open `~/workspace/nanochat/scripts/tok_train.py`. Let's walk it:

```python
# 1. Load a chunk of training text (2B chars ~ 500M tokens-ish).
train_text = load_text_from_shards(...)

# 2. Build a BPE trainer. nanochat uses a Rust backend via rustbpe.
from nanochat.tokenizer import HFTokenizer
tokenizer = HFTokenizer()
tokenizer.train(train_text, vocab_size=2**15)  # 32768

# 3. Save the merges and vocab.
tokenizer.save(out_dir)
```

That's the essence. Real code has progress bars and error handling, but the algorithm is standard BPE.

Key design decisions in nanochat:
- Vocab size **32,768** (= 2^15). Power-of-2 is nice for GPU math.
- **Byte-level**: no pre-tokenization of special characters.
- Trained on **English-heavy web text**, so non-English compresses worse.
- Includes **special tokens** for chat (`<|im_start|>`, `<|im_end|>`, `<|user|>`, etc.).

## Evaluating a tokenizer

Open `~/workspace/nanochat/scripts/tok_eval.py`. It computes:
- **Compression ratio**: bytes per token. Higher = better compression. GPT-2 gets ~4 bytes/token on English; nanochat gets ~4.5 on its training domain.
- **Coverage** of special characters.
- **Comparison** against tiktoken's GPT-2 and GPT-4 encodings on the same text.

Compression matters because it's your effective context length. If your tokenizer takes 5000 tokens to encode what another takes 4000, your 8K context window is worth 20% less.

## Chat templates

A chat-capable tokenizer has extra special tokens to wrap messages:

```
<|im_start|>user
What's 2+2?<|im_end|>
<|im_start|>assistant
4<|im_end|>
```

nanochat's tokenizer.py defines:
- `<|bos|>` (beginning of sequence)
- `<|user_start|>`, `<|user_end|>`
- `<|assistant_start|>`, `<|assistant_end|>`
- `<|tool_start|>`, `<|tool_end|>` (for tool use)
- `<|python_start|>`, `<|python_end|>` (for code execution)
- `<|output_start|>`, `<|output_end|>` (tool outputs)

These are not "text" - they're control tokens the model learns to produce at appropriate times.

During SFT (Module 5), the tokenizer formats conversations using these tokens. The model learns: "after `<|user_end|>`, I should produce `<|assistant_start|>` then a helpful response then `<|assistant_end|>`."

## Common pitfalls

1. **Tokenizer mismatch**: if you train with one tokenizer and infer with another, results are garbage. Always pair them.
2. **Non-English text**: GPT-2 tokenizer splits Chinese into 2-3 tokens per character. Your context shrinks dramatically.
3. **Trailing space vs leading space**: `"hello"` and `" hello"` are often *different* tokens in BPE. Prompt engineering subtlety.
4. **Numbers**: digit-level tokenization vs chunky tokenization of numbers matters for math. Llama-3 went digit-level for this reason.

## Visualize this

**Watch BPE build its vocabulary step by step:**

```viz
{"viz": "bpe_trainer"}
```

Start with one-char-per-token. Press **Next merge** or **Auto-run**. Each step finds the most frequent pair and merges them into a new token. Watch common patterns like "lo", "low", "est", "newest" emerge as merge by merge. This is literally the algorithm tiktoken and nanochat's tokenizer use — just on 2B characters instead of 20.

**The BPE algorithm, precisely**:

```
  Iteration 0:
  corpus (chars):   l o w   l o w e r   l o w l y
  vocab: {l, o, w, e, r, y, SPACE}   size = 7
  char pairs counts: (l,o)=3, (o,w)=3, (w,SPACE)=2, (w,e)=1, (e,r)=1, (l,y)=1, ...

  [highest: (l,o)=3 → merge into 'lo']

  Iteration 1:
  corpus:           lo w   lo w e r   lo w l y
  vocab: {..., lo}  size = 8
  pair counts: (lo,w)=3, (w,SPACE)=2, ...

  [highest: (lo,w)=3 → merge into 'low']

  Iteration 2:
  corpus:           low   low e r   low l y
  vocab: {..., low}  size = 9

  ...continue 32,000 times...

  Final vocab: {" the", "ing", "tion", "Paris", ..., 32000 tokens}
```

Each iteration = one new token in the vocabulary = one merge rule.

**What you'd see in nanochat's trained tokenizer**:

```
  Top merges (most frequent pairs that became tokens):

  merge 256: " " + "t"    →  " t"
  merge 257: "t" + "h"    →  "th"
  merge 258: " " + "a"    →  " a"
  merge 259: "h" + "e"    →  "he"
  merge 260: " t" + "he"  →  " the"     ← the word "the" is often one token!
  merge 261: " " + "o"    →  " o"
  merge 262: "i" + "n"    →  "in"
  merge 263: " " + "s"    →  " s"
  ...
  merge 5000: "Paris"      →  (one token, because it appeared often in training data)
  ...
  merge 32000: last one reached
```

English function words end up as dedicated tokens. Rare scientific terms get split.

**Compression ratio visualized** (why BPE beats char-level for LLMs):

```
  "The quick brown fox jumps over the lazy dog."

  Char-level:  44 chars  →  44 tokens
  GPT-2 BPE:   44 chars  →  10 tokens       ← 4.4× compression
  GPT-4 BPE:   44 chars  →  9 tokens        ← 4.9× compression
  nanochat:    44 chars  →  10 tokens       ← 4.4× compression
```

More compression = longer effective context for same token budget.

**Same sentence, different languages** (the known non-English bias):

```
  Same meaning: "Good morning, how are you?"

  English:  "Good morning, how are you?"      →  6 tokens   (GPT-4)
  Spanish:  "Buenos días, ¿cómo estás?"        →  12 tokens
  Chinese:  "早上好，你怎么样？"                  →  18 tokens
  Arabic:   "صباح الخير، كيف حالك؟"             →  17 tokens

  Non-English users pay 2-3× for the same context. Newer tokenizers
  (Llama-3, GPT-4o) have narrowed this gap.
```

**Try it yourself**: paste text in multiple languages at https://tiktokenizer.vercel.app/ - the dramatic difference is real.

## Exercises

1. With tiktoken, tokenize `"hello"` vs `" hello"`. Are they the same token? (No.) Check:
   ```python
   import tiktoken
   enc = tiktoken.get_encoding("cl100k_base")
   print(enc.encode("hello"))       # e.g. [15339]
   print(enc.encode(" hello"))      # e.g. [24748]
   ```

2. Measure compression: take a 10KB text file, encode it, compute `bytes/tokens`. Do this for an English wikipedia article and for a Chinese one. Compare.

3. Read `~/workspace/nanochat/nanochat/tokenizer.py` fully. Find the list of special tokens. Find the `render_conversation` or similar method - that's where chat templates get applied.

4. Run `python -m scripts.tok_eval` in nanochat (after downloading data and training). See a compression comparison against tiktoken.

## Next

`03_mixed_precision_bf16_fp16.md` - speeding up training with smaller number formats.
