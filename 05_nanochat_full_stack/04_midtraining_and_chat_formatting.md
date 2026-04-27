# 5.4 - Midtraining and Chat Formatting

After pretraining you have a "base model" - good at continuing text in whatever style it was trained on. But chatting with it is awkward: it doesn't know it's supposed to take turns, doesn't know when to stop, rambles.

**Midtraining** (and/or SFT, depending on terminology) bridges pretraining to chat behavior by teaching the model chat structure.

## What pretraining didn't teach

A base model never saw examples like:

```
<|user_start|>What's 2+2?<|user_end|><|assistant_start|>4<|assistant_end|>
```

It just saw a firehose of web text. So if you prompt a base model with "What's 2+2?" it might produce:

```
What's 2+2? 3+3? 4+4? In this puzzle book, each page asks...
```

It continues the style, not the conversation. Midtraining fixes this.

## The chat template

A chat conversation becomes a token sequence:

```
<|bos|><|user_start|>user_msg_1<|user_end|><|assistant_start|>assistant_reply_1<|assistant_end|><|user_start|>user_msg_2<|user_end|><|assistant_start|>assistant_reply_2<|assistant_end|>
```

At inference time, you format the conversation this way, append `<|assistant_start|>` to signal "model, your turn", and generate until `<|assistant_end|>`. The model learns:
- Produce a coherent assistant response after `<|assistant_start|>`.
- Stop with `<|assistant_end|>` (not ramble forever).
- Don't produce `<|user_start|>` tokens itself (don't speak for the user).

## Midtraining vs SFT (terminology)

Different authors use these differently. One common convention:

- **Midtraining**: a short phase *between* pretraining and SFT, on general-purpose instruction-tuning data. Adjusts the model to the chat format without deep task-specific tuning.
- **SFT (Supervised Fine-Tuning)**: longer training on curated (prompt, response) pairs, often domain-specific or higher-quality.

nanochat doesn't cleanly separate these - `chat_sft.py` does both in one stage, using SmolTalk (an open instruction dataset) + synthetic identity conversations. The effect is the same: the model learns chat format and picks up instruction-following behavior.

## Loss masking: the key technique

**See it live first**:

```viz
{"viz": "sft_loss_mask"}
```

Switch between scenarios. Green tokens (assistant) contribute to the loss; red tokens (user) don't. Each token carries its mask bit (1 or 0). Notice: even in tool-use scenarios, tool *outputs* have mask=0 — model reads them but doesn't try to generate them.

During SFT, the training data is (user_message, assistant_response) pairs. We compute loss **only** on the assistant's tokens, not the user's.

Why? Because:
- The user's tokens are **input** to the model, not something it should predict.
- Forcing the model to predict user tokens would distort its behavior.

Implementation:

```python
# encode the full conversation
ids = [<|bos|>, u1, u2, u3, <|user_end|>, <|assistant_start|>, a1, a2, <|assistant_end|>]
# loss mask: 1 where we compute loss, 0 where we don't
mask = [0, 0, 0, 0, 0, 0, 1, 1, 1]
# (note: <|assistant_start|> itself is also typically 0 - we want the model to PRODUCE assistant tokens
# after it, not predict the start marker itself)
```

nanochat's tokenizer implements `encode_chat_with_mask` that returns both.

In training:
```python
logits = model(ids)                                      # (T, V)
loss_per_pos = F.cross_entropy(logits, targets, reduction='none')   # (T,)
loss = (loss_per_pos * mask).sum() / mask.sum()         # average only on assistant
```

## The SmolTalk dataset

HuggingFace's [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) is ~1M general-purpose instruction-following conversations, a widely-used open SFT dataset.

Examples:
- "Write a haiku about summer."
- "Explain recursion with an example."
- "Translate this to French: ..."

It's diverse enough to teach general chat behavior without being domain-specific. See `tasks/smoltalk.py` for how nanochat loads it.

## Identity conversations

nanochat also adds a small set (~2000) of *identity* conversations:

```
User: What's your name?
Assistant: I'm nanochat, a small language model...
```

Purpose: give the model a consistent persona. Otherwise when you ask "who are you?" it'll hallucinate randomly.

See `dev/gen_synthetic_data.py` - how these are generated (using a bigger LLM to produce diverse phrasings of the same personality facts).

## Putting it together

A pretrained base model + SFT on (SmolTalk + identity conversations) = a model that:
- Follows instructions.
- Has a personality.
- Uses chat formatting correctly.
- Stops at `<|assistant_end|>`.

It's not good yet - maybe kindergarten-level. But it's a chatbot. That's the qualitative leap.

## Walkthrough of chat formatting in code

Open `~/workspace/nanochat/nanochat/tokenizer.py`. Find methods like:
- `encode_chat(messages)` - renders + tokenizes.
- `render_user_message`, `render_assistant_message` - individual pieces.

Open `~/workspace/nanochat/tasks/smoltalk.py`. See how SmolTalk examples get converted into (tokens, mask) pairs for training.

Open `~/workspace/nanochat/tasks/common.py`. Shared helpers used across task datasets.

## Visualize this

**Midtraining / SFT token structure**:

```
  Training example during SFT:

  <|bos|><|user_start|>What is 2+2?<|user_end|><|assistant_start|>4.<|assistant_end|>
  
  Split for training:
    x =        [bos, us, W, i, s, ?, ue, as, 4, .]       (input)
    y =        [us, W, i, s, ?, ue, as, 4, ., ae]        (target, shifted by 1)
    loss_mask = [0,  0, 0, 0, 0, 0,  0, 1, 1, 1]         (compute loss only on assistant)
```

**Why loss masking matters**:

```
  WITHOUT loss mask (wrong):
    Model learns to predict user tokens too.
    Effect: it might try to "complete" user messages, mimicking users.
    Result: weird conversational behavior, model asking its own questions.

  WITH loss mask (right):
    Model learns only: "given a formatted conversation, produce a good
    assistant response and stop at <|assistant_end|>".
    Result: clean, role-appropriate behavior.
```

**Packing conversations into training sequences**:

```
  naive packing (waste):
  seq 1: <conv1 tokens> [PAD PAD PAD PAD PAD ... padding ...]   seq_len 2048
  seq 2: <conv2 tokens> [PAD PAD PAD ...]
  (most tokens are padding, wasted compute)

  dense packing (nanochat-style):
  seq: <bos><conv1 tokens><bos><conv2 tokens><bos><conv3 tokens>...   seq_len 2048
  (each seq contains multiple conversations, minimal waste)

  BUT: attention must not cross conversation boundaries!
  Solution: attention mask that resets at each <bos>.
       Conv 1 cannot attend to Conv 2.
       Conv 2 cannot attend to Conv 1.
       Implemented in nanochat/flash_attention.py.
```

**Chat template format comparison**:

```
  OpenAI format (ChatGPT):
  [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"}
  ]

  Llama-2 format:
  "<s>[INST] <<SYS>>\nYou are helpful.\n<</SYS>>\n\nHi [/INST] Hello! </s>"

  Llama-3 format:
  "<|begin_of_text|><|start_header_id|>system<|end_header_id|>
  You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>
  Hi<|eot_id|><|start_header_id|>assistant<|end_header_id|>
  Hello!<|eot_id|>"

  nanochat format:
  "<|bos|><|user_start|>Hi<|user_end|><|assistant_start|>Hello!<|assistant_end|>"
```

Different models = different special tokens. Your tokenizer's `apply_chat_template` handles this. Using the wrong template for a given model produces bad outputs.

**Dataset composition for SFT**:

```
  SmolTalk (general instruction-following):
  ──────────────────────────────────────────
  "Write a haiku about summer."
  "Explain recursion."
  "Translate to French: Hello world"
  ...
  → teaches broad helpfulness
  ~1M conversations total

  Identity conversations (~2000):
  ──────────────────────────────────────────
  "What's your name?" → "I'm nanochat, trained by..."
  "Who made you?" → "I was built with nanochat..."
  ...
  → teaches consistent persona

  Tool-use examples (optional):
  ──────────────────────────────────────────
  "What's 4237 * 89?"
  → "<|python_start|>4237*89<|python_end|><|output_start|>377093<|output_end|>
     The answer is 377,093."
  → teaches tool use

  Mixed together, shuffled, packed into sequences.
```

## Exercises

1. Use the nanochat tokenizer to encode a sample conversation:
   ```python
   from nanochat.tokenizer import Tokenizer
   tok = Tokenizer.load(...)
   ids, mask = tok.encode_chat_with_mask([
       {"role": "user", "content": "What's 2+2?"},
       {"role": "assistant", "content": "4."},
   ])
   print(ids)
   print(mask)
   # decode to inspect
   print(tok.decode(ids))
   ```
   Verify mask has 1s only where assistant tokens are.

2. Read `tasks/smoltalk.py`. How many lines? What does its iterator yield?

3. Imagine training without the loss mask (treating user tokens same as assistant). Predict what would go wrong. (Answer: model would learn to produce user-sounding text too, get confused about roles.)

4. In `dev/gen_synthetic_data.py`, find where the prompts to the bigger LLM are defined. Note that making the identity consistent takes surprising care.

## Next

`05_sft_supervised_fine_tuning.md` - the full SFT script.
