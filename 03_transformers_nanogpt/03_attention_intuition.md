# 3.3 - Attention, First Pass: The Intuition

This is the lesson. Attention is the idea that made transformers work. If you get this, you get the transformer.

We'll do it twice: first as a story, no math. Then, in Lesson 3.4, we'll formalize it into matmuls.

## The problem attention solves

Suppose the input is: "The animal didn't cross the street because it was too tired."

To understand "it", the model needs to know "it" refers to "animal". The word "it" by itself has no information about animals - it's a pronoun, could point to anything. To resolve it, the model has to **look back at the earlier words** and figure out which one is the referent.

Before transformers, models used RNNs (recurrent neural networks) that processed text one word at a time, carrying a hidden state forward. But a hidden state can only hold so much, and faraway references often got lost. Transformers solve this by letting every word **directly access** every other word in one operation.

## The mechanism, in words

For each position in the sequence, the model does three things:

1. **Ask a question.** Each token generates a **query**: "what am I looking for?"
2. **Offer information.** Each token also generates a **key** ("what I am") and a **value** ("what I can provide").
3. **Match and retrieve.** The token with query `q` looks at all other tokens, computes how much each of their keys `k` matches `q`, and uses those match scores to build a weighted average of their values `v`.

Concretely, for "it":
- Query: "I'm a pronoun looking for a referent."
- Each other word's key says what it is: "animal" says "noun, subject, living creature." "street" says "noun, location." Etc.
- The query-key match with "animal" is high; with "street" is lower.
- "it"'s output becomes a weighted mix of other words' values, dominated by "animal".

After this operation, the vector at position "it" has absorbed information about "animal". The word is now context-aware.

You do this for *every* position simultaneously, every layer. Deep networks of attention build increasingly abstract, context-rich representations.

## "Every word talks to every word"

The crucial property: attention connects every position to every other position in **one operation**. Not sequential, not through a bottleneck. Direct.

This is why transformers:
- Are highly parallelizable (all positions computed at once).
- Handle long-range dependencies well.
- Scale beautifully to billions of parameters.

And why they're expensive:
- Computing all-pairs interactions is O(T^2) where T is sequence length. 8K tokens means 64M pairs. 128K tokens means 16B pairs. That's why long context is hard.

## Causal (for language models)

When training a language model, we want the model to predict the *next* token from the tokens *before* it. We can't let a token at position `i` look at tokens at positions `j > i` - that's cheating (it'd see the future).

So we **mask** those connections: for each position `i`, only positions `0..i` contribute. This is **causal attention** or **autoregressive attention**. GPT uses it. (BERT, by contrast, is non-causal - it can see both directions - which makes it great for understanding but not for generation.)

The mask looks like a lower-triangular matrix of 1s (allowed) and 0s (blocked):

```
token 0 can attend to: [0]
token 1 can attend to: [0, 1]
token 2 can attend to: [0, 1, 2]
...
```

Triangular mask. We'll see it explicitly next lesson.

## A picture

Imagine a 5-token sentence. Draw a 5x5 grid. Each cell (i, j) is "how much does token i attend to token j". For causal attention, cells above the diagonal are zero:

```
         target j
       0    1    2    3    4
q  0  1.0  -    -    -    -
u  1  0.3  0.7  -    -    -
e  2  0.1  0.2  0.7  -    -
r  3  0.2  0.1  0.3  0.4  -
y  4  0.1  0.0  0.4  0.2  0.3
  i
```

Each row sums to 1 (because we apply softmax across the row). For token 2, it attends mostly to token 2 itself (0.7) and a bit to tokens 0 and 1.

The model learns these weights. After training, the weights will reflect meaningful linguistic patterns: nouns attend to their adjectives, verbs attend to their subjects, etc. People have analyzed transformer attention patterns and found interpretable "heads" - one for coreference, one for syntactic agreement, etc.

## Where this is in nanoGPT

`model.py`, class `CausalSelfAttention`, method `forward`:

```python
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)   # compute Q, K, V
# reshape into multi-head form (next lesson)
if self.flash:
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, ..., is_causal=True)
else:
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    y = att @ v
```

Three lines of substance:
1. Compute Q, K, V from input.
2. Score (dot-product of Q with K, masked, softmaxed) -> attention weights.
3. Weighted sum of V's with those weights -> output.

That's all. Next lesson makes each line concrete.

## Visualize this

**Attention pattern for "it" in "The animal didn't cross the street because it was too tired"**:

```
                 attention weight for "it" looking back
  The        [0.04]   ▮
  animal     [0.62]   ▮▮▮▮▮▮▮▮▮▮▮▮▮   ←── big attention, found referent!
  didn't     [0.03]
  cross      [0.02]
  the        [0.04]
  street     [0.08]   ▮▮
  because    [0.04]
  it         [0.10]
  was        [0.03]
  too        [0.00]
  tired      [0.00]

  (causal mask: "it" can't see "was too tired")
```

That's what a well-trained attention head for coreference resolution looks like. Visualize your own model's attention:

```python
# after training nanoGPT, extract attention from a layer:
import torch.nn.functional as F
with torch.no_grad():
    # ... forward pass, grabbing attention weights at a block ...
    attn_weights  # shape (batch, n_head, T, T)
    
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.imshow(attn_weights[0, 0].cpu(), cmap='viridis')  # head 0
plt.xlabel("key position (attended to)")
plt.ylabel("query position (attending)")
plt.colorbar()
plt.savefig("attention_head0.png")
```

Different heads often specialize - one for position, one for syntax, one for coreference. Running this for each of your 12 heads is enlightening.

**The causal mask, pictorially**:

```
  query pos →   0    1    2    3    4
                ┌────────────────────┐
            0   │ ✓   .    .    .    │
   key      1   │ ✓   ✓    .    .    │
   pos      2   │ ✓   ✓    ✓    .    │     lower-triangular
   ↓        3   │ ✓   ✓    ✓    ✓    │     ✓ = allowed
            4   │ ✓   ✓    ✓    ✓    ✓   │     . = -inf (blocked)
                └────────────────────┘

  Token at position i can only attend to positions 0..i.
  Prevents "cheating" by seeing the future.
```

**Interactive**: https://bbycroft.net/llm - scroll to the attention section, click a token, watch the attention lines light up to other tokens. The single best attention visualization ever built. Open it now.

**Or click around a simulated attention head right here:**

```viz
{"viz": "attention_interactive"}
```

Click any token. You'll see its attention pattern — a blue arc to each token it looks at, with thickness proportional to attention weight. Try:
- Click **"it"** on head 1 (Coreference): it attends strongly to **"animal"**. Classic pronoun resolution.
- Click **"was"** on head 3 (Syntactic): it attends to **"animal"** (its subject).
- Click any token on head 2 (Local): mostly attends to the previous token.

Real transformer heads specialize like this. You're looking at a hand-crafted example of a real phenomenon.

**Another great tool**: BertViz (https://github.com/jessevig/bertviz) - visualize attention patterns in real pretrained models (BERT, GPT-2) inside a Jupyter notebook.

## Exercise

1. Stare at the `CausalSelfAttention.forward` method. Don't try to understand it in detail yet. Identify the three steps above.
2. Read "Attention Is All You Need" figure 2 (the architecture diagram). You don't need to understand it all; just see the "multi-head attention" box and "add & norm" around it.
3. Think: if your input is 1024 tokens, what's the shape of the attention score matrix before softmax? (Hint: it's T×T. So 1024×1024.)

## Next

`04_attention_mathematically.md` - the math, precisely. Q, K, V, softmax, mask.
