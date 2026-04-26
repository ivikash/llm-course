# Module 1 Capstone - A Neural Bigram Language Model

You built a bigram model by counting pairs (Lesson 0.3). Now build the same idea, but as a *neural network trained by gradient descent*. This is the smallest possible bridge between "counting" and "deep learning." The whole thing is ~50 lines.

## The plan

Train a model that, given one character, outputs a probability distribution over the next character. Same goal as Lesson 0.3's bigram model. But this time, the probabilities come from a trainable weight matrix, not a lookup of counts.

**The weight matrix** `W` has shape `(vocab_size, vocab_size)`. `W[i, j]` = logit for next char being `j` given current char is `i`. Softmax each row to get probabilities.

Training: cross-entropy loss + gradient descent. The result will be equivalent to counting pairs, but via a completely different route - the route that scales to GPT.

## The code

Save as `~/workspace/llm-course/exercises/03_neural_bigram.py`:

```python
import torch
import torch.nn.functional as F

torch.manual_seed(42)

# 1) Load text, build vocab
with open("~/workspace/nanoGPT/data/shakespeare_char/input.txt") as f:
    text = f.read()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

# 2) Encode text as a big list of integer IDs
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
print(f"vocab_size={vocab_size}, data length={len(data)}")

# 3) Build training pairs (x[i], y[i]) = (char_i, char_i+1)
xs = data[:-1]
ys = data[1:]
print(f"num training pairs: {len(xs)}")

# 4) The "model": a single weight matrix.
#    W[i] is the unnormalized log-probability (logits) over next char, given current char is i.
W = torch.randn(vocab_size, vocab_size, requires_grad=True)

# 5) Training loop
lr = 50.0  # big lr because only 1 param tensor and small model
for step in range(200):
    # forward: for each x, look up its row of W = logits
    logits = W[xs]                # shape: (N, vocab_size)
    loss = F.cross_entropy(logits, ys)

    # backward
    W.grad = None
    loss.backward()

    # update
    with torch.no_grad():
        W -= lr * W.grad

    if step % 20 == 0:
        print(f"step {step}: loss={loss.item():.4f}")

# 6) Sample from the trained model
def sample(seed="T", max_new=500):
    ix = stoi[seed]
    out = [seed]
    for _ in range(max_new):
        logits = W[ix]
        probs = F.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1).item()
        out.append(itos[ix])
    return "".join(out)

print("\n--- sample ---\n")
print(sample())
```

Run it:

```bash
python ~/workspace/llm-course/exercises/03_neural_bigram.py
```

You'll see loss drop from ~4.8 (random) down to ~2.48 (matches the counting bigram's perplexity). Then you'll see Shakespeare-flavored gibberish.

## What just happened (re-read this carefully)

1. We represented each possible character as an integer.
2. The model is one matrix `W` of shape `(vocab_size, vocab_size)`, `requires_grad=True`. That's ~(65x65) = 4225 parameters. Tiny.
3. "Forward pass": given input characters `xs`, look up rows of `W`. Those are **logits**.
4. "Loss": cross-entropy against the true next characters `ys`.
5. "Backward pass": `loss.backward()` fills in `W.grad` via autograd.
6. "Update": manually subtract `lr * grad` from `W`. This is SGD (stochastic gradient descent), the simplest optimizer.
7. Repeat 200 times. Loss drops. Model learns.

**This is a real neural network.** Admittedly the laziest possible one - no non-linearity, no hidden layer, just a lookup-matrix parameterized by gradient descent. But the *machinery* - forward, loss, backward, update - is exactly what a 70-billion parameter model does. The rest of the course is about making the forward pass smarter.

## The key insight

The bigram-via-counting and the bigram-via-gradient-descent converge to **the same answer**. If you count pairs and normalize, you get `W[i, j] = P(j | i)`. If you train this neural model with cross-entropy, you approach `W[i, j] ≈ log P(j | i)` (up to constant). Same information, encoded differently.

**So why bother with the neural version?** Because now you can replace `W[xs]` - a lookup - with *anything*. Embed the character into a vector, pass it through a transformer, project back to a distribution. Same loss, same training loop. But now you can look at 2048 tokens of context instead of 1. Now you can learn rich representations. Now your model generalizes. **That is the whole journey from here to GPT-4.**

## Compare against `nanoGPT/model.py`

Open that file. Find the `GPT.forward` method. You'll see:

```python
def forward(self, idx, targets=None):
    ...
    tok_emb = self.transformer.wte(idx)      # (b, t, n_embd) - token embeddings
    pos_emb = self.transformer.wpe(pos)      # (t, n_embd) - position embeddings
    x = self.transformer.drop(tok_emb + pos_emb)
    for block in self.transformer.h:
        x = block(x)                          # attention + MLP
    x = self.transformer.ln_f(x)
    if targets is not None:
        logits = self.lm_head(x)              # (b, t, vocab_size)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
```

Our bigram's forward: `logits = W[xs]`. Three characters of code.
GPT-2's forward: embed, add position, pass through N transformer blocks, final norm, linear to logits. Still ends with a logits tensor and a `cross_entropy` call.

**Same framing. Much richer mechanism.** Module 3 is just expanding each of those inner lines.

## Visualize this

**Watch your model learn, in real time**:

```python
# After training, plot loss curve:
import matplotlib.pyplot as plt
losses_history = []  # collect loss.item() every 10 steps
# ...run training, appending to losses_history...
plt.plot(losses_history)
plt.xlabel("step"); plt.ylabel("loss")
plt.savefig("bigram_loss.png")
```

Expected: loss drops sharply in first 20 steps, then plateaus near ~2.48. That's the model learning the bigram distribution.

**Visualize what the model learned** (after training):

```python
# W has shape (vocab_size, vocab_size)
# Each row is logits for "what comes next given row-index char"
import matplotlib.pyplot as plt
import torch.nn.functional as F

probs = F.softmax(W.detach(), dim=-1)
plt.figure(figsize=(10, 10))
plt.imshow(probs.numpy(), cmap='viridis', aspect='auto')
plt.xticks(range(vocab_size), chars, rotation=90, fontsize=6)
plt.yticks(range(vocab_size), chars, fontsize=6)
plt.xlabel("next char"); plt.ylabel("current char")
plt.title("Learned transition probabilities (neural bigram)")
plt.colorbar()
plt.tight_layout()
plt.savefig("neural_bigram_probs.png")
```

Compare to the heatmap from the counting bigram in Module 0. **They should look nearly identical.** That's the point: gradient descent + cross-entropy converges to the same answer as counting. You've proven the claim empirically.

**Watch it train visually**: https://playground.tensorflow.org/ - same concept (gradient descent on a simple model) but with a visual network. Watch neurons activate, loss decrease.

## Challenge exercises

1. **Extend to trigram.** Make `W` of shape `(vocab_size, vocab_size, vocab_size)` so the model looks at 2 previous chars. Watch loss drop further but training get slower and memory balloon. This demonstrates why brute-force context is doomed.

2. **Add a hidden layer.** Instead of direct lookup, embed each char as a vector `E[char]` (shape `(n_embd,)`), then linear to logits:
   ```python
   embed = nn.Embedding(vocab_size, 32)
   linear = nn.Linear(32, vocab_size)
   # forward:
   x = embed(xs)      # (N, 32)
   logits = linear(x) # (N, vocab_size)
   ```
   No more capable than bigram (still only looks at 1 char), but you've introduced an **embedding**. Module 2 expands from here.

3. **Use batches.** Right now we do `logits = W[xs]` over the entire 1M-char dataset. Not bad for a tiny model, but for real training we sample mini-batches. Add:
   ```python
   for step in range(1000):
       idx = torch.randint(0, len(xs), (256,))
       batch_x = xs[idx]
       batch_y = ys[idx]
       logits = W[batch_x]
       loss = F.cross_entropy(logits, batch_y)
       ...
   ```

## What you now know

- A language model is a function from token-so-far to a probability distribution over next token.
- The simplest neural version is a single weight matrix, trained by gradient descent with cross-entropy.
- Training: forward, loss, backward, update. Repeat.
- GPT is "same thing, with a transformer in the middle."

## What's next

**Module 2** - we introduce non-linearities, proper layers, and the full training-loop anatomy. Then, in Module 3, we replace the trivial lookup with a transformer block and use it to train on Shakespeare for real (the nanoGPT default config).

Open `02_deep_learning/01_a_neuron.md` when ready.
