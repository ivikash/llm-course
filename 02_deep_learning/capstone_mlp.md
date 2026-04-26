# Module 2 Capstone - A 2-Layer MLP Character Predictor

Extend the bigram from Module 1's capstone into a proper neural network with:
- An embedding layer
- A hidden layer with a GELU activation
- A linear output layer
- Trained with AdamW, cross-entropy, mini-batches

You'll see every piece from Module 2 in use. ~80 lines.

Save as `~/workspace/llm-course/exercises/04_mlp.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- data ---
with open("~/workspace/nanoGPT/data/shakespeare_char/input.txt") as f:
    text = f.read()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long, device=device)

# use context of 8 chars to predict the 9th
context = 8
def get_batch(batch_size=64):
    idx = torch.randint(0, len(data) - context - 1, (batch_size,), device=device)
    x = torch.stack([data[i:i+context] for i in idx])           # (B, context)
    y = data[idx + context]                                       # (B,)
    return x, y

# --- model ---
class MLP(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden=128, context=8):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim * context, hidden)
        self.fc2 = nn.Linear(hidden, vocab_size)
    def forward(self, x):                     # x: (B, context) integer IDs
        e = self.emb(x)                       # (B, context, embed_dim)
        e = e.flatten(1)                      # (B, context*embed_dim)
        h = F.gelu(self.fc1(e))               # (B, hidden)
        return self.fc2(h)                    # (B, vocab_size)

model = MLP(vocab_size).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)

# --- train ---
for step in range(3000):
    x, y = get_batch()
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 300 == 0:
        print(f"step {step}: loss={loss.item():.4f}")

# --- sample ---
model.eval()
seed = "ROMEO:  "[:context]
out = list(seed)
with torch.no_grad():
    for _ in range(500):
        x = torch.tensor([[stoi[c] for c in out[-context:]]], device=device)
        logits = model(x)
        probs = F.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1).item()
        out.append(itos[ix])
print("".join(out))
```

Run:

```bash
python ~/workspace/llm-course/exercises/04_mlp.py
```

Expected: loss drops from ~4.2 to ~2.0-2.3. Samples are noticeably more word-like than the bigram.

## What you've built

- An **embedding** layer: each character -> 32-dim vector.
- A **hidden layer**: concatenate 8 embeddings (256 numbers), pass through linear + GELU -> 128 numbers.
- An **output layer**: linear from 128 to vocab_size -> logits.
- Trained with **AdamW**, **cross-entropy loss**, **mini-batches**.

This is actually a classical "Bengio 2003" style neural language model - the ancestor of modern LLMs. With a context of 8 chars and a single hidden layer, it's already qualitatively better than bigram. Now imagine replacing the "concat-and-linear" with a transformer, context of 2048, billions of parameters, trained on all the internet. That's GPT.

## Visualize this

**Data flow through our MLP, shape by shape**:

```
  input batch: characters like "ROMEO:"
                                   │
                                   ▼
  Tokenized:  [15, 17, 12, 5, 17, 27]  shape (B=64, context=8)
                                   │
                                   ▼
  Embedding: each char → 32-dim vector
                                       shape (B=64, context=8, 32)
                                   │
                                   ▼
  Flatten:                        shape (B=64, 256)
                                   │
                                   ▼
  Linear fc1:                     shape (B=64, 128)
                                   │
                                   ▼
  GELU (nonlinearity):             same
                                   │
                                   ▼
  Linear fc2:                     shape (B=64, vocab_size=65)
                                   │
                                   ▼
  Softmax → probability distribution over next char
```

When you read nanoGPT's model.py next module, this exact pattern appears - just with attention in place of `fc1`.

**Sample outputs over training** (run this to see):

```python
# after every 500 training steps:
model.eval()
print(sample(seed="ROMEO:  ", max_new=100))
model.train()
```

You'll observe a progression like:
- Step 0: "ROMEO:  xW!4kq9a7$kl..." (pure noise)
- Step 500: "ROMEO:  aht ssto tsere t wh t sou..." (starts to look like text)
- Step 3000: "ROMEO:  thou shalt not ate the streteth of..." (almost words)

Watching your model learn Shakespeare over time is one of the most satisfying experiences in ML. Do it.

**Plot the training curve**:

```python
losses = []
# in training loop:
if step % 50 == 0:
    losses.append(loss.item())

# after training:
import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel("step × 50"); plt.ylabel("loss")
plt.savefig("mlp_loss.png")
```

You'll see the shape: rapid drop from ~4.2, then slow decay toward ~2.0. Classic.

## Try these modifications

1. Increase `context` to 16. Does loss go lower? (Yes - more signal.)
2. Increase `hidden` to 512. Does it help? By how much? (Plot it.)
3. Add a second hidden layer. Does it help more?
4. Add `nn.Dropout(0.2)` between fc1 and fc2. Does train/val gap change?

## What you're ready for now

You've trained neural networks. You know forward, loss, backward, update. You've used embeddings, linear layers, activations, AdamW. The next module replaces `fc1` with **attention + MLP blocks** - that's the transformer. Every concept around it is already yours.

## Next

Module 3. Open `03_transformers_nanogpt/01_tokenization.md`.
