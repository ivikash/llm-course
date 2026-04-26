# 03 - What Is a Language Model, Physically?

This is your first real lesson. By the end, you'll have a working "language model" on your laptop, built from scratch in Python, no libraries beyond the standard library.

It won't be a transformer. It won't be good. But it will be a language model, and after this lesson you'll know in your bones what "predicting the next word" means.

## The idea in one sentence

**A language model is a function that takes a sequence of words and gives you a probability distribution over what word comes next.**

Let's break that down.

- "A function that takes a sequence of words" - e.g. you give it `["the", "cat", "sat", "on", "the"]`.
- "Gives you a probability distribution" - it returns a mapping like:
  ```
  "mat"    -> 0.40
  "floor"  -> 0.12
  "couch"  -> 0.08
  "dog"    -> 0.005
  ...
  "pineapple" -> 0.00001
  ```
- The probabilities sum to 1.0 across all possible words.
- We then **sample** from this distribution to pick the next word. Or if we want deterministic output, we just take the one with the highest probability ("mat" here).
- To generate a longer piece of text, we add the chosen word to the sequence and call the function again.

That's it. GPT-4 is this. Claude is this. Everything else - attention, layers, training loops, scaling - is implementation detail serving this one idea.

## The simplest possible language model: the bigram model

Forget neural networks for a moment. Here's the oldest, dumbest "language model" that still technically meets the definition:

**A bigram model just counts pairs.** Given a big pile of text, count how often each word is followed by each other word. Turn those counts into probabilities. Done.

Example: suppose you trained on exactly this text:

```
the cat sat
the cat slept
the dog ran
the dog sat
```

You'd count pairs like:
- `the` is followed by `cat` 2 times, `dog` 2 times. -> P(cat | the) = 0.5, P(dog | the) = 0.5.
- `cat` is followed by `sat` 1 time, `slept` 1 time. -> P(sat | cat) = 0.5, P(slept | cat) = 0.5.
- `dog` is followed by `ran` 1 time, `sat` 1 time. -> P(ran | dog) = 0.5, P(sat | dog) = 0.5.

Now to generate text:
1. Start with a word, e.g. "the".
2. Look up what can follow "the". Sample from {cat: 0.5, dog: 0.5}. Get "cat".
3. Look up what can follow "cat". Sample from {sat: 0.5, slept: 0.5}. Get "slept".
4. Look up what can follow "slept". Nothing in training data. Stop.

Output: "the cat slept". Not bad! Completely circular on real data but the machinery is right.

## Build it yourself

Save this as `~/workspace/llm-course/exercises/01_bigram.py` and run it.

```python
# A bigram character-level language model, in pure Python.
# ~30 lines, no libraries.

import random
from collections import Counter, defaultdict

# 1. Load some text. Any text. Let's use nanoGPT's Shakespeare.
with open("~/workspace/nanoGPT/data/shakespeare_char/input.txt") as f:
    text = f.read()

# We'll work character-level (simpler than words). So a "token" here is one character.
# 2. Build bigram counts: how often does char Y follow char X?
counts = defaultdict(Counter)
for a, b in zip(text, text[1:]):
    counts[a][b] += 1

# 3. Turn counts into probabilities.
probs = {}
for a, ctr in counts.items():
    total = sum(ctr.values())
    probs[a] = {b: c / total for b, c in ctr.items()}

# 4. Sample! Start from a seed character, generate 500 characters.
def sample_next(char):
    dist = probs[char]
    chars, weights = zip(*dist.items())
    return random.choices(chars, weights=weights, k=1)[0]

out = ["R"]
for _ in range(500):
    out.append(sample_next(out[-1]))
print("".join(out))
```

Run it:

```bash
python ~/workspace/llm-course/exercises/01_bigram.py
```

You'll get something like:

```
R g, bed, fred thinde:
CENBRUCo he be s sitherean?
UTou,
T mand tendorinoud herd, s I; t tinearin's Wand w t b
```

Gibberish, but Shakespeare-flavored gibberish. The spacing is right. The punctuation pattern is Shakespeare-ish. Even capital letters appear after newlines. You see? **The model learned something real, just by counting pairs.**

## Now look what's missing

The bigram model only looks at the **previous 1 character** to predict the next. That's why it produces gibberish - it has no sense of words, phrases, meaning, or long-range structure.

What if we looked at the previous 2 characters? That's a **trigram** model. 3? 4-gram. 10? 10-gram. You'd need exponentially more data, and still not really understand anything.

**The transformer's entire job is to fix this.** A transformer is a language model whose "previous context" can be thousands of tokens long, AND it learns rich representations of those tokens so it can *generalize*, not just memorize pairs.

That's the whole plot of this course.

## Wait, what's the connection to a real GPT?

Inside `nanoGPT/model.py` and `nanochat/nanochat/gpt.py`, buried in a lot of code, is the same fundamental recipe:

1. Take a sequence of tokens.
2. Compute a probability distribution over the next token. (They call this `logits` - raw numbers - then apply softmax to get probabilities.)
3. During training, compare the predicted distribution to the real next token and measure error.
4. During generation, sample from the distribution.

The machinery around this - embeddings, attention, layer norm, etc. - is all just a much smarter way to compute step 2. Our bigram model "computes step 2" using a giant lookup table. GPT "computes step 2" using billions of parameters and attention. Same goal.

I'll show you this directly. Open `~/workspace/nanoGPT/model.py` and search for `F.cross_entropy` (or jump to line ~190). You'll see something like:

```python
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
```

That's the "compare predicted distribution to real next token and measure error" step. Almost the whole model is in service of computing good `logits`.

## Exercises

1. **Run** the bigram script above. Modify the seed character from "R" to something else. Generate longer output.
2. **Extend** to a trigram model. Count triples `(a, b, c)` where `c` follows the pair `(a, b)`. Sample using the last two chars as context. Output should look noticeably less random. Hint: change `defaultdict(Counter)` to be keyed by pairs.
3. **Think**: A bigram model with a vocabulary of 100 unique chars needs a table of 100 x 100 = 10,000 numbers to represent. What if you used words instead of characters, with a vocab of 50,000 words? You'd need 50,000 x 50,000 = 2.5 billion numbers for bigrams, and 50,000^3 = 125 trillion for trigrams. That's why nobody uses n-gram models for serious LLMs. This is why we need neural networks - to *compress* this knowledge into a finite set of parameters.

## Visualize this

**See it yourself (essential)**:
- **Tiktokenizer** (https://tiktokenizer.vercel.app/) - paste any text, see how real LLMs split it into tokens. Colored. Takes 10 seconds.
- **3Blue1Brown: "What is a GPT?"** (https://www.youtube.com/watch?v=wjZofJX0v4M) - 27 minutes, visual walkthrough. Watch after this lesson.
- **bbycroft.net/llm** (https://bbycroft.net/llm) - 3D animation of a GPT processing tokens. Click through.

**Visualize your bigram model after running it**:

```python
# after building probs dict in 01_bigram.py, add:
import matplotlib.pyplot as plt
import numpy as np

# build a heatmap of char -> char transition probabilities
chars_list = sorted(probs.keys())
n = len(chars_list)
matrix = np.zeros((n, n))
for i, a in enumerate(chars_list):
    for j, b in enumerate(chars_list):
        matrix[i, j] = probs[a].get(b, 0)

plt.figure(figsize=(10, 10))
plt.imshow(matrix, cmap='hot')
plt.xticks(range(n), chars_list, rotation=90, fontsize=6)
plt.yticks(range(n), chars_list, fontsize=6)
plt.xlabel("next char"); plt.ylabel("current char")
plt.title("Bigram transition probabilities")
plt.colorbar()
plt.tight_layout()
plt.savefig("bigram_heatmap.png")
```

Open `bigram_heatmap.png`. **Bright cells = "this letter usually follows this one."** You'll see:
- A bright column at the space " " and newline positions (everything gets followed by them).
- "q" almost always followed by "u".
- Vowels have broad rows (many things follow them).
- Capital letters mostly follow newlines.

You've seen the language model's "knowledge" as a picture.

## The punchline

You have just built a language model. A real one, if a weak one.

Everything we do from now on is:
- **Module 1-2**: learn the math and neural-network machinery.
- **Module 3**: replace the lookup-table "count pairs" approach with a transformer that *learns* what matters.
- **Module 4-5**: add the supporting infrastructure - tokenizers, fine-tuning, RL, evaluation.
- **Module 6-8**: scale it up and organize humans around it.

Congratulations. You're no longer a complete beginner. You know what the thing *is*.

## Next

Module 1 starts. Open `01_foundations/01_vectors_and_matrices.md`.
