# Visualizations: Where to See, Not Just Read

This course is text-heavy by necessity. But some concepts are vastly easier with visuals. Here's a curated list of the best free interactive and visual resources for every major topic.

Open them *as you read* the corresponding lesson. They'll save you hours of struggle.

---

## Neural network fundamentals

### Neural Network Playground (TensorFlow)
https://playground.tensorflow.org/

**Use with**: Module 2 (deep learning basics).

Drag-and-drop neural network. Watch it classify data in real time. Change activations, layer counts, see effects. The best intro-to-NNs tool ever built.

### 3Blue1Brown's Neural Network Series (YouTube)
https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

**Use with**: Module 1-2.

Grant Sanderson's legendary animated series. Each video is ~20 min. Explains backprop, gradient descent, and (in recent episodes) transformers. Beautiful and correct. **Essential viewing.**

### ConvNet Playground (for CNNs)
https://poloclub.github.io/cnn-explainer/

**Use with**: Module 9.1 (ViT, for the CNN background).

Interactive CNN filter visualization.

---

## The transformer

### LLM Visualization (Brendan Bycroft)
https://bbycroft.net/llm

**Use with**: Module 3 (the transformer).

**This is the single best transformer visualization.** Click through a 3D rendered GPT-2 as it processes tokens. See every tensor's shape, every matmul, every layer. Extraordinary.

### The Illustrated Transformer (Jay Alammar)
https://jalammar.github.io/illustrated-transformer/

**Use with**: Module 3 Lesson 3-5 (attention).

Jay Alammar's blog posts are classics. Static images but each is carefully designed to build intuition.

Related:
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
- [The Illustrated BERT, ELMo](https://jalammar.github.io/illustrated-bert/)

### 3Blue1Brown: Attention in Transformers
https://www.youtube.com/watch?v=eMlx5fFNoYc

**Use with**: Module 3 Lessons 3-4.

Grant's explanation of attention and the Q, K, V matrices. Released 2024, very modern.

### Transformer Circuits (Anthropic)
https://transformer-circuits.pub/

**Use with**: advanced Module 3, Module 7.

Mechanistic interpretability. Visualizes how transformers actually work internally. Deep, slow read. Earth-shattering if you engage.

---

## Attention mechanisms

### Attention Visualizer (BertViz)
https://github.com/jessevig/bertviz

**Use with**: Module 3 Lesson 4-5.

Visualize attention weights from pretrained BERT, GPT-2. Hover over tokens, see what they attend to.

### LLM Attention Heatmap (any Jupyter notebook)
You can cheaply make your own:
```python
import seaborn as sns
sns.heatmap(attention_weights[0, 0].cpu().numpy())
```

Useful on your own trained nanoGPT.

---

## Backprop and gradient descent

### 3Blue1Brown: Backpropagation, intuitively
https://www.youtube.com/watch?v=Ilg3gGewQ5U

**Use with**: Module 2 Lesson 4.

The best backprop explanation, visually.

### Micrograd (Karpathy)
https://github.com/karpathy/micrograd

**Use with**: Module 2 Lesson 4.

200 lines of Python implementing autograd from scratch. Watch Karpathy's [explanation video](https://www.youtube.com/watch?v=VMj-3S1tku0) - 2.5 hours, builds the whole thing from scratch.

---

## Scaling laws

### Tim Dettmers's scaling laws post
https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/

**Use with**: Module 6 Lesson 9.

Dense but crisp analysis of scaling.

### Chinchilla original paper Figure 1
Just flip through the paper - the plots are the core content.

---

## Diffusion models

### Diffusion Explainer
https://poloclub.github.io/diffusion-explainer/

**Use with**: Module 9.3.

Interactive visualization of the diffusion process.

### The Annotated Diffusion Model
https://huggingface.co/blog/annotated-diffusion

**Use with**: Module 9.3.

Walk through DDPM with code + math + explanations. Excellent.

### Jay Alammar's Illustrated Stable Diffusion
https://jalammar.github.io/illustrated-stable-diffusion/

**Use with**: Module 9.3-9.4.

---

## Tokenization

### Tiktoken playground
https://tiktokenizer.vercel.app/

**Use with**: Module 3 Lesson 1, Module 4 Lesson 2.

Paste text, see how GPT-2 / GPT-4 tokenize it. Color-coded tokens. Instant understanding.

### OpenAI Tokenizer
https://platform.openai.com/tokenizer

Same idea, official.

---

## GPUs and compute

### GPU Trainer Profiler (PyTorch Profiler)
Use with PyTorch:
```python
with torch.profiler.profile(...) as p:
    train_step()
print(p.key_averages().table())
```

**Use with**: Module 6 Lesson 5.

Not exactly a visualization, but shows where time is spent.

### Tim Dettmers's GPU guide
https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/

**Use with**: Module 6 Lessons 1-2.

Comparative tables of every major GPU generation. Updated occasionally.

---

## Karpathy's "Zero to Hero" videos

https://karpathy.ai/zero-to-hero.html

**Use with**: the entire course.

Karpathy's lecture series is the closest thing to a companion to this entire course. Notably:
- **Let's build GPT** (2-hour deep dive, builds a mini-GPT from scratch).
- **Let's build the GPT tokenizer**.
- **State of GPT** (conceptual overview).
- **Micrograd** (backprop from scratch).
- **makemore** series (language models from scratch).

Watch at 1.5x. Essential.

---

## Hugging Face Spaces

https://huggingface.co/spaces

**Use with**: every lesson.

Interactive demos. Try LLaVA, Whisper, Stable Diffusion in your browser. No setup.

---

## Arena-style leaderboards (for seeing real outputs)

- **Chatbot Arena** (https://chat.lmsys.org): pairwise human votes on real chatbots.
- **Artificial Analysis** (https://artificialanalysis.ai): latency/cost/quality comparisons.
- **Open LLM Leaderboard** (https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard): benchmark scores.

**Use with**: Module 5 Lesson 7, Module 8 Lesson 4.

---

## Your own visualizations

Once you train your own models (Module 3+ capstones), you should make:

1. **Loss curves**: train and val, over steps. wandb does this automatically.
2. **Attention heatmaps**: visualize what your model learned.
3. **Embedding plots**: t-SNE or UMAP of token embeddings.
4. **Sample outputs over training**: see how quality evolves.

These will be your own personal visual understanding. No tutorial can replace this.

---

## Closing thought

The best "visualization" is a trained model you can poke at. After Module 3's capstone, you have one. After Module 5's, you have a chatbot. Play with them. Look inside them. That's how understanding becomes intuition.
