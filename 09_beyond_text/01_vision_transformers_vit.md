# 9.1 - Vision Transformers (ViT)

Before 2020, computer vision was dominated by Convolutional Neural Networks (CNNs) - specialized architectures with built-in assumptions about 2D structure (locality, translation invariance).

Then the ViT paper asked: **what if we just use transformers on images, pretending they're sequences?** Turns out: with enough data and compute, it works as well as or better than CNNs.

## The idea: images as sequences

To feed an image into a transformer, you need to turn it into a sequence of vectors.

ViT's recipe:
1. Take an RGB image. Say 224×224×3.
2. Split into 16×16 patches. That's 14×14 = 196 patches.
3. Flatten each patch: 16×16×3 = 768 numbers per patch.
4. Linearly project each patch to a vector of dimension `d_model` (e.g. 768).
5. Add positional embeddings (which patch # is this?).
6. Prepend a special `[CLS]` token (will hold the aggregated image representation).
7. Run the resulting 197-length sequence through a transformer encoder.
8. Take the `[CLS]` output, run through a classifier head → class label.

That's it. Same transformer blocks as BERT/GPT. Same attention mechanism. Same MLP.

## Why this works

The transformer doesn't care what's in the tokens. Each token is just a vector. The self-attention learns which patches are informative for each other.

Surprising properties:
- With enough data (ImageNet-21K, JFT-300M), ViT beats CNNs.
- Self-attention heads learn meaningful things: some attend locally, some globally, some to specific object parts.
- Scales beautifully: bigger ViT + more data → better.

## ViT variants

- **ViT-Base** (86M params): 12 layers, 768 dim, 12 heads.
- **ViT-Large** (307M): 24 layers, 1024 dim.
- **ViT-Huge** (632M): 32 layers, 1280 dim.

The same L/d/h scaling you see in LLMs.

## Training tricks

ViT needs lots of data to outperform CNNs. Strategies:
- **Pretrain on huge datasets** (ImageNet-21K, JFT-300M), then fine-tune on the target.
- **Data augmentation**: RandAugment, CutMix, Mixup.
- **Regularization**: dropout, stochastic depth.

For smaller datasets, CNN-transformer hybrids (DeiT, Swin Transformer) often outperform pure ViT.

## Self-supervised vision pretraining

For labeled ImageNet, you need labels. But we don't have labels for internet-scale image data. Enter self-supervised learning:

### Contrastive (DINO, SimCLR, MoCo)
Take two augmentations of the same image. Force their representations to be similar. Force representations of different images to be different.

### Masked (MAE - Masked Autoencoders)
Mask 75% of patches. Reconstruct them. Force the model to infer global structure.

Both work well with ViT. MAE is simpler, matches BERT's masked-language-model approach but for images.

## ViT in the wild

Once trained, you use ViTs for:
- Image classification (direct).
- Image embeddings (feed to other systems - retrieval, clustering).
- Pretraining backbone for detection, segmentation (with modifications).
- As the vision encoder in multi-modal LLMs (LLaVA uses CLIP's ViT).

## In code (using HuggingFace)

```python
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

image = Image.open("cat.jpg")
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

predicted_class = logits.argmax(-1).item()
print(model.config.id2label[predicted_class])
```

Four lines to run a state-of-the-art image classifier. Compare to what CNNs required a decade ago.

## ViT vs CNNs today

Current state (2024-26):
- Pure ViTs win at scale (billions of parameters, billions of images).
- CNNs (or hybrids) still win on efficiency and smaller data.
- Most modern vision-language models use ViT-based vision encoders.
- Segment Anything (SAM), CLIP, DINOv2 all use ViT.

CNN architectures are still widely deployed but research has shifted to transformers.

## Visualize this

**An image as patches, pictorially**:

```
  Original image (224 × 224 pixels, RGB):
  ┌────────────────────────────────────────┐
  │  ╱╲ ╱╲ ╱╲ ╱╲ ╱╲ ╱╲ ╱╲ ╱╲ ╱╲ ╱╲ ╱╲ ╱╲  │  224 px tall
  │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  │
  │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  │
  │ ... your cat photo ...                  │
  │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  │
  └────────────────────────────────────────┘
         ↓  cut into 16×16 patches  ↓
  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
  │P0│P1│P2│P3│P4│P5│P6│P7│P8│P9│10│11│12│13│
  ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
  │14│15│16│17│18│19│20│21│22│23│24│25│26│27│
  ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
  │28│29│30│31│32│33│34│35│36│37│38│39│40│41│
  ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
  ...
  └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
   14 columns × 14 rows = 196 patches

   each patch: 16 × 16 × 3 (RGB) = 768 numbers
   flatten → a vector of 768 numbers
   linear project → a vector of n_embd (e.g. 768)

   → now each image is a sequence of 196 "tokens"
   → feed to transformer exactly like text
```

That's the entire ViT insight.

**Text transformer vs ViT**:

```
  Text transformer (GPT-2):          Image transformer (ViT):
  ──────────────────────              ──────────────────────────

  "hello world" → tokens             image → split into patches
  [512, 1059]                         [14×14 = 196 patches]
      ↓                                    ↓
  embed each token                    embed each patch
  (lookup table)                       (linear projection)
      ↓                                    ↓
  add position embed                  add position embed
      ↓                                    ↓
  ┌──────────────┐                    ┌──────────────┐
  │ Transformer   │                   │ Transformer   │
  │ (12 layers)    │                  │ (12 layers)    │
  │ attention +   │                   │ attention +    │
  │ MLPs          │                   │ MLPs           │
  └──────┬───────┘                    └──────┬───────┘
         ↓                                    ↓
  predict next token                    take [CLS] output
                                         → classification head
                                             → class label

  Same architecture. Different task heads.
```

**Attention patterns in a trained ViT**:

```
  Visualize ViT attention maps (each head of each layer):

  Query patch: center of image
  Attention weights shown as brightness:

  Layer 1, head 1:         Layer 6, head 3:        Layer 11, head 7:
  ░░░░░░░░░░░░░░            ▒▒▒▒▒▒▒▒▒▒▒▒            ▓▓▓▓▓▓▓▓▓▓▓▓
  ░░░░▒▒▒▒▒▒░░░░           ▒▒▒░░░░░░░▒▒▒            ░░░░▓▓▓▓▓▓░░░░
  ░░▒▒▓▓▓▓▒▒░░░░            ▒░░░░░░░░░░░▒            ░░▓▓▓▓▓▓▓▓░░░░
  ░░▒▒▓▓██▓▓▒▒░░            ▒░░░▓▓▓▓░░░░░            ░░▓▓██████▓▓░░
  ░░░░▒▒▓▓▒▒░░░░            ▒▒░░▓▓▓▓░░░░▒            ░░▓▓██████▓▓░░
  ░░░░░░░░░░░░░░            ▒▒▒░░░░░░░▒▒▒            ░░░░▓▓▓▓▓▓░░░░

  Early: local (neighboring patches)
  Middle: long-range selective
  Late: object-focused
```

Different from CNN: ViT doesn't hard-code local bias. It can learn both local and global attention from data.

**ViT scaling (why data matters)**:

```
  ViT vs CNN, accuracy on ImageNet:

  Training data            CNN (ResNet)   ViT
  ───────────────────       ────────────   ───
  1M images (ImageNet)       76%          72%     CNN wins
  14M images (ImageNet-21K)  78%          78%     tied
  300M images (JFT-300M)      80%           84%     ViT wins ↑
  10B images                 82%           88%     ViT wins more ↑

  CNNs have built-in biases (locality).
  ViTs must learn everything.
  With enough data, ViT's flexibility wins.
```

**ViT sizes in the wild**:

```
  ViT-Base   (86M params)   12 layers, 768 dim, 12 heads
  ViT-Large  (307M)          24 layers, 1024 dim, 16 heads
  ViT-Huge   (632M)          32 layers, 1280 dim, 16 heads
  ViT-G      (2B)            48 layers, 1408 dim, 16 heads (ViT-22B exists too)

  Model in serious use today:
    CLIP uses ViT-Large   (by OpenAI)
    LLaVA uses CLIP's ViT-Large
    Stable Diffusion uses ViT-Large as text-to-image encoder
    DINOv2 uses ViT-L/H for self-supervised vision features
```

**The full ViT architecture**:

```
  Image (224×224×3)
       │
       ▼
  Split into 16×16 patches  → 196 patches of 768 values each
       │
       ▼
  Linear projection          → 196 vectors of n_embd
       │
       ▼
  Prepend learnable [CLS] token → 197 tokens
       │
       ▼
  Add positional embeddings   → encodes which patch is which
       │
       ▼
  [transformer blocks × N]    → same as GPT!
       │
       ▼
  Take [CLS] output            → aggregated image representation
       │
       ▼
  Linear classifier            → class logits
```

## Exercises

1. Read the [ViT paper](https://arxiv.org/abs/2010.11929) abstract + Figure 1. Understand the patchify step.

2. Run a pretrained ViT on your own images using HuggingFace (code above). Check predictions.

3. Visualize attention from a pretrained ViT. Some tokens attend locally, some globally. Use tools like [ViT Explorer](https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb).

4. Think: why would training on patches make the model "translation invariant" in a different way than CNNs? (CNNs enforce it by weight sharing; ViT learns it statistically from data.)

## Next

`02_clip_and_embeddings.md` - linking images and text.
