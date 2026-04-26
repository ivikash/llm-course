# 9.2 - CLIP and Multimodal Embeddings

CLIP (Contrastive Language-Image Pre-training) is OpenAI's 2021 model that learned a **shared embedding space** for images and text. It's the quiet backbone of most modern visual AI - Stable Diffusion uses it, LLaVA uses it, DALL-E 2 uses it.

Understanding CLIP = understanding the vocabulary of modern vision-language systems.

## The core idea

Train two encoders jointly:
- **Image encoder** (ViT or ResNet): image → vector.
- **Text encoder** (Transformer): text → vector.

For matched (image, caption) pairs, make the two vectors similar. For mismatched pairs, make them dissimilar.

After training: an image of a cat and the text "a photo of a cat" produce similar embeddings. An image of a dog and "a photo of a cat" produce dissimilar ones.

## The loss: contrastive (InfoNCE)

For a batch of N (image, text) pairs:

1. Encode all N images → `I_1, ..., I_N` (each a vector).
2. Encode all N texts → `T_1, ..., T_N`.
3. Compute similarity matrix: `S[i,j] = cosine_similarity(I_i, T_j)`. Shape (N, N).
4. The **diagonal** should be high (matched pairs). The **off-diagonal** should be low.

Loss (symmetric cross-entropy over rows and columns):

```python
# Temperature-scaled logits
logits = (I @ T.T) * exp(tau)   # (N, N)

# Loss: target is the diagonal (identity)
targets = torch.arange(N)
loss_i = F.cross_entropy(logits, targets)        # image → text
loss_t = F.cross_entropy(logits.T, targets)      # text → image
loss = (loss_i + loss_t) / 2
```

This is called **InfoNCE** or **NT-Xent**. It's the standard contrastive objective.

## Training data

OpenAI's original CLIP: 400M (image, alt-text) pairs scraped from the web. Alt-text is free supervision - people wrote captions so screen readers could describe the image.

Scale was crucial. Smaller datasets don't produce useful CLIP models.

## Why this works

After training:
- Images about "dogs" (any breed, any angle) cluster in embedding space.
- Texts about "dogs" cluster near the same region.
- You have an **aligned multimodal space**.

This alignment lets you:
1. **Search images by text** (retrieval).
2. **Classify images zero-shot** (by comparing to text descriptions).
3. **Guide image generation** (diffusion models use CLIP to steer toward text).
4. **Generate captions** (retrieval-augmented).

## Application 1: zero-shot image classification

No fine-tuning needed.

```python
import clip
import torch
from PIL import Image

model, preprocess = clip.load("ViT-B/32")
image = preprocess(Image.open("photo.jpg")).unsqueeze(0)
text = clip.tokenize([
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a bird",
])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    # normalize + similarity
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100 * image_features @ text_features.T).softmax(dim=-1)

print(similarity)   # probabilities over the three labels
```

This is **zero-shot**: you didn't train anything for "cat/dog/bird" classification. The model just knows.

With carefully chosen prompt templates ("a photo of a [X]"), CLIP ViT-L matches supervised ImageNet models.

## Application 2: image retrieval

Index millions of images by encoding each into a vector. Store vectors in a vector database (FAISS, Pinecone). At query time: encode the query text, find nearest image vectors.

Powers: Google Images, Pinterest, Unsplash search, photo libraries on your phone.

## Application 3: conditioning diffusion models

In Stable Diffusion:
1. User provides text prompt.
2. CLIP text encoder converts prompt to embeddings.
3. Diffusion U-Net cross-attends to those embeddings at each denoising step.
4. Generated image is steered toward the prompt.

No CLIP → no text-to-image.

## Application 4: vision-language models (VLMs)

In LLaVA:
1. Image → CLIP's image encoder → visual embedding.
2. A projector maps it to the LLM's embedding dim.
3. LLM sees it as "visual tokens" alongside text.

The LLM inherits CLIP's visual understanding for free.

## Successor models

CLIP was 2021. The field has moved:

- **OpenCLIP** (LAION): open-source reproduction at larger scale. Often better than original CLIP.
- **SigLIP** (Google 2023): replaces softmax-cross-entropy with sigmoid loss. Simpler, scales better, often preferred for new projects.
- **EVA-CLIP** (BAAI): larger, stronger.
- **DFN** (Apple): trained on heavily filtered data, very strong.
- **MetaCLIP** (Meta): published data + training recipe.

If you're picking a CLIP today, SigLIP or OpenCLIP ViT-L/14 are usually right.

## Limitations

- **Fine-grained distinctions**: CLIP confuses similar things ("golden retriever" vs "labrador retriever" may fail).
- **Counting**: CLIP doesn't count well.
- **Spatial relations**: "cat on top of dog" vs "dog on top of cat" - CLIP often treats as same.
- **Text in images**: reads text roughly but not precisely (SigLIP and newer are better).
- **Bias**: inherits biases from web data.

## How to use one today

### Via HuggingFace

```python
from transformers import CLIPModel, CLIPProcessor
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# same zero-shot classification pattern
inputs = processor(
    text=["a cat", "a dog"], images=image, return_tensors="pt", padding=True
)
outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=1)
```

### Via `open_clip`

```python
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k'
)
tokenizer = open_clip.get_tokenizer('ViT-B-32')
```

More model options (SigLIP, EVA, various training datasets).

## Key paper

Radford et al. 2021: ["Learning Transferable Visual Models From Natural Language Supervision"](https://arxiv.org/abs/2103.00020). 48 pages. Read the abstract + Figure 1 + intro. That's enough.

Figure 1 is beautiful: shows the contrastive training setup on the left and zero-shot classification on the right, same architecture.

## Exercises

1. Run the zero-shot classification snippet on 5 photos from your phone. Try "a photo of a [X]" for various X. Which ones work, which fail?

2. Compute: for CLIP ViT-L/14, a 224x224 image produces ~50 tokens of embeddings. What's the total compute to encode 1 million images on an A100? (Hint: ~1 hour at good batch size.)

3. Use CLIP to build a simple image retrieval: encode ~100 images, encode a query text, find the closest. ~20 lines.

4. Read the CLIP paper's Section 3.1 (training details) once you're comfortable with the basics.

## Next

`03_diffusion_models.md` - how CLIP's text embeddings steer image generation.
