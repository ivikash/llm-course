# Module 3 - The Transformer via nanoGPT

This is the heart of the course. By the end, `nanoGPT/model.py` will not be mysterious - it will be obvious. And once you understand nanoGPT, every other transformer (Llama, Claude, Gemini) is a variation you can pick up in an afternoon.

## Lessons

- `01_tokenization.md` - turn text into integers. Character vs BPE.
- `02_embeddings.md` - integers become vectors.
- `03_attention_intuition.md` - the "tokens talking" operation, first pass, no math.
- `04_attention_mathematically.md` - Q, K, V, the softmax, the mask.
- `05_multi_head_attention.md` - several smaller attentions in parallel.
- `06_mlp_block.md` - the other half of a transformer block.
- `07_layernorm_and_residual.md` - the glue that makes deep nets train.
- `08_full_block_walkthrough.md` - put it together: one complete block.
- `09_positional_encoding.md` - how the model knows word order.
- `10_lm_head_and_sampling.md` - turning vectors back into words.
- `11_train_py_walkthrough.md` - every line of `nanoGPT/train.py`.
- `capstone_shakespeare.md` - train the baby GPT yourself on Shakespeare.

## The pacing

Module 3 is ~15-25 hours. Don't rush. Attention is worth understanding three times over. If you feel confused, re-read. Talk to me. The third-pass "aha" moment is where real understanding crystallizes.

## Before you start

Open `~/workspace/nanoGPT/model.py` in a tab. You'll be looking at specific lines in each lesson.

Also open `~/workspace/nanoGPT/train.py` - we'll dissect the training loop at the end.
