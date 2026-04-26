# Module 2 - Deep Learning from Zero

A neural network is a function with many knobs. Training = turning the knobs.
This module builds the full training-loop vocabulary, which we'll use in every
remaining module.

## Lessons

- `01_a_neuron.md` - A "neuron" is `y = w @ x + b`. Then many in parallel -> a `nn.Linear` layer. We build one by hand.
- `02_activations_nonlinearity.md` - Why stacking linear layers alone is pointless (algebraically equivalent to one layer). Enter ReLU, GELU, Tanh, SiLU. What each one looks like and when to use it.
- `03_loss_functions.md` - MSE (for regression), cross-entropy (for classification / LLMs). When you'd use each. Numerically-stable implementations.
- `04_backprop_and_sgd.md` - The algorithm that computes gradients. Walkthrough of a 2-layer MLP by hand. Then `loss.backward()` in PyTorch does it all. The concept of a computation graph.
- `05_optimizers_adam_adamw.md` - Plain SGD is slow. Momentum helps. Adam adds per-parameter learning rates. AdamW fixes weight-decay. Every modern LLM uses AdamW (or a variant like Muon, which nanochat also uses). Line reference: `train.py` configure_optimizers method in nanoGPT.
- `06_regularization_dropout_weightdecay.md` - Keeping the model from memorizing. Dropout (randomly zero activations), weight decay (L2 on weights).
- `07_training_loop_anatomy.md` - The 10 lines at the center of every training script: forward, loss, zero_grad, backward, step. Point to these lines in `nanoGPT/train.py`.
- `capstone_mlp.md` - Train a multi-layer perceptron to classify MNIST digits (or to continue character sequences like our bigram but with a hidden layer). CPU-friendly.

Each lesson ~20-40 minutes reading + exercises. Module 2 is ~10 hours total.

## Why this matters for LLMs

Every concept in Module 2 is present inside a transformer:
- `nn.Linear` layers (neurons in parallel): every attention projection, every MLP block.
- Activations (GELU inside the MLP block of `nanoGPT/model.py`, SiLU/SwiGLU in `nanochat/nanochat/gpt.py`).
- Cross-entropy loss: the LM objective.
- AdamW optimizer: nanoGPT's default.
- Dropout + weight decay: standard regularization used in both.

If you understand Module 2, the transformer in Module 3 looks like "a particular way of stitching these familiar pieces together." That's the right mental model.
