# 1.3 - Calculus for Learning (Just Enough, Intuitive)

"I don't know calculus" is a fair worry. The good news: for deep learning, you need *one* idea from calculus - the gradient - and PyTorch computes it for you. Understanding what it represents is sufficient.

If you can internalize this lesson, you've crossed a big bridge. This is where most self-taught people stall, so go slowly.

## The one idea: derivatives are "how fast is this changing?"

A **derivative** of a function `f(x)` at a point `x` is a single number answering: "if I nudge `x` slightly up, how much does `f(x)` change?"

Written: `df/dx` or `f'(x)`. Just a number.

Examples you can check by hand:

- `f(x) = x`. If you nudge `x` up by 0.001, `f(x)` goes up by 0.001. So `df/dx = 1` everywhere.
- `f(x) = 2x`. Nudge `x` up by 0.001, `f(x)` goes up by 0.002. `df/dx = 2`.
- `f(x) = x^2`. At `x=3`, nudging `x` by 0.001 gives: `f(3.001) = 9.006001`. So `f` went up by ~0.006. Derivative is ~6 at `x=3`. In general, `df/dx = 2x`. At `x=5`, derivative is 10. Bigger `x`, steeper slope.

Visualize it: the derivative is the **slope of the function's graph** at that point. Positive slope: function going up as we move right. Negative slope: function going down. Zero slope: flat (maximum or minimum).

## Why we care: gradient descent

<div data-viz="gradient_descent"></div>
*Adjust learning rate and starting point; watch the ball roll (or diverge).*

Here's the magic connection. Suppose you have some function `L(w)` that depends on a weight `w`, and you want to find the `w` that makes `L` as small as possible. (`L` stands for "loss" - the thing you want to minimize.)

If `dL/dw > 0` at your current `w`, it means increasing `w` increases `L`. So to decrease `L`, you should decrease `w`.

If `dL/dw < 0`, increasing `w` decreases `L`. So to decrease `L`, you should increase `w`.

In both cases: **move `w` in the *opposite* direction of the derivative.**

Formally: `w_new = w_old - lr * (dL/dw)`

Where `lr` is the **learning rate** - a small positive number (e.g. 0.001) that controls how big a step you take. Too big: you overshoot and bounce around. Too small: you crawl.

That rule - "nudge each weight opposite to its derivative" - is called **gradient descent**. It's the core of every neural network training algorithm.

## From one weight to millions

In a real neural net you don't have one `w`, you have millions or billions of weights. A **gradient** is just the collection of all the individual derivatives, one per weight.

- If `L` depends on weights `w_1, w_2, ..., w_n`, the gradient is the vector `(dL/dw_1, dL/dw_2, ..., dL/dw_n)`.
- For each weight, you do the same nudge: `w_i_new = w_i_old - lr * (dL/dw_i)`.
- You update *all* weights simultaneously in one step.

That is **gradient descent** on a neural network.

## Chain rule (the one rule you need to know the name of)

Functions are nested: you compute `y = f(g(h(x)))`. How do you compute `dy/dx`? You multiply the derivatives of each stage.

```
dy/dx = f'(g(h(x))) * g'(h(x)) * h'(x)
```

Don't worry about the formula. Just take this: **to find the derivative of a composition, you multiply the derivatives of the pieces.**

This matters because a neural network IS a composition of many functions (one per layer). Layer 1 output feeds into Layer 2 feeds into Layer 3 ... feeds into Loss. To find the derivative of Loss with respect to a weight in Layer 1, you apply the chain rule through all the intervening layers.

The algorithm that does this efficiently, starting from the loss and working backwards, is called **backpropagation** ("backprop"). Module 2 Lesson 4 is dedicated to it. Good news: PyTorch does it for you with one line (`loss.backward()`).

## Why this works in practice

A neural net has millions of weights. For each, PyTorch computes the derivative of the loss with respect to that weight. Then the optimizer nudges each weight slightly in the direction that reduces the loss. Do this a few hundred thousand times, and the model gets good. That's training.

It is genuinely not much more complicated than that. The *details* of which weights matter, how fast to nudge, and how to compute gradients efficiently for a billion-parameter model on 8 GPUs - that's the rest of this course. But the core idea is: **slopes tell you which way is downhill; keep walking downhill.**

## Gradients in PyTorch: a taste

You don't compute derivatives by hand. Ever. This is all you do:

```python
import torch

w = torch.tensor([3.0], requires_grad=True)  # "this weight needs gradients"
x = torch.tensor([2.0])

loss = (w * x - 10)**2   # just some loss function. goal: minimize.

loss.backward()           # PyTorch walks backward, fills in .grad

print(w.grad)  # tensor([-16.]), i.e. dL/dw = -16 at w=3
# interpretation: the loss decreases if we INcrease w.

# one gradient-descent step:
with torch.no_grad():     # don't track gradients during the update
    w -= 0.01 * w.grad
    w.grad = None         # clear old gradients

print(w)  # tensor([3.16]) - we moved toward the minimum
```

Run that. Change the initial `w`. Loop it 100 times. Watch the loss go down.

## Visualize this

**Gradient descent is walking downhill**:

```
  loss
    │                     ● start (random weights)
    │                    ╱
    │                   ╱
    │                  ╱  ← each step: -lr × gradient
    │                 ╱
    │                ╱
    │               ╱
    │              ●●●  current position
    │                ╲
    │                 ╲
    │                  ╲
    │                   ●●  minimum (trained model)
    │
    └─────────────────────────────────► weight value
```

The slope (the derivative) at each point tells you which direction is "downhill." Gradient descent follows it.

**Watch this in real time**: run the exercise snippet below. Every step prints `w` and `loss`. You'll see loss drop smoothly as `w` walks to its optimal value.

**3Blue1Brown's visual explanation**: https://www.youtube.com/watch?v=IHZwWFHWa-w (15 min) - literally shows a ball rolling down a loss surface. Watch once, remember forever.

**Or watch it live here:**

```viz
{"viz": "gradient_descent"}
```

Press **Run**. The ball rolls down the loss curve L(w) = (w-5)². It converges to w=5 (the minimum) — if the learning rate is right.

**Now break it:**
- Set learning rate to **0.01** (crawls; doesn't reach the minimum).
- Set learning rate to **1.1** (bounces around, never settles).
- Set learning rate to **1.2** (explodes to infinity — DIVERGED).

This is the most important hyperparameter in ML. Now you've felt why.

**Learning rate tuning intuition**:

```
  Too small lr:                Just right:              Too big lr:
  ●                            ●                        ●
   ●                            ●                         ●
    ●                            ●                    ●
     ●                            ●                       ●
      ●                            ●                  ●
       ●                            ●                        ●
        ●                            ●●●              ●
  (crawls, never reaches)      (converges fast)       (bounces, diverges)
```

We'll return to this picture in Module 4 Lesson 5 (Learning Rate Schedules).

## Exercise - do this one, it teaches you a lot

Save as `~/workspace/llm-course/exercises/02_gradient_descent.py`:

```python
import torch

# goal: find w such that w*2 = 10, i.e. w = 5
# loss = (w*2 - 10)^2
w = torch.tensor([0.0], requires_grad=True)
lr = 0.05

for step in range(50):
    loss = (w * 2 - 10)**2
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        w.grad = None
    print(f"step {step}: w={w.item():.4f}, loss={loss.item():.4f}")
```

Run it. Watch `w` creep toward 5, loss toward 0. Congrats - you just trained a neural network with one parameter.

Then:
1. Try `lr = 0.6` - too big - watch training explode.
2. Try `lr = 0.001` - too small - watch it crawl.
3. Change the target to 7 (change `10` to `14`). Does `w` go to 7?
4. Make `w` a 3-vector. Make the loss `((w * torch.tensor([1.,2.,3.])).sum() - 100)**2`. Does each weight adjust differently? (Yes - each has its own gradient.)

## What to internalize

- Derivative = slope = "how much does output change per nudge of input".
- Gradient = collection of derivatives, one per parameter.
- Gradient descent = nudge each parameter opposite its derivative, by a little bit, repeatedly.
- PyTorch computes all gradients for you with `loss.backward()`.
- Chain rule + backprop = how you compute gradients through a deep net, cheaply.

You don't need to solve calculus problems. You need to be comfortable with the *idea*: slopes tell you which way is downhill.

## Next

`04_probability_basics.md` - softmax and cross-entropy, the last math you need.
