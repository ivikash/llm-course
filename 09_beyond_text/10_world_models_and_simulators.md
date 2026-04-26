# 9.10 - World Models and Simulators

The most speculative lesson in this course. Cutting-edge, unresolved, potentially huge. Worth understanding because it's where several research threads converge.

## The claim

A **world model** is a model that learns the dynamics of an environment - how states evolve given actions. If we train big enough generative models on enough diverse video / sensor / action data, do they become general-purpose simulators?

Strong form: "Sora is an early AGI precursor because it's learning physics implicitly."
Weak form: "Video models learn useful representations of the world that help downstream tasks."

Both positions have smart advocates and smart skeptics.

## History

### Classical world models (pre-deep-learning)
Explicit physics simulations: finite-element methods for solid mechanics, CFD for fluids, Unity/Unreal for games. Painstakingly hand-crafted.

### Neural world models (2018+)

**Ha and Schmidhuber 2018, "World Models"**: agents that learned a generative model of their environment in latent space, then used it to plan without interacting with the real environment. Trained via VAE + RNN. Limited scope (small toy domains).

**Dreamer** series (Hafner et al. 2019, 2020, 2023):
- Train a latent-state world model.
- Do RL entirely in imagination (rollouts of the learned model).
- Apply to Atari, physical control, Minecraft.
- DreamerV3 (2023) solved tasks zero-shot using world model rollouts.

### Large-scale world models (2023+)

**Genie** (DeepMind 2024): given a single 2D game image, generates a playable game - you can press "left/right/jump" and the model shows what would happen. Trained on 200,000+ hours of game footage without any labels.

**Genie 2** (DeepMind 2024): 3D environments. Given an image, generates explorable, consistent 3D worlds for ~1 minute.

**Sora** (OpenAI 2024): OpenAI described Sora as exhibiting "emerging simulation capabilities" - object permanence, 3D consistency, interactions - though not robust.

**UniSim** (Stanford / Google 2023): video generation conditioned on actions. "Here's a scene, here's an action (robot arm grabs cup), here's the result."

## The intuition

Why might video models be world models?

To predict the next frame correctly, a model has to understand:
- What objects are in the scene.
- How they move (gravity, inertia, friction).
- Cause and effect (I drop a glass → it falls → it breaks).
- Object permanence (hidden things still exist).
- 3D structure (even from 2D images).

Do these emerge from scale? Somewhat. Current evidence:
- Sora demonstrates impressive short-term physical plausibility.
- Genie generates navigable worlds.
- Dreamer solves control tasks via imagination.

But they break in revealing ways:
- Objects teleport or multiply.
- Gravity inverts mid-scene.
- Causality violated (water flows uphill, glass unbreaks).
- Long-horizon consistency fails.

So world models learn *some* physics, not reliably.

## Implications if world models work

### For robotics
Robots need to plan. Planning requires simulating actions. Currently done in hand-coded simulators (MuJoCo, Isaac Sim). A learned world model could:
- Simulate novel environments zero-shot.
- Train robotic policies in simulation.
- Provide unlimited training data.

### For games
Procedural content beyond scripted assets. Infinite novel game worlds.

### For science
Simulate experiments faster than real experiments. Protein folding, materials science.

### For AGI discussions
Sutskever, Hassabis, and others have argued that world-modeling is necessary for general intelligence. An agent that can simulate outcomes of hypothetical actions is closer to "reasoning about the world".

## Implications if they don't

Current models may be **memorizing trajectories** rather than learning underlying physics. Memorization doesn't generalize to novel situations.

Counterarguments:
- Sora generates novel scenes it never saw - so it's not pure memorization.
- But: it breaks when asked for novel physics (objects behaving in unusual ways).

The debate is unresolved. Research direction for the 2025-2030 decade.

## Embodied AI (adjacent)

Robotics labs combine:
- **Vision models** (perceive environment).
- **LLMs** (reason about tasks).
- **Action policies** (control motors).
- **World models** (plan by imagining).

Examples:
- Google's RT-1, RT-2 (robotic transformers).
- Tesla's Optimus.
- Figure, 1X, various startups.
- Physical Intelligence (pi-0): robot foundation model.
- Stanford Mobile ALOHA, GR00T (NVIDIA), Helix (Figure).

Robotics has been "5 years away" for 40 years. The LLM+vision+world-model stack has finally moved the needle in 2023-2025. Still far from reliable general-purpose robots.

## Video games as training grounds

- **SIMA** (DeepMind 2024): agent playing many different video games via vision + natural language instructions.
- **OpenAI Five / AlphaStar** (earlier): RL agents for specific games.
- Future: general-purpose agents in open-world games like Minecraft.

Games are tractable because they have rules, clear rewards, and unlimited training data.

## Related concept: simulators for LLM training

A separate use of "world model" phrase: training LLMs inside simulations.

- **Verifiers** that check LLM outputs (math, code, game states).
- **Agentic simulations** where LLMs interact and learn from consequences.
- **SIM-TO-REAL**: train in simulation, deploy in reality.

DeepSeek's R1 and OpenAI's o1 can be viewed as LLMs learning in "simulators" of reasoning problems (verifiable math).

## Research frontier

Active questions:
- Do video models actually learn physics, or just plausible-looking trajectories?
- Can a Sora-style model power a robot?
- What's the right abstraction: pixel-level video, latent states, symbolic world models?
- How do you evaluate world-modeling capability rigorously?

No consensus yet. Very active area.

## Key papers / references

- Ha and Schmidhuber 2018, "World Models": https://arxiv.org/abs/1803.10122
- Dreamer V3 (Hafner 2023): https://arxiv.org/abs/2301.04104
- Genie (Bruce 2024): https://sites.google.com/view/genie-2024/
- Genie 2 (DeepMind 2024): https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/
- Sora's technical report (OpenAI 2024): https://openai.com/index/video-generation-models-as-world-simulators/
- UniSim (Yang 2023): https://arxiv.org/abs/2310.06114

## Criticism worth engaging with

Yann LeCun (Meta): video generation ≠ understanding. Argues that current models are "autoregressive" generators that don't have a real causal model of the world. He advocates "JEPA" (Joint Embedding Predictive Architectures) as an alternative.

Gary Marcus: deep learning fundamentally can't do common sense reasoning without symbolic components.

These critiques are partially right. Current models break in ways that suggest shallow physical understanding. But they're improving fast.

## What to expect in 2026-2028

- Longer-coherence video (5+ minutes).
- Action-conditioned video (controllable simulation).
- Robotics foundation models that generalize across tasks.
- Hybrid symbolic-neural world models.

If 2020-2024 was "LLMs took off", 2024-2028 may be "world models took off".

## Exercises

1. Read Sora's technical blog (10 min). Note the "world simulator" framing.

2. Try Genie 2 if you have access (public demos on DeepMind's blog).

3. Play with a Dreamer implementation from GitHub - watch it learn a control task.

4. Think: what task would convince you a video model is a world model? (Hint: holdout physics novel situations where the model has never seen similar examples.)

5. Read Yann LeCun's public statements on JEPA. Form your own view.

## Next

`capstone_multimodal_experiment.md` - Module 9 capstone.
