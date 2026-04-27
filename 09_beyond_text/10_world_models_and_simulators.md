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

## Visualize this

**What we mean by "world model"**:

```
  Bad world model:              Good world model:
  (memorizes trajectories)      (learns physics)

  Input: "a glass falls off    Input: "a glass falls off
          a table"                     a table"
          │                             │
          ▼                             ▼
  Output: video of that           Output: video showing
  specific falling glass it       ANY glass falling,
  has seen before.                 obeying gravity, shattering
                                   realistically, fluids spilling.

  Memorization doesn't generalize.
  Physics does.
```

**The world model hypothesis, visualized**:

```
  Hypothesis: "A sufficiently capable video generator learns physics."

  Evidence FOR:
    ✓ Sora shows emergent object permanence
    ✓ Genie generates playable game worlds from images
    ✓ Dreamer's world models enable RL in imagination

  Evidence AGAINST:
    ✗ Sora breaks physics on edge cases
    ✗ Objects teleport / duplicate
    ✗ Causality violated
    ✗ Can't extrapolate to novel situations

  Current state: partial world-modeling. Not a true simulator.
```

**Spectrum from video generation → world simulator**:

```
       mere video synthesis              true world simulator
       (pattern matching)                   (physics + reasoning)
                │                                  │
       ────────●──────────────●──────────────●──────
                              ↑                ↑
                              │                │
                    Sora (2024)               "AGI"
                    Genie (2024)             (not there yet)
                    Veo 3 (2025)
                    
       ← we are HERE
```

**Dreamer: learning a world model for RL**:

```
  Agent in an environment:
    - takes action
    - observes new state
    - gets reward

  Traditional RL:
    try actions in the REAL environment
    → slow, expensive (real robots, real world)

  Dreamer:
    ┌──────────────────────────┐
    │ Learn a latent world model  │
    │ (predicts next state given  │
    │  current state + action)    │
    └──────────────┬──────────────┘
                   │
                   ▼
    train the RL policy ENTIRELY in imagination:
      current state + action → model predicts next state
      model predicts reward
      policy learns from imagined rollouts

    Only real environment needed for occasional grounding.
    Massively reduces sample complexity.

    DreamerV3 (2023) solved many tasks zero-shot using this idea.
```

**Genie: 2D games from single images**:

```
  Input: a single video game screenshot
  ┌──────────────────┐
  │ 🎮 platformer      │
  │  /\                │
  │ /  \    player     │
  │ ━━━━━ ground       │
  └──────────────────┘
          │
          ▼
  Genie model (1B+ params, trained on 200k hours of game video)
          │
          ▼
  Interactive playable game!

  User presses [LEFT]: next frame shows player moving left
  User presses [JUMP]: player arcs in trajectory
  User presses [RIGHT]: continues right
  ...

  Trained unsupervised from video (no labels of actions).
  Latent "actions" emerge from training.
```

**The skeptical view** (Yann LeCun et al.):

```
  LeCun: "Video generation models are NOT world models."

  His argument:
    - They are autoregressive generators
    - They don't have a causal model of reality
    - They generate what LOOKS plausible, not what IS

  His alternative:
    JEPA (Joint Embedding Predictive Architectures)
    Predict representations, not pixels
    More grounded in causal structure

  Still: LeCun's approach hasn't dominated yet.
  Sora-style continues to improve rapidly.
  Debate active, unresolved.
```

**Embodied AI: robots using world models**:

```
  2023-2024 trend: robot foundation models.

  ┌───────────────────────────────────────┐
  │ Robot sees its environment (camera)     │
  │         │                                 │
  │         ▼                                 │
  │   Vision-Language-Action (VLA) model      │
  │     - Vision: camera feed                 │
  │     - Language: task instruction          │
  │     - Action: motor commands              │
  │                                           │
  │         │                                 │
  │         ▼                                 │
  │   Robot executes motion                   │
  └───────────────────────────────────────┘

  Trained on:
    - Teleoperation data (human demo)
    - Video data
    - Simulated environments

  Examples:
    RT-1, RT-2 (Google): generalist robotic transformer
    Pi-0 (Physical Intelligence 2024)
    GR00T (NVIDIA 2024)
    Figure, 1X, Unitree humanoids (various)

  Still: robots are brittle in practice. "5 years away" syndrome.
```

**SIMA: generalist game-playing agent**:

```
  DeepMind's SIMA (2024):
    One agent that plays many video games.
    Takes:
      - visual input (screen)
      - natural language instruction ("go left", "find the key")
    Outputs:
      - game actions (keyboard/mouse equivalents)

  Demonstrated on:
    Goat Simulator
    Valheim
    No Man's Sky
    many more

  Generalization: ability to follow instructions in NEW games.
  Not yet human-level, but impressive.
```

**What about AGI?** (the speculation):

```
  Some prominent researchers argue:
    "True AGI requires a world model."
    "A model that can PREDICT and PLAN in the physical world."
    "Current LLMs are language models. AGI needs more."

  Counter-argument:
    "Current LLMs are already partial world models."
    "With enough data + compute, they'll become full ones."
    "Don't need explicit physics; scale provides it."

  Nobody knows who's right. It's partly a definitional question.
  Bet on both directions hedges well.
```

**What to watch for in 2025-2028**:

```
  Research signals that world models are "working":
  
    ✓ Video models that extrapolate novel physics ("what if gravity
      reversed here?")
    ✓ Robots that learn new tasks from 1 demonstration
    ✓ Game agents that generalize to unseen games
    ✓ Sora-style models used for actual simulation (weather, fluid)
    ✓ Improvements on ARC-AGI / Humanity's Last Exam requiring world reasoning

  Research signals that they're NOT:
    ✗ Continued struggle with causal reasoning
    ✗ Video generators break on novel situations
    ✗ Robots plateau at narrow demonstrations
    ✗ Benchmarks that require real simulation consistently fail
```

**Real-world applications (today)**:

```
  Working, now:
    ✓ Architectural visualization
    ✓ Product mockups
    ✓ Storyboards (with editing)
    ✓ Social media content

  Emerging:
    ~ Robot training simulators
    ~ Climate modeling assistants
    ~ Drug discovery (physical molecule simulation)
    ~ Game engine AI (procedural content)

  Far future:
    ✗ Fully autonomous robots in unstructured environments
    ✗ Scientific discovery from simulation alone
    ✗ Human-like understanding of physical world
```

**Honest conclusion**:

```
  World models are a REAL capability direction.
  Sora / Genie / Dreamer are REAL progress.

  But we're further from "AI that truly simulates reality"
  than hype suggests.

  Likely path: incremental improvements over 5-10 years.
  Watch for: integration with robotics, novel physics tests.
  Don't bet: on "world models solved in 2025."
```

## Exercises

1. Read Sora's technical blog (10 min). Note the "world simulator" framing.

2. Try Genie 2 if you have access (public demos on DeepMind's blog).

3. Play with a Dreamer implementation from GitHub - watch it learn a control task.

4. Think: what task would convince you a video model is a world model? (Hint: holdout physics novel situations where the model has never seen similar examples.)

5. Read Yann LeCun's public statements on JEPA. Form your own view.

## Next

`capstone_multimodal_experiment.md` - Module 9 capstone.
