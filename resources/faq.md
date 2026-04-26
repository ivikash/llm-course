# FAQ

## "How long will this take?"

Realistically: 3-6 months of evening/weekend study for a total of 80-120 hours of focused work. Faster if you study 3-4 hours a day, slower if you only have a few hours per week. This isn't a sprint.

## "Do I need a GPU?"

For Modules 0-3: no. Your laptop is fine. The Shakespeare toy GPT trains on CPU in ~30 minutes.

For Modules 4+: yes, ideally. Options:
- Your own NVIDIA GPU (RTX 3060+ or better)
- Apple Silicon Mac (slower but workable)
- Rent cloud GPUs (Lambda, Runpod, Vast.ai) - $1-3/hour

Module 6 teaches you how to do the cloud part safely.

## "Can I skip modules?"

No. Each builds on the previous. If you skip Module 1 because "math is boring", Module 3 will crush you and you'll quit. Seriously - do them in order.

You can *gloss over* sections, but don't skip lessons.

## "I'm stuck. What do I do?"

1. Re-read the lesson slowly.
2. Try the exercises. The exercises aren't busywork - they force you to discover what you didn't understand.
3. Ask me. Literally: "explain Module 3 Lesson 4 section X again." I have both codebases open.
4. Take a break. Sleep on it. Come back. Many concepts click on the third pass, not the first.

## "Is Python really enough?"

Yes. No JavaScript, no C++, no R. Pure Python with PyTorch. Occasional bash for shell scripts.

Bonus: if you know any C/CUDA/Rust, Module 6 will make more sense. But it's not required.

## "Which language should I focus on?"

Python. Period. 99% of this course is Python. Some bash.

Ignore: JavaScript, Java, Go, C++. They're not part of the core LLM workflow.

## "What editor should I use?"

Whatever you're comfortable with. If you want AI-assisted editing, Cursor or VS Code with Copilot/Claude extensions are popular. Vim/Emacs work too.

## "Should I use Anaconda?"

No. Modern Python (3.10+) with `pip` or `uv` is enough. Anaconda adds complexity without much benefit for this work.

## "Should I memorize the code?"

No. Understand what it does. Memorization doesn't help; being able to find and read code does.

## "What if the paper or technique is outdated by the time I reach Module 7?"

It will happen. This is a fast-moving field. But the fundamentals (attention, backprop, optimizers, evaluation) don't change year-to-year. New techniques build on them. You'll be fine.

## "I know a little ML already. Should I skip anything?"

If you're confident you understand: tensors and shapes (1.5), cross-entropy loss (1.4), and you've trained a simple neural net (2.x), you can skim Modules 1 and 2 and focus on Module 3 onwards. Don't completely skip them - there are nanoGPT-specific framings you'll want.

## "Is the course written by a human or an AI?"

Both. I'm Kiro CLI, an AI agent. I wrote this course by reading both codebases carefully and structuring lessons around the code. I've aimed for "what a senior engineer would explain to a patient beginner." I'm here to expand any lesson and answer questions at depth.

## "What if I disagree with something in the course?"

Great - push back. Real learning happens when you question things. I'll back up claims with specifics from the code, and I'll admit when I'm uncertain.

## "What should my end-state look like?"

By completion, you should be able to:

1. Given a new LLM paper on arXiv, extract its core idea in 30 minutes.
2. Given a working codebase like nanochat, make a specific modification to test a hypothesis.
3. Run that modified pipeline on a cloud GPU, interpret the metrics, decide next step.
4. Explain your finding in a one-page memo a non-researcher can follow.
5. Know enough of the vocabulary and architecture to hold your own in a research discussion with PhDs.

You don't have to match a top LLM researcher. You do need to be a competent, curious, careful practitioner. This course will get you there.

## "After the course, what's next?"

- Contribute to open-source (HF, nanochat itself, research repos).
- Reproduce a recent paper.
- Build something you care about - not a toy, a real thing you use.
- Get a job or start a project where you apply this.
- Keep reading papers. 1-2 per week for the rest of your career.

## "Why do I sometimes feel dumb?"

Because you're learning hard things. Everyone in this field has had that feeling, including people at the top. It's the price of admission. Keep going.
