# 8.10 - Staying Current in a Field That Moves Weekly

The LLM field publishes thousands of papers per year. No one can read them all. This lesson is about filtering the firehose while staying genuinely informed.

## The firehose problem

### The numbers

- arXiv cs.LG + cs.CL: 1000+ papers per month.
- HuggingFace model releases: hundreds per week.
- LLM-related tweets on X: millions per week.
- Major model releases: 2-3 per month.

If you try to read it all, you'll:
- Fall behind.
- Feel anxious.
- Actually learn less (shallow passes, no retention).

You need filters.

## Filter 1: follow curators, not firehoses

Good curators:

### Newsletters
- [Interconnects](https://interconnects.ai) by Nathan Lambert - weekly LLM deep dives.
- [Import AI](https://importai.substack.com) by Jack Clark - weekly AI news + analysis.
- [The Batch](https://www.deeplearning.ai/the-batch/) by Andrew Ng - weekly ML news.
- [Latent Space](https://www.latent.space) - popular LLM newsletter.
- [Alpha Signal](https://alphasignal.ai) - daily AI updates.

### Twitter/X accounts worth following
- **For transformer/LLM fundamentals**: @karpathy, @ylecun, @jeffdean, @sama
- **For infrastructure/optimization**: @tri_dao, @StasBekman
- **For open models**: @winglian (Axolotl), @Teknium1, @erhartford
- **For analysis**: @srush_nlp, @jackclarkSF, @natolambert
- **For research taste**: @fchollet, @ylecun, @kipperrii

(Verify these handles - they change.)

### YouTube
- 3Blue1Brown: animations of transformer concepts.
- Yannic Kilcher: paper reviews, 30-90 min per paper.
- Andrej Karpathy: the definitive series on LLMs.
- Two Minute Papers: fast summaries.

### Blogs
- [Lilian Weng's blog](https://lilianweng.github.io/) - deep technical posts.
- [Eleuther AI blog](https://blog.eleuther.ai/).
- [Anthropic's research blog](https://www.anthropic.com/research) - interpretability.
- [DeepMind blog](https://deepmind.google/discover/blog/).

### Paper aggregators
- [arXiv-sanity](http://arxiv-sanity-lite.com/) - personalized arXiv feed.
- [Hugging Face Daily Papers](https://huggingface.co/papers) - community-voted.
- [Papers with Code](https://paperswithcode.com/) - with benchmarks and implementations.
- [LifeArchitect AI papers](https://lifearchitect.ai/papers/) - curated big-picture view.

Pick 3-5 sources. Not 30. Too many and you read none of them.

## Filter 2: signal > noise metrics

A paper's real value is usually proportional to:

- **Strong authors** (not just big labs, but authors with track record of reproducible wins).
- **Open-source release** of code and weights.
- **Clean writing** (often correlates with clean thinking).
- **Moves a meaningful metric on a hard benchmark.**
- **Enables new applications** (not just improves existing ones).

Red flags:
- Press-release-first research.
- "SOTA" without comparison to strong baselines.
- Tiny improvements on saturated benchmarks.
- No code, no data, no reproducibility details.
- Authors' previous papers didn't pan out.

## Filter 3: personal curiosity

Most important filter. What do *you* care about?

- Inference efficiency? Follow vLLM, TensorRT-LLM updates, quantization papers.
- Reasoning? Follow o1/R1-style RL papers, chain-of-thought work.
- Multimodal? Follow VLM releases, image/video generation.
- Agents? Follow tool-use, code-gen, workflow papers.
- Infra? Follow distributed training frameworks, GPU kernel libraries.

You don't need to be current on everything. Be deep on 1-2 areas.

## Filter 4: time budget

Set a weekly budget. Example:

- Monday 30 min: scan Newsletter + Twitter highlights from the weekend.
- Wednesday 1 hr: deep-read one paper from Monday's list.
- Friday 30 min: catch up on what you missed. Plan reading for next week.

That's 2 hours per week. Enough to stay broadly current + occasionally deep. Less and you fall behind; more is diminishing returns.

## Filter 5: communities

Active Discord/Slack communities to be lurking in:

- **Eleuther AI** Discord: research-heavy, open source.
- **HuggingFace** Discord: library/model focus.
- **LMSys** Discord: Chatbot Arena folks, eval focus.
- **PyTorch** Slack / Discord: framework internals.
- **#nanoGPT** / #nanochat on Karpathy's Discord.

Don't spend hours a day. Drop in once a week.

Conferences, in-person:
- **NeurIPS** (December): biggest, most comprehensive.
- **ICML** (July): another major.
- **ICLR** (April/May): rising in LLM relevance.
- Local meetups if available.

You don't need to attend conferences to follow them. Many talks are on YouTube shortly after.

## Depth over breadth (when to go deep)

Once a quarter, pick **one paper** you'll truly master.
- Re-implement the main result.
- Write a blog post about it.
- Talk about it with peers.

This "one deep dive per quarter" habit compounds. After 3 years, you've deeply understood 12 important papers. That's a tremendous base.

## The half-life of knowledge

Rough estimates for LLM knowledge:

- **Fundamentals** (attention, backprop, gradient descent): decades. Always relevant.
- **Architecture recipes** (RoPE, GQA, RMSNorm): ~5 years.
- **Specific models** (GPT-4, Claude 3): 1-3 years before obsolete.
- **Benchmark numbers**: 6-12 months before outdated.
- **Prompt engineering tricks**: sometimes weeks.

Invest heavily in the deep-half-life knowledge. Shallow-half-life stuff is worth tracking, not memorizing.

## Build your own knowledge base

As you learn things, **write them down**:
- A personal wiki (Obsidian, Notion, just Markdown).
- Tagged by topic: `#attention`, `#optimizers`, `#evals`, etc.
- Include: key papers, your notes, code snippets, open questions.

After a year: you have a searchable body of personal expertise. It's immensely valuable.

## Habits to cultivate

1. **Weekly arXiv scan** (30 min).
2. **One paper pass 2** per week.
3. **One deep dive** per quarter.
4. **One reproduction** per year.
5. **Share publicly** 3-5 times per year (blog, tweet thread, talk).
6. **Talk to someone smarter** monthly.

None of these are heroic. All of them compound.

## Habits to avoid

1. Anxious doom-scrolling on Twitter.
2. Bookmarking papers and never reading.
3. Being "too busy" to learn for 6 months at a time.
4. Only consuming content, never producing.
5. Tracking metrics but not doing experiments.

## The meta-lesson

Staying current is a skill, like running. You can train it or let it atrophy.

The most successful senior researchers I know have a **rhythm** - not manic consumption, but steady attention to the field, with deep investment in what matters to them.

Find your rhythm. Protect it.

## Exercises

1. Audit: what are your current sources? (List them.)
2. Subscribe to 3 newsletters from the list above. Unsubscribe from 3 other noisy ones.
3. Set weekly calendar reminders for your reading time. Treat them like meetings.
4. Start a personal knowledge base. Even if just one Markdown file. Add notes as you read.
5. Pick one paper this week and deeply engage with it. Write a 1-page summary.

## Conclusion of Module 8

You've reached the end of the course.

Module 8 has been about the non-code parts of ML work - taste, communication, evaluation, team-building, career-long learning. These are the 80% of being senior that nobody writes textbooks about.

No capstone for Module 8. The capstone is your career itself.

Before you go: open `README.md` at the top of the course. Reflect on how far you've come. Text in Module 0 probably looked scary. Now you can write it yourself.

Keep going. Be generous with what you learn. The field needs more people who are both technically strong and humane.

Good luck.
