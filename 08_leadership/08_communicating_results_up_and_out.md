# 8.8 - Communicating Results Up and Out

Most ML research dies in obscurity not because it was bad, but because nobody communicated it well. Being a great communicator is 40% of being a great ML leader.

## The audiences

Different audiences need different communication.

### Audience A: your team
Weekly. Deep technical detail. Graphs, numbers, Slack threads. Your team can handle `val_bpb = 0.74804 ± 0.003`.

### Audience B: your org / manager
Monthly or per-milestone. Some technical detail, more context. Charts with labels, narrative paragraphs. Your manager wants to know what's working and what you need.

### Audience C: leadership
Per-quarter or per-release. Almost no technical detail. Two-number summary: capability gained, cost. Leaders want to know: "Are we winning?"

### Audience D: public
Per-paper or per-launch. Marketing polish. Narrative arc. Broad audience.

Same finding, four different communications. Skilled leaders rewrite for each.

## The one-page memo

For audiences B and C, the one-pager is your workhorse. Every important finding gets one.

Structure:

```
Title: [Specific claim in 10 words or less]

Context (1-2 sentences): Why does this matter? What problem is it solving?

Finding (2-3 sentences): The main result, quantified.

Implications (2-3 bullets): What should we do differently because of this?

Evidence (1 graph, 1 table): The one plot and one table that prove the finding.

Limitations (1-2 bullets): What we don't know, what could be wrong.

Next steps (1-2 bullets): What to do next, by when, who.
```

One page. One graph. One table. Not two pages.

**Test**: can a smart non-specialist read it in 2 minutes and understand the main point?

## The three-number summary

For executives, reduce everything to three numbers:

1. **Capability**: what the model can do now, in a single benchmark or user-facing metric.
2. **Cost**: $ per operation (training, inference, both).
3. **Risk**: what could go wrong (accuracy gap, safety incident, compute overrun).

This forces discipline. If you can't reduce your project to 3 numbers, you probably don't understand it yet.

## Headline graphs

The single most important visual.

Rules:
1. **One message per graph.** If it's saying two things, make two graphs.
2. **Self-contained caption.** A reader skimming figures should understand.
3. **Clear labels.** Every axis. Units included.
4. **Meaningful colors.** Not rainbow. Not hard-to-distinguish.
5. **Baseline included.** Your improvement is meaningless without context.

### Good:
"Val loss over 100K steps for 3 architectures (solid: proposed; dashed: baseline). Each curve averaged over 3 seeds. Shaded region: std. Proposed architecture achieves 5% lower loss at matched compute."

### Bad:
[A graph with 10 lines, no legend, three axes, rainbow colors, titled "Results"]

## Meeting hygiene

Meetings with leadership:

1. **Start with the conclusion.** Not the journey.
2. **Pre-read materials.** Send your memo 24hrs in advance.
3. **Know your audience's context.** What did they read last?
4. **Have a "so what?"** Every fact you mention should answer "so what?"
5. **Plan for the one-question attack.** What's the most likely hostile question? How do you respond?

Bad meeting: "We ran many experiments. First I'll show X, then Y, then Z..."
Good meeting: "Our model is now 15% cheaper to serve. Here's how. [Goes straight to the graph.]"

## Writing for range

Advance preparation for each audience:

- Longest first: write the detailed team-facing version.
- For manager: cut to 2 pages. Add an executive summary.
- For leadership: cut to 1 page, with 3-number summary at top.
- For public: rewrite for narrative.

Each compression is work, but the reader's time savings are multiples of yours.

## The "why should I care?" test

Every piece of communication should answer this in the first sentence or two:

- Why should my manager care?
- Why should product care?
- Why should the VP care?

Different answers for each. If you can't articulate the answer, you haven't earned their attention.

## Handling hostile questions

Common questions you'll face:

### "Why aren't we better than OpenAI?"

Translation: leadership is worried about competitiveness.

Bad response: "It's complicated, our architecture is different..."

Good response: "We're building for a different use case. For our target metric [X], we're within 5% of the best. Here's what we'd need to close that gap: [specific plan]."

### "Can we train a model with capability X by date Y?"

Translation: someone wants to commit to something.

Bad response: "We'll try!"

Good response: "With our current compute and team size, capability X has a 30% probability by date Y. To increase to 80%, we'd need [specific resources]. What's the actual business need?"

### "Why did this experiment cost so much?"

Translation: someone is looking at the GPU bill.

Bad response: Defensive explanation.

Good response: "The experiment delivered on its scope; however, the value-per-dollar was suboptimal because [specific reason]. Going forward, we'll [specific change]."

## Writing principles

- **Short sentences.** Long ones lose readers.
- **Active voice.** "We trained the model" > "The model was trained."
- **Concrete over abstract.** "15% cheaper" > "significantly more efficient."
- **Own your claims.** "We believe" > "it seems that."
- **Label uncertainty explicitly.** "Confidence: high / medium / low."
- **No jargon for non-technical audiences.** "Loss" → "quality metric."
- **Short paragraphs.** One idea each.

## Public communication

Posting externally (blog, Twitter, paper) has additional considerations:

- **Review for confidentiality** before posting.
- **Overclaim is worse than underclaim.** Under-claim and let readers be impressed.
- **Always include limitations.** Reviewers will find them anyway; be first.
- **Respond to feedback gracefully.** The internet is loud. Engage with good-faith critics, ignore bad-faith.

## Handling failure in public

When your project fails or a release has issues:

1. Acknowledge immediately.
2. Explain what you know.
3. Describe what you're doing about it.
4. Commit to a follow-up with more info.

Don't spin. Don't minimize. Don't go dark. Bad-news communication done well actually builds trust.

## Templates to build

Your team will produce the same communications repeatedly. Build templates:

- **Weekly status**: team-internal.
- **Monthly memo**: for your manager.
- **Quarterly review**: for leadership.
- **Release notes**: for product / customers.
- **Postmortem template**: for failures.

A reusable template saves hours per occurrence and enforces good structure.

## The meta-skill

The secret is: **communication forces clarity**. If you can't write up what you did, you don't understand it yet.

Good researchers use writing as a thinking tool. Write the memo early. Discover you don't know the answer. Go figure it out. Revise the memo.

Most junior researchers write up at the end. They've already moved on in their head. The communication is hollow.

Senior researchers write as they think. The memo evolves with their understanding. When they're done, the memo is excellent because the thinking was structured.

## Exercises

1. Take the most recent experiment or project you ran. Write it up in the one-page memo format.

2. Reduce it to a three-number summary. Which 3 numbers matter?

3. Design the one headline graph. What does it show? What's the message?

4. Write the same finding for three audiences: team, manager, leadership. Compare.

## Next

`09_research_taste.md` - the hardest-to-teach skill.
