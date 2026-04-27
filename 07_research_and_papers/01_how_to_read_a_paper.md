# 7.1 - How to Read a Research Paper

The skill of reading papers separates people who can keep up with the field from people who can't. Done right, you can extract the core idea of a new LLM paper in 30 minutes.

## The three-pass method

Standard technique, adapted from Keshav's classic [How to Read a Paper](https://web.stanford.edu/class/cs219a/readings/keshav-2013-how-to-read-a-paper.pdf).

### Pass 1: the bird's-eye view (5-10 minutes)

Goal: decide whether to spend more time on this paper.

Read, in order:
1. **Title and authors**. Authors tell you about lineage (DeepMind paper vs CMU paper vs OpenAI paper have different cultures and conventions).
2. **Abstract**. The claim.
3. **Introduction's last paragraph** ("In this paper, we..."). The contribution.
4. **Section headings**. The structure.
5. **Figures** and **tables**. The results.
6. **Conclusion's first paragraph**. The bottom line.
7. **References**: skim titles. What does this paper build on? Are there 2-3 papers I should read first?

After 10 minutes you should be able to say in a sentence: "This paper proposes X, and shows Y on Z benchmarks."

If the paper doesn't look useful, stop here. Most papers you start don't make it past pass 1. That's correct.

### Pass 2: deeper understanding (1 hour)

Goal: understand what they did, well enough to explain it to a peer.

- Read each section carefully.
- But ignore proofs, detailed derivations.
- Pay attention to figures - good papers have a "headline figure" that tells the whole story.
- Make notes: restate ideas in your own words.
- Mark what you don't understand.

End of pass 2: you can summarize the paper in a paragraph, including their experimental setup and main result.

### Pass 3: re-implement (half a day to days)

Goal: believe (or not) the claim, deeply.

- Read the methods section carefully, with the code open (if available).
- Consider: "Could I replicate this experiment?"
- Work through the math if important.
- Do a small-scale reproduction if possible.

You do pass 3 for maybe 5-10 papers per year. Pass 1 for hundreds.

## Applied: "Attention Is All You Need"

Let's walk pass 1 of the transformer paper right now. Open [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762).

- **Title**: "Attention Is All You Need". 2017, Google Brain + Google Research.
- **Abstract claim**: new architecture (Transformer) based solely on attention, outperforms RNNs on translation with less training time.
- **Intro last paragraph**: proposes the Transformer, trains on English-German translation, achieves state-of-the-art BLEU.
- **Section headings**: 1. Intro. 2. Background. 3. Model Architecture. 4. Why Self-Attention. 5. Training. 6. Results. 7. Conclusion.
- **Figure 1**: the architecture diagram. Two stacks (encoder, decoder). Multi-head attention boxes.
- **Figure 2**: attention operation in more detail.
- **Tables**: BLEU scores for English-German and English-French, model variants.
- **Conclusion**: state-of-the-art on translation, plans to apply to other problems.

After pass 1: "This paper proposes the Transformer, a new seq2seq architecture replacing RNNs with stacked multi-head self-attention. It gets better BLEU on translation in less training time. Encoder-decoder structure, 6 layers each, 8 attention heads, 512 embedding dim."

Not full understanding. But enough to know: this is foundational, worth pass 2. (And pass 3 already happened for us in Module 3.)

## Red flags in papers

Learn to spot weak papers fast:

- **Single-number metric with no error bars**: variance is hidden.
- **Cherry-picked baselines**: "we outperform GPT-2" (but not GPT-3).
- **No ablation studies**: you don't know if the improvement is from the proposed idea or an incidental change.
- **Evaluation-only on one benchmark**: overfitting to it is easy.
- **"Emergent" or "AGI" language without evidence**: marketing.
- **Missing training details**: data mix, hyperparameters, seeds. Makes reproduction impossible.
- **Code not available or not runnable**: 90% of claims unverifiable.

Conversely, good signals:
- Strong baselines, multiple benchmarks.
- Ablations that remove each component.
- Released code, checkpoints, training scripts.
- Clear statement of limitations.
- Replicated by others afterward.

## Tools for managing papers

- **Zotero / Mendeley**: PDF libraries with tags and notes.
- **ar5iv.org**: better rendering of arXiv papers.
- **alphaxiv.org**: arxiv with comments and links.
- **scholar.google.com**: find citing papers ("who built on this?").
- **Semantic Scholar**: paper search with citation graphs.
- **Papers with Code**: links papers to implementations and benchmarks.

For this course, a plain notes file per paper you read is enough.

## Using LLMs to read papers

Legit and useful, if you're careful:

- **Summarize**: paste the abstract and figures into Claude/GPT, ask for a 3-sentence summary.
- **Explain a section**: "What does equation 4 mean?" - with context.
- **Find related**: "What papers should I read before this one?"

Risks:
- LLMs **hallucinate**, especially about specific numbers and methods.
- Always cross-check key claims against the paper itself.

Don't let LLMs replace reading the paper. Use them to accelerate, not skip.

## Visualize this

**The three-pass reading workflow**:

```
  ┌─ Pass 1 (5-10 min) ─────────────────────────┐
  │  Title, abstract, figures, conclusion       │
  │  Question: is this worth more time?          │
  │  Outcome: 90% of papers stop here.          │
  └──────────────────────────────────────────────┘
              │ (worth it)
              ▼
  ┌─ Pass 2 (30-60 min) ────────────────────────┐
  │  Methodology, experiments, results           │
  │  Question: do I understand the idea?         │
  │  Outcome: can explain to a peer.             │
  └──────────────────────────────────────────────┘
              │ (really important)
              ▼
  ┌─ Pass 3 (several hours to days) ────────────┐
  │  Re-implement, verify claims, read refs      │
  │  Question: could I have done this?           │
  │  Outcome: deep mastery, ~5-10 papers/year.   │
  └──────────────────────────────────────────────┘
```

**Which papers get which passes**:

```
  Year 1 (you):
    Pass 1:   ~200 papers
    Pass 2:   ~40  papers
    Pass 3:   ~5   papers
    → Broad awareness + targeted depth

  Senior researcher:
    Pass 1:   ~500 papers
    Pass 2:   ~100 papers
    Pass 3:   ~15-20 papers
    → Encyclopedic breadth + deep specialization
```

**The anatomy of a paper (what to read when)**:

```
  ┌──────────────────────────────────────┐
  │ TITLE                      ←── pass 1 │
  │ Authors, affiliations      ←── pass 1 │
  ├──────────────────────────────────────┤
  │ ABSTRACT                   ←── pass 1 │
  ├──────────────────────────────────────┤
  │ 1. Introduction                       │
  │   (last paragraph)          ←── pass 1│
  ├──────────────────────────────────────┤
  │ 2. Related Work            ←── pass 2 │
  ├──────────────────────────────────────┤
  │ 3. Method                             │
  │   Main Figure              ←── pass 1 │
  │   Key equations            ←── pass 2 │
  │   Algorithm details         ←── pass 3 │
  ├──────────────────────────────────────┤
  │ 4. Experiments                         │
  │   Main table                ←── pass 1 │
  │   Ablations                 ←── pass 2 │
  │   Hyperparameters          ←── pass 3 │
  ├──────────────────────────────────────┤
  │ 5. Discussion / Limitations ←── pass 2│
  ├──────────────────────────────────────┤
  │ 6. Conclusion (1st paragraph)← pass 1 │
  ├──────────────────────────────────────┤
  │ References                  ←── pass 2 │
  │ Appendix                    ←── pass 3 │
  └──────────────────────────────────────┘
```

**Red flags to look for in pass 1**:

```
  ✗ No code released and no weights
  ✗ Vague evaluation (one metric, one dataset)
  ✗ No ablations
  ✗ Authors have no track record of reproducing their claims
  ✗ "SOTA" without showing baselines clearly
  ✗ Claims that seem too good (2× improvement on everything)
  ✗ No limitations section

  ✓ Open-source release
  ✓ Multiple benchmarks, multiple seeds
  ✓ Thorough ablations
  ✓ Clear limitations
  ✓ Follows from prior work logically
  ✓ Small, precise claim backed by evidence
```

**Your paper-reading log (keep this)**:

```
  papers.md
  ─────────

  ## 2026-04-26
  ### Attention Is All You Need (Vaswani 2017)
  - Pass 2 done.
  - Key insight: attention alone replaces recurrence.
  - 6 encoder + 6 decoder layers; each with multi-head attention + FFN.
  - What's different in modern LLMs: pre-norm, RoPE, decoder-only.
  - Status: foundational. Re-read yearly.

  ## 2026-04-28
  ### Llama 3 paper (Meta 2024)
  - Pass 1 done.
  - Key claim: 405B params trained on 15.6T tokens.
  - Key data innovation: aggressive data filtering.
  - Worth pass 2 for architecture details.

  ## 2026-05-01
  ...
```

Over a year, 50 papers read = a solid foundation.

**Using LLMs to accelerate (but not replace) reading**:

```
  DON'T:  "Claude, summarize this paper" → read summary → move on.

  DO:     Read the paper yourself first.
          Then ask Claude: "Explain equation 4 in simpler terms."
          Then ask: "What's the relationship to paper X?"
          Cross-check claims against the original.

  LLMs hallucinate about papers. Always verify specific numbers/claims.
```

## A weekly practice

Set a habit:
- Monday: scan arXiv's cs.CL listing (5 minutes, see what's new).
- Pick 1-2 papers for pass 1 this week (30 min).
- Maybe 1 paper for pass 2 (1 hr).
- Maintain a notes file with 3-5 lines per paper.

Two years of this and you have ~200 papers in your head. You know the field.

## Exercises

1. Open ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762). Do pass 1. Write your one-sentence summary.

2. Go to arxiv.org, browse cs.CL from the last 3 days. Pick any 2 titles that interest you. Do pass 1 on each. Write summaries.

3. Open any recent arXiv paper on a new LLM (e.g. DeepSeek-V3, Llama-3). Do pass 1. Pick one figure and describe what it tells you.

4. Set a calendar reminder: "10 min arXiv pass Mondays."

## Next

`02_core_papers_you_must_know.md` - the reading list.
