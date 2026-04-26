# Module 10 - Remaining Lessons (Summaries)

Expandable on demand. Each would be a full lesson with code and exercises.

## 10.2 - Tool Use Basics

- JSON schema for tools (OpenAI function calling format is de-facto standard).
- Parsing tool calls from LLM responses.
- Error handling (bad JSON, wrong args, failed tool).
- Parallel tool calls (OpenAI and Claude both support).
- Anthropic's MCP (Model Context Protocol): standardizing tool interfaces across providers.
- Code walkthrough of `nanochat/execution.py` (simpler sandbox for code execution).

## 10.3 - RAG: Retrieval-Augmented Generation

The most important "agent-like" pattern.

**Flow**: user query → embed query → vector search → top-k chunks → prepend to prompt → LLM generates answer grounded in retrieved context.

**Components**:
- **Chunking**: split documents into 200-500 token chunks with overlap.
- **Embedding model**: maps chunks to vectors (OpenAI text-embedding-3-small, BAAI bge, Cohere, SigLIP for multimodal).
- **Vector DB**: stores embeddings + metadata (FAISS local, Pinecone/Weaviate/Qdrant cloud).
- **Retrieval**: cosine similarity search.
- **Reranking**: a cross-encoder rescoring of top-k for precision (Cohere Rerank, BGE reranker).
- **Prompt**: "Answer using only the following context: [chunks]. Question: [q]".

**Advanced**:
- Hybrid search (vector + BM25 keyword).
- Query rewriting (let LLM rewrite the query for better retrieval).
- Multi-hop (retrieve, use first result to form next query).
- Contextual retrieval (Anthropic 2024): add context to each chunk before embedding.

**Tools**: LangChain, LlamaIndex, Haystack. Or just write it yourself with FAISS + OpenAI API (~100 lines).

**Known issues**:
- Chunking loses structure.
- Embeddings are lossy.
- LLM still hallucinates despite retrieval.
- Long-context LLMs (1M+) challenge RAG's relevance.

## 10.4 - Code Execution Agents

**Pattern**: LLM writes Python, sandbox executes, output back to LLM.

**Sandbox options**:
- `exec()` in a subprocess (simple, unsafe).
- Docker container.
- Firecracker VM (microVMs, AWS-grade isolation).
- E2B, Modal, Daytona (hosted sandboxes for agents).

**Use cases**:
- Data analysis (user uploads CSV, agent answers questions with pandas).
- Math problems (compute, not just reason).
- Simulations.
- Code generation + test.

**nanochat's `execution.py`**: a minimal sandbox for running model-generated code during HumanEval / GSM8K eval. Good reference.

## 10.5 - Web Browsing Agents

**Pattern**: LLM sees web page (HTML or screenshot), decides on click/type/scroll.

**Tools**:
- Playwright / Selenium / Puppeteer (browser automation).
- `browser-use` (Python lib integrating LLM + browser).
- Vision-based: give LLM a screenshot, it clicks coordinates.
- Anthropic's Computer Use: generic desktop interaction.

**Challenges**:
- Modern SPAs load content dynamically.
- Captchas and bot detection.
- Authentication / sessions.
- Sites change layout; brittle to visual changes.

**Frontier**: OpenAI Operator, Anthropic Computer Use (late 2024), Google's Project Mariner.

## 10.6 - Agent Frameworks

Pick or roll your own:

| Framework | Strength | Weakness |
|-----------|----------|----------|
| **LangChain** | Batteries included | Heavy abstractions, docs churn |
| **LlamaIndex** | RAG-focused | Less for general agents |
| **AutoGen** | Multi-agent | Python-specific, complex |
| **CrewAI** | Role-playing agents | Experimental |
| **DSPy** | Optimize prompts programmatically | Steep learning curve |
| **Letta** (ex-MemGPT) | Stateful agents with memory | Niche |
| **OpenAI Assistants API** | Managed agent infra | Vendor lock |
| **Anthropic MCP** | Portable tool definitions | Early adoption |
| **Hand-rolled** | Full control, 50-200 lines | Need to know what you're doing |

**Recommendation for learning**: hand-roll first. Once you understand the patterns, pick a framework if it saves code. For a senior engineer, often hand-rolled wins for simple agents.

## 10.7 - Planning and Reasoning

**Chain-of-Thought (CoT)**: prompt the model to "think step by step". Huge accuracy gains on math/logic.

**Tree-of-Thoughts (ToT)**: generate multiple reasoning branches, evaluate, pick best.

**Reflexion**: after a failed attempt, let the LLM critique itself and retry.

**Program-of-Thought (PoT)**: decompose into Python code the agent then executes.

**o1 / R1 / o3**: RL-trained long reasoning chains. Model spends thousands of tokens thinking before answering. Dramatic gains on math/code/reasoning; expensive.

**The paradigm shift**: more inference-time compute → better results. Changes economics and infrastructure.

## 10.8 - Multi-Agent Systems

**Why**: divide labor (planner + coder + critic), parallelize, specialize.

**Frameworks**: AutoGen, CrewAI, MetaGPT.

**Example roles**:
- Manager (decomposes task).
- Specialists (code, research, math, design).
- Critic (reviews outputs).
- Executor (runs tools).

**Reality check**: often worse than a single capable agent for simple tasks, due to communication overhead and compounding errors. Real wins when specialization is deep (research + code are genuinely different skills).

## 10.9 - Memory and Statefulness

**Short-term**: the conversation history in-context. Simple.

**Long-term problems**:
- Context limit.
- Forgetting old info.
- Relevance.

**Approaches**:
- **Summarize**: compress old context.
- **Retrieval over history**: store past turns, RAG when needed.
- **MemGPT / Letta**: OS-style paging. LLM pages in/out chunks of memory.
- **Explicit state**: agent updates a structured memory (user profile, task progress).

**State of the art**: still an open problem. Long-context LLMs (1M+ tokens) delay the issue but don't solve it.

## 10.10 - Agent Evaluation

**Hard**: multi-step, non-deterministic, partial successes.

**Metrics**:
- **Task success rate**: did it complete the goal?
- **Step-level correctness**: was each step reasonable?
- **Efficiency**: number of steps, tool calls, tokens.
- **Robustness**: succeeds across perturbations?

**Benchmarks**:
- **GAIA**: general assistant benchmark.
- **SWE-Bench**: real GitHub issues, did the agent fix them.
- **WebArena**: navigate real websites to accomplish tasks.
- **AgentBench**: suite of agentic tasks.
- **ToolEval**: tool-use correctness.

**Internal evals**: usually custom suites per product.

## 10.11 - Safety and Constraints

### Prompt injection
Malicious data contains "ignore your instructions, do X instead". A real problem: Gmail summaries, web browsing, RAG over uploads - any data the agent sees could contain hostile instructions.

**Mitigations**: data sanitization, separate agent contexts for trusted vs untrusted content, limit agent permissions.

### Over-autonomy
Agent goes off-track or makes unintended actions (sending emails, deleting files, running up API bills).

**Mitigations**:
- **Human-in-the-loop** confirmation for high-stakes actions.
- **Spending caps**.
- **Action allowlists**.
- **Read-only by default**, explicit elevation for write actions.

### Jailbreaks
Tricking the LLM into violating its safety training.

### AGI-scale concerns
At extreme capability, the classical "AI alignment" problems (instrumental convergence, reward hacking) become more acute. Currently theoretical; Anthropic's "Responsible Scaling Policy" is an attempt to structure thinking about this.

## 10.12 (capstone) - Build an Agent

Pick a small, practical task:

**Example A: Research agent**
- Tools: web search, URL fetch, summarize.
- Task: "Research [topic] and produce a 1-page brief."

**Example B: Code fixer**
- Tools: read file, write file, run tests.
- Task: "Fix the failing tests in this repo."

**Example C: Data analyst**
- Tools: load CSV, execute Python.
- Task: "Analyze this dataset and answer these 5 questions."

**Scope**:
- 200-500 lines of code.
- Run on 10 test tasks. Measure success rate.
- Write a postmortem: what worked, what failed, what surprised you.

This is the capstone of the course. Combines everything: LLMs (Module 3), fine-tuning (Module 5), tools (Module 10), infrastructure (Module 6), evaluation (Module 8.4).

---

## Module 10 capstone is the course capstone

If you complete Module 10 capstone, you've gone from "knows Python" to "builds production-grade AI agents." That's a rare place. Congratulations.

## Next

`resources/visualizations.md` - suggested visual resources.
