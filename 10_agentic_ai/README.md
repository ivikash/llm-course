# Module 10 - Agentic AI

**Agentic AI** = LLMs that don't just produce text, but take actions in the world. Call tools, browse the web, execute code, interact with APIs, and chain many steps to accomplish goals.

This is arguably the biggest practical frontier of 2024-2026. Every big AI lab is shipping agents (OpenAI's o1 reasoning, Claude Computer Use, Devin, Cursor, Replit Agent).

## Lessons

### `01_what_is_an_agent.md`
- Definition: LLM + tool-use loop + memory.
- Spectrum: single-turn tool call → autonomous agent.
- "Agentic capability" = planning, tool use, self-correction, memory.
- When do you need an agent vs. direct LLM call?

### `02_tool_use_basics.md`
- Function calling / tool calling in GPT/Claude APIs.
- JSON schemas for tools.
- The "think → call → observe → think" loop.
- ReAct (Reasoning + Acting) pattern.
- Code example: simple weather-looking agent.

### `03_retrieval_augmented_generation_rag.md`
- The "use external knowledge" pattern.
- Vector databases (FAISS, Pinecone, Weaviate, Chroma, Qdrant).
- Embedding models (OpenAI embeddings, BAAI bge, Cohere).
- Chunking strategies.
- Hybrid search (vector + BM25).
- Reranking.
- How chatbots with proprietary knowledge are built.

### `04_code_execution_agents.md`
- The LLM writes Python, a sandbox executes it, output goes back.
- OpenAI Code Interpreter, Claude's analysis tool, nanochat's execution.py.
- Sandbox security (Docker, gVisor, Firecracker).
- Data analysis agents, math agents.

### `05_web_browsing_agents.md`
- LLM navigates a browser (Selenium, Playwright, browser-use).
- Sees HTML / screenshots, issues actions (click, type, scroll).
- OpenAI's Operator, Anthropic's Computer Use.
- Challenges: captchas, rate limits, authentication.

### `06_agent_frameworks.md`
- LangChain: the original, heavy abstraction.
- LlamaIndex: RAG-focused.
- AutoGen (Microsoft): multi-agent.
- CrewAI: role-playing multi-agent.
- DSPy (Stanford): program synthesis for prompts.
- Letta (ex-MemGPT): stateful agents.
- OpenAI Assistants API.
- Anthropic MCP (Model Context Protocol): tool sharing standard.
- Which to use: usually "write it yourself" beats frameworks for simple cases.

### `07_planning_and_reasoning.md`
- Chain-of-Thought (CoT): think step by step.
- Tree of Thoughts (ToT): explore multiple reasoning paths.
- Reflexion: self-critique and revise.
- o1 / R1 style: RL-trained long reasoning chains.
- How inference-time compute shifted the frontier.

### `08_multi_agent_systems.md`
- One LLM doing everything vs multiple specialized agents collaborating.
- "Manager", "coder", "critic" role splits.
- When multi-agent helps (parallel work, specialization) and when it doesn't (overhead).
- Examples: MetaGPT, AutoGen, ChatDev.

### `09_memory_and_statefulness.md`
- Short-term: in-context memory (the conversation).
- Long-term: vector store, database, updated summaries.
- MemGPT / Letta: OS-like paging.
- Relevance filtering, forgetting, consolidation.
- A key unsolved problem.

### `10_agent_evaluation.md`
- Evaluating agents is hard (multi-step, non-deterministic).
- Task success rate.
- Step-level vs trajectory-level.
- Benchmarks: GAIA, SWE-Bench, WebArena, AgentBench.
- Building your own eval.

### `11_safety_and_constraints.md`
- Prompt injection (user hides instructions in data).
- Jailbreaks.
- Over-autonomy (agent does too much).
- Guardrails: allowlists, confirmation steps, sandboxing.
- The "AI takeover" concerns at extreme scale.

### `capstone_build_an_agent.md`
- Build a small, useful agent. Examples:
  - Research agent: given a question, searches + summarizes.
  - Code assistant: given a bug, reproduces, fixes, tests.
  - Data analyst: given a CSV, answers questions via code execution.
- Use function calling or a minimal framework.
- Evaluate: run 10 test cases, measure success rate, latency.

## Why this matters

Simple LLM APIs are commodities. Competitive advantage increasingly comes from:
- How well you use them in agents.
- The tools and data you give them.
- The memory and context you manage.
- The evaluation you run.

Agentic AI is where "AI engineering" becomes a distinct discipline from "prompt engineering" or "model training."

## The honest state of agents (as of 2026)

What works:
- Tool use on well-defined tasks (SQL generation, code fixes in familiar contexts).
- RAG on focused corpora.
- Code execution for data analysis.
- Human-in-the-loop for high-stakes actions.

What partially works:
- Multi-step reasoning (often needs guardrails to avoid drift).
- Web browsing (brittle, fails on modern SPAs).
- Software engineering (SWE-Bench: best is ~50-60% of real bugs).

What doesn't work yet:
- Fully autonomous long-horizon tasks.
- Reliable planning for novel goals.
- Multi-agent collaboration at scale.
- Safety for open-ended agents.

The capability gap is where research lives.

## Reading list for Module 10

Required:
- Yao et al 2022, "ReAct: Synergizing Reasoning and Acting in Language Models".
- Lewis et al 2020, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (RAG).
- Schick et al 2023, "Toolformer: Language Models Can Teach Themselves to Use Tools".
- Shinn et al 2023, "Reflexion: Language Agents with Verbal Reinforcement Learning".
- Greenblatt 2023 "SWE-Bench" paper.

Nice-to-have:
- MemGPT paper (Packer 2023).
- "The Rise and Potential of Large Language Model Based Agents: A Survey".
- OpenAI's o1 system card.
- Anthropic's "Building effective agents" blog post (Dec 2024).

## Your relevance for internal / enterprise codebases

The link you shared (`TISDeepReasoning`) is exactly the kind of agentic codebase you'll encounter. After Module 10, you'll have the conceptual vocabulary:
- Tool definitions.
- Agent loops.
- RAG over internal docs.
- Memory management.
- Guardrails and safety.

Combined with your code-reading skills from Modules 3-5, you'll be able to navigate agentic codebases quickly.
