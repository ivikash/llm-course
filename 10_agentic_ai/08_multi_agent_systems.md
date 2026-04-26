# 10.8 - Multi-Agent Systems

Instead of one LLM doing everything, multiple specialized LLMs collaborate. Sometimes helpful, often cargo-cult.

## When multi-agent actually helps

1. **Role specialization**: different prompts / tools / models per role.
2. **Parallel work**: subtasks that can run concurrently.
3. **Adversarial review**: one agent proposes, another critiques.
4. **Longer effective context**: each agent has its own context budget.

## When it doesn't help (which is often)

1. **Simple tasks**: one good agent beats three mediocre agents.
2. **Communication overhead**: agents waste tokens talking to each other.
3. **Compounding errors**: each agent introduces noise; noise compounds.
4. **Debugging**: traces across multiple agents are painful.

**Rule**: try single-agent first. Add more only when you have a specific reason.

## Common multi-agent patterns

### Pattern 1: Manager + workers

One "manager" agent decomposes tasks and delegates to worker agents.

```
User → Manager
         ├─> Researcher
         ├─> Coder
         └─> Writer
Manager ← results ← workers
Manager → final answer
```

Use case: complex projects. Research-heavy tasks.

### Pattern 2: Critic + proposer

One agent proposes, another critiques. Loop until the critic is satisfied.

```
Proposer: "Here's a plan..."
Critic: "Plan has issues X, Y. Revise."
Proposer: "Revised plan..."
Critic: "LGTM."
```

Use case: code review, creative writing, plans that need scrutiny.

### Pattern 3: Debate / consensus

N agents independently solve, then vote or debate.

```
Agent A: "Answer: X."
Agent B: "Answer: Y. Disagree with A because..."
Agent C: "Adjudicating... X is more defensible."
Final: X
```

Use case: uncertain problems where diverse perspectives reduce bias.

### Pattern 4: Assembly line

Each agent handles one stage in a pipeline.

```
Raw input → Parser → Summarizer → Formatter → Output
```

Use case: document processing, ETL-like flows.

### Pattern 5: Role-play committee

Agents with named personas (CEO, engineer, lawyer, etc.) discuss a decision.

Works for brainstorming. Weak for real decisions - agents are often sycophantic.

## Frameworks

### AutoGen (Microsoft)

Most mature multi-agent framework.

```python
from autogen import ConversableAgent

coder = ConversableAgent(name="coder", system_message="You write Python.", ...)
reviewer = ConversableAgent(name="reviewer", system_message="You review code.", ...)
coder.initiate_chat(reviewer, message="Write me a fibonacci function.")
```

The two agents converse until one of them says "TERMINATE".

### CrewAI

Role-based. Each agent has a goal, backstory, tools.

```python
researcher = Agent(role="researcher", goal="...", tools=[search_tool])
writer = Agent(role="writer", goal="...", tools=[])
crew = Crew(agents=[researcher, writer], tasks=[...])
crew.kickoff()
```

### MetaGPT

Simulates a software company: CEO → PM → Architect → Engineer → QA. Each is an agent.

Fun demos, production reliability iffy.

### Hand-rolled

Multi-agent in 100 lines of Python is very doable. Simulate each agent as a function that takes (memory, tools, task) → (action, new_memory). A simple scheduler routes messages.

## Specific research directions

### Society of Mind (Marvin Minsky's concept, applied)
A team of narrow agents adding up to something general. Research, not production.

### Agent-based simulations
Simulate behavior of crowds of agents (e.g., Stanford's Generative Agents paper) to study emergent behavior.

Paper (Park 2023): https://arxiv.org/abs/2304.03442

### Emergent communication
Let agents invent their own language. Academic curiosity; not practical.

### Hierarchical agents
Meta-agents that spawn sub-agents. For very large tasks.

## Tradeoffs

| Factor | Single agent | Multi-agent |
|--------|--------------|-------------|
| **Simplicity** | Simple | Complex |
| **Latency** | Low | High (agents take turns) |
| **Cost** | Low | High (multiple LLM calls) |
| **Debuggability** | Easy | Hard |
| **Error propagation** | Contained | Compounds |
| **Parallel work** | No | Yes (if designed for it) |
| **Quality** | Limited by one model | Can exceed via specialization |
| **Context limits** | Shared | Each agent has own context |

## Concrete example: researcher + summarizer

Two-agent task: read 10 articles on a topic, produce a structured brief.

```python
from openai import OpenAI
import json
client = OpenAI()

# Agent 1: researcher (has web search tool)
# Agent 2: summarizer (has no tools, just reads notes)

def researcher(topic):
    # uses search + fetch tools, returns list of facts
    ...

def summarizer(notes):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Summarize as a structured brief:\n{notes}"}]
    )
    return response.choices[0].message.content

topic = "State of reasoning LLMs 2025"
notes = researcher(topic)    # single agent with tools
brief = summarizer(notes)    # single agent without tools, small model
print(brief)
```

**Is this really multi-agent?** Barely. It's more like a pipeline. But it demonstrates role specialization: the researcher uses tools, the summarizer doesn't, and we use a cheaper model for the summarizer.

This simple pipeline often outperforms fancy multi-agent frameworks because it's easier to debug and has clearer responsibilities.

## Anti-patterns

### 1. "Let's have 5 agents discuss"

Adding agents doesn't automatically improve quality. Often the opposite - groupthink, sycophancy.

### 2. Infinite back-and-forth

Agents loop: "Thanks!" / "You're welcome!" / "No thanks to you!" - termination is a real design concern.

### 3. Opaque authority

Unclear which agent has the final say. Decisions made by consensus of weak agents.

### 4. Over-specialization

Breaking down a task into 20 agents with tiny roles. Each adds overhead; net effect is worse.

## My honest take

As of 2026, multi-agent is **overhyped in frameworks**, **undervalued in specific scenarios**.

Overhyped: "My agents autonomously run a company!" demos rarely hold up to real scrutiny.

Undervalued: simple two-agent patterns (proposer + critic, researcher + writer) can meaningfully improve quality for the right tasks.

## Reading

- **AutoGen** (Wu 2023): https://arxiv.org/abs/2308.08155
- **Generative Agents (Smallville)** (Park 2023): https://arxiv.org/abs/2304.03442
- **MetaGPT** (Hong 2023): https://arxiv.org/abs/2308.00352
- **Debate as a tool** (Irving 2018, revisited 2023): https://arxiv.org/abs/1805.00899

## Exercises

1. Implement proposer + critic for code writing. Proposer writes, critic reviews, proposer revises. Up to 3 rounds.

2. Try AutoGen's tutorial (15 minutes). Note the abstractions.

3. Think: what's the simplest task where multi-agent genuinely helps vs. hurts? (My guess: writing that benefits from critique.)

## Next

`09_memory_and_statefulness.md` - making agents remember.
