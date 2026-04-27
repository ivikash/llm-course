# 10.6 - Agent Frameworks

At some point you'll consider: "should I use a framework?" This lesson gives you an honest map.

## The claim

**For simple agents (1-3 tools, clear loop): hand-rolled is better.** ~50 lines of Python beats importing anything.

**For complex agents (many tools, complex routing, team collaboration): frameworks help.** Pick carefully.

Most popular frameworks optimize for demos, not production. Use them knowingly.

## The framework landscape (2026)

### LangChain

**Age**: Oct 2022. Oldest.
**Pitch**: "batteries included for LLM apps."
**Strengths**: huge ecosystem, integrates with everything, good for prototyping, lots of tutorials.
**Weaknesses**: heavy abstractions, docs churn constantly, debugging is harder because the abstraction gets in the way.
**When to use**: rapid prototype, you want integrations with 100+ LLMs/vector DBs/tools and don't want to write them.
**When NOT**: production, where you want to understand every line.

```python
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI

tools = [Tool(name="calc", func=lambda x: str(eval(x)), description="...")]
agent = initialize_agent(tools, ChatOpenAI(model="gpt-4o"), agent=AgentType.OPENAI_FUNCTIONS)
agent.run("what is 23*98?")
```

### LlamaIndex

**Age**: Nov 2022.
**Pitch**: "data framework for LLMs" - RAG-focused.
**Strengths**: best in class for RAG pipelines, lots of data connectors, good indexing primitives.
**Weaknesses**: fewer agent features than LangChain. Also has abstraction overhead.
**When to use**: your agent is primarily a RAG system over your docs.
**When NOT**: heavy tool use, multi-agent.

### AutoGen

**Age**: 2023 (Microsoft).
**Pitch**: "multi-agent conversations."
**Strengths**: conversation between multiple agents, human-in-the-loop, Microsoft-backed.
**Weaknesses**: Python-only, complex for simple tasks.
**When to use**: agents that need to dialogue with each other (coder + critic, researcher + writer).
**When NOT**: single-agent pipelines.

### CrewAI

**Age**: 2024.
**Pitch**: "role-playing autonomous AI agents."
**Strengths**: simple mental model (agents have roles, goals, tools), lots of marketing.
**Weaknesses**: immature, production reliability iffy.
**When to use**: experimental multi-agent workflows with clear role divisions.
**When NOT**: production.

### DSPy

**Age**: 2023 (Stanford).
**Pitch**: "programming, not prompting." Compile prompts from declarative programs.
**Strengths**: unique: treats prompting as a compilation problem, optimizes prompts with data.
**Weaknesses**: steep learning curve, different mental model.
**When to use**: research, when you want to treat LLMs as compositionally programmable, when you have good eval data.
**When NOT**: one-off projects.

### Letta (ex-MemGPT)

**Age**: 2023 (UC Berkeley).
**Pitch**: "stateful agents with memory."
**Strengths**: first-class support for persistent memory, agents that maintain long-term state.
**Weaknesses**: niche.
**When to use**: chatbots that need to remember users across sessions.

### OpenAI Assistants API

**Age**: Nov 2023.
**Pitch**: managed agent infrastructure from OpenAI.
**Strengths**: file search, code interpreter, function calling, threads all managed.
**Weaknesses**: vendor lock-in, less flexible than building your own.
**When to use**: quick prototypes with OpenAI models, you're OK being tied to OpenAI.
**When NOT**: multi-provider, custom tooling.

### Anthropic MCP (Model Context Protocol)

**Age**: Nov 2024.
**Pitch**: an open standard for defining tools/resources across any AI client.
**Strengths**: portable - write an MCP server once, works with Claude Desktop, Cursor, VS Code, etc.
**Weaknesses**: early adoption.
**When to use**: you're exposing tools for multiple clients to use.

Not a framework per se, more a protocol. But increasingly important.

### Haystack

**Age**: pre-LLM era (2020), updated.
**Pitch**: RAG-focused framework.
**Use**: less popular now vs. LangChain/LlamaIndex, but production-tested.

### Vercel AI SDK

**Language**: TypeScript.
**Use**: if your frontend is TS/Next.js.

### Inngest Agent Kit / Mastra

**Age**: 2024-2025.
**Pitch**: production-focused, durable workflows.

## The "hand-rolled" alternative

For 80% of use cases, this is the right answer:

```python
import openai, json

client = openai.OpenAI()

tools = [...]   # your JSON Schema definitions
TOOL_REGISTRY = {"tool_name": tool_function, ...}

def run_agent(initial_message, max_steps=10):
    messages = [{"role": "user", "content": initial_message}]
    for _ in range(max_steps):
        r = client.chat.completions.create(
            model="gpt-4o", messages=messages, tools=tools
        )
        m = r.choices[0].message
        messages.append(m.model_dump())
        if not m.tool_calls:
            return m.content
        for tc in m.tool_calls:
            args = json.loads(tc.function.arguments)
            result = TOOL_REGISTRY[tc.function.name](**args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })
    return "Max steps reached."
```

~30 lines. Debug-transparent. You understand every line. For simple agents, this beats any framework.

Add features as needed:
- Retry on tool error.
- Structured logging.
- Max tool calls per step.
- Context compression.

## Visualize this

**Framework complexity vs control**:

```
  Hand-rolled           Tiny framework        Full framework
  (30 lines)            (LangChain lite)      (LangChain / AutoGen)
                                                 
  Full control           Some abstraction       Maximum features
  Easy to debug           Slightly faster         Harder to debug
  Fits in your head       Fits mostly             Big API surface
  You code everything     Provides common          Pre-built for many
                           patterns                 use cases

  Use for:                Use for:                 Use for:
  learning                small projects           complex multi-agent
  simple agents           rapid prototyping        enterprise apps
  production              well-understood patterns  when you need breadth
```

**Framework choice matrix**:

```
  Your need                    Recommended
  ────────────────────────     ──────────────────────
  Learning agents               hand-rolled (know the pattern)
  Simple RAG + chat              LlamaIndex or hand-rolled
  Complex RAG                    LlamaIndex
  Multi-agent collaboration      AutoGen or CrewAI
  Integrate many LLMs + tools    LangChain (widest support)
  Optimize prompts               DSPy
  Stateful persistent agents     Letta (ex-MemGPT)
  Managed infra                  OpenAI Assistants API
  Cross-provider tools           MCP server
```

**Decision flowchart**:

```
  "Should I use a framework?"
        │
        ▼
  How many tools?
        │
        ├── 1-5 ──▶  Hand-roll (30-50 lines)
        │
        ├── 5-20 ──▶ LangChain OR hand-rolled with routing
        │
        └── 20+ ──▶  LangChain + custom routing

  Is this RAG-heavy?
        │
        ├── Yes ──▶  LlamaIndex
        │
        └── No ──▶   (use above choice)

  Multi-agent?
        │
        ├── Yes ──▶  AutoGen or CrewAI
        │
        └── No ──▶   Single-agent (above)
```

**Pros and cons at a glance**:

```
  LangChain (2022+)
  ─────────────────
  ✓ Most integrations (LLMs, vector DBs, tools)
  ✓ Large community, lots of tutorials
  ✗ Heavy abstractions, docs churn
  ✗ Debugging is painful (stack traces through abstractions)
  Best for: rapid prototyping, integrations.

  LlamaIndex (2022+)
  ──────────────────
  ✓ RAG-focused, best-in-class chunking/querying
  ✓ Good data connectors
  ✗ Less versatile for general agents
  Best for: knowledge-base chat, RAG pipelines.

  AutoGen (Microsoft, 2023)
  ──────────────────────────
  ✓ First-class multi-agent conversation
  ✓ Human-in-the-loop support
  ✗ Complex for single-agent
  ✗ Python-only
  Best for: researcher-critic, planner-executor patterns.

  CrewAI (2024)
  ──────────────
  ✓ Role-based agents (researcher, writer, reviewer)
  ✓ Clean API
  ✗ Newer, production reliability iffy
  Best for: prototyping multi-role workflows.

  DSPy (Stanford, 2023)
  ──────────────────────
  ✓ Declarative: "here's what I want; compile the prompts"
  ✓ Automatic prompt optimization given eval data
  ✗ Steep learning curve
  Best for: research; when you have eval data and want to optimize.

  Letta / MemGPT (2023)
  ──────────────────────
  ✓ Persistent memory primitives
  ✓ Agents with long-term state
  ✗ Niche
  Best for: personal assistants with memory.

  OpenAI Assistants API
  ─────────────────────
  ✓ Managed: file search, code interpreter, threads
  ✓ Minimal code
  ✗ Vendor lock-in (OpenAI only)
  ✗ Opaque (can't see internal prompts)
  Best for: quick prototypes with OpenAI.

  MCP (Anthropic, 2024)
  ──────────────────────
  ✓ Cross-provider tool standard
  ✓ Reusable tool servers
  ○ Still emerging
  Best for: tool authors; future-proof setup.
```

**Same agent, 3 different frameworks** (compare verbosity):

```
  Task: "Given a math question, compute via calculator tool."

  Hand-rolled (30 lines):
  ────────────────────────
  def calculator(expr): return eval(expr, {}, {})
  tools = [{...schema...}]
  messages = [{"role":"user","content":"..."}]
  while True:
      r = client.chat.completions.create(...)
      if not r.choices[0].message.tool_calls: break
      for tc in ...: messages.append({"role":"tool",...})

  LangChain (~10 lines but hidden complexity):
  ─────────────────────────────────────────────
  from langchain.agents import initialize_agent, Tool
  from langchain_openai import ChatOpenAI
  tools = [Tool(name="calc", func=calculator, description="...")]
  agent = initialize_agent(tools, ChatOpenAI(model="gpt-4o"),
                             agent=AgentType.OPENAI_FUNCTIONS)
  print(agent.run("What is 237 * 491?"))

  DSPy (declarative):
  ───────────────────
  import dspy
  class Calc(dspy.Signature):
      "Compute math expressions"
      question: str = dspy.InputField()
      answer: str = dspy.OutputField()
  # DSPy compiles the prompt from examples
```

Each has tradeoffs. Pick the one that matches your goal.

**Watch out for abstraction leakage**:

```
  Problem: when a framework abstracts too much, bugs hide.

  Example:
    You use LangChain's AgentExecutor.
    Agent keeps looping forever.
    You can't figure out why.
    Logs hidden inside abstractions.

    vs hand-rolled:
    You have 30 lines of Python.
    Add print statements.
    Immediately see: "agent keeps calling search_web with same query".
    Fix: dedupe tool calls.

  Rule: prototype with framework, PRODUCTIONIZE by understanding
  what the framework does and (possibly) replacing with custom code.
```

**Observability stacks** (critical for production):

```
  LangSmith (LangChain's):
    traces every step of every agent run.
    see prompts, outputs, tool calls, errors.

  Helicone:
    LLM-call logging.
    costs, latencies, errors.

  Braintrust:
    eval + monitoring combined.

  Arize Phoenix (open source):
    tracing with span-level detail.

  Hand-rolled:
    structured logging to JSON.
    analyze later with jq / DuckDB / Pandas.

  → Whatever framework you pick, HAVE observability.
  → You'll need it when things go wrong.
```

**My honest recommendation**:

```
  For this course's capstone (your first serious agent):

  Step 1: Hand-roll in ~50 lines.
  Step 2: Get it working end-to-end.
  Step 3: If you hit real pain → reach for LangChain.
  Step 4: If you hit framework pain → back to hand-roll.

  Many production agents end up as careful hand-rolls + helper libraries.
  Few end up as pure framework code.

  Why: the SPECIFIC combination of tools/flow/errors for YOUR problem
  rarely matches what a framework provides out of the box.
```

## Criteria for picking

Ask:

1. **How many tools?**
   - 1-5: hand-rolled.
   - 6-20: consider a light framework.
   - 20+: framework or build your own routing.

2. **RAG primary or secondary?**
   - Primary: LlamaIndex.
   - Secondary: LangChain or hand-rolled.

3. **Single agent or multi-agent?**
   - Single: hand-rolled.
   - Multi: AutoGen, CrewAI, or hand-rolled with message passing.

4. **Production or experiment?**
   - Production: lean toward hand-rolled or tiny framework.
   - Experiment: any framework is fine.

5. **Team size?**
   - Solo: whatever works for you.
   - Team: consider the framework's ecosystem and hiring pool.

6. **Vendor concerns?**
   - Multi-provider: avoid OpenAI Assistants API.
   - Tied to OpenAI OK: Assistants can be convenient.

## Common mistakes

### 1. Over-frameworking
Starting with LangChain for a 3-tool agent, then spending days fighting abstractions. Start simple.

### 2. Getting stuck in v1 frameworks
LangChain underwent a major rewrite (v0.1 → 0.2 → 0.3). Code from 2023 tutorials often breaks. Check dates.

### 3. Trusting framework output
"LangChain worked" might mean it ran without errors, not that it performed well. Always eval.

### 4. Missing the MCP shift
If you're building reusable tools, target MCP. Otherwise you might have to rewrite when MCP adoption accelerates.

### 5. Ignoring observability
Frameworks hide logs. Use LangSmith, Helicone, Arize, Braintrust, or your own structured logging.

## My recommendations

For this course's capstone (a small practical agent):

1. **Start hand-rolled.** You know enough now (Lessons 10.1-10.5).
2. **If RAG-heavy, consider LlamaIndex.**
3. **If you want to try frameworks**, LangChain is the widest-known.
4. **If exposing tools to Claude Desktop / Cursor / etc.**, implement MCP.
5. **Try DSPy later** once you want to think about prompt programming as compilation.

## Exercises

1. Take the hand-rolled 30-line agent and add: retry on tool error, max 3 retries per tool call.

2. Rewrite the same agent in LangChain. Count lines. Note abstraction.

3. Read the MCP spec (~30 minutes). Build a toy MCP server exposing one tool. Connect with Claude Desktop.

4. Read LlamaIndex's "5-line RAG" quickstart. Try it on the `~/workspace/llm-course` docs themselves.

## Next

`07_planning_and_reasoning.md` - making agents think harder before acting.
