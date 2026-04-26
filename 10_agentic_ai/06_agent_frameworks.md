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
