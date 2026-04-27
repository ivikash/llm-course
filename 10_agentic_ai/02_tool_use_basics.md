# 10.2 - Tool Use Basics

Tools are the agent's hands. This lesson covers the mechanics: how to define tools, how the LLM calls them, how you handle the response.

## The function-calling interface

**Inspect a real tool-call JSON live:**

```viz
{"viz": "tool_call_inspector"}
```

Switch between examples (weather, search, SQL, parallel-calls). Hover any yellow-highlighted piece for what it means. This is exactly what the OpenAI API returns when the LLM decides to invoke a tool.


Modern LLM APIs (OpenAI, Anthropic, Google, Mistral) support native tool calling. You pass tool definitions alongside the prompt. The model returns either text or a structured tool call.

### OpenAI format (de facto standard)

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    }
}]
```

That's a JSON Schema. The model uses:
- **name**: to match the intended tool.
- **description**: to decide *whether* to use it.
- **parameters**: to format the arguments.

Writing good descriptions is more important than you'd expect. Bad: "weather tool". Good: "Get the current weather for a city. Use this when the user asks about weather, temperature, forecast, or outdoor conditions."

### The model's response

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [{
    "id": "call_abc123",
    "type": "function",
    "function": {
      "name": "get_weather",
      "arguments": "{\"city\": \"Paris\", \"unit\": \"celsius\"}"
    }
  }]
}
```

Two important things:
- `content` is `null` (the model chose to call a tool, not to respond).
- `arguments` is a JSON *string* (not parsed object). You parse it yourself.

### The loop

```python
import json, openai

client = openai.OpenAI()

def call_tool(name, args):
    if name == "get_weather":
        # actual implementation
        return f"The weather in {args['city']} is 22°C, sunny."

messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]

while True:
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, tools=tools
    )
    msg = response.choices[0].message
    messages.append(msg.model_dump())

    if not msg.tool_calls:
        print(msg.content)
        break

    for tc in msg.tool_calls:
        result = call_tool(tc.function.name, json.loads(tc.function.arguments))
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result,
        })
```

~30 lines. That's the entire agent mechanism.

## Parallel tool calls

OpenAI and Claude both support calling multiple tools in one response:

```python
"I need to fetch the weather in Paris AND convert from EUR to USD."
# → tool_calls has TWO entries, executable in parallel.
```

Your loop should handle lists of tool_calls. Saves round-trips.

## Error handling

Things that go wrong:

### Bad JSON in arguments
Rare but happens. Older models do this more. Solution: wrap `json.loads` in try/except; feed the error back as tool output so the model can self-correct.

### Wrong tool name
Model hallucinates a tool that doesn't exist. Solution: feed back "Tool XYZ does not exist. Available tools: [list]."

### Tool raises an exception
Your function crashes. Don't let the exception kill the loop. Catch, format as string, feed back. The model can often recover.

### Tool returns something huge
Logs, HTML dumps. Can blow past context. Truncate or summarize before inserting.

## Anthropic (Claude) format

Nearly identical to OpenAI, with different key names:

```python
tools = [{
    "name": "get_weather",
    "description": "...",
    "input_schema": {"type": "object", "properties": {...}, "required": [...]},
}]
```

Response structure is slightly different (Claude uses `tool_use` content blocks). Porting between OpenAI and Claude is a 1-hour job.

## Google Gemini

Uses similar JSON Schema format with minor differences.

## Visualize this

**A tool definition, inspected**:

```
  tool = {
      "type": "function",
      "function": {
          "name": "get_weather",                     ← unique name
          "description": "Get current weather.",      ← USED BY LLM to
                                                       decide when to call
          "parameters": {
              "type": "object",                        ← always "object"
              "properties": {
                  "city": {
                      "type": "string",
                      "description": "City name"       ← helps LLM
                  },
                  "unit": {
                      "type": "string",
                      "enum": ["C", "F"],               ← restrict values
                  },
              },
              "required": ["city"],                     ← which are mandatory
          },
      }
  }

  This JSON Schema teaches the LLM:
    - the tool exists
    - what it does
    - what arguments it takes
    - what's optional
```

**Lifecycle of a tool call**:

```
  User: "What's the weather in Paris in Celsius?"
       │
       ▼
  LLM decides: "I should call get_weather."
       │
       ▼
  LLM returns:
  {
    "tool_calls": [{
      "id": "call_abc123",
      "name": "get_weather",
      "arguments": '{"city": "Paris", "unit": "C"}'
    }]
  }
       │
       ▼
  Your code parses:
    name = "get_weather"
    args = {"city": "Paris", "unit": "C"}
       │
       ▼
  Your code runs:
    result = get_weather_impl(city="Paris", unit="C")
           = "18°C, partly cloudy"
       │
       ▼
  Feed result back:
  {"role": "tool", "tool_call_id": "call_abc123", "content": "18°C, partly cloudy"}
       │
       ▼
  LLM sees result. Decides:
    "I have the answer. Done."
       │
       ▼
  LLM final response: "The weather in Paris is 18°C and partly cloudy."
```

**Parallel tool calls** (modern models):

```
  User: "What's the weather in Paris AND what's 237 × 491?"
       │
       ▼
  LLM decides: call BOTH tools simultaneously.
       │
       ▼
  LLM returns:
  {
    "tool_calls": [
      {"id": "call_1", "name": "get_weather", "arguments": "..."},
      {"id": "call_2", "name": "calculator",  "arguments": "..."}
    ]
  }
       │
       ▼
  Your code runs BOTH in parallel (or sequential, both work):
    result_1 = get_weather(...)
    result_2 = calculator(...)
       │
       ▼
  Feed both results back:
  [{"role": "tool", "tool_call_id": "call_1", ...},
   {"role": "tool", "tool_call_id": "call_2", ...}]
       │
       ▼
  LLM synthesizes: "The weather is ... and 237 × 491 = ..."
```

**Error handling (essential in production)**:

```
  Good agent loop:

  try:
      result = my_tool(**args)
  except Exception as e:
      result = f"Error: {type(e).__name__}: {e}"

  Feed the error back to the LLM. It will often:
    - Retry with different args
    - Try a different tool
    - Apologize to user if it can't recover

  Don't silently swallow errors. Don't let exceptions kill the agent.
```

**Native tool calling vs ReAct prompting**:

```
  Native (OpenAI, Claude, Gemini, Mistral):
  ──────────────────────────────────────
  API understands tools. Parses JSON. Returns structured tool_calls.

  Pros: reliable, structured, supports parallel calls.
  Cons: tied to specific API format.

  ReAct prompting (works on any LLM):
  ──────────────────────────────────
  You prompt the LLM with:
    "Respond in this format:
     Thought: ...
     Action: tool_name
     Action Input: {json args}

     Then I'll say:
     Observation: result

     Repeat until done, then:
     Final Answer: ..."

  You parse this text yourself.
  Works on any LLM (including local Llama / Mistral).

  Pros: works everywhere.
  Cons: brittle (model might break the format).
  Usage today: mostly with open-source LLMs that lack native tool calling.
```

**MCP (Model Context Protocol): the cross-provider standard**:

```
  Problem: tools defined differently for OpenAI, Claude, Gemini, Mistral.
          Reimplement each. Ugh.

  Solution: MCP (Anthropic, Nov 2024).

  ┌──────────────────────────────────────────────┐
  │ MCP Server                                    │
  │  - defines tools once                          │
  │  - exposes resources (files, data)             │
  │  - standard protocol                            │
  └─────────────┬────────────────────────────────┘
                │
                │ standard MCP protocol
                │
    ┌───────────┼───────────┬───────────┐
    ▼           ▼           ▼           ▼
  Claude    Cursor      VS Code    Your app
  Desktop             (Copilot)    (anything)

  Write tools once, use everywhere MCP-compliant.
  Increasingly adopted through 2025-26.
```

**Tool design principles**:

```
  GOOD tool:
    ✓ One clear purpose
    ✓ Descriptive name ("search_arxiv" vs "search1")
    ✓ Clear parameter names
    ✓ Useful description (teaches the LLM when to use)
    ✓ Returns structured output (JSON preferred)
    ✓ Handles errors gracefully (returns error strings, doesn't crash)
    ✓ Idempotent (safe to call twice)

  BAD tool:
    ✗ "do_stuff(query)" - no clarity
    ✗ Mega-tool that does everything
    ✗ Silently fails (returns "" on error)
    ✗ Side effects without confirmation (sends email on every call)
```

**Tool granularity choice**:

```
  Option A: few broad tools
    search(type, query)
      type ∈ {web, wikipedia, arxiv}

  Option B: many specific tools
    search_web(query)
    search_wikipedia(query)
    search_arxiv(query)

  Tradeoffs:
    Option A: fewer tools (LLM easier to manage) but more decision inside tool.
    Option B: more tools (LLM picks) but clearer intent.

  Rule of thumb:
    < 5 tools: keep them specific (Option B).
    5-15 tools: mix; maybe group related ones under umbrella tool.
    > 20 tools: use router pattern (dispatch in the code).
```

**Security considerations**:

```
  Every tool is an entry point. Treat LLM output like untrusted user input.

  Tool calls your code runs:
    ┌──────────────────────┐
    │ LLM produces args     │
    │ (attacker-controllable│
    │  if prompt-injected)   │
    └──────────┬───────────┘
               ▼
    ┌──────────────────────┐
    │ Your tool function    │
    │ executes those args    │
    └──────────────────────┘

  Danger:
    - File operations → path traversal attacks
    - Shell commands → command injection
    - Database queries → SQL injection
    - HTTP calls → SSRF / exfiltration

  Mitigations:
    - Whitelist paths / patterns
    - Parameterized queries (never format strings in)
    - No shell command execution with LLM-provided args
    - Rate limit destructive ops
    - Human-in-the-loop for high-stakes actions
```

## Anthropic's MCP (Model Context Protocol)

Released November 2024. A **standard** for how tools are defined and shared across providers.

Idea: instead of each integration hardcoding Anthropic/OpenAI/Google tool formats, you implement an MCP server once. Any MCP-compliant client (Claude Desktop, Cursor, VS Code, etc.) can use it.

Growing rapidly. If you're building tools you want widely reusable, target MCP.

Reference: https://modelcontextprotocol.io

## Tool design principles

### Good tools

- **Clear purpose**: one job per tool.
- **Predictable behavior**: same inputs → same outputs.
- **Structured outputs**: JSON is better than prose.
- **Failure modes documented**: what happens when the tool errors.
- **Idempotent when possible**: calling twice should be safe.

### Bad tools

- "do_everything(query: str)" that tries to interpret requests.
- Silent failures.
- Outputs too large to fit in context.
- Side effects that aren't reversible without user confirmation.

## Tool granularity

Should `search_web` be one tool, or `google_search` + `wikipedia_search` + `arxiv_search` as three? Tradeoffs:

- **One tool, parametrized**: fewer tools, more decisions inside the tool. Model needs less reasoning.
- **Many specific tools**: model picks the right one. More reasoning required but more controllable.

Rule of thumb: if you have 3-10 tools, keep them specific. If you have 50+, consolidate via router tools.

## Tool call parsing from raw text (for non-native-calling models)

Older or open models without function calling can still "call tools" via ReAct-style prompting:

```
You have access to these tools:
- search_web(query: str)
- calculator(expr: str)

Use them as:
Action: search_web
Action Input: "Paris weather"

Observation: [you'll see the result here]

Then respond with:
Final Answer: ...

User: What's the weather in Paris?
Assistant: Thought: I need to look up Paris weather.
Action: search_web
Action Input: "Paris weather today"
```

You parse `Action:` / `Action Input:` patterns from the response, execute, insert `Observation:`, continue.

Brittle but works. Replaced by native function calling when available.

## Schema validation

For production, validate tool call arguments against your JSON Schema:

```python
import jsonschema
jsonschema.validate(args, tool_schema["parameters"])
```

Catches malformed args before your tool implementation sees them.

## Security: tool call as code path

Every tool call is the LLM asking your code to do something. Treat it accordingly:

- **Sanitize inputs**: if a tool arg goes into a shell command, quote it. If into SQL, parameterize.
- **Don't trust URLs**: the LLM might produce attacker-controlled URLs.
- **Rate limit destructive ops**: if the LLM is calling "delete_user", have guardrails.
- **Human-in-the-loop for big actions**: payments, emails, deletions.

See Lesson 10.11 for deeper safety.

## nanochat's `execution.py` as reference

Open `~/workspace/nanochat/nanochat/execution.py`. ~200 lines. Implements:
- Run arbitrary Python in a subprocess.
- Capture stdout/stderr.
- Timeout.
- Return structured result.

It's a minimal tool for the "code execution" tool in an agent. Good reference for implementing your own.

## Exercises

1. Write tool definitions for a calculator, a web search, and a current-time-getter. Use them with OpenAI's API.

2. Add error handling: make your calculator tool return `"Error: division by zero"` instead of crashing. Verify the LLM handles it gracefully.

3. Implement ReAct-style prompting for a model that doesn't support native tool calling (e.g., Mistral-7B or a local Llama). Parse `Action:` / `Observation:` pairs.

4. Read Anthropic MCP docs. Implement a toy MCP server that exposes one tool. Test with Claude Desktop.

5. Read `nanochat/execution.py`. Understand the subprocess + timeout pattern.

## Next

`03_retrieval_augmented_generation_rag.md` - the most important agent-adjacent pattern.
