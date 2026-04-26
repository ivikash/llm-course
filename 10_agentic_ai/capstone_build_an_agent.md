# Module 10 Capstone - Build an Agent

Combine everything: tools, reasoning, memory (optional), safety. Ship something.

## The deliverable

A working agent that:
1. Solves a specific problem you actually care about.
2. Uses at least 2 tools.
3. Has some error handling.
4. Has basic eval (10+ test cases).
5. Has a README and code you can show.

## Pick one

Pick something YOU find genuinely useful. Motivation matters.

### Option A: Personal research agent

Goal: given a topic, produce a 1-page brief.

**Tools**:
- `search_web(query)` - DuckDuckGo or Brave search API.
- `fetch_url(url)` - fetch and extract main content.

**Loop**: search → pick top 3-5 results → fetch each → summarize → compose brief.

**Stretch**: allow follow-up questions that trigger more search.

**Eval**: 10 research topics. Does the brief cover the key facts? LLM-as-judge scoring.

### Option B: Code fixer

Goal: given a failing pytest output, produce a PR-ready fix.

**Tools**:
- `read_file(path)` - read source files.
- `write_file(path, content)` - write files.
- `run_tests(test_name)` - run pytest, capture output.
- `ls(path)` - list directory.

**Loop**: read failing test + the function it tests → hypothesize fix → write → re-test → iterate.

**Stretch**: work against a real repo's issues (SWE-Bench Lite).

**Eval**: 10 synthetic bugs you inject. Does the agent fix them?

### Option C: Data analyst

Goal: given a CSV and a question, produce an answer + chart.

**Tools**:
- `execute_python(code)` - sandboxed Python.
- `read_csv(path)` - convenience, could just be in execute_python.

**Loop**: load CSV → explore schema → write analysis code → execute → iterate → produce answer + plot.

**Eval**: 10 questions on a dataset of your choice. Are the answers right?

### Option D: Customer support chatbot

Goal: answer questions from a knowledge base with RAG.

**Tools**: RAG over your docs (Lesson 10.3).

**Stretch**: add escalation tool for when RAG doesn't have the answer.

**Eval**: 10 questions from your docs. Faithfulness score.

### Option E: Your own idea

If you have a specific problem, go for it. Agents work best when they solve *your* problems.

## Build plan

Budget: 1-2 weekends (15-25 hours).

### Session 1: scope and MVP (4-6 hours)

- Pick option.
- Define exact scope (write it down).
- Write 5 test cases first. Actual tasks you want it to solve.
- Minimal agent: 1 tool, simple loop, works on 1-2 test cases.

### Session 2: iterate (4-6 hours)

- Add remaining tools.
- Handle errors.
- Run against all test cases.
- Fix the failures.

### Session 3: evaluate and polish (4-6 hours)

- Measure success rate on test cases.
- Identify failure modes.
- Add eval logging (cost, steps, time).
- Write README.

### Session 4: polish and share (3-5 hours)

- Clean up code.
- Add safety (dry-run mode, cost cap, etc.).
- Publish on GitHub.
- Write a blog post about what you learned.

## Example: skeleton for Option A (research agent)

```python
import os, json
from openai import OpenAI
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup

client = OpenAI()

def search_web(query):
    with DDGS() as ddgs:
        return [
            {"title": r["title"], "url": r["href"], "snippet": r["body"]}
            for r in ddgs.text(query, max_results=5)
        ]

def fetch_url(url):
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        # strip scripts, styles
        for t in soup(["script", "style", "nav", "footer"]):
            t.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return text[:5000]  # truncate
    except Exception as e:
        return f"Error: {e}"

tools = [
    {"type":"function","function":{
        "name":"search_web",
        "description":"Search the web. Returns list of {title, url, snippet}.",
        "parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
    {"type":"function","function":{
        "name":"fetch_url",
        "description":"Fetch and extract main text from a URL.",
        "parameters":{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}}},
]

def research(topic, max_steps=15):
    messages = [
        {"role":"system","content":"You are a research agent. Use tools to gather information, then write a 300-word structured brief on the topic. Include citations."},
        {"role":"user","content":f"Research: {topic}"},
    ]
    cost_estimate = 0.0

    for step in range(max_steps):
        r = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
        m = r.choices[0].message
        messages.append(m.model_dump())

        # simple cost tracking
        cost_estimate += r.usage.prompt_tokens * 0.00015 / 1000 + r.usage.completion_tokens * 0.0006 / 1000
        if cost_estimate > 0.50:
            return "COST CAP EXCEEDED."

        if not m.tool_calls:
            return m.content

        for tc in m.tool_calls:
            args = json.loads(tc.function.arguments)
            if tc.function.name == "search_web":
                result = json.dumps(search_web(args["query"]))
            elif tc.function.name == "fetch_url":
                result = fetch_url(args["url"])
            else:
                result = f"Unknown tool: {tc.function.name}"
            messages.append({"role":"tool","tool_call_id":tc.id,"content":result[:4000]})

    return "MAX STEPS REACHED."

# Quick eval
test_topics = [
    "state of reasoning LLMs 2025",
    "differences between PostgreSQL and MySQL",
    "best Python web frameworks 2024",
]
for topic in test_topics:
    print(f"\n=== {topic} ===\n{research(topic)}\n")
```

~80 lines. A working research agent with cost cap. Extend from here.

## The report

Write `agent_report.md`:

```markdown
# [Your Agent Name]

## What it does
[1-2 sentences]

## Architecture
[Diagram or list: tools, loop, any frameworks]

## Results
- Test cases: N
- Success rate: X%
- Average cost per task: $Y
- Average steps per task: Z

## Interesting failures
[3-5 examples of failures and what caused them]

## What I'd do differently
[Reflection]

## Code
[GitHub link]
```

Put it in your portfolio. Serious ML roles care about "have you shipped?"

## Extensions

Once the basic agent works, try:

### A. Add RAG
Give the agent access to a corpus of your own documents via vector search.

### B. Add memory
Remember user preferences across sessions.

### C. Add reasoning
Reflexion-style retries. Or use o1/R1 as the backbone.

### D. Make it multi-agent
Planner + executor + critic.

### E. Publish as an MCP server
So Claude Desktop / Cursor can use your tools.

### F. Deploy
Wrap in a web UI or Slack bot.

## What this capstone earns you

You have:
- A working agent you built.
- A repo you can show in interviews.
- Real numbers (cost, success rate).
- Understanding of all the trade-offs.

Almost nobody interviewing for "AI engineer" roles in 2026 has shipped an agent themselves. You will. That's rare.

## End of the course

After Module 10 capstone, you've:

- Learned the math (Module 1).
- Built a transformer from scratch (Module 3).
- Trained a chatbot end-to-end (Module 5).
- Used cloud GPUs (Module 6).
- Read the canonical papers (Module 7).
- Thought about team leadership (Module 8).
- Covered multimodal AI (Module 9).
- Built a production-ish agent (Module 10).

You are now genuinely fluent in AI.

What's next:
- Keep doing. Shipping is the skill.
- Read one paper per week.
- Contribute to open source.
- Get a role where you get to apply this.

This course gave you scaffolding. Your career is built on it.

Good luck.
