# 10.4 - Code Execution Agents

Give an LLM a Python interpreter and suddenly it can compute, analyze data, call APIs, and produce plots. This is one of the most powerful agentic patterns - widely deployed (OpenAI Code Interpreter, Claude analysis tool, Cursor, Replit Agent, Devin).

## The core loop

```
1. LLM receives a task.
2. LLM writes Python code to solve part of it.
3. Sandbox executes the code.
4. Output (stdout, errors, figures) goes back to LLM.
5. LLM interprets, writes more code or answers.
```

That's it. A tool (`execute_python`) that takes a string and returns stdout. Everything else is the LLM's reasoning about what to run.

## A tiny implementation

```python
import subprocess, openai, json

def execute_python(code: str) -> str:
    result = subprocess.run(
        ["python", "-c", code],
        capture_output=True, text=True, timeout=30
    )
    return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

tools = [{
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": "Execute Python code. Returns stdout and stderr.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        }
    }
}]

# standard agent loop here (see 10.2)
```

This will work for safe math problems. **Do not run on untrusted input without sandboxing.** See safety section below.

## What code execution unlocks

### Data analysis

User uploads a CSV. Agent:
1. Reads it with pandas.
2. Computes statistics.
3. Generates charts.
4. Explains findings.

```
User: Analyze sales.csv. What was the biggest-growing product last quarter?
Agent:
    Action: execute_python
    Code: import pandas as pd; df = pd.read_csv("sales.csv"); ...
    Observation: "Product X grew 142% QoQ..."
Agent: Product X had the biggest growth at 142%. Here's a chart:
```

### Mathematical reasoning

Reliable math. LLMs are bad at arithmetic; Python isn't.

```
User: What's 23498 * 8722 - (234^5)?
Agent: I'll compute this.
    Action: execute_python
    Code: print(23498 * 8722 - 234**5)
    Observation: -719516625060
Agent: The answer is -719,516,625,060.
```

### Web / API interactions

Agent writes code to:
- `requests.get(url)` to fetch data.
- Call APIs (e.g., OpenAI, Google Maps, internal services).
- Scrape pages.

### Code debugging

Cursor, Aider, and similar: agent writes test, runs, sees failure, fixes, runs again.

### Simulations

Monte Carlo, agent-based models, physics sims - anything expressible as Python.

## Sandbox options

You must NOT run LLM-generated code on your production machine. Sandbox it.

### Level 1: subprocess with timeout

```python
subprocess.run(..., timeout=30)
```

**Problems**: can still touch filesystem, make network calls, exfiltrate data.

Okay for trusted internal use. Not okay for public-facing.

### Level 2: Docker container

Run each code execution in a fresh ephemeral container:

```python
subprocess.run([
    "docker", "run", "--rm", "--network=none", "--memory=512m",
    "python:3.11", "python", "-c", code
], timeout=30)
```

`--network=none` disables network. `--rm` cleans up. `--memory` limits RAM.

Still vulnerable to container escapes (rare but real).

### Level 3: gVisor / Firecracker

Serious isolation. gVisor (Google) intercepts syscalls. Firecracker (AWS Lambda's backend) runs tiny VMs per request, spinning up in ~125ms.

### Level 4: hosted sandbox services

Don't build this yourself:
- **E2B** (e2b.dev): SDK for starting sandboxed containers. Agent-friendly.
- **Modal** (modal.com): serverless Python with fast cold starts.
- **Daytona**: dev environments as a service, also works for agents.
- **Replit**: hosted code execution.

For production agents, use one of these. They've solved the security you haven't.

### nanochat's execution.py

Look at `~/workspace/nanochat/nanochat/execution.py`. ~200 lines. Runs model-generated code for the HumanEval benchmark. Uses subprocess + timeout. No strong sandbox - because it's for eval, not public deployment.

Useful to study the structure:
- Parse code from model output.
- Spawn subprocess with timeout.
- Capture stdout/stderr.
- Format result as structured output.

## Making code execution agents reliable

### Iteration

Agents should retry on failure. The `(code → error → fix)` loop is where they shine.

```python
attempts = 0
while attempts < 5:
    result = execute_python(code)
    if "Error" not in result:
        break
    # LLM sees the error and writes new code
    attempts += 1
```

### Context preservation

State (variables, imports) doesn't persist across subprocess calls. Options:
- Use a long-running Jupyter kernel (persistent state).
- Feed previous code + new code to each call.
- Use IPython via a persistent subprocess.

Production systems (Code Interpreter, Claude analysis) run a Jupyter kernel per session.

### Error interpretation

Help the LLM debug. Format errors clearly:

```
File "<code>", line 3
   df = pd.read_csv('missing.csv')
FileNotFoundError: [Errno 2] No such file or directory
```

Don't truncate stack traces. The LLM reads them.

### Preinstalled libraries

Sandbox should have common libs: pandas, numpy, matplotlib, requests, sklearn. Otherwise the LLM keeps trying `pip install` which might not work.

### File I/O

If user uploads files, put them in a known directory. Tell the LLM ("files are in /data/"). 

### Plotting

Matplotlib figures: save as PNG, return a file path or base64. Display to user.

## Concrete: minimal data analyst agent

```python
import json
from openai import OpenAI
import subprocess, base64, os

client = OpenAI()

def execute(code):
    # NOTE: in production, sandbox this!
    result = subprocess.run(
        ["python", "-c", code],
        capture_output=True, text=True, timeout=30,
        cwd="/tmp/agent_workdir"
    )
    return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

tools = [{
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": "Run Python. State does NOT persist between calls. Files in /tmp/agent_workdir.",
        "parameters": {"type":"object","properties":{"code":{"type":"string"}},"required":["code"]},
    }
}]

os.makedirs("/tmp/agent_workdir", exist_ok=True)
# assume user put data.csv there

messages = [
    {"role": "system", "content": "You are a data analyst. User has data.csv in working directory."},
    {"role": "user", "content": "Summarize data.csv. What's the most interesting trend?"},
]

for _ in range(10):
    r = client.chat.completions.create(model="gpt-4o", messages=messages, tools=tools)
    m = r.choices[0].message
    messages.append(m.model_dump())
    if not m.tool_calls:
        print(m.content); break
    for tc in m.tool_calls:
        out = execute(json.loads(tc.function.arguments)["code"])
        messages.append({"role":"tool","tool_call_id":tc.id,"content":out[:4000]})
```

70-80 lines for a data analyst agent. With E2B sandbox, it's production-safe.

## Evaluation

Benchmarks:
- **HumanEval** (code): does generated code pass tests? nanochat evaluates this.
- **MBPP**: simpler Python problems.
- **SWE-Bench**: real GitHub issues. Much harder.
- **DS-1000**: data-science specific.

## Commercial code-execution agents

- **OpenAI Code Interpreter** (part of ChatGPT Plus): Python + Jupyter-style UX. Great for non-technical users.
- **Claude Analysis Tool**: similar, built into Claude.
- **Cursor / Aider / Continue / Windsurf**: AI code editors with execution + testing.
- **Devin** (Cognition): autonomous software agent.
- **Replit Agent**: end-to-end app building.
- **GitHub Copilot Workspace**: PR-level code agent.

Every one uses the same core pattern. Differentiation is in UX, sandbox quality, and reliability.

## Exercises

1. Build the minimal data analyst agent above. Give it a CSV from Kaggle. Try various questions.

2. Add iteration: when code fails, the agent sees the error and retries up to 5 times.

3. Replace subprocess with E2B (~10 line swap). Now it's sandboxed.

4. Add matplotlib plot-saving: agent writes figures to `/tmp/agent_workdir/figs/`, you display them.

5. Read `nanochat/execution.py`. Identify the timeout + capture_output pattern.

## Next

`05_web_browsing_agents.md` - the next frontier.
