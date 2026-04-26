# 10.5 - Web Browsing Agents

An LLM that navigates a browser - clicking, typing, scrolling, filling forms. This is the frontier of 2024-2026 agent research. Impressive and still unreliable.

## The vision

User: "Book me a flight from SFO to NYC on Tuesday, aisle seat, under $400."

Agent: opens a browser, searches Google Flights, applies filters, clicks through to the airline, fills the form, stops before payment for confirmation.

This is what OpenAI Operator, Anthropic Computer Use, and Google Project Mariner aim for.

## The two architectural approaches

### Approach A: DOM-based

The agent sees the HTML of the page. Picks an element. Clicks or types.

**Tools**:
- `get_page()` → returns simplified DOM (or accessibility tree).
- `click(selector)` → clicks a CSS selector or element index.
- `type(selector, text)` → types into an input.
- `navigate(url)` → goes to URL.

**Advantages**:
- Deterministic (same DOM → same action).
- Can use structured elements (form labels, aria attributes).
- Compact representation.

**Disadvantages**:
- Modern SPAs (React, Vue, Angular) have ugly DOM. Deep nesting, dynamic IDs.
- Visual-only elements (images, icons) not represented in text.

Examples: Selenium, Playwright, browser-use (Python lib).

### Approach B: Vision-based

The agent sees a screenshot. Predicts (x, y) to click or types text.

**Tools**:
- `screenshot()` → returns image.
- `click(x, y)` → clicks coordinates.
- `type(text)` → types at current focus.
- `scroll(amount)`.

**Advantages**:
- Works on any UI (including desktop apps, games).
- Handles visual-only elements.
- Matches how humans use computers.

**Disadvantages**:
- Slow (image understanding is expensive per call).
- Precise coordinates can be fragile across window sizes.
- Requires multimodal LLM.

Examples: Anthropic Computer Use, OpenAI CUA (Computer-Using Agent), Adept ACT-1.

### Hybrid (most practical)

Use DOM when structured, screenshots for visual understanding:
- DOM for form fields, links, buttons with text labels.
- Screenshots for graphs, images, ambiguous UI.

## Tools and frameworks

### `browser-use` (Python, open source)

```python
from browser_use import Agent
from langchain_openai import ChatOpenAI

agent = Agent(
    task="Go to Hacker News, find the top story, summarize it.",
    llm=ChatOpenAI(model="gpt-4o"),
)
await agent.run()
```

Works out of the box. DOM-based. Good starting point.

### Playwright (Python/JS)

Lower-level, programmable browser automation. You build the loop yourself:

```python
from playwright.async_api import async_playwright
async with async_playwright() as p:
    browser = await p.chromium.launch()
    page = await browser.new_page()
    await page.goto("https://example.com")
    await page.fill("input[name=q]", "hello")
    await page.click("button[type=submit]")
```

Combine with an LLM for decisions. Most production browser agents use Playwright underneath.

### Anthropic Computer Use (API)

Launched Oct 2024. Claude can emit screenshot and click actions as tool calls. You provide a VM (or Docker), Claude controls it.

Demo: asks Claude to complete multi-step tasks in a browser. Works, sometimes.

### OpenAI Operator (product)

Launched Jan 2025 (ChatGPT Pro feature). Fully managed - OpenAI runs the browser, you describe tasks. Stable for e-commerce, less so for complex flows.

### Adept (defunct 2024)

Pioneered web agents. Products absorbed by Amazon.

### Project Mariner (Google)

Announced Dec 2024. Chrome extension. Browser-integrated agent.

## What breaks

Common failure modes:

### 1. Dynamic content / AJAX
Page loads, then content appears 2 seconds later via JS. Agent clicks too early. Solution: wait conditions (`wait_for_selector`).

### 2. Captchas
Every serious site has them. LLMs can't reliably solve captchas (deliberately - it's a cat-and-mouse).

### 3. Authentication
Complex OAuth flows, 2FA, cookie popups, modal dialogs. Agents fumble.

### 4. Dropdowns and autocompletes
Typing "new yo" triggers a dropdown. Agent has to select the right suggestion, not just hit enter.

### 5. Rate limits and bot detection
Sites block automated traffic. Cloudflare, Datadome, etc. detect you.

### 6. Layout changes
Page updates, selectors break.

### 7. Long-term tasks
Multi-page checkouts. Agent forgets what it was doing 3 pages ago.

Modern models (GPT-4o, Claude 3.5) handle basic flows. Complex flows (insurance claims, flight bookings with preferences) still fail 50%+ of the time.

## Benchmarks

- **WebArena** (Princeton 2023): 812 real-world tasks across 5 websites. Leading agents get 20-40% success.
- **Mind2Web**: diverse browsing tasks, large scale.
- **AgentBench**: general agent tasks including browsing.
- **WebVoyager**: real websites, multimodal.

A 60% success rate is impressive today. Humans are ~90%+. Gap = research opportunity.

## Production considerations

### Idempotency
Agent clicks submit twice. Order placed twice. Disaster.

Solution: confirmation UI for any state-changing action. Always have a "dry run" mode.

### Recovery
Agent crashes mid-task. Can it resume? Usually no. State management is hard.

### Logging
Log everything: DOM snapshots, screenshots, model responses, tool calls. For debugging and compliance.

### Authentication
Never let the LLM see your password. Use password managers, session cookies passed by the user, or OAuth tokens. Consider running the agent in a browser that already has sessions.

### Legal / TOS
Many sites prohibit automated access. Check ToS. Don't build agents for things that violate them.

## A minimal browser agent

```python
from playwright.async_api import async_playwright
from openai import OpenAI
import asyncio, json

client = OpenAI()

async def run_task(task):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        tools = [
            {"type":"function","function":{"name":"navigate","description":"Go to a URL.",
                "parameters":{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}}},
            {"type":"function","function":{"name":"click","description":"Click an element by CSS selector.",
                "parameters":{"type":"object","properties":{"selector":{"type":"string"}},"required":["selector"]}}},
            {"type":"function","function":{"name":"type_text","description":"Type text into an input.",
                "parameters":{"type":"object","properties":{"selector":{"type":"string"},"text":{"type":"string"}},"required":["selector","text"]}}},
            {"type":"function","function":{"name":"get_content","description":"Get visible text of page.",
                "parameters":{"type":"object","properties":{}}}},
        ]

        messages = [
            {"role":"system","content":"You control a browser. Use tools to complete the task."},
            {"role":"user","content":task}
        ]

        for _ in range(20):
            r = client.chat.completions.create(model="gpt-4o", messages=messages, tools=tools)
            m = r.choices[0].message
            messages.append(m.model_dump())
            if not m.tool_calls:
                print("Done:", m.content); break

            for tc in m.tool_calls:
                args = json.loads(tc.function.arguments)
                name = tc.function.name
                try:
                    if name == "navigate":
                        await page.goto(args["url"]); out = f"At {args['url']}"
                    elif name == "click":
                        await page.click(args["selector"]); out = "Clicked"
                    elif name == "type_text":
                        await page.fill(args["selector"], args["text"]); out = "Typed"
                    elif name == "get_content":
                        out = (await page.inner_text("body"))[:3000]
                except Exception as e:
                    out = f"Error: {e}"
                messages.append({"role":"tool","tool_call_id":tc.id,"content":out})

        await browser.close()

asyncio.run(run_task("Go to news.ycombinator.com and tell me the top 3 stories."))
```

~60 lines. Works on simple sites. Fragile on complex.

## Exercises

1. Run the snippet above. Try different tasks: Wikipedia lookups, Google searches, weather checks.

2. Install `browser-use` (Python). Compare to hand-rolled version.

3. Try Anthropic Computer Use demo (Oct 2024). Note failures.

4. Read the WebArena paper. Look at the tasks. What's hard about them?

5. Think: what's the single blocker for web agents to be 10x more reliable? (My bet: better planning / error recovery, not better vision.)

## Next

`06_agent_frameworks.md` - survey of the frameworks you might use.
