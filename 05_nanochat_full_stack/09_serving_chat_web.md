# 5.9 - Serving: chat_web.py

You have a trained model and a fast inference engine. Now let users talk to it. `scripts/chat_web.py` serves a ChatGPT-like web UI.

## Launch

```bash
python -m scripts.chat_web
```

Output:
```
Serving at http://0.0.0.0:8000
```

Open that URL in your browser. Type in the chat box. Watch tokens stream in.

## What's in the script

Open `~/workspace/nanochat/scripts/chat_web.py`. ~400 lines. Components:

1. **aiohttp server** - a tiny async Python web server.
2. **Static file route** - serves `nanochat/ui.html`.
3. **WebSocket endpoint** - `/ws` - for streaming chat.
4. **The engine** - loaded once at startup, handles all requests.
5. **Tokenizer** - for formatting conversations.

## The flow

```
Client opens browser
  ↓
Server sends ui.html
  ↓
Client JS connects to /ws WebSocket
  ↓
Client types message, sends JSON {role: user, content: "..."}
  ↓
Server: updates conversation history, formats with chat template, tokenizes
  ↓
Server: calls engine.generate_stream(tokens, ...)
  ↓
For each generated token:
  decoded = tokenizer.decode([token])
  send decoded over websocket
  ↓
Client appends incoming tokens to assistant message, displays in DOM
  ↓
When <|assistant_end|> is produced, server sends "done"
```

Under ~1 second latency from "user types" to "tokens start streaming back" on a local GPU.

## The engine integration

```python
# simplified from chat_web.py
engine = Engine(model=model, max_seq_len=2048, batch_size=1)

async def chat(ws, msg_list):
    tokens = tokenizer.encode_chat(msg_list + [{"role": "assistant", "content": ""}])
    engine.prefill(tokens)
    for _ in range(max_new_tokens):
        logits = engine.last_logits()
        next_tok = sample(logits, temperature=0.8)
        if next_tok == eot_token: break
        engine.decode_step(next_tok)
        await ws.send_json({"token": tokenizer.decode([next_tok])})
    await ws.send_json({"done": True})
```

(Actual code is slightly more involved: batching, interruption, error handling.)

## The UI (ui.html)

Open `~/workspace/nanochat/nanochat/ui.html`. ~500 lines of HTML/CSS/JS. No framework - pure DOM manipulation. Implements:

- Chat bubbles (user vs assistant styling)
- Input box with "send" button
- WebSocket connection to the server
- Markdown rendering (for code blocks, formatting)
- Token-level streaming (new text gets appended live)

A good example of "how simple a chat UI can be." Production chat apps (Claude.ai, ChatGPT) are more elaborate (history, context settings, image upload, voice, etc.) but the fundamental loop is the same.

## Production concerns (not in nanochat, but you should know)

1. **Auth**: anyone who knows your IP can chat. Add auth for real deployment.

2. **Rate limiting**: one user with a script could hammer the model. Limit requests per IP/user.

3. **Persistence**: nanochat doesn't save conversations. Production: store in a DB.

4. **Moderation**: toxic prompts, dangerous outputs. Add input/output filtering.

5. **HTTPS**: use nginx or caddy as a reverse proxy.

6. **Scalability**: one process handles one conversation at a time in nanochat. For many users, use a batching server (vLLM, TGI, SGLang).

7. **Monitoring**: log requests, track latency, track token counts for billing.

These are a whole other career's worth of skills. For now: nanochat teaches you the model + inference part, and you know that production adds the above layers.

## Alternative serving: chat_cli

If you don't want a web UI:

```bash
python -m scripts.chat_cli -p "What is 2+2?"
# or interactive:
python -m scripts.chat_cli
```

Much simpler: a Python REPL-style loop that uses the same Engine. Good for debugging and scripting.

## What you get from this lesson

- How a chat interface works from browser to model and back.
- How streaming generation is implemented.
- How the engine fits into a server.
- What's missing vs production (auth, batching, moderation, HTTPS).

The big insight: **a chat interface is simple**. The hard part is training the model. Once you have a good model, wrapping it in a UI is a weekend project.

## Visualize this

**The full request lifecycle**:

```
  User                    Browser            Server                 Model
  ────                    ───────            ──────                 ─────
  types "Hi"              display            listening
      │                      │                  │
      │ sends msg             │                  │
      ├──────────────────────▶│                  │
      │                       │ WebSocket        │
      │                       ├─────────────────▶│
      │                       │                  │
      │                       │                  │ tokenize
      │                       │                  │ append to convo
      │                       │                  │ engine.prefill(tokens)
      │                       │                  │      │
      │                       │                  │      ▼
      │                       │                  │    Model  (~50ms)
      │                       │                  │
      │                       │                  │ for token in generate():
      │                       │                  │   stream over WS
      │                       │                  ├────▶│
      │                       │◀──── token: "H" ──│    │
      │                       │                   │    │
      │ sees "H"               │ DOM update        │    │
      │                       │◀──── token: "i" ──│    │
      │                       │                   │    │
      │ sees "Hi"              │ DOM update        │    │
      │                       │◀──── token: "!"──│    │
      │ sees "Hi!"             │                   │    │
      │                       │◀──── token: EOT ──│    │
      │                       │                   │    │
      │                       │  "done" signal    │    │
      │                       │◀──────────────────│    │
      │ cursor idle             │                  │    │
```

Streaming tokens = responsive UX. User sees first letter in ~200ms.

**The server in 80 lines of Python**:

```python
# scripts/chat_web.py (simplified)
from aiohttp import web
from nanochat.engine import Engine
import nanochat.tokenizer as tok

engine = Engine.from_checkpoint("sft/latest")  # loads once at startup

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    conv = []
    async for msg in ws:
        data = msg.json()
        conv.append({"role": "user", "content": data["content"]})
        tokens = tok.render_chat(conv + [{"role": "assistant", "content": ""}])
        engine.prefill(tokens)
        assistant_text = ""
        for _ in range(800):
            token = engine.decode_step_and_sample(temperature=0.8)
            if token == tok.EOT_TOKEN: break
            text = tok.decode([token])
            assistant_text += text
            await ws.send_json({"token": text})
        conv.append({"role": "assistant", "content": assistant_text})
        await ws.send_json({"done": True})
    return ws

app = web.Application()
app.router.add_get("/ws", websocket_handler)
app.router.add_static("/", path=".")   # serves ui.html
web.run_app(app, port=8000)
```

That's a ChatGPT-like interface in ~80 lines. The hard part was training the model; serving is easy.

**What ui.html does** (client side):

```
  index.html
    │
    ▼
  <div id="messages">  ← chat bubbles rendered here
  <input>              ← user types here
    │
    ▼
  JavaScript:
    1. connect to /ws
    2. on submit: send {"content": inputText}
    3. on message("token"): append to current assistant bubble
    4. on message("done"): cursor-blink off, allow new input
```

~500 lines of plain HTML/CSS/JS. No React, no Vue, no build step. Open `nanochat/ui.html` and read it.

**Latency breakdown**:

```
  user hits Enter
        │
    ~10 ms → WebSocket message hits server
        │
    ~50 ms → tokenize conversation, render chat template
        │
    ~100 ms → engine.prefill (process full prompt)
        │
    ~30 ms → first token sampled, sent to browser
        │
    ~10 ms → browser renders letter
        │
  TOTAL: ~200 ms to first letter
        │
  then ~30 ms per subsequent token (decode)
```

Fast enough to feel interactive.

## Exercises

1. Train a small nanochat model (even on CPU, even tiny). Launch `chat_web`. Have a conversation.

2. Open `ui.html` in a text editor. Change the background color. Reload your browser. See the change. You're a web developer now.

3. Read `chat_web.py`'s request handler. Trace one user message through to the engine and back.

4. Find the streaming logic. Note that tokens are sent as they're generated, not batched at the end. Why? (Lower time-to-first-token is a UX win.)

5. Bonus: modify `chat_cli.py` to keep conversation history. Currently it might (or might not, depending on version) reset per message.

## Next

`10_speedrun_end_to_end.md` - run the whole speedrun.sh and narrate what happens.
