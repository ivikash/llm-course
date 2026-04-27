# 10.9 - Memory and Statefulness

An LLM has no memory. Each API call starts fresh. If your agent is going to act over time - hours, days, months - it needs memory. This lesson covers the approaches.

## The memory levels

### Level 1: In-context (short-term)

The conversation history in the current request. Trivial.

```python
messages = [
    {"role": "user", "content": "I'm Vikas."},
    {"role": "assistant", "content": "Hi Vikas!"},
    {"role": "user", "content": "What's my name?"},    # will answer "Vikas"
]
```

**Limits**:
- Context budget (128K-1M tokens max in 2026).
- Cost grows with context length.
- "Lost in the middle": models under-attend to middle content.

Works for single-session chats. Fails for long-running agents.

### Level 2: Summarization

As context grows, periodically summarize older turns and compress.

```
[Turn 1-50] → summary: "User is Vikas, a developer learning about ML. Discussed transformers..."
[Turn 51-60] → full text
```

The LLM or a smaller model does the summarization.

**Pros**: simple, preserves essential info.
**Cons**: summarization loses detail; biases accumulate.

### Level 3: Retrieval-based memory (vector store)

Treat each past exchange as a document. At query time, retrieve relevant ones.

```python
# store every turn as an embedding
for turn in past_turns:
    vector_db.add(embed(turn.content), metadata=turn)

# at query time
relevant = vector_db.search(embed(current_query), k=5)
context = [r.content for r in relevant] + current_turn
```

**Pros**: scales to arbitrary history.
**Cons**: noisy retrieval, context assembly gets complex.

Basically RAG over the agent's own history. Most "long-term memory" in agents works this way.

### Level 4: Structured memory

Extract specific facts and store them in a database.

```
User profile:
- name: Vikas
- occupation: software engineer
- prefers_language: English
- projects_in_progress: ["LLM course", ...]
- last_interaction: 2026-04-26
```

**Pros**: reliable recall, queryable, editable.
**Cons**: requires extraction logic, schema design, drift over time.

### Level 5: Hierarchical / OS-like (MemGPT, Letta)

Model has different "memory levels":
- **Core** (always in context): persistent user profile, key facts.
- **Recall** (retrievable): episodic memories.
- **Archive** (offloaded): old stuff, can be paged back.

The model can self-manage what lives where via tool calls: `read_memory`, `write_memory`, `archive`.

MemGPT paper (Packer 2023): https://arxiv.org/abs/2310.08560. Now the startup Letta: https://letta.com

### Level 6: Graph / knowledge-base

Like Level 4 but structured as a knowledge graph. Entities + relationships.

For agents that build up complex understanding (personal assistants, research agents), this can be powerful but high-engineering.

## What to remember

**Don't try to remember everything.** Pick:

### Must remember
- User identity (who they are, what they want).
- Explicit preferences ("I prefer metric units").
- Commitments made ("You agreed to X by Thursday").
- Ongoing projects / context.

### Should remember
- Past questions (for continuity).
- Documents / files referenced.
- Decisions made.

### Don't bother
- Small talk.
- Most chitchat.
- Task-specific state that's irrelevant next session.

## The extraction problem

For structured memory, you need to decide what to extract.

Simple approach: after each conversation, ask the LLM to extract facts:

```python
extraction_prompt = f"""Extract user facts from this conversation:
{conversation}

Output JSON: {{"facts": ["...", "..."]}}"""
```

Facts are appended to the user's memory record.

More sophisticated:
- **Incremental updates**: "Update existing profile to reflect new info."
- **Deduplication**: "Is this a new fact, or restatement of existing?"
- **Contradiction resolution**: "This contradicts an old fact. Which is correct?"

All of this is prompted to an LLM at session end.

## Forgetting

Memory grows indefinitely. Eventually:
- Old info becomes irrelevant.
- Retrieval quality degrades.
- Storage / compute costs grow.

Strategies:
- **Decay**: recent memories weighted higher in retrieval.
- **Periodic consolidation**: LLM compresses old memories into summaries.
- **Explicit expiration**: "remember this for one week."
- **User control**: "Forget X, Y, Z."

## A minimal memory-augmented agent

```python
import json, sqlite3, openai
client = openai.OpenAI()

# SQLite-backed memory
conn = sqlite3.connect("memory.db")
conn.execute("CREATE TABLE IF NOT EXISTS facts (user_id, fact, created_at)")

def get_facts(user_id):
    return [row[1] for row in conn.execute(
        "SELECT * FROM facts WHERE user_id=?", (user_id,)
    ).fetchall()]

def add_fact(user_id, fact):
    conn.execute("INSERT INTO facts VALUES (?, ?, datetime('now'))", (user_id, fact))
    conn.commit()

def chat(user_id, user_msg):
    facts = get_facts(user_id)
    facts_str = "\n".join(f"- {f}" for f in facts) or "(none yet)"
    system = f"You are an assistant. Known facts about this user:\n{facts_str}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg}
        ],
    )
    reply = response.choices[0].message.content

    # Extract new facts
    extract = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content":
            f"From this user message, extract durable facts about them (if any). "
            f"Reply JSON: {{\"facts\": [...]}}\n\nMessage: {user_msg}"}],
        response_format={"type": "json_object"},
    )
    new_facts = json.loads(extract.choices[0].message.content).get("facts", [])
    for f in new_facts:
        add_fact(user_id, f)

    return reply

# Test
print(chat("user1", "Hi, I'm Vikas."))
print(chat("user1", "I love Thai food."))
print(chat("user1", "What do you know about me?"))   # Should mention both
```

60 lines. A real memory-augmented agent. Facts persist across sessions because SQLite does.

## Letta / MemGPT in practice

For richer memory without DIY:

```python
from letta_client import Letta
letta = Letta(base_url="http://localhost:8283")
agent = letta.create_agent(name="assistant", model="gpt-4o-mini")
response = letta.send_message(agent_id=agent.id, message="Hi, I'm Vikas.")
```

Letta manages core/recall/archive memory automatically. Useful for sophisticated personal assistants.

## Challenges

### 1. Preventing hallucinated memories
Agent says "Remember when we talked about X?" when X never happened. Extraction discipline and verification help.

### 2. Stale memories
User updates profile; old memory contradicts. Needs update / reconciliation logic.

### 3. Privacy
Memories contain sensitive info. Encrypt at rest. User-deletable. GDPR-compliant.

### 4. Multi-user memories
If memory is accidentally shared across users, privacy disaster. Strict user_id partitioning.

### 5. Retrieval precision
Semantic retrieval returns "related" memories, not always "right" ones. Filter carefully.

## Memory in production chatbots

- **ChatGPT Memory** (April 2024): OpenAI introduced persistent memory. Users can see/edit.
- **Claude Projects**: scoped memory per project.
- **character.ai**: persistent character memory.

Production systems increasingly use a mix: explicit user profile + retrieval over past chats.

## Key reading

- **MemGPT** (Packer 2023): https://arxiv.org/abs/2310.08560
- **Letta docs**: https://docs.letta.com
- **Generative Agents** (Park 2023) uses a memory stream model.
- **Mem0** (OSS memory layer): https://github.com/mem0ai/mem0

## Visualize this

**The memory hierarchy**:

```
  Level 1: in-context (short-term)
  ─────────────────────────────────
  ┌───────────────────────────┐
  │  Current conversation      │  limited by model context (128K-1M)
  │  "Hi" "Hello" "2+2" "4"    │  disappears when session ends
  └───────────────────────────┘

  Level 2: summarization
  ───────────────────────
  ┌───────────────────────────┐
  │  Summary of old turns       │   compressed older context
  │  "User is Vikas, developer  │
  │   learning ML..."           │
  └───────────────────────────┘

  Level 3: retrieval (RAG-style)
  ───────────────────────────────
  ┌───────────────────────────┐
  │  Every turn stored as       │  retrieve when relevant
  │  embedding in vector DB    │
  └───────────────────────────┘

  Level 4: structured facts
  ──────────────────────────
  ┌───────────────────────────┐
  │  User: Vikas                │
  │  Job: software eng           │
  │  Prefers: English             │
  │  Projects: [LLM course, ...]  │
  └───────────────────────────┘

  Level 5: OS-like (MemGPT / Letta)
  ──────────────────────────────────
  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
  │ Core (hot)  │  │ Recall       │  │ Archive     │
  │ always in   │  │ (retrieve)   │  │ (offloaded) │
  │ context     │  │              │  │              │
  └─────────────┘  └─────────────┘  └─────────────┘

  Level 6: graph / knowledge base
  ────────────────────────────────
       Vikas ─── works_at ─── [company]
         │
         │── learning ───── LLMs
         │
         │── uses ───────── nanoGPT ─── forked ─── [repo]
```

Each level adds capability at complexity cost.

**In-context (simplest) pros and cons**:

```
  Model sees full conversation in every call.

  Pros:
    ✓ Simple, no infrastructure needed
    ✓ Perfect recall within context
    ✓ Works out of the box

  Cons:
    ✗ Context size limits (even 1M tokens eventually runs out)
    ✗ Cost grows linearly with context length
    ✗ "Lost in the middle": models attend poorly to middle content
    ✗ Disappears when session ends
```

**Retrieval-based memory flow**:

```
  Session start: user says "Hello!"
      │
      ▼
  Embed user message → store with metadata
      │
      ▼
  Session continues. User asks: "Did I tell you about my project?"
      │
      ▼
  Embed query. Search memory.
      │
      ▼
  Retrieved: "User's project: LLM course, ML learning, building nanoGPT-style..."
      │
      ▼
  Inject into current context:
      System: You know this about the user: ...
      User: Did I tell you about my project?
      │
      ▼
  LLM answers with continuity.
```

**Structured memory (best for products)**:

```
  Schema:
    user_id: INTEGER
    name: STRING
    preferences: JSON
    facts: LIST[STRING]
    last_seen: TIMESTAMP

  Extraction (after every conversation):
    LLM reads conversation → extracts facts as JSON
    Example output:
      {"facts": ["Vikas is learning Python-to-LLM.",
                 "Vikas prefers Markdown notes.",
                 "Vikas has access to nanoGPT and nanochat."]}

  Storage:
    INSERT INTO facts VALUES ('vikas', 'is learning...', NOW())

  Retrieval (before every conversation):
    SELECT fact FROM facts WHERE user_id = 'vikas'
    → prepend to system prompt

  Pros:
    ✓ Editable (user can see / correct / delete facts)
    ✓ Efficient (small, tabular queries)
    ✓ Debuggable (you can inspect what's remembered)
```

**60-line memory-augmented agent**:

```python
import sqlite3, json, openai

client = openai.OpenAI()
conn = sqlite3.connect("memory.db")
conn.execute("CREATE TABLE IF NOT EXISTS facts (user_id, fact, created_at)")

def get_facts(uid):
    return [r[1] for r in conn.execute(
        "SELECT * FROM facts WHERE user_id=?", (uid,)
    ).fetchall()]

def add_fact(uid, fact):
    conn.execute("INSERT INTO facts VALUES (?, ?, datetime('now'))", (uid, fact))
    conn.commit()

def chat(uid, msg):
    facts = get_facts(uid)
    facts_str = "\n".join(f"- {f}" for f in facts) or "(none yet)"
    system = f"You are an assistant. Known facts about this user:\n{facts_str}"

    reply = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":system},
                  {"role":"user","content":msg}]
    ).choices[0].message.content

    # Extract new facts
    extract = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user",
                   "content":f"Extract durable facts about this user from their message."
                             f" Reply JSON: {{\"facts\":[...]}}\n\nMessage: {msg}"}],
        response_format={"type":"json_object"},
    ).choices[0].message.content

    for fact in json.loads(extract).get("facts", []):
        add_fact(uid, fact)

    return reply

# Test multi-session:
print(chat("user1", "Hi, I'm Vikas."))
print(chat("user1", "I love Thai food."))
print(chat("user1", "What do you know about me?"))
# → should mention name and food preference
```

Real memory across sessions, persisted in SQLite. ~60 lines.

**The forgetting problem**:

```
  Memory grows indefinitely.
  After 6 months:
    - 10,000+ facts stored
    - Many outdated ("last_seen: 2024-01-01")
    - Many redundant ("user likes Thai" × 10)
    - Retrieval quality degrades (too much noise)

  Strategies:

  1. Decay:
     Score facts by recency; older ones weight less in retrieval.

  2. Consolidation:
     Periodically: LLM summarizes many facts into fewer.
     "User has mentioned Thai food 10 times" → "User loves Thai food"

  3. Explicit expiration:
     Mark facts "transient" vs "permanent".
     Delete transients after 30 days.

  4. User control:
     Let user see/edit/delete facts they don't want remembered.
     Essential for trust.
```

**MemGPT / Letta: OS-like memory**:

```
  Memory is divided into regions:

  Core memory (always in context):
    {user_name: "Vikas", pronouns: "he/him",
     current_goal: "learning ML"}
    → 1-2KB, stays in every LLM call.

  Recall memory (retrievable):
    past conversation turns
    stored in vector DB
    retrieved when topically relevant

  Archival memory (offloaded):
    old conversations, large documents
    paged into recall when needed

  Tools the agent gets:
    - read_memory(query)       ← search
    - insert_memory(content)    ← add
    - delete_memory(id)         ← remove
    - modify_core(key, value)   ← update pinned memory

  Agent manages its own memory via tool calls.
  Mimics human "I'll remember this" behavior.
```

**ChatGPT's memory feature (April 2024+)**:

```
  User says:
    "Remember that I prefer concise responses."
  ChatGPT says:
    "Got it. I'll keep responses concise."

  Behind the scenes:
    → OpenAI stores: "User prefers concise responses."
    → Added to system prompt in future conversations.
    → User can view/edit in Settings → Personalization → Memory.

  This is basically the structured memory pattern,
  in a production UX.
```

**Privacy considerations**:

```
  Memories often contain sensitive info:
    - User's name, location, relationships
    - Health conditions, political views
    - Work secrets, financial data

  Must-dos:
    ✓ Encrypt at rest
    ✓ User-visible: show what's remembered
    ✓ User-deletable: right to be forgotten
    ✓ Multi-user isolation (no cross-user leaks!)
    ✓ Expiration policies
    ✓ Compliance (GDPR, etc.)

  Real-world failures:
    "AI assistant remembered user A's password.
     User B accidentally gets it in response."
  → always partition by user_id strictly.
```

**The memory-capability tradeoff**:

```
  More memory = more personalized = better UX
    ↕ (but)
  More memory = more context = more confused = more expensive

  Sweet spot:
    Keep 10-50 "important" facts in core.
    Use retrieval for long history.
    Periodically consolidate.

  Don't try to remember everything.
  Remember what matters for the user's goals.
```

**Real products using memory**:

```
  ChatGPT Memory (April 2024):
    User-visible + editable fact list.

  Claude Projects:
    Per-project memory scope.

  character.ai:
    Persistent character personalities + user history.

  Letta (startup from MemGPT authors):
    Hosted stateful agents as a service.

  Mem0 (open source):
    Memory layer you can drop into any LLM app.
```

**Common failures**:

```
  1. Memories contradicting:
     User said "I love cats" in January.
     User said "I hate cats" in April (got allergies).
     Agent pulls old fact. Says something weird.
     Fix: timestamp facts; prefer newer; detect contradictions.

  2. Hallucinated memories:
     Agent says "Remember when we talked about X?"
     You never did.
     Fix: strict extraction; verify facts are in the source.

  3. Privacy leaks:
     Multi-user system; bug allows user A to see user B's memory.
     Fix: test isolation rigorously.

  4. Stale memories:
     Facts about your life 3 years ago, still cited.
     Fix: expire / decay old facts.
```

## Exercises

1. Extend the 60-line snippet: instead of SQLite, use a vector DB. Embed facts. At query time, retrieve relevant facts, not all.

2. Add deduplication: don't store "user is named Vikas" twice.

3. Add forgetting: delete facts older than 30 days unless marked important.

4. Try Letta locally (Docker compose). Build a chatbot that remembers you across restarts.

5. Think: what's the hardest part of memory? (My vote: contradiction resolution.)

## Next

`10_agent_evaluation.md` - measuring agent quality.
