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

## Exercises

1. Extend the 60-line snippet: instead of SQLite, use a vector DB. Embed facts. At query time, retrieve relevant facts, not all.

2. Add deduplication: don't store "user is named Vikas" twice.

3. Add forgetting: delete facts older than 30 days unless marked important.

4. Try Letta locally (Docker compose). Build a chatbot that remembers you across restarts.

5. Think: what's the hardest part of memory? (My vote: contradiction resolution.)

## Next

`10_agent_evaluation.md` - measuring agent quality.
