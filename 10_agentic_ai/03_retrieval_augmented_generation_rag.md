# 10.3 - RAG: Retrieval-Augmented Generation

The single most important agent-adjacent pattern. RAG lets an LLM answer questions using information that wasn't in its training data - your company's docs, recent news, a codebase, a personal knowledge base.

## The problem

LLMs have two limitations:
1. **Knowledge cutoff**: trained on data up to some date. Newer facts are invisible.
2. **No access to your private data**: company wikis, personal files, customer records.

You *could* fine-tune a model on your data every time it changes. Expensive and error-prone.

**RAG is the cheaper, more flexible alternative**: at query time, find relevant documents and prepend them to the prompt. The LLM reasons over the retrieved context.

## The core flow

```
User query: "What's our refund policy for Pro subscribers?"
  ↓
1. Embed the query → vector
  ↓
2. Search vector DB → top-k relevant document chunks
  ↓
3. Prepend chunks to the prompt:
   "Context: [chunk 1] [chunk 2] [chunk 3]
    Question: What's our refund policy for Pro subscribers?"
  ↓
4. LLM generates answer grounded in the context
  ↓
Response: "Pro subscribers can request a refund within 30 days..."
```

## Components

### 1. Documents

Your knowledge base. PDFs, HTML pages, Markdown, database rows, anything with text.

Step 1: extract text from each source. Tools:
- **Unstructured** (Python lib, `unstructured.io`): handles PDFs, DOCX, HTML, etc.
- **LlamaParse**: better for complex PDFs.
- **BeautifulSoup** / **trafilatura** for HTML.

### 2. Chunking

Split each document into chunks. Why?
- Embedding models have input limits (512 or 8192 tokens).
- Smaller chunks = more focused matches in retrieval.

**Strategies**:
- **Fixed size**: 500 tokens with 50-token overlap. Simple, often fine.
- **Sentence / paragraph boundaries**: preserve semantic units.
- **Markdown structure**: split by headings.
- **Semantic chunking**: use embeddings to find "topic shifts". More complex.

Rule of thumb: 200-500 tokens per chunk, 10-20% overlap. Start simple.

### 3. Embeddings

Convert each chunk to a vector so you can compute similarity.

Popular embedding models (2024-2026):

| Model | Dims | Notes |
|-------|------|-------|
| OpenAI `text-embedding-3-small` | 1536 | Cheap, fast, strong. Good default. |
| OpenAI `text-embedding-3-large` | 3072 | Best OpenAI. ~$0.13/1M tokens. |
| Cohere `embed-v3` | 1024 | Strong multilingual. |
| Voyage AI `voyage-3` | 1024 | High quality, specialized variants. |
| BAAI `bge-large-en-v1.5` | 1024 | Open, free, run locally. |
| BAAI `bge-m3` | 1024 | Multilingual, dense + sparse. |

For most uses: OpenAI `text-embedding-3-small` is hard to beat for cost.

```python
from openai import OpenAI
client = OpenAI()
resp = client.embeddings.create(
    input=["your text here"], model="text-embedding-3-small"
)
embedding = resp.data[0].embedding   # list of 1536 floats
```

### 4. Vector database

Stores embeddings + metadata, supports fast nearest-neighbor search.

| DB | Type | Notes |
|----|------|-------|
| **FAISS** | Library (Meta) | Local, free, fast. No persistence. |
| **Chroma** | Library | Local, persistent, simple API. Good for starting. |
| **Qdrant** | Self-hosted or cloud | Open source, strong filtering. |
| **Weaviate** | Self-hosted or cloud | Full-featured. |
| **Pinecone** | Cloud only | Managed, scales well. Paid. |
| **pgvector** | Postgres extension | Great if you already use Postgres. |
| **LanceDB** | Embedded | Arrow-native, modern. |

For learning: Chroma or FAISS. For production: Qdrant, pgvector, or Pinecone.

Minimal FAISS example:

```python
import faiss
import numpy as np

embeddings = np.array([...])   # (N, 1536)
index = faiss.IndexFlatL2(1536)
index.add(embeddings)

query_emb = np.array([...])    # (1, 1536)
distances, indices = index.search(query_emb, k=5)
# indices[0] is the k nearest document indices
```

### 5. Retrieval

Given query, return top-k relevant chunks:

```python
query_embedding = embed(query)
distances, indices = index.search(query_embedding, k=5)
retrieved_chunks = [documents[i] for i in indices[0]]
```

### 6. Prompt assembly

```python
context = "\n\n".join([f"[{i}] {chunk}" for i, chunk in enumerate(retrieved_chunks)])

prompt = f"""Answer using only the context below. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {query}
Answer:"""
```

That's the RAG prompt. Send to LLM. Get answer.

## Advanced RAG techniques

### Hybrid search (vector + keyword)

Vector search catches semantic matches ("refund" ≈ "money back"). Keyword search (BM25) catches exact terms ("SKU-12345").

Combine both, merge scores:

```python
vector_results = vector_db.search(query_embedding, k=20)
keyword_results = bm25_search(query, k=20)
merged = reciprocal_rank_fusion(vector_results, keyword_results)
top_k = merged[:5]
```

Significant quality improvement for most use cases.

### Reranking

Top-k retrieval is noisy. Use a **cross-encoder** to rerank: it takes (query, chunk) and scores how relevant. Slow (can't precompute like embeddings) but precise.

```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('BAAI/bge-reranker-large')
scores = reranker.predict([(query, chunk) for chunk in candidates])
sorted_chunks = [c for _, c in sorted(zip(scores, candidates), reverse=True)][:5]
```

Pipeline: retrieve 50 with vector search → rerank to top 5 → prompt.

Cohere Rerank API is the commercial counterpart. Widely used.

### Query rewriting

User query is often suboptimal for retrieval. Let the LLM rewrite first:

```python
expanded_query = llm("Rewrite this query for retrieval: " + original_query)
retrieved = retrieve(expanded_query)
```

Or HyDE (Hypothetical Document Embeddings): LLM drafts a fake answer; you embed the fake answer and search for real similar docs. Often better than raw query embedding.

### Multi-hop retrieval

For complex questions: retrieve, use first results to form a follow-up query, retrieve again.

```
Q: Which product did we release in Q4 2023?
First retrieval: company timeline → product X released 2023-11-15
Second query: "product X features"
Second retrieval: product X spec → features
Final answer uses both.
```

Chaining adds latency but answers more complex questions.

### Contextual retrieval (Anthropic, 2024)

Before embedding each chunk, use an LLM to add *context*:

```
Original chunk: "The refund is 30 days."
With context: "This chunk is from the Pro subscription refund policy. Original: The refund is 30 days."
```

Embed the contextualized version. Significantly better retrieval for documents where chunks reference entities not mentioned in the chunk itself.

Reference: https://www.anthropic.com/news/contextual-retrieval

## When RAG works well

- Well-defined knowledge base (documentation, policies, product catalog).
- Questions that can be answered from a single chunk or small set.
- Up-to-date responses needed.
- Limited need for deep reasoning across many sources.

## When RAG fails

- **Synthesis across many documents**: "What have all our customers complained about?" needs aggregation, not retrieval.
- **Numerical / tabular data**: RAG over Excel sheets often fails. Use SQL agents instead.
- **Complex reasoning**: the retrieved chunks may not contain enough for multi-step deduction.
- **Time-sensitive**: if facts changed after your indexing, RAG is wrong.

## Evaluation

Measuring RAG quality is two-fold:

1. **Retrieval quality**: did we find the right docs?
   - Precision@k, Recall@k on labeled data.
2. **Generation quality**: did the LLM use the docs correctly?
   - Faithfulness: does the answer match the retrieved context?
   - Relevance: does it answer the question?

Tools: RAGAS, TruLens, Arize.

## The "long context" alternative

Modern LLMs (GPT-4 Turbo 128K, Gemini 1.5 1M+, Claude 3.5 200K) can hold entire books in context. So: instead of RAG, just paste everything?

Works for small corpora (<1M tokens). Breaks down for larger:
- Cost: $1M context per query = $10 at current rates.
- Latency: slow.
- "Lost in the middle": models attend poorly to middle of long contexts.

For large or frequently-updated corpora, RAG still wins. Hybrid approaches are emerging.

## Frameworks

- **LangChain** + **LlamaIndex**: the two biggies. Batteries-included but heavyweight.
- **Haystack**: another option, less popular.
- **Hand-rolled**: 100 lines for a basic RAG. Often preferred once you know the pattern.

## A minimal RAG in 60 lines

```python
import faiss, openai
from pathlib import Path

client = openai.OpenAI()

def embed(text):
    r = client.embeddings.create(input=text, model="text-embedding-3-small")
    return r.data[0].embedding

# 1. Load and chunk documents
docs = []
for p in Path("knowledge_base").glob("*.md"):
    text = p.read_text()
    # simple chunking: 500-char overlapping
    for i in range(0, len(text), 450):
        docs.append(text[i:i+500])

# 2. Embed
import numpy as np
embeddings = np.array([embed(d) for d in docs]).astype("float32")

# 3. Build index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# 4. Query
def answer(question):
    q_emb = np.array([embed(question)]).astype("float32")
    _, indices = index.search(q_emb, k=3)
    context = "\n\n".join(docs[i] for i in indices[0])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        }],
    )
    return response.choices[0].message.content

print(answer("How do I return a product?"))
```

60 lines, real RAG. You can expand with reranking, hybrid search, etc.

## Exercises

1. Build the 60-line RAG above. Use your own `.md` files.

2. Add BM25 hybrid search (use `rank-bm25` library). Measure quality gain.

3. Swap OpenAI embeddings for `bge-large-en-v1.5` from HuggingFace. Free.

4. Try contextual retrieval: rewrite each chunk with Claude to add context before embedding. Compare quality.

5. Read Anthropic's [Contextual Retrieval blog post](https://www.anthropic.com/news/contextual-retrieval).

## Next

`04_code_execution_agents.md` - one of the most powerful agent patterns.
