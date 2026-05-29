# CLAUDE.md — SupportDoc Agent

This file gives you full context to help build the SupportDoc Agent project.
Read every section before writing any code.

---

## What This Project Is

SupportDoc Agent answers questions about Microsoft Intune support documentation
using an agentic RAG pipeline. It is NOT a simple retrieve-and-generate system.
The LLM makes routing and quality decisions at multiple points — it does not
blindly retrieve and generate every time.

**Author:** Manoj Kumar
**GitHub:** github.com/mojomahendia
**Stack:** Python · LangGraph · LangChain · ChromaDB · RAGAs · LangSmith · Streamlit

---

## Hard Rules — Follow These Always

1. Never write code Manoj cannot explain line by line.
2. Never move to the next step until the checkpoint for the current step passes.
3. Every decision goes in `notes.md` as it is made — not after.
4. Commit to git after every step.
5. No Jupyter notebooks — everything runs as `.py` scripts.
6. No `requirements.txt` — use `pyproject.toml` with `uv`.
7. Never use globals or shared mutable state — all state flows through LangGraph.
8. One file per node in `graph/nodes/`. One file per prompt in `prompts/`.
9. All prompts are module constants — never defined inside node functions.
10. Embedding model and ChromaDB path are defined once in `config/settings.py`
    and imported everywhere — never hardcoded twice.

---

## Project Structure

```
SupportDoc-Agent/
├── app.py                          # Streamlit UI entry point
├── main.py                         # CLI runner for testing
├── run_ingestion.py                # One-time ingestion script
├── notes.md                        # Engineering decision log — fill as you build
├── pyproject.toml
├── .env                            # API keys — never commit
├── .gitignore
├── eval/
│   ├── eval_dataset.json           # 20 hand-written Q&A pairs
│   └── run_eval.py                 # RAGAs evaluation script
└── supportdoc_agent/
    ├── config/
    │   └── settings.py             # All env vars + constants — single source of truth
    ├── models/
    │   └── llm.py                  # Shared ChatOpenAI instance
    ├── prompts/                     # One file per node prompt
    │   ├── router_prompt.py
    │   ├── rewriter_prompt.py
    │   ├── grader_prompt.py
    │   └── generator_prompt.py
    ├── data/
    │   └── support_docs.py         # Curated list of Intune article dicts
    ├── ingestion/
    │   ├── loader.py               # URL → LangChain Documents
    │   └── chunker.py              # Chunk + embed + store ChromaDB
    └── graph/
        ├── state.py                # SupportDocState TypedDict
        ├── graph.py                # Compiled StateGraph
        └── nodes/
            ├── router.py
            ├── rewriter.py
            ├── retriever.py
            ├── grader.py
            ├── generator.py
            └── direct_answer.py
```

---

## Environment Variables

`.env` file needs exactly these keys:

```env
OPENAI_API_KEY=sk-...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=SupportDocAgent
USER_AGENT=supportdoc-agent/1.0
```

---

## config/settings.py — Define Once, Import Everywhere

```python
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY  = os.environ["OPENAI_API_KEY"]
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY", "")

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL       = "gpt-4o-mini"
CHROMA_DIR      = "./chroma_db"
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200
TOP_K           = 5
MAX_RETRIEVAL_ATTEMPTS = 2
```

**Why this matters:** embedding model defined here and imported by both
`chunker.py` and the retriever node. Mismatched models produce silent
retrieval failures with no error message.

---

## Graph Architecture

### The 7-key State

```python
# graph/state.py
from typing import TypedDict
from langchain_core.documents import Document

class SupportDocState(TypedDict):
    query:            str              # Original question. Never modified after entry.
    rewritten_query:  str              # Improved query from rewriter. Used on retry.
    route:            str              # 'retrieve' or 'direct_answer'. Set by router.
    documents:        list[Document]   # Retrieved chunks from ChromaDB.
    relevance:        str              # 'relevant' or 'not_relevant'. Set by grader.
    retrieval_count:  int              # How many retrieval attempts so far. Max = 2.
    generation:       str              # Final answer string. Set by generator.
```

**Rule:** every node takes the full state dict and returns only the keys it
modifies as a partial dict. LangGraph merges it back. No node touches keys
it does not own.

### Node Contracts

| Node | Reads | Writes |
|------|-------|--------|
| `query_router` | `query` | `route` |
| `query_rewriter` | `query`, `retrieval_count` | `rewritten_query`, `retrieval_count` (+1) |
| `retriever` | `rewritten_query` if set, else `query` | `documents` |
| `relevance_grader` | `documents`, `query` | `relevance`, `documents` (filtered) |
| `generator` | `documents`, `query` | `generation` |
| `direct_answer` | `query` | `generation` |

### Graph Edges

```
entry_point ──────────────────────► query_router
query_router ──(route=retrieve)───► query_rewriter
query_router ──(route=direct)─────► direct_answer ──► END
query_rewriter ───────────────────► retriever
retriever ────────────────────────► relevance_grader
relevance_grader ─(relevant)──────► generator ──────► END
relevance_grader ─(not_relevant    ► query_rewriter   ← THE CYCLE
                   + count < 2)
relevance_grader ─(not_relevant    ► generator ──────► END
                   + count >= 2)
```

**Key:** the rewriter → retriever edge is a cycle. LangGraph supports this
natively. Do not use a while loop. The graph handles it through edges.

---

## Ingestion Pipeline

### loader.py

**What it does:** fetches each Intune article URL, strips HTML noise,
returns `list[Document]` with full metadata.

**HTML cleaning — two layers:**

Layer 1 — decompose noise tags:
```python
_NOISE_TAGS = ["script", "style", "nav", "header", "footer",
               "aside", "form", "button", "svg"]

_NOISE_CLASSES = ["toc", "feedback", "toolbar", "breadcrumb",
                  "sidebar", "banner", "alert", "table-of-contents",
                  "action-container", "article-header", "auth-page-section",
                  "learn-ask", "summarize", "reading-mode", "feedback-section"]

_NOISE_IDS = ["toc", "article-header", "ms--action-bar"]
```

Layer 2 — post-extraction line filter:
```python
_NOISE_LINES = {
    "table of contents", "exit editor mode", "ask learn",
    "reading mode", "add to plan", "copy markdown", "feedback",
    "summarize this article for me", "print", "read in english",
    "access to this page requires authorization",
}
# Also drop any line with fewer than 4 words
```

**Metadata on every Document:**
```python
metadata={
    "title":    article["title"],
    "url":      article["url"],       # used for citations
    "source":   article["source"],    # "microsoft_learn" or community blog
    "url_type": article["url_type"],
    "category": article["category"],  # "enrollment", "app_management", etc.
    "platform": ",".join(article["platform"]),  # must be string, not list
    "priority": article["priority"],  # 1 = primary, 2 = supplementary
}
```

**Rate limiting:** `time.sleep(0.5)` between every request. Do not remove.

**Error handling:** broad `except Exception` per article — one bad URL must
not crash the full ingestion run.

### chunker.py

**Splitter:** `RecursiveCharacterTextSplitter`
- `chunk_size=1000`, `chunk_overlap=200`
- `separators=["\n\n", "\n", ".", " ", ""]`
- Use `split_documents()` not `split_text()` — preserves metadata

**Add positional metadata to every chunk:**
```python
chunk.metadata["doc_index"]    = doc_index      # which document
chunk.metadata["chunk_index"]  = chunk_index    # position within doc
chunk.metadata["total_chunks"] = len(chunks)    # how many chunks this doc produced
```

**Storage:**
```python
Chroma.from_documents(
    documents=all_chunks,
    embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    persist_directory=CHROMA_DIR,
)
```

**Near-duplicate handling:** community blog sources (Prajwal Desai,
Anoop C Nair) only included for categories not already covered by
`microsoft_learn` source. Checked via `source` and `category` metadata.

### run_ingestion.py

Three steps with explicit logging:
1. Load documents — abort if 0 docs returned
2. Chunk + embed + store
3. Verify: print `vectorstore._collection.count()` (expect 200+),
   run 3 smoke test queries and print top result title + preview

**Re-ingestion:**
```bash
rm -rf ./chroma_db/ && python run_ingestion.py
```
ChromaDB appends to existing collections — always delete before re-running.

---

## Node Implementation Guide

### Node 1: Query Router

**File:** `graph/nodes/router.py`
**Prompt file:** `prompts/router_prompt.py`

Output must be exactly `'retrieve'` or `'direct_answer'` — nothing else.
Use `with_structured_output` or enum-constrained prompting to enforce this.

Checkpoint:
- `'How do I fix Intune enrollment error 80180014?'` → `'retrieve'`
- `'What time is it?'` → `'direct_answer'`

---

### Node 2: Query Rewriter

**File:** `graph/nodes/rewriter.py`
**Prompt file:** `prompts/rewriter_prompt.py`

Increments `retrieval_count` by 1 on every call.
Research step-back prompting (arxiv 2310.06117) and HyDE (arxiv 2212.10496).
Implement one, document which and why in `notes.md`.

Checkpoint: `'Intune not working'` must produce a meaningfully more
specific and technical rewrite — not just a rephrasing.

---

### Node 3: Retriever

**File:** `graph/nodes/retriever.py`

Uses `rewritten_query` if set in state, falls back to `query`.
Load persisted ChromaDB — do not re-create it.

```python
Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
```

`k=TOP_K` from settings. Document why in `notes.md`.

---

### Node 4: Relevance Grader ← THE AGENTIC CORE

**File:** `graph/nodes/grader.py`
**Prompt file:** `prompts/grader_prompt.py`

For EACH document: ask LLM if it is relevant to the query — YES or NO.
Filter out NO documents.
If zero pass → `relevance='not_relevant'`.
If any pass → `relevance='relevant'`, return filtered documents only.

Use `with_structured_output` — LLM must not return 'Yes, this is relevant.'
It must return exactly YES or NO.

Checkpoint: `'What is the capital of France?'` must return
`not_relevant` and trigger the retry loop. Verify in LangSmith trace.

---

### Node 5: Generator

**File:** `graph/nodes/generator.py`
**Prompt file:** `prompts/generator_prompt.py`

Grounding instruction must:
- Force answer only from provided context
- Return `'I don't have enough information.'` if context is empty
- Require citing the source URL for each factual claim

Checkpoint:
- Ask a real Intune question — answer must cite a source URL
- Ask a question not in the corpus — must return the fallback string

---

### Node 6: Direct Answer

**File:** `graph/nodes/direct_answer.py`

Simple node — answer the query directly with the LLM, no retrieval.
Used for greetings, general knowledge, out-of-scope questions.

---

## Evaluation

### eval_dataset.json — 20 entries, hand-written

```json
[
  {
    "question": "...",
    "ground_truth": "...",
    "source_url": "https://learn.microsoft.com/...",
    "contexts": ["chunk text 1", "chunk text 2"]
  }
]
```

Coverage requirements:
- Enrollment, apps, policies, compliance, certificates
- 3-4 hard questions where answer spans multiple chunks
- 2-3 questions where answer is NOT in corpus — test fallback path
- Do NOT generate with ChatGPT — write from Intune knowledge

### run_eval.py

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
```

RAGAs Dataset fields: `question`, `answer`, `contexts` (list of strings),
`ground_truth`.

Save results table to file. Print all 4 scores.
Target: at least 2 metrics above 0.70 at baseline.

### Retrieval improvement (pick based on weakest metric)

| Weakest metric | Fix |
|---------------|-----|
| `context_recall` low | `MultiQueryRetriever` — 3 query variations, union results |
| `context_precision` low | Cohere Reranker — rerank top-20 to top-5 |
| `faithfulness` low | Stricter generator prompt |

Apply fix in retriever node only — do not restructure the graph.
Record before/after delta in `notes.md`.

---

## Streamlit UI — app.py

Minimal. Three elements only:
1. Text input for query
2. Submit button
3. Answer display showing: final answer, source URLs cited,
   number of retrieval attempts

Deploy to Streamlit Community Cloud:
- API keys go in Streamlit secrets, not `.env`
- `pyproject.toml` or `requirements.txt` must be at repo root

---

## pyproject.toml

```toml
[project]
name = "supportdoc-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "langchain",
    "langchain-openai",
    "langchain-community",
    "langgraph",
    "chromadb",
    "streamlit",
    "python-dotenv",
    "ragas",
    "langsmith",
    "beautifulsoup4",
    "requests",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

Install with: `uv pip install -e .`

---

## .gitignore

```
.env
chroma_db/
.venv/
__pycache__/
*.pyc
.DS_Store
```

---

## notes.md — Decisions to Document

Write spoken-word answers (full sentences, not bullets) to every question
below. These become interview answers.

### Data Ingestion
- What splitter class? Exact `chunk_size` and `chunk_overlap` in characters?
- What embedding model? Why over `text-embedding-ada-002`?
- Why ChromaDB? When would you use Pinecone or Weaviate instead?
- Why FAISS was ruled out?
- How many total chunks? How did you verify ingestion quality?

### Graph Architecture
- Why LangGraph StateGraph over a LangChain sequential chain?
- Walk through each node: what state keys does it read? What does it write?
- Why LLM-as-judge in relevance grader over cosine similarity threshold?
- What is the maximum retrieval attempt count? Why that number?
- What happens when both attempts fail? What does the user see?

### Query Handling
- What does the query rewriter do? Which technique — step-back or HyDE?
- Why rewrite on retry rather than re-running the original?
- What query types does the router classify as `direct_answer`?

### State Design
- Why TypedDict for state?
- Why is `query` never modified after entry?
- Why `list[Document]` for `documents` rather than `list[str]`?

### Evaluation
- What are the 4 RAGAs baseline scores?
- What was the weakest metric? What fix was applied?
- What was the before/after delta? Why did it improve?

### Deployment
- Where is the app deployed? What is the live URL?
- What does LangSmith show that print statements cannot?
- What is the average end-to-end latency for a query that triggers one retry?

---

## Checkpoints — Do Not Skip

| Step | Checkpoint |
|------|-----------|
| 1 | `python -c 'import langchain, langgraph, chromadb, ragas'` runs clean |
| 2 | `run_ingestion.py` completes · `vectorstore._collection.count() > 200` |
| 3 | `from supportdoc_agent.graph.state import SupportDocState` imports cleanly |
| 4a | Router: enrollment error → `retrieve` · time question → `direct_answer` |
| 4b | Rewriter: `'Intune not working'` produces specific technical rewrite |
| 4c | Retriever: `'MDM certificate expired'` returns relevant chunks |
| 4d | Grader: `'What is the capital of France?'` → `not_relevant` in LangSmith |
| 4e | Generator: real question → answer with source URL cited |
| 5 | `app.invoke({'query':'...','retrieval_count':0})` returns non-empty `generation` |
| 6 | `eval_dataset.json` has 20 entries, all 4 fields, no ChatGPT generation |
| 7 | `run_eval.py` completes · at least 2 metrics above 0.70 |
| 8 | At least one metric improved by 0.05 after retrieval fix |
| 9 | App live at public URL · demoed on phone · URL in README |

