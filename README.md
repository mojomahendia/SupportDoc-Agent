# SupportDoc Agent

> Agentic RAG pipeline that answers Microsoft Intune support questions using self-correcting retrieval, LLM-as-judge relevance grading, and source-cited generation.

**Live demo:** `https://your-app.streamlit.app` · **Stack:** Python · LangGraph · LangChain · ChromaDB · RAGAs · LangSmith

---

## Why I Built This

I spent 4 years resolving Microsoft Intune escalations. The same questions came in every week — enrollment errors, MDM certificate failures, app deployment issues. I started building RAG systems to answer those questions automatically. SupportDoc Agent is that idea built to production quality. I am not moving away from my support experience — I am applying it at a different layer of the stack.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│  Query Router   │──── Can answer directly? ──YES──► Direct LLM Answer
└────────┬────────┘
         │ needs docs
         ▼
┌─────────────────┐
│ Query Rewriter  │  Improves query before retrieval (step-back prompting)
└────────┬────────┘
         ▼
┌─────────────────┐
│    Retriever    │  ChromaDB semantic search · top-k chunks
└────────┬────────┘
         ▼
┌─────────────────┐
│ Relevance Grader│  LLM-as-judge · grades each chunk YES/NO
└────────┬────────┘
         │
         ├── RELEVANT ──────────────► Generator ──► Answer + citations
         │
         └── NOT RELEVANT ──────────► Rewriter (loop · max 2 attempts)
                                           │
                                    Still failing?
                                           │
                                           ▼
                               "I don't have enough information."
```

### State — what flows through the graph

| Key | Purpose |
|-----|---------|
| `query` | Original user question. Never modified after entry. |
| `rewritten_query` | Improved query from the rewriter node. Used on retry. |
| `route` | `retrieve` or `direct_answer`. Set by the router. |
| `documents` | List of LangChain Document objects from the retriever. |
| `relevance` | `relevant` or `not_relevant`. Set by the grader. |
| `retrieval_count` | Integer. How many retrieval attempts so far. Max = 2. |
| `generation` | The final answer string. Set by the generator. |

---

## Project Structure

```
SupportDoc-Agent/
├── app.py                          # Streamlit UI entry point
├── main.py                         # CLI runner for testing
├── run_ingestion.py                # One-time ingestion script
├── notes.md                        # Engineering decision log
├── pyproject.toml
├── eval/
│   ├── eval_dataset.json           # 20 Q&A pairs (hand-written)
│   └── run_eval.py                 # RAGAs evaluation script
└── supportdoc_agent/
    ├── config/settings.py          # Centralised env vars + constants
    ├── models/llm.py               # Shared ChatOpenAI client
    ├── prompts/                    # One file per node prompt
    ├── data/support_docs.py        # Curated list of Intune article URLs
    ├── ingestion/
    │   ├── loader.py               # URL → LangChain Documents
    │   └── chunker.py              # Chunk + embed + store ChromaDB
    └── graph/
        ├── state.py                # SupportDocState TypedDict
        ├── graph.py                # Compiled StateGraph
        └── nodes/                  # One file per node
```

---

## Data Ingestion Pipeline

The ingestion pipeline runs **once** before the graph is used. It builds the ChromaDB vector store from a curated corpus of 30+ Microsoft Intune troubleshooting articles.

```bash
python run_ingestion.py
```

### What it does

```
intune_articles (URLs)
       ↓
   loader.py    →  HTTP GET → BeautifulSoup HTML cleaning → list[Document]
       ↓
   chunker.py   →  RecursiveCharacterTextSplitter → OpenAI embeddings → ChromaDB
       ↓
   ./chroma_db/ →  Persisted vector store · ready for retrieval
```

### Ingestion decisions

**Splitter:** `RecursiveCharacterTextSplitter` with `chunk_size=1000`, `chunk_overlap=200`.
Chose this over `SemanticChunker` because Microsoft Learn articles are already paragraph-structured — the document's own formatting is a reliable semantic boundary. `SemanticChunker` would add embedding cost and latency at ingestion time with marginal quality gain on structured technical prose.

**Embedding model:** `text-embedding-3-small`.
Newer than `text-embedding-ada-002`, higher MTEB benchmark scores, 5× cheaper. Did not use `text-embedding-3-large` — quality gain from doubled dimensions is negligible at ~500 chunk corpus scale.

**Vector store:** ChromaDB with local persistence.
Chose ChromaDB over Pinecone because the corpus is ~500 chunks, deployment is single-developer, and local persistence adds zero network latency on retrieval. Would switch to Pinecone for multi-developer or high-query-volume production use.

**HTML cleaning:** Two-layer approach — (1) BeautifulSoup strips noise tags and MS Learn-specific UI chrome classes; (2) post-extraction line filter removes known toolbar strings and lines under 4 words. Microsoft Learn pages contain significant non-content UI (toolbar buttons, auth banners, AI widgets) that pollutes chunks if not stripped.

**Metadata on every chunk:**

```python
chunk.metadata = {
    "title":       "...",
    "url":         "...",        # source citation
    "category":    "enrollment",
    "platform":    "Windows",
    "priority":    1,
    "doc_index":   0,            # which document
    "chunk_index": 3,            # position within document
    "total_chunks": 12,
}
```

Metadata survives chunking. Every chunk knows its source URL — this is what makes citations possible in the generator.

**Re-ingestion:**

```bash
rm -rf ./chroma_db/ && python run_ingestion.py
```

ChromaDB appends to existing collections rather than overwriting. Always delete `./chroma_db/` before re-running.

### Verification

`run_ingestion.py` self-verifies after storing:

```
Total chunks in ChromaDB: 487

Query : 'Intune enrollment error 80180014'
  Top result : Troubleshooting device enrollment in Intune
  Preview    : Error 80180014 occurs when the MDM...

Query : 'MDM certificate expired'
  Top result : Certificate troubleshooting for Intune
  Preview    : When the MDM push certificate expires...
```

---

## Key Engineering Decisions

**1. LangGraph StateGraph over a LangChain sequential chain**
A sequential chain is a fixed pipeline — input flows A → B → C with no branching or loops. LangGraph gives conditional edges and native cycle support. The relevance grader → rewriter → retriever retry loop cannot be expressed in a chain at all. The moment you need "if this, go back there", you need a graph.

**2. LLM-as-judge for relevance grading over cosine similarity threshold**
A cosine threshold is fast but brittle — a chunk about "certificate renewal" might score 0.75 similarity against "enrollment error 80180014" because vocabulary overlaps, but it is not useful for answering that question. An LLM reasons about relevance semantically. Tradeoff: ~300ms and one API call per chunk. Worth it for the quality gain at this corpus scale.

**3. Query rewriting on retry instead of re-running the original**
The original query failed retrieval once. Running it again returns identical chunks — nothing changes. The rewriter makes the query more specific and technical, increasing the chance that semantic search finds a relevant match on the second pass.

**4. ChromaDB over Pinecone for local persistence**
No managed infrastructure, no API latency on retrieval, no credentials to rotate. For a single-developer project with ~500 chunks, eliminating retrieval network round-trips and infrastructure overhead is the correct tradeoff.

**5. Manual source curation over automated scraping**
Community blogs (Prajwal Desai, Anoop C Nair) sometimes cover the same Intune topics as Microsoft Learn, just worded differently. Used domain knowledge to include community sources only for topics not covered by Microsoft Learn, marked with `priority: 2` in metadata. More accurate than a similarity threshold at this corpus size.

---

## Evaluation

Evaluated using RAGAs on a hand-written dataset of 20 Intune Q&A pairs covering enrollment, app management, policies, compliance, and certificates. Dataset includes 3 questions where the answer is not in the corpus — testing the fallback path.

### Baseline scores

| Metric | What it measures | Baseline | After fix | Delta |
|--------|-----------------|----------|-----------|-------|
| `faithfulness` | Answer supported by retrieved chunks — no hallucination | — | — | — |
| `answer_relevancy` | Answer addresses the question — on-topic, not evasive | — | — | — |
| `context_precision` | Retrieved chunks are relevant to the question | — | — | — |
| `context_recall` | Chunks contain all information needed to answer | — | — | — |

> Fill in your actual scores after running `python eval/run_eval.py`

### Running evaluation

```bash
python eval/run_eval.py
```

---

## Tech Stack

| Component | Technology | Decision |
|-----------|-----------|---------|
| Graph orchestration | LangGraph StateGraph | Conditional edges + native cycle support |
| LLM | GPT-4o-mini | Cost-efficient for multi-node graph with grading calls |
| Embeddings | text-embedding-3-small | Higher MTEB scores than ada-002 at 5× lower cost |
| Vector store | ChromaDB | Local persistence, zero retrieval latency, built-in metadata filtering |
| Document loading | LangChain + BeautifulSoup | Two-layer HTML cleaning for MS Learn UI chrome |
| Evaluation | RAGAs | 4-metric evaluation: faithfulness, relevancy, precision, recall |
| Observability | LangSmith | Node-by-node trace for every graph invocation |
| UI | Streamlit | Minimal deployment surface, public URL in one command |
| Package management | uv | 10-100× faster than pip, built-in lock files |

---

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) installed
- OpenAI API key
- LangSmith API key ([free tier](https://smith.langchain.com))

### Setup

```bash
git clone https://github.com/mojomahendia/SupportDoc-Agent
cd SupportDoc-Agent

uv venv && source .venv/bin/activate
uv pip install -e .
```

### Environment variables

Create a `.env` file at the project root:

```env
OPENAI_API_KEY=sk-...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=SupportDocAgent
USER_AGENT=supportdoc-agent/1.0
```

### Run ingestion (once)

```bash
python run_ingestion.py
```

### Run CLI

```bash
python main.py
```

### Run UI

```bash
streamlit run app.py
```

---

## Observability

Every query is traced node-by-node in LangSmith. Set `LANGCHAIN_TRACING_V2=true` in your `.env` and open [smith.langchain.com](https://smith.langchain.com) to see:

- Which route the query took (`retrieve` vs `direct_answer`)
- Whether the relevance grader triggered a retry
- Exact chunks passed to the generator
- End-to-end latency per node

---

## Author

**Manoj Kumar** · M.Sc. Computer Science (AI & ML), Scaler Neoversity · CGPA 3.7

[GitHub](https://github.com/mojomahendia) · [LinkedIn](#)
