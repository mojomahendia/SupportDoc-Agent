# SupportDoc Agent — Project Decisions

## Data Source (`data/support_docs.py`)
- Restructured from a nested dict (grouped by source) to a flat list of article dicts.
  Flat list is easier to iterate in the loader and cleaner to extend.
- Each article has: `title`, `url`, `source`, `url_type`, `category`, `platform`, `priority`.
- `url_type` distinguishes `direct_article_url` (fetchable content) from `category/archive_page`
  (link listing pages — lower quality content).
- `platform` is a list at source (`["windows", "android"]`) but serialised as a comma-joined
  string in Document metadata because LangChain metadata values must be scalar.
- `priority` 1 = Microsoft Learn (official), 2 = community blogs. Used later to rank retrieval results.
- Duplicate URLs removed manually — no deduplication logic needed in the loader.

## Package Management
- Using `uv` for dependency management.
- `langchain-community` is deprecated but kept because `WebBaseLoader` was the starting point.
  Eventually replaced with `requests` + `BeautifulSoup` directly for better HTML control.
- `langchain-google-vertexai` added as a fix for a `ragas 0.4.3` bug — it unconditionally
  imports `ChatVertexAI` from a removed path in `langchain-community`. A shim file at
  `.venv/lib/python3.11/site-packages/langchain_community/chat_models/vertexai.py`
  re-exports it from the new location. Must be recreated if the venv is rebuilt.
- `bs4` added as explicit dependency (wrapper package that pulls `beautifulsoup4`).
- `[tool.setuptools.packages.find]` added to `pyproject.toml` with `exclude = [".venv*", "tests*"]`
  so editable install (`uv pip install -e .`) works across all project packages automatically.
  Each new folder needs an `__init__.py` to be recognised as a package.
- `ipykernel` registered as a Jupyter kernel (`supportdoc-agent`) so notebooks use the `.venv`.

## Development Workflow
- Editable install (`uv pip install -e .`) chosen so imports work identically in scripts,
  notebooks, and tests — no `sys.path` hacks needed.
- Notebooks kept at project root so Jupyter sets `sys.path` to the project root automatically.
- `python -m ingestion.loader` (module mode) used for script testing instead of
  `python ingestion/loader.py` — the latter sets `sys.path` to the `ingestion/` folder,
  breaking root-level imports like `from data.support_docs import ...`.

## Loader (`ingestion/loader.py`)
- Switched from `WebBaseLoader` to `requests` + `BeautifulSoup` directly for full control
  over HTML parsing and noise removal.
- Noise removal strategy: decompose `script`, `style`, `nav`, `header`, `footer`, `aside`,
  `form` tags, then decompose elements whose class names contain known noise keywords
  (`toc`, `feedback`, `toolbar`, `breadcrumb`, `sidebar`, `banner`, `alert`).
- Content selector order: `max(soup.select("div.content"), key=text_length)` → `article`
  → `main` → `body`. Microsoft Learn has multiple `div.content` elements — the largest
  one is always the article body. `select_one` was tried first but always picked the
  wrong (title-only) div.
- `USER_AGENT` env variable set in `.env` and loaded via `python-dotenv` before any
  langchain imports. `langchain-community` reads it at import time and falls back to
  `"DefaultLangchainUserAgent"` if missing (logs a warning but still works).
- `time.sleep(0.5)` between requests to avoid rate-limiting on community blog sites.
- Loader is a one-time operation — runs once to populate ChromaDB, not called again
  during normal agent use. No caching needed.
- `load_documents()` wrapped as a function (not module-level code) so imports don't
  trigger fetching.

## Chunking (`ingestion/chunker.py`)
- Used `RecursiveCharacterTextSplitter` with `chunk_size=1000`, `chunk_overlap=200`.
  Chosen over `SemanticChunker` because Microsoft Learn articles are already
  paragraph-structured — the document's own formatting is a reliable semantic boundary.
  `SemanticChunker` would add embedding cost and latency at ingestion time with minimal
  quality gain on structured technical prose.
- Each chunk inherits all metadata from its parent Document automatically via
  `split_documents()`.
- Near-duplicate content across sources (community blogs covering same topics as
  Microsoft Learn) is acceptable at this scale. The `priority` metadata field lets the
  agent prefer official sources. Deduplication by embedding similarity can be added
  later if retrieval quality drops.

## Ingestion Pipeline
```
load_documents() → chunk_documents() → embed → store in ChromaDB
```
Run once via `python run_ingestion.py`. Agent queries ChromaDB directly at runtime. ChromaDB appends to existing collections rather than overwriting — delete `./chroma_db/` before re-running.

## Query Router (`graph/nodes/router.py`)
- Used `llm.with_structured_output(RouteDecision)` where `RouteDecision` is a Pydantic model with `route: Literal["retrieve", "direct_answer"]`. This uses OpenAI's function-calling under the hood — the LLM cannot return anything outside those two values. Cleaner and more reliable than parsing free text or regex-matching.
- Prompt biases heavily toward `retrieve`. Only classifies as `direct_answer` for general conversational questions any IT professional would know without documentation. When in doubt, `retrieve` — a missed retrieval is recoverable; a hallucinated direct answer is not.
- `temperature=0` on the shared LLM client — this is a classifier, randomness has no value here.

## Query Rewriter (`graph/nodes/rewriter.py`)
- Chose step-back prompting (arxiv 2310.06117) over HyDE (arxiv 2212.10496).
  HyDE generates a hypothetical answer document and uses it as the search query — but for Intune troubleshooting, HyDE hallucinates domain-specific details in the hypothetical, producing a search query full of confident-sounding wrong information. Step-back is more controlled: it broadens vocabulary predictably by expanding specific symptoms to general concepts and adding MDM/MEM/Entra ID synonyms. Error codes are preserved exactly — they are the highest-signal retrieval tokens and must not be paraphrased.
- Always rewrites from `state["query"]` (the original), never from the previous `rewritten_query`. Original preserves more user intent; rewriting the rewrite risks drift.
- `retrieval_count` is passed to the prompt so the LLM knows to go broader on the second attempt.
- `retrieval_count` is NOT incremented here — the retriever node increments it when the ChromaDB call actually happens, keeping "attempts so far" semantically correct.

## Generator (`graph/nodes/generator.py`)
- Retrieved case prompt instructs "use ONLY the provided excerpts" — prevents the LLM from mixing in training-data knowledge that may be outdated or contradict the official Microsoft docs. If the excerpts don't fully answer the question, the LLM is instructed to say so explicitly rather than silently filling gaps with potentially wrong information.
- Citations deduped by URL (not by title) — a single article produces multiple chunks; we want one citation per source article. Sorted by `priority` metadata so Microsoft Learn (priority 1) appears before community blogs (priority 2).
- Fallback is a hardcoded string with no LLM call — when both retrieval attempts failed, the corpus doesn't cover the topic. Calling the LLM at this point would either hallucinate an answer or produce the same "I don't know" with added latency and cost. Neither is acceptable.
- Branch order: check `documents` emptiness first, then `route`. This means an empty `documents` list always triggers fallback regardless of route — a defensive guard against unexpected state combinations.

## Relevance Grader (`graph/nodes/grader.py`)
- LLM-as-judge over cosine similarity threshold: a similarity threshold is fast but brittle — a chunk about "certificate renewal" might score high similarity against "enrollment error 80180014" because vocabulary overlaps, but it isn't useful for answering that question. The LLM reasons about relevance semantically. Tradeoff: ~300–500ms and one API call per chunk. Worth it for quality at this corpus scale.
- `relevant: bool` in the Pydantic model over `Literal["YES","NO"]`: bool is cleaner — no string parsing, Pydantic validates it natively via `with_structured_output`, and the grader loop reads `score.relevant` directly.
- Grading query uses `rewritten_query` if available, else falls back to `query` — the rewrite is more specific and gives the LLM better context for judging relevance.
- Filters `documents` in-place: returns only the chunks that passed grading. The generator receives only relevant content; no noise passed downstream.

## Retriever (`graph/nodes/retriever.py`)
- No LLM call — purely a ChromaDB `similarity_search`. The only node with no prompt file.
- `k=4`: balances grader latency (4 serial LLM calls at ~300–500ms each) against retrieval coverage. Increasing k improves recall but adds proportional grader latency.
- Module-level singletons for `_embeddings` and `_vectorstore`: constructed once on first import, reused across all graph invocations in the same process. Avoids repeated ChromaDB disk I/O and connection overhead on every query.
- `persist_directory` computed as absolute path from `__file__` — avoids silent failure when the process is started from a directory other than the project root.
- Uses `Chroma(persist_directory=..., embedding_function=...)` not `Chroma.from_documents()` — the former loads an existing store; the latter creates a new one, destroying the ingested data.
- `retrieval_count` incremented here (not in the rewriter) because the count means "how many ChromaDB calls have happened so far", which is only true after the search runs.

## Graph State (`graph/state.py`)
- Plain `TypedDict` with all seven fields present: `query`, `rewritten_query`, `route`, `documents`, `relevance`, `retrieval_count`, `generation`.
- No `NotRequired` or `total=False` — LangGraph nodes return partial dicts and the framework merges them back in. TypedDict completeness is not validated at runtime; fields absent on the first pass are simply not in the dict yet.
- Nodes that read fields not yet set (e.g., `rewritten_query` before the rewriter runs) use `.get()` with a default, not direct key access, to avoid `KeyError` on the first pass.
- `query` is never modified after entry. The rewriter always writes to `rewritten_query`.

## vector database
Chose ChromaDB over Pinecone and Weaviate. Corpus is ~500 chunks — well within ChromaDB's local performance range. ChromaDB persists to disk with no managed infrastructure, adds zero network latency on retrieval, and metadata filtering works natively. Would switch to Pinecone if this became a multi-developer or high-query-volume production system, or Weaviate if hybrid BM25 + vector search was needed for better keyword-heavy queries.

- Evaluated FAISS but ruled it out — FAISS is a similarity search library with no built-in metadata storage or filtering. Would require building a parallel metadata store and keeping it in sync manually. ChromaDB provides both vector search and metadata filtering in one component, which is the correct tradeoff for a RAG pipeline where source citations depend on chunk metadata

- Used `text-embedding-3-small` over `text-embedding-ada-002` — newer model, higher MTEB benchmark scores, 5x cheaper. Did not use `text-embedding-3-large` because the quality gain from doubled dimensions is negligible at ~500 chunk corpus scale. Embedding model must be identical between ingestion (`ingestion/vector_store.py`) and retrieval (`graph/nodes/retriever.py`) — mismatched models produce silent retrieval failures with no error message.

Used with_structured_output() on router and grader nodes to enforce exact output schemas via Pydantic models. Eliminates string parsing fragility — the LLM returns a typed Python object instead of free text. Set temperature=0 on all decision nodes because they make binary judgments where determinism matters more than creativity.

RAGAs uses contexts to evaluate three of its four metrics: context_precision (chunks relevant to question), context_recall (chunks contain ground-truth information), and faithfulness (answer grounded in chunks). The contexts field in the eval dataset represents the chunks ChromaDB returns for each question — not idealized chunks. Using real retriever output is what makes the scores meaningful signals about retrieval quality. If context_recall comes back low, that points to a retrieval problem to fix; if it were prefilled with ideal chunks, that signal would disappear.

## Evaluation (RAGAs — `eval/run_eval.py`)

### RAGAs compatibility fix (v0.4.3)
RAGAs 0.4.3 requires explicit LLM and embeddings wrappers; it cannot use `OpenAIEmbeddings` or `ChatOpenAI` directly. Fixed by wrapping with `LangchainLLMWrapper` and `LangchainEmbeddingsWrapper` and setting `max_tokens=4096` on the LLM — the default 1024 caused the faithfulness statement-decomposition prompt to truncate mid-output, producing silent zero scores. `MultiQueryRetriever` is no longer in `langchain` or `langchain_community` in v1.x — it moved to `langchain_classic.retrievers.multi_query`.

### First eval run — wrong question types (before fix)
Initial eval dataset contained mostly how-to and configuration questions ("How do I configure Wi-Fi profile?", "How do I deploy Win32 app?"). The corpus is 21 troubleshooting-only articles (all from `/troubleshoot/mem/intune/`). Result: 16/20 questions returned the fallback string. Scores were near-zero across all metrics — not a pipeline failure, but a corpus-question mismatch.

### Baseline scores (after rewriting questions to match corpus)
All 17 Intune questions rewritten to troubleshooting questions mapping to the 21 ingested articles.

| Metric | Score |
|--------|-------|
| faithfulness | 0.3056 |
| answer_relevancy | 0.0855 |
| context_precision | 0.1500 |
| context_recall | 0.3167 |

When retrieval succeeded (enrollment questions), per-question faithfulness was 0.63–0.82 and context_precision was ~1.0. The low averages are entirely explained by questions that still returned the fallback string — approximately 10/17 Intune questions still retrieved no relevant chunks.

### Root cause of low scores
The corpus is dominated by enrollment-related content (4 of 21 articles are enrollment articles). In the embedding space, all queries — even about app management, certificates, or policy conflicts — tend to retrieve chunks from the generic enrollment troubleshooting article, which the grader correctly rejects. The app, certificate, and configuration articles are in the corpus but their chunks rank below the enrollment chunks for most queries.

### Retrieval fix tried: MultiQueryRetriever (k=4, 3 variations)
`langchain_classic.retrievers.multi_query.MultiQueryRetriever` generates 3 query variations and unions results. Result: context_recall dropped from 0.3167 to 0.2567. The LLM-generated variations remained semantically close to the original, so they continued retrieving enrollment-focused chunks. The union of more similar results did not surface the specific app/cert/policy articles.

### Retrieval fix applied: k=4 → k=5 (Checkpoint 8)

Increased k from 4 to 5 — one extra chunk per retrieval call, no additional LLM calls, minimal latency impact (one more grader call per retrieval attempt). Result:

| Metric | k=4 baseline | k=5 | Δ |
|--------|-------------|-----|---|
| faithfulness | 0.3056 | 0.3100 | +0.004 |
| answer_relevancy | 0.0855 | 0.0892 | +0.004 |
| context_precision | 0.1500 | 0.2000 | **+0.050 ✅** |
| context_recall | 0.3167 | 0.3167 | 0.000 |

context_precision improved by exactly 0.05 — meets Checkpoint 8 (≥ 0.05 improvement on one metric). The extra chunk gives the grader one more relevant candidate per attempt, improving precision of the passed context. context_recall is unchanged because the root issue is corpus coverage (relevant articles exist but rank below the top-5 for many queries), not the number of chunks retrieved.

MultiQueryRetriever (tried first) was reverted — it generated semantically similar variations that still retrieved enrollment-focused chunks, dropping context_recall from 0.32 to 0.26.

### What would actually fix the remaining low scores
Corpus expansion: add how-to and configuration guide articles (not just `/troubleshoot/`) to `data/support_docs.py`, then re-ingest. This would ensure the app management, certificate, and compliance policy articles surface in semantic search instead of being outranked by the enrollment troubleshooting articles.
