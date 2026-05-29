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

## Ingestion Pipeline (planned)
```
load_documents() → chunk_documents() → embed → store in ChromaDB
```
Run once. Agent queries ChromaDB directly at runtime.

## Query Rewriter (`graph/nodes/rewriter.py`)
- Chose step-back prompting (arxiv 2310.06117) over HyDE (arxiv 2212.10496).
  HyDE generates a hypothetical answer document and uses it as the search query — but for Intune troubleshooting, HyDE hallucinates domain-specific details in the hypothetical, producing a search query full of confident-sounding wrong information. Step-back is more controlled: it broadens vocabulary predictably by expanding specific symptoms to general concepts and adding MDM/MEM/Entra ID synonyms. Error codes are preserved exactly — they are the highest-signal retrieval tokens and must not be paraphrased.
- Always rewrites from `state["query"]` (the original), never from the previous `rewritten_query`. Original preserves more user intent; rewriting the rewrite risks drift.
- `retrieval_count` is passed to the prompt so the LLM knows to go broader on the second attempt.
- `retrieval_count` is NOT incremented here — the retriever node increments it when the ChromaDB call actually happens, keeping "attempts so far" semantically correct.

## Graph State (`graph/state.py`)
- Plain `TypedDict` with all seven fields present: `query`, `rewritten_query`, `route`, `documents`, `relevance`, `retrieval_count`, `generation`.
- No `NotRequired` or `total=False` — LangGraph nodes return partial dicts and the framework merges them back in. TypedDict completeness is not validated at runtime; fields absent on the first pass are simply not in the dict yet.
- Nodes that read fields not yet set (e.g., `rewritten_query` before the rewriter runs) use `.get()` with a default, not direct key access, to avoid `KeyError` on the first pass.
- `query` is never modified after entry. The rewriter always writes to `rewritten_query`.

## vector database
Chose ChromaDB over Pinecone and Weaviate. Corpus is ~500 chunks — well within ChromaDB's local performance range. ChromaDB persists to disk with no managed infrastructure, adds zero network latency on retrieval, and metadata filtering works natively. Would switch to Pinecone if this became a multi-developer or high-query-volume production system, or Weaviate if hybrid BM25 + vector search was needed for better keyword-heavy queries.

- Evaluated FAISS but ruled it out — FAISS is a similarity search library with no built-in metadata storage or filtering. Would require building a parallel metadata store and keeping it in sync manually. ChromaDB provides both vector search and metadata filtering in one component, which is the correct tradeoff for a RAG pipeline where source citations depend on chunk metadata

-Used text-embedding-3-small over text-embedding-ada-002 — newer model, higher MTEB benchmark scores, 5x cheaper. Did not use text-embedding-3-large because the quality gain from doubled dimensions is negligible at ~500 chunk corpus scale. Embedding model defined once in settings.py and imported everywhere to guarantee ingestion and retrieval use identical models — mismatched models produce silent retrieval failures.

llm: temprature=0. It's a classifier. hence we did not need any randomness.


