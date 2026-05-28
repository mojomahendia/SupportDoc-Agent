 Plan: Load intune_articles into LangChain Documents

 Context

 data/support_docs.py has 90 article dicts (URLs +
 metadata). We need ingestion/loader.py
 to fetch each URL's content and return a list of LangChain
  Document objects with full
 metadata attached — ready for chunking and embedding into
 ChromaDB.

 Key constraints

 - 29 entries: url_type == "direct_article_url" — unique,
 real article pages
 - 61 entries: url_type == "category/archive_page" — many
 share the same URL (e.g. 10+
 Prajwal Desai articles all point to
 https://www.prajwaldesai.com/intune/). Must
 deduplicate fetches.
 - LangChain Document.metadata values must be scalar —
 platform (a list) must be
 serialized as a comma-joined string e.g. "windows,android"
 - WebBaseLoader is already the intended loader (broken
 import exists in loader.py)

 Implementation — ingestion/loader.py

 Imports

 from langchain_community.document_loaders import
 WebBaseLoader
 from langchain_core.documents import Document
 from bs4 import SoupStrainer
 from data.support_docs import intune_articles
 import logging

 Public function signature

 def load_documents() -> list[Document]:

 Logic (step by step)

 1. Deduplicate URLs — build url_cache: dict[str, str]
 mapping each unique URL to
 its fetched page content. This avoids re-fetching archive
 pages shared by many articles.
 2. Fetch content per unique URL using WebBaseLoader:
   - Pass a bs4_strainer targeting <article>, <main>, or
 <div class="content"> to
 strip nav/footer/scripts and return clean text.
   - Wrap each fetch in try/except — log failures and
 continue (don't crash).
   - Store result in url_cache[url] = page_content.
 3. Build one Document per article entry from
 intune_articles:
   - page_content = cached content for that article's URL
 (or skip if fetch failed)
   - metadata = all article fields with platform joined as
 comma string:
 {
     "title":    article["title"],
     "url":      article["url"],
     "source":   article["source"],
     "url_type": article["url_type"],
     "category": article["category"],
     "platform": ",".join(article["platform"]),
     "priority": article["priority"],
 }
 4. Log summary — total attempted, loaded successfully,
 skipped.
 5. Return list[Document].

 Rate limiting

 Add a small delay (time.sleep(0.5)) between fetches of
 unique URLs to avoid being
 rate-limited by community blog sites.

 Files to modify

 ┌─────────────────────┬────────────────────────────────┐
 │        File         │             Change             │
 ├─────────────────────┼────────────────────────────────┤
 │ ingestion/loader.py │ Full implementation (fix typo, │
 │                     │  add logic)                    │
 └─────────────────────┴────────────────────────────────┘

 Files NOT touched

 - data/support_docs.py — read-only input
 - ingestion/chunker.py — out of scope
 - main.py — out of scope

 Verification

 # Quick smoke test from project root
 from ingestion.loader import load_documents
 docs = load_documents()
 print(len(docs))           # expect up to 90
 print(docs[0].metadata)    # should show all 7 metadata
 fields
 print(docs[0].page_content[:200])  # should show real
 article text
 