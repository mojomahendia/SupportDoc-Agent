import logging
import os
import time

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.documents import Document

from data.support_docs import intune_articles

load_dotenv()

logger = logging.getLogger(__name__)

_NOISE_TAGS = ["script", "style", "nav", "header", "footer", "aside", "form"]
_NOISE_CLASSES = ["toc", "feedback", "toolbar", "breadcrumb", "sidebar", "banner", "alert"]


def _fetch_content(url: str) -> str:
    headers = {"User-Agent": os.environ.get("USER_AGENT", "supportdoc-agent/1.0")}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()

    for tag in soup.find_all(class_=lambda c: c and any(x in " ".join(c) for x in _NOISE_CLASSES)):
        tag.decompose()

    content_divs = soup.select("div.content")
    main = max(content_divs, key=lambda el: len(el.get_text()), default=None) or soup.find("article") or soup.find("main") or soup.body
    return main.get_text(separator="\n", strip=True) if main else ""


def load_documents() -> list[Document]:
    docs = []
    failed = 0

    for article in intune_articles:
        try:
            content = _fetch_content(article["url"])
            if not content:
                raise ValueError("Empty content after parsing")

            doc = Document(
                page_content=content,
                metadata={
                    "title":    article["title"],
                    "url":      article["url"],
                    "source":   article["source"],
                    "url_type": article["url_type"],
                    "category": article["category"],
                    "platform": ",".join(article["platform"]),
                    "priority": article["priority"],
                },
            )
            docs.append(doc)
            logger.info("Loaded: %s", article["title"])
        except Exception as exc:
            failed += 1
            logger.warning("Failed to load '%s' (%s): %s", article["title"], article["url"], exc)

        time.sleep(0.5)

    logger.info("Done — %d loaded, %d failed", len(docs), failed)
    return docs
