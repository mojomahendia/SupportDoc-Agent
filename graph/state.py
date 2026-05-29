from typing import TypedDict

from langchain_core.documents import Document


class SupportDocState(TypedDict):
    query: str
    rewritten_query: str
    route: str
    documents: list[Document]
    relevance: str
    retrieval_count: int
    generation: str
