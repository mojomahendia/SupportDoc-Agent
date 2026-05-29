from langchain_core.messages import HumanMessage, SystemMessage

from graph.nodes._llm import llm
from graph.state import SupportDocState
from prompts.generator_prompt import (
    DIRECT_HUMAN_TEMPLATE,
    GENERATOR_SYSTEM_PROMPT,
    RETRIEVED_HUMAN_TEMPLATE,
)

_FALLBACK = (
    "I wasn't able to find relevant information in the Intune documentation corpus "
    "to answer your question. Please check the Microsoft Intune troubleshooting "
    "documentation at learn.microsoft.com or contact Microsoft Support for further assistance."
)


def _build_citations(documents) -> str:
    seen = {}
    for doc in documents:
        url = doc.metadata.get("url", "")
        if url and url not in seen:
            seen[url] = doc.metadata
    sorted_sources = sorted(seen.values(), key=lambda m: m.get("priority", 99))
    lines = [f"- {m.get('title', 'Source')} — {m.get('url', '')}" for m in sorted_sources]
    return "\n\n[Sources]\n" + "\n".join(lines)


def generator_node(state: SupportDocState) -> dict:
    documents = state.get("documents") or []
    route = state.get("route", "retrieve")
    query = state["query"]

    # Fallback — both retrieval attempts exhausted with no relevant docs
    if route == "retrieve" and not documents:
        return {"generation": _FALLBACK}

    # Direct answer — no retrieval needed
    if route == "direct_answer":
        response = llm.invoke([
            SystemMessage(GENERATOR_SYSTEM_PROMPT),
            HumanMessage(DIRECT_HUMAN_TEMPLATE.format(query=query)),
        ])
        return {"generation": response.content}

    # Retrieved answer — ground in excerpts, append citations
    context = "\n\n---\n\n".join(doc.page_content for doc in documents)
    response = llm.invoke([
        SystemMessage(GENERATOR_SYSTEM_PROMPT),
        HumanMessage(RETRIEVED_HUMAN_TEMPLATE.format(query=query, context=context)),
    ])
    return {"generation": response.content + _build_citations(documents)}
