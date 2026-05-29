from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from graph.nodes._llm import llm
from graph.state import SupportDocState
from prompts.grader_prompt import GRADER_SYSTEM_PROMPT


class RelevanceScore(BaseModel):
    relevant: bool


_chain = llm.with_structured_output(RelevanceScore)


def grader_node(state: SupportDocState) -> dict:
    grading_query = state.get("rewritten_query") or state["query"]
    filtered = []

    for doc in state.get("documents", []):
        score = _chain.invoke([
            SystemMessage(GRADER_SYSTEM_PROMPT),
            HumanMessage(f"Question: {grading_query}\nDocument chunk: {doc.page_content}"),
        ])
        if score.relevant:
            filtered.append(doc)

    if filtered:
        return {"documents": filtered, "relevance": "relevant"}
    return {"documents": [], "relevance": "not_relevant"}
