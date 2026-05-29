from langchain_core.messages import HumanMessage, SystemMessage

from graph.nodes._llm import llm
from graph.state import SupportDocState
from prompts.rewriter_prompt import REWRITER_SYSTEM_PROMPT


def rewriter_node(state: SupportDocState) -> dict:
    retrieval_count = state.get("retrieval_count", 0)

    attempt_context = f"Retrieval attempt: {retrieval_count + 1} of 2"
    if retrieval_count >= 1:
        attempt_context += "\nThe previous rewrite did not retrieve relevant documents — go broader this time."

    human_msg = f"Original question: {state['query']}\n{attempt_context}"

    response = llm.invoke([
        SystemMessage(REWRITER_SYSTEM_PROMPT),
        HumanMessage(human_msg),
    ])

    return {"rewritten_query": response.content.strip()}
