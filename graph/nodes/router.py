from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from graph.nodes._llm import llm
from graph.state import SupportDocState
from prompts.router_prompt import ROUTER_SYSTEM_PROMPT


class RouteDecision(BaseModel):
    route: Literal["retrieve", "direct_answer"]


chain = llm.with_structured_output(RouteDecision)


def router_node(state: SupportDocState) -> dict:
    decision = chain.invoke([
        SystemMessage(ROUTER_SYSTEM_PROMPT),
        HumanMessage(state["query"]),
    ])
    return {"route": decision.route}
