from langgraph.graph import END, START, StateGraph

from graph.nodes.generator import generator_node
from graph.nodes.grader import grader_node
from graph.nodes.retriever import retriever_node
from graph.nodes.rewriter import rewriter_node
from graph.nodes.router import router_node
from graph.state import SupportDocState


def _route_decision(state: SupportDocState) -> str:
    return state["route"]


def _grade_decision(state: SupportDocState) -> str:
    if state.get("relevance") == "relevant":
        return "generator"
    if state.get("retrieval_count", 0) >= 2:
        return "generator"
    return "rewriter"


builder = StateGraph(SupportDocState)

builder.add_node("router", router_node)
builder.add_node("rewriter", rewriter_node)
builder.add_node("retriever", retriever_node)
builder.add_node("grader", grader_node)
builder.add_node("generator", generator_node)

builder.add_edge(START, "router")

builder.add_conditional_edges("router", _route_decision, {
    "retrieve": "rewriter",
    "direct_answer": "generator",
})

builder.add_edge("rewriter", "retriever")
builder.add_edge("retriever", "grader")

builder.add_conditional_edges("grader", _grade_decision, {
    "relevant": "generator",
    "rewriter": "rewriter",
    "generator": "generator",
})

builder.add_edge("generator", END)

graph = builder.compile()
