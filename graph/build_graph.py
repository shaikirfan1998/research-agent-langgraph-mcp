"""Wires the four agent nodes into a LangGraph StateGraph."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from .agents import (
    critic_node,
    make_researcher_node,
    planner_node,
    route_after_critique,
    writer_node,
)
from .state import ResearchState


def build_graph(tools: list):
    """tools: MCP tools discovered from the running MCP server, bound into
    the researcher agent."""
    graph = StateGraph(ResearchState)

    graph.add_node("plan", planner_node)
    graph.add_node("research", make_researcher_node(tools))
    graph.add_node("write", writer_node)
    graph.add_node("critique", critic_node)
    graph.add_node("save", _make_save_node(tools))

    graph.set_entry_point("plan")
    graph.add_edge("plan", "research")
    graph.add_edge("research", "write")
    graph.add_edge("write", "critique")
    graph.add_conditional_edges(
        "critique", route_after_critique, {"write": "write", "save": "save"}
    )
    graph.add_edge("save", END)

    return graph.compile()


def _make_save_node(tools: list):
    save_tool = next(t for t in tools if t.name == "save_report")

    async def save_node(state: ResearchState) -> dict:
        # Pull a title out of the first line of the report if possible
        first_line = state["report"].splitlines()[0].lstrip("# ").strip()
        title = first_line or state["question"]
        result = await save_tool.ainvoke(
            {"title": title, "markdown_content": state["report"]}
        )
        return {"saved_path": str(result)}

    return save_node
