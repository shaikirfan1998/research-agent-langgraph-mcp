"""
The four agents in the graph. Each is a plain async function of
(state, config) -> partial state update, which is the shape LangGraph
node functions take. The `tools` list (MCP tools, loaded once at startup)
is closed over via functools.partial in build_graph.py so these stay
easy to unit test.
"""

from __future__ import annotations

import json

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from .state import ResearchState

MODEL_NAME = "claude-sonnet-4-6"
MAX_REVISIONS = 2


def _llm(temperature: float = 0.3) -> ChatAnthropic:
    return ChatAnthropic(model=MODEL_NAME, temperature=temperature)


# ---------------------------------------------------------------------------
# Planner: turns a broad question into 2-4 concrete research subtopics
# ---------------------------------------------------------------------------
async def planner_node(state: ResearchState) -> dict:
    system = SystemMessage(content=(
        "You are a research planner. Given a research question, break it "
        "into 2-4 concrete, non-overlapping subtopics that together would "
        "answer it well. Respond ONLY with a JSON array of short strings, "
        "e.g. [\"subtopic one\", \"subtopic two\"]. No prose, no markdown fences."
    ))
    human = HumanMessage(content=state["question"])
    response = await _llm().ainvoke([system, human])

    text = response.content.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        subtopics = json.loads(text)
        assert isinstance(subtopics, list) and all(isinstance(s, str) for s in subtopics)
    except Exception:  # noqa: BLE001
        subtopics = [state["question"]]  # fall back to treating it as one topic

    return {"subtopics": subtopics[:4], "revision_count": 0}


# ---------------------------------------------------------------------------
# Researcher: a ReAct agent (LangGraph prebuilt) with MCP tools bound in.
# It runs once per subtopic and produces a finding.
# ---------------------------------------------------------------------------
def make_researcher_node(tools: list):
    react_agent = create_react_agent(_llm(temperature=0), tools)

    async def researcher_node(state: ResearchState) -> dict:
        findings = []
        for subtopic in state["subtopics"]:
            prompt = (
                f"Research this subtopic thoroughly: '{subtopic}'\n"
                f"(This is part of answering the broader question: '{state['question']}')\n\n"
                "Use the web_search and arxiv_search tools as needed. "
                "When you have enough information, respond with a concise, "
                "well-sourced summary (with inline URLs) of what you found. "
                "Do not call save_report — that happens later."
            )
            result = await react_agent.ainvoke(
                {"messages": [HumanMessage(content=prompt)]}
            )
            content = result["messages"][-1].content
            findings.append({"subtopic": subtopic, "content": content})

        return {"findings": findings}

    return researcher_node


# ---------------------------------------------------------------------------
# Writer: synthesizes findings (and any critic feedback) into a report
# ---------------------------------------------------------------------------
async def writer_node(state: ResearchState) -> dict:
    findings_block = "\n\n".join(
        f"### {f['subtopic']}\n{f['content']}" for f in state["findings"]
    )
    revision_note = ""
    if state.get("critique"):
        revision_note = (
            f"\n\nA previous draft was reviewed and needs revision. "
            f"Critic feedback to address:\n{state['critique']}"
        )

    system = SystemMessage(content=(
        "You are a research writer. Synthesize the findings below into a "
        "clear, well-structured markdown report answering the original "
        "question. Include a short executive summary, a section per "
        "subtopic, and a 'Sources' section listing URLs mentioned in the "
        "findings. Do not fabricate sources."
    ))
    human = HumanMessage(content=(
        f"Question: {state['question']}\n\nFindings:\n{findings_block}{revision_note}"
    ))
    response = await _llm(temperature=0.4).ainvoke([system, human])

    return {"report": response.content}


# ---------------------------------------------------------------------------
# Critic: reviews the draft, approves it or sends it back with feedback
# ---------------------------------------------------------------------------
async def critic_node(state: ResearchState) -> dict:
    system = SystemMessage(content=(
        "You are a strict editorial critic. Evaluate whether the report "
        "fully and accurately answers the original question, is well "
        "sourced, and is well organized. Respond ONLY with JSON: "
        '{"approved": true|false, "feedback": "..."}. '
        "If approved is true, feedback should be empty."
    ))
    human = HumanMessage(content=(
        f"Original question: {state['question']}\n\nDraft report:\n{state['report']}"
    ))
    response = await _llm(temperature=0).ainvoke([system, human])

    text = response.content.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        parsed = json.loads(text)
        approved = bool(parsed.get("approved", False))
        feedback = parsed.get("feedback", "")
    except Exception:  # noqa: BLE001
        approved, feedback = True, ""  # fail open rather than loop forever

    revision_count = state.get("revision_count", 0)
    # Force approval once we've hit the revision cap, so the graph terminates.
    if not approved and revision_count >= MAX_REVISIONS:
        approved = True

    return {
        "approved": approved,
        "critique": feedback,
        "revision_count": revision_count + (0 if approved else 1),
    }


def route_after_critique(state: ResearchState) -> str:
    return "save" if state["approved"] else "write"
