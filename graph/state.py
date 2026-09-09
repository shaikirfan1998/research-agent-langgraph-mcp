"""Shared state schema passed between nodes in the LangGraph graph."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class Finding(TypedDict):
    subtopic: str
    content: str


class ResearchState(TypedDict):
    # Set once at the start
    question: str

    # Produced by the planner node
    subtopics: list[str]

    # Produced by the researcher node, one entry per subtopic.
    # `operator.add` lets multiple graph steps append to this list
    # instead of overwriting it.
    findings: Annotated[list[Finding], operator.add]

    # Produced by the writer node, overwritten on each revision
    report: str

    # Produced by the critic node
    critique: str
    approved: bool
    revision_count: int

    # Final output path from the save step
    saved_path: str
