"""
MCP server exposing research tools over stdio.

This is a *real* MCP server — it speaks the Model Context Protocol, the same
way a Slack/GitHub/Notion MCP server would. The agent process (main.py)
connects to this as a subprocess and discovers its tools dynamically via
MCP's tool-listing handshake, rather than importing Python functions
directly. That's the whole point of using MCP here: tools are decoupled
from the agent process and can be swapped, versioned, or run on a different
machine without touching agent code.

Run standalone for debugging:
    python server/mcp_server.py
"""

from __future__ import annotations

import datetime
import os

import arxiv
from duckduckgo_search import DDGS
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("research-tools")

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for a query and return titles, URLs, and snippets.

    Args:
        query: The search query.
        max_results: Max number of results to return (default 5).
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:  # noqa: BLE001
        return f"web_search failed: {e}"

    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "")
        href = r.get("href", "")
        body = r.get("body", "")
        lines.append(f"{i}. {title}\n   URL: {href}\n   {body}")
    return "\n\n".join(lines)


@mcp.tool()
def arxiv_search(query: str, max_results: int = 5) -> str:
    """Search arXiv for papers relevant to a query.

    Args:
        query: The search query (topic, keywords, or author).
        max_results: Max number of papers to return (default 5).
    """
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = list(search.results())
    except Exception as e:  # noqa: BLE001
        return f"arxiv_search failed: {e}"

    if not results:
        return "No papers found."

    lines = []
    for i, paper in enumerate(results, start=1):
        authors = ", ".join(a.name for a in paper.authors[:3])
        if len(paper.authors) > 3:
            authors += " et al."
        lines.append(
            f"{i}. {paper.title} ({paper.published.year})\n"
            f"   Authors: {authors}\n"
            f"   URL: {paper.entry_id}\n"
            f"   Abstract: {paper.summary[:300].replace(chr(10), ' ')}..."
        )
    return "\n\n".join(lines)


@mcp.tool()
def save_report(title: str, markdown_content: str) -> str:
    """Save a finished research report to disk as a markdown file.

    Args:
        title: Short title used to build the filename.
        markdown_content: The full report body in markdown.
    """
    safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in title)
    safe_title = safe_title.strip().replace(" ", "_")[:60] or "report"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{safe_title}.md"
    path = os.path.join(REPORTS_DIR, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    return f"Report saved to {os.path.abspath(path)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
