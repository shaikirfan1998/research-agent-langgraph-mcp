"""
AI Research Agent — entrypoint.

Launches the MCP tool server as a subprocess, connects to it over stdio,
loads its tools into LangChain/LangGraph via langchain-mcp-adapters, and
runs the planner -> researcher -> writer -> critic graph.

Usage:
    python main.py "How is agentic AI being adopted in telecom networks?"
"""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

from graph.build_graph import build_graph

load_dotenv()

SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "server", "mcp_server.py")


async def run(question: str) -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY (see .env.example)", file=sys.stderr)
        sys.exit(1)

    client = MultiServerMCPClient(
        {
            "research-tools": {
                "command": sys.executable,
                "args": [SERVER_SCRIPT],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()
    print(f"Connected to MCP server. Tools available: {[t.name for t in tools]}\n")

    app = build_graph(tools)

    print(f"Question: {question}\n{'=' * 60}")
    final_state = None
    async for event in app.astream(
        {"question": question, "findings": [], "revision_count": 0},
        stream_mode="values",
    ):
        final_state = event
        if "subtopics" in event and event.get("subtopics"):
            print(f"[plan] Subtopics: {event['subtopics']}")
        if event.get("findings") and len(event["findings"]) == len(event.get("subtopics", [])):
            print(f"[research] Gathered {len(event['findings'])} findings")
        if event.get("critique") is not None and event.get("approved") is False:
            print(f"[critique] Sent back for revision: {event['critique'][:200]}")
        if event.get("saved_path"):
            print(f"[save] {event['saved_path']}")

    print(f"\n{'=' * 60}\nFINAL REPORT\n{'=' * 60}\n")
    print(final_state["report"])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python main.py "<research question>"')
        sys.exit(1)
    asyncio.run(run(" ".join(sys.argv[1:])))
