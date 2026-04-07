"""
Dynamic subagent -- team leader spawning subagents.

A team leader can spawn ephemeral subagents in addition to
delegating to its registered members.

Run:
    python cookbook/95_dynamic_subagent/04_team.py
"""

from agno.agent import Agent, SubAgentConfig
from agno.models.openai import OpenAIChat
from agno.team import Team

writer = Agent(
    model=OpenAIChat(id="gpt-4o"),
    name="writer",
    role="Writes clear prose from structured data.",
)

team = Team(
    name="content_team",
    model=OpenAIChat(id="gpt-4o"),
    members=[writer],
    description="Content team that can spawn specialist agents on demand.",
    instructions=[
        "Delegate prose writing to the writer member.",
        "For fact-checking or specialised analysis, use spawn_agent to create a focused specialist.",
    ],
    enable_dynamic_subagents=True,
    subagent_config=SubAgentConfig(
        inject_session_state=True,
    ),
    markdown=True,
)

team.print_response(
    "Create a blog post about the history of the Python programming language. "
    "Fact-check any dates before passing them to the writer.",
    stream=True,
)
