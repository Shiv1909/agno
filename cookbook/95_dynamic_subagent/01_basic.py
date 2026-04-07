"""
Dynamic subagent -- basic example.

The researcher agent decides mid-run that it needs a specialist
to write a formal summary. It spawns one on the spot.

Run:
    python cookbook/95_dynamic_subagent/01_basic.py
"""

from agno.agent import Agent, SubAgentConfig
from agno.models.openai import OpenAIChat

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    name="researcher",
    description="A research agent that can delegate writing tasks to a specialist.",
    instructions=[
        "When you need a polished formal summary, use spawn_agent to create a writing specialist.",
        "Pass all relevant facts to the specialist's task.",
    ],
    enable_dynamic_subagents=True,
    subagent_config=SubAgentConfig(),
    markdown=True,
)

agent.print_response(
    "Research the key benefits of async programming in Python and produce a formal executive summary.",
    stream=True,
)
