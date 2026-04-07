"""
Dynamic subagent -- with inherited tools.

The parent agent has a DuckDuckGo search tool.
It spawns a subagent that is allowed to use search too.

Run:
    python cookbook/agents/dynamic_subagent/02_with_tools.py

Requirements:
    pip install duckduckgo-search
"""

from agno.agent import Agent, SubAgentConfig
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    name="coordinator",
    description="Coordinator that can spawn search specialists.",
    instructions=[
        "Use spawn_agent to delegate deep-dive searches to a specialist.",
        "Request tools=['search'] when spawning so the specialist can search the web.",
    ],
    tools=[DuckDuckGoTools()],
    enable_dynamic_subagents=True,
    subagent_config=SubAgentConfig(
        allowed_tools=["search"],
        allow_tool_selection=True,
    ),
    markdown=True,
)

agent.print_response(
    "Find the latest developments in quantum computing and summarize them.",
    stream=True,
)
