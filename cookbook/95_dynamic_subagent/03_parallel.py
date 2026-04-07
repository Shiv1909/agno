"""
Dynamic subagent -- parallel spawning.

The orchestrator spawns three specialists in parallel (one LLM turn,
three concurrent tool calls). Results arrive together.

Run:
    python cookbook/95_dynamic_subagent/03_parallel.py
"""

from agno.agent import Agent, SubAgentConfig
from agno.models.openai import OpenAIChat

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    name="orchestrator",
    description="Orchestrator that parallelises analysis across specialists.",
    instructions=[
        "When asked to analyse a topic from multiple angles, call spawn_agent three times "
        "in the SAME response -- once for each angle. This runs them in parallel.",
        "Angles: technical, business, and societal impact.",
        "After receiving all results, synthesise them into a final report.",
    ],
    enable_dynamic_subagents=True,
    subagent_config=SubAgentConfig(
        max_concurrent=5,
    ),
    markdown=True,
)

agent.print_response(
    "Analyse the impact of large language models from three angles.",
    stream=True,
)
