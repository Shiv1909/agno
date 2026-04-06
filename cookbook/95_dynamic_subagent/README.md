# Dynamic Subagents

Spawn ephemeral, fully-capable subagents mid-run. The LLM decides when a specialist is needed, creates one, uses it, and it is automatically discarded.

## Examples

| File | Demonstrates |
|------|-------------|
| `01_basic.py` | Agent spawns a writing specialist for one task |
| `02_with_tools.py` | Subagent inherits search tool from parent |
| `03_parallel.py` | Three specialists spawned concurrently in one LLM turn |
| `04_team.py` | Team leader spawns subagents alongside registered members |

## Quick Start

```python
from agno.agent import Agent, SubAgentConfig
from agno.models.openai import OpenAIChat

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    enable_dynamic_subagents=True,
    subagent_config=SubAgentConfig(markdown=True),
)
```

## SubAgentConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | Model | None | Model for subagents. Inherits parent's when None. |
| `tools` | List | None | Tools always available to subagents. |
| `allowed_tools` | List[str] | None | Tool names the LLM may request at spawn time. |
| `inherit_parent_tools` | bool | False | Give subagents all parent tools. |
| `inherit_parent_knowledge` | bool | False | Share parent's knowledge base. |
| `inject_session_state` | bool | False | Pass parent session_state read-only to subagent. |
| `max_concurrent` | int | 5 | Max parallel subagents per run. |
| `allow_tool_selection` | bool | True | LLM can pick tools from allowed_tools at spawn time. |
| `allow_model_override` | bool | False | LLM can specify a different model at spawn time. |
| `markdown` | bool | False | Subagent formats output as markdown. |
| `reasoning` | bool | False | Enable step-by-step reasoning in subagent. |
| `tool_call_limit` | int | None | Cap tool calls per subagent run. |
| `retries` | int | 0 | Retry count on subagent failure. |
| `knowledge` | KnowledgeProtocol | None | Dedicated knowledge base for subagents. |
| `pre_hooks` / `post_hooks` | List | None | Hooks for subagent runs. |

## How It Works

1. `enable_dynamic_subagents=True` adds a `spawn_agent` tool to the agent/team.
2. The LLM calls `spawn_agent(role, instructions, task, ...)` during its tool loop.
3. An ephemeral `Agent` is built from the `SubAgentConfig` template + spawn-time overrides.
4. The subagent runs (`db=None` -- no persistence, no session created).
5. The result string is returned as the tool output into the parent's message history.
6. The subagent object goes out of scope and is garbage collected.

Parallel spawning works automatically: if the LLM calls `spawn_agent` multiple times
in a single response, they run concurrently via async tool execution.
