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
    subagent_config=SubAgentConfig(),
)
```

## SubAgentConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inherit_parent_tools` | bool | False | Give subagents all of the parent's tools. |
| `allowed_tools` | List[str] | None | Tool names the LLM may request at spawn time. `None` means all parent tools are eligible. |
| `allow_tool_selection` | bool | True | Let the LLM choose which tools to give each subagent at spawn time. |
| `context_heavy_tools` | List[str] | None | Tool names that return large outputs. Injected into guidance as "always route via spawn_agent". |
| `model_tiers` | Dict[str, str] | None | Map of tier label → model ID (e.g. `{"fast": "gpt-4o-mini", "standard": "gpt-4o"}`). |
| `allow_model_tier_selection` | bool | False | Expose `model_tier` parameter in `spawn_agent` so the LLM can pick a cost-appropriate model. |
| `tier_hints` | Dict[str, str] | None | Optional map of tier label → hint text shown to the LLM. Merged with built-in defaults for `fast`, `standard`, `powerful`. |
| `inject_session_state` | bool | False | Embed parent `session_state` as read-only JSON in the subagent's context. |
| `max_concurrent` | int | 5 | Maximum simultaneous subagents (enforced via semaphore). |

## How It Works

1. `enable_dynamic_subagents=True` adds a `spawn_agent` tool to the agent/team.
2. The LLM calls `spawn_agent(role, instructions, task, ...)` during its tool loop.
3. An ephemeral `Agent` is built from the `subagent_template` (if set) via deep-copy with per-spawn overrides, or from the parent's model if no template is provided.
4. The subagent runs with `db=None` — no persistence, no session created.
5. The result string is returned as the tool output into the parent's message history.
6. The subagent object goes out of scope and is garbage collected.

## Context Isolation

Subagent tool outputs stay inside the subagent's own context — they **never** appear in the parent's message history. This is architecturally different from `CompressionManager`, which compresses tool results that are already in context after the fact. Subagent isolation prevents them from entering the parent's context in the first place.

## Parallel Spawning

Parallel spawning works automatically: if the LLM calls `spawn_agent` multiple times in a single response, they run concurrently via async tool execution. The `max_concurrent` parameter caps simultaneous subagents.

## Tool Delegation

When `allow_tool_selection=True`, the LLM may request specific parent tools for each subagent via the `tools` parameter. Only the named `Function` objects are extracted from the parent's toolkits — the entire toolkit is never delegated wholesale, preserving the whitelist guarantee.

## Configuring the Subagent Template

Use `subagent_template` to configure what spawned agents look like (model, markdown, knowledge, etc.). `SubAgentConfig` controls spawn-time *policy* — it intentionally does not duplicate Agent configuration fields.

```python
from agno.agent import Agent, SubAgentConfig
from agno.models.openai import OpenAIChat

template = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    markdown=True,
)

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    enable_dynamic_subagents=True,
    subagent_template=template,
    subagent_config=SubAgentConfig(
        context_heavy_tools=["query_db", "read_file"],
        model_tiers={"fast": "gpt-4o-mini", "standard": "gpt-4o"},
        allow_model_tier_selection=True,
        max_concurrent=3,
    ),
)
```

## Running the Examples

```bash
# Set up the demo environment first
./scripts/demo_setup.sh

# Run examples
.venvs/demo/bin/python cookbook/95_dynamic_subagent/01_basic.py
.venvs/demo/bin/python cookbook/95_dynamic_subagent/02_with_tools.py
.venvs/demo/bin/python cookbook/95_dynamic_subagent/03_parallel.py
.venvs/demo/bin/python cookbook/95_dynamic_subagent/04_team.py
```
