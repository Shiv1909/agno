# Sub-Agents Cookbook

Examples and guidance for Agno's `SubAgent` and `AsyncSubAgent` APIs.

---

## What This Covers

This cookbook is for delegation patterns where a parent `Agent` hands work to
child targets with explicit context boundaries.

Supported target types:

- `Agent`
- `Team`

Supported delegation controls:

- history inheritance
- session-state inheritance
- read-only memory inheritance
- optional session-state merge-back
- async task lifecycle management

---

## Core Model

```python
from agno.agent import Agent, AsyncSubAgent, SubAgent
from agno.team import Team

research_team = Team(model=..., members=[...])
coder_agent = Agent(model=..., tools=[...])

coordinator = Agent(
    model=...,
    subagents=[
        SubAgent(
            name="research_team",
            description="Blocking research team",
            target=research_team,
            inherit_history=True,
            history_strategy="last_n",
            history_runs=2,
            inherit_session_state=True,
            session_state_strategy="filtered",
            session_state_keys=["repo_path", "ticket_id"],
        ),
        AsyncSubAgent(
            name="coder",
            description="Background implementation agent",
            target=coder_agent,
            inherit_session_state=True,
            session_state_strategy="filtered",
            session_state_keys=["repo_path", "ticket_id"],
        ),
    ],
)
```

Defaults stay conservative. Nothing is inherited unless you configure it.

| Class | Target | Context | Remote support |
|-------|--------|---------|---------------|
| `SubAgent` | `Agent` or `Team` | sync + async | No |
| `AsyncSubAgent` | `Agent` or `Team` | async only | Yes (`url=`) |

---

## Examples

| File | What it shows |
|------|---------------|
| `01_team_targets.py` | `SubAgent(target=Team(...))` + `AsyncSubAgent(target=Agent(...))` with history and session-state controls |

---

## Prerequisites

`01_team_targets.py` uses Azure OpenAI and WSL:

```bash
pip install agno openai
```

Environment variables:

```bash
set AZURE_OPENAI_API_KEY=your-key
set AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
set AZURE_OPENAI_DEPLOYMENT=gpt-4o
set AZURE_OPENAI_API_VERSION=2024-10-21
```

WSL must be installed for the example backend:

```bash
wsl --install
```

Run:

```bash
python cookbook/95_subagents/01_team_targets.py
```

---

## Decision Guide

```text
Do you need a parent agent to delegate work?
    Yes -> use SubAgent / AsyncSubAgent

Do you need the child target to be multi-member?
    Yes -> pass target=Team(...)

Do you need background execution?
    Yes -> use AsyncSubAgent

Do you need a remote HTTP target?
    Yes -> use AsyncSubAgent(url=...)

Do you need explicit context boundaries?
    Yes -> configure inherit_history / inherit_session_state /
           inherit_memories / inherit_tools / inherit_knowledge
```
