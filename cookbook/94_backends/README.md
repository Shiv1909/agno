# Backends, Sandboxes & Sub-Agents

Three independent, composable capabilities for Agno agents.

---

## The Three Capabilities

### 1. Sub-Agents — Agents that delegate to other agents

Define specialised child agents directly on a parent agent. Each sub-agent
becomes its own named tool so the LLM calls `researcher(task=...)` directly.

```python
from agno.agent import Agent, SubAgent, AsyncSubAgent

agent = Agent(
    model=...,
    subagents=[
        SubAgent(
            name="researcher",
            description="Deep-dive research specialist",
            tools=[DuckDuckGoTools()],   # any tools — no backend required
        ),
        AsyncSubAgent(
            name="coder",
            description="Python coding assistant",
            tools=[PythonTools()],
            # url="https://remote-coder.example.com"  # optional HTTP transport
        ),
    ],
)
```

**No backend required.** Sub-agents can have web search tools, database tools,
or any other Agno tools — backends are entirely optional.

| Class | Context | Remote support |
|-------|---------|---------------|
| `SubAgent` | sync + async | No |
| `AsyncSubAgent` | async only | Yes (`url=`) |

---

### 2. BackendToolkit — File and shell tools for agents

Wraps any backend and registers its operations as callable tools on an agent.
The agent can then read, write, edit, search, and execute through a uniform interface.

```python
from agno.tools.backend import BackendToolkit

toolkit = BackendToolkit(backend)           # 8 tools registered automatically
agent = Agent(tools=[toolkit])
```

Restrict what the agent can do:

```python
# Read-only agent
toolkit = BackendToolkit(backend, include_tools=["ls", "read_file", "grep", "glob"])

# No reading of sensitive files
toolkit = BackendToolkit(backend, exclude_tools=["read_file", "download_files"])
```

**Requires a backend** (see below). Does not require sub-agents.

---

### 3. Backends & Sandboxes — Where files live and code runs

The execution environment. Every backend implements the same `BackendProtocol`
(8 file ops). Sandbox backends additionally expose `execute()`.

| Backend | Execute | Where code runs | Files live |
|---------|---------|----------------|------------|
| `FilesystemBackend` | No | N/A | Local disk |
| `LocalShellBackend` | Yes | Host machine | Local disk |
| `WSLBackend` | Yes | WSL Linux (bash) | Windows disk |
| `CompositeBackend` | Optional | Routes to children | Multiple locations |
| `E2BSandbox` | Yes | E2B cloud container | E2B storage |
| `DaytonaSandbox` | Yes | Daytona cloud VM | Daytona storage |
| `ModalSandbox` | Yes | Modal cloud | Modal storage |
| `RunloopSandbox` | Yes | Runloop cloud | Runloop storage |

**Usable standalone.** A backend can exist without BackendToolkit or sub-agents.
BackendToolkit is just the adapter that turns a backend into agent tools.

---

## Using Them Together

The three capabilities compose freely — use any combination:

```python
# Sub-agents + backend tools (most common)
coordinator = Agent(
    model=...,
    subagents=[
        SubAgent(name="coder", description="...", tools=[BackendToolkit(WSLBackend(...))]),
        SubAgent(name="analyst", description="...", tools=[BackendToolkit(WSLBackend(...))]),
    ],
    tools=[BackendToolkit(WSLBackend(...))],  # coordinator can also use backend directly
)

# Backend tools only — no sub-agents
agent = Agent(model=..., tools=[BackendToolkit(FilesystemBackend(root="/project"))])

# Sub-agents only — no backend
agent = Agent(
    model=...,
    subagents=[SubAgent(name="researcher", description="...", tools=[DuckDuckGoTools()])],
)

# CompositeBackend — split across multiple backends
composite = CompositeBackend(
    default=FilesystemBackend(root="/project"),
    routes={"/memories/": FilesystemBackend(root="/notes")},
)
agent = Agent(model=..., tools=[BackendToolkit(composite)])
```

---

## Examples

| File | What it shows |
|------|--------------|
| `01_filesystem_backend.py` | BackendToolkit + FilesystemBackend — file ops only, no shell |
| `02_local_shell_backend.py` | BackendToolkit + LocalShellBackend — file ops + shell on host |
| `03_wsl_backend.py` | BackendToolkit + WSLBackend — shell runs in WSL Linux, files on Windows disk |
| `04_e2b_sandbox.py` | BackendToolkit + E2BSandbox — isolated cloud execution |
| `05_composite_backend.py` | BackendToolkit + CompositeBackend — path routing across backends |
| `06_subagent_parallel.py` | SubAgent API — coordinator with coder + analyst sub-agents on WSLBackend |

---

## Prerequisites

```bash
# All examples
pip install agno openai

# WSL examples (03, 06)
# WSL must be installed: wsl --install

# E2B sandbox (04)
pip install e2b-code-interpreter
```

## Environment Variables

All examples use Azure OpenAI:

```bash
set AZURE_OPENAI_API_KEY=your-key
set AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
set AZURE_OPENAI_DEPLOYMENT=gpt-4o
set AZURE_OPENAI_API_VERSION=2024-10-21   # optional
```

E2B example additionally needs:
```bash
set E2B_API_KEY=your-e2b-key
```

---

## Decision Guide

```
Do you need agents to delegate tasks to other agents?
    Yes → use SubAgent / AsyncSubAgent on Agent(subagents=[...])

Do you need agents to read/write files or run shell commands?
    Yes → use BackendToolkit(backend)

Which backend?
    No shell needed          → FilesystemBackend
    Shell, Windows + Linux   → WSLBackend
    Shell, local only        → LocalShellBackend
    Shell, isolated/cloud    → E2BSandbox / DaytonaSandbox / ModalSandbox / RunloopSandbox
    Multiple locations       → CompositeBackend
```
