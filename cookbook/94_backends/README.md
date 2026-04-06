# Backends & Sandboxes Cookbook

Examples and guidance for Agno execution backends and `BackendToolkit`.

This cookbook is intentionally backend-focused. Subagent delegation examples
live in [cookbook/95_subagents](../95_subagents).

---

## What This Covers

Backends define where files live and where commands run.

`BackendToolkit` adapts a backend into agent tools such as:

- `ls`
- `read_file`
- `write_file`
- `edit_file`
- `grep`
- `glob`
- `execute` when supported by the backend

---

## Backend Types

| Backend | Execute | Where code runs | Files live |
|---------|---------|----------------|------------|
| `FilesystemBackend` | No | N/A | Local disk |
| `LocalShellBackend` | Yes | Host machine | Local disk |
| `WSLBackend` | Yes | WSL Linux | Windows disk |
| `CompositeBackend` | Optional | Routed to children | Multiple locations |
| `E2BSandbox` | Yes | E2B cloud container | E2B storage |
| `DaytonaSandbox` | Yes | Daytona cloud VM | Daytona storage |
| `ModalSandbox` | Yes | Modal cloud | Modal storage |
| `RunloopSandbox` | Yes | Runloop cloud | Runloop storage |

---

## Basic Usage

```python
from agno.agent import Agent
from agno.backends.filesystem import FilesystemBackend
from agno.tools.backend import BackendToolkit

backend = FilesystemBackend(root="/project", virtual_mode=True)
toolkit = BackendToolkit(backend)

agent = Agent(model=..., tools=[toolkit])
```

Restrict the available backend tools:

```python
toolkit = BackendToolkit(backend, include_tools=["ls", "read_file", "grep", "glob"])
toolkit = BackendToolkit(backend, exclude_tools=["read_file", "download_files"])
```

---

## Examples

| File | What it shows |
|------|---------------|
| `01_filesystem_backend.py` | BackendToolkit + FilesystemBackend for file operations only |
| `02_local_shell_backend.py` | BackendToolkit + LocalShellBackend for local shell execution |
| `03_wsl_backend.py` | BackendToolkit + WSLBackend for Linux commands on Windows |
| `04_e2b_sandbox.py` | BackendToolkit + E2BSandbox for isolated cloud execution |
| `05_composite_backend.py` | CompositeBackend routing paths across multiple backends |

---

## Prerequisites

```bash
pip install agno openai
```

WSL example:

```bash
wsl --install
```

E2B example:

```bash
pip install e2b-code-interpreter
set E2B_API_KEY=your-e2b-key
```

Azure OpenAI examples:

```bash
set AZURE_OPENAI_API_KEY=your-key
set AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
set AZURE_OPENAI_DEPLOYMENT=gpt-4o
set AZURE_OPENAI_API_VERSION=2024-10-21
```

---

## Decision Guide

```text
Do you need shell execution?
    No -> FilesystemBackend
    Yes -> continue

Are you on Windows and need Linux tools?
    Yes -> WSLBackend
    No -> continue

Do you need isolation from your host machine?
    No -> LocalShellBackend
    Yes -> E2BSandbox / DaytonaSandbox / ModalSandbox / RunloopSandbox

Do you need paths routed across multiple locations?
    Yes -> CompositeBackend
```
