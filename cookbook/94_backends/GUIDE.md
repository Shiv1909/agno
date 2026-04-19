# Backends & Sandboxes Guide

This guide is about execution environments and file access, not delegation.
Subagent material lives in [cookbook/95_subagents](../95_subagents).

---

## The Core Problem

A plain LLM agent can generate text, but it cannot interact with the filesystem
or execute commands unless you give it a backend through tools.

Backends solve that by defining where files live and where code runs.

---

## 1. BackendProtocol

Every backend implements the same file operations:

- `ls(path)`
- `read(path, offset, limit)`
- `write(path, content)`
- `edit(path, old, new)`
- `grep(pattern, path)`
- `glob(pattern, path)`
- `upload_files(files)`
- `download_files(paths)`

Sandbox-style backends additionally support:

- `execute(command, timeout)`

This common interface means your agent code does not need to change when you
swap execution environments.

---

## 2. BackendToolkit

`BackendToolkit` exposes backend operations as callable tools on an agent.

```python
from agno.agent import Agent
from agno.tools.backend import BackendToolkit

toolkit = BackendToolkit(backend)
agent = Agent(model=..., tools=[toolkit])
```

You can also restrict capabilities:

```python
BackendToolkit(backend, include_tools=["ls", "read_file", "grep", "glob"])
BackendToolkit(backend, exclude_tools=["read_file", "download_files"])
```

That is the backend-layer least-privilege control.

---

## 3. Backend Types

### FilesystemBackend

File operations only. No command execution.

Use when: the agent needs to inspect or edit files safely.

### LocalShellBackend

Adds command execution on the host machine.

Use when: local trusted development.

### WSLBackend

Runs commands in WSL Linux while keeping files on Windows disk.

Use when: you need Linux tooling on Windows.

### E2BSandbox / DaytonaSandbox / ModalSandbox / RunloopSandbox

Run code in isolated remote environments.

Use when: you need sandboxing or cloud isolation.

### CompositeBackend

Routes different path prefixes to different backends.

Use when: different files belong in different trust zones or storage locations.

---

## 4. Example Routing Pattern

```python
composite = CompositeBackend(
    default=FilesystemBackend(root="/project"),
    routes={
        "/memories/": FilesystemBackend(root="/persistent-notes"),
        "/sandbox/": E2BSandbox(api_key="..."),
    },
)
```

This lets one agent:

- read and write project files
- persist notes somewhere else
- run risky code in an isolated sandbox

through one toolkit interface.

---

## 5. Decision Guide

```text
Do you need shell execution?
    No -> FilesystemBackend
    Yes -> continue

Are you on Windows and need Linux tools?
    Yes -> WSLBackend
    No -> continue

Do you need isolation from the host?
    No -> LocalShellBackend
    Yes -> E2BSandbox / DaytonaSandbox / ModalSandbox / RunloopSandbox

Do you need multiple storage or execution roots?
    Yes -> CompositeBackend
```

---

## 6. Relationship to Subagents

Backends and subagents are separate concerns:

- backends decide where code runs and where files live
- subagents decide how a coordinator delegates work to child targets

If you want delegated `Agent` or `Team` targets with history/session-state
controls, use the dedicated subagent cookbook instead of this backend guide.
