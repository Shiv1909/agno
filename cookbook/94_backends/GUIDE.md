# Backends, Sandboxes & Sub-Agents — Why and How

A practical guide to understanding what these systems do, why they exist,
and how they make AI agents genuinely useful in real workflows.

---

## The Core Problem

A vanilla LLM agent can only do one thing: generate text. It cannot read
a file on your disk, run a Python script, check whether a test passed, or
install a package. The moment you need the agent to actually *do* something
in the real world — touch files, execute code, call shell tools — you need
a backend.

Without backends, every agent workflow looks like this:

```
User: "Refactor my code"
Agent: "Here is how I would refactor it: ..."   ← just words
```

With backends:

```
User: "Refactor my code"
Agent: reads the file → makes the edit → runs the tests → reports results
```

That is the entire motivation. Backends turn agents from text generators
into workers that can actually complete tasks.

---

## 1. BackendProtocol — The Unified Interface

### What it is

`BackendProtocol` is a standard interface that every backend implements.
It defines 8 file operations:

| Operation | What it does |
|-----------|-------------|
| `ls(path)` | List files and directories |
| `read(path, offset, limit)` | Read file content, paginated |
| `write(path, content)` | Create a new file |
| `edit(path, old, new)` | Replace exact text in a file |
| `grep(pattern, path)` | Search file contents |
| `glob(pattern, path)` | Find files by name pattern |
| `upload_files(files)` | Transfer files into the backend |
| `download_files(paths)` | Transfer files out of the backend |

`SandboxBackendProtocol` extends this with:

| Operation | What it does |
|-----------|-------------|
| `execute(command, timeout)` | Run a shell command |
| `id` | Unique identifier for this backend instance |

### Why this matters

Because the interface is the same for every backend, your agent code never
changes when you switch execution environments. You write the agent once and
swap the backend:

```python
# Development: local WSL
backend = WSLBackend(root="/workspace")

# Production: isolated cloud sandbox
backend = E2BSandbox(api_key="...")

# The agent code is identical either way
agent = Agent(tools=[BackendToolkit(backend)])
```

This is the same principle as database drivers — you write SQL once and
switch between SQLite, Postgres, and MySQL by changing the connection string.
Backends do the same for agent execution environments.

---

## 2. BackendToolkit — How Agents Use Backends

### What it is

`BackendToolkit` is the adapter layer that takes any backend and registers
its operations as callable tools for an Agno agent. When the LLM decides
it needs to read a file, it calls `read_file`. When it needs to run a
command, it calls `execute`. The toolkit handles the call, formats the
result, and returns it to the model as a tool message.

```
Agent LLM → decides to call "write_file"
    → BackendToolkit.write_file("/stats.py", "import random...")
    → backend.write("/stats.py", "import random...")
    → disk / WSL / E2B cloud (depending on which backend is wired in)
    → "Written: /stats.py" returned to LLM
```

### Why this matters

Without `BackendToolkit`, you would have to manually write a separate tool
function for every operation on every backend. With it, you get 8 or 9
tools registered automatically just by passing your backend in:

```python
# 8 tools registered automatically (9 if backend supports execute)
toolkit = BackendToolkit(backend)
```

You can also restrict which tools the agent can use:

```python
# Read-only agent — can only inspect, cannot modify
toolkit = BackendToolkit(backend, include_tools=["ls", "read_file", "grep", "glob"])

# Write-only agent — cannot read sensitive files
toolkit = BackendToolkit(backend, exclude_tools=["read_file", "download_files"])
```

This gives you fine-grained control over what each agent can do.

---

## 3. The Sandbox Backends — Where Code Actually Runs

### FilesystemBackend — Read and write, nothing more

The simplest backend. Gives the agent direct access to files on disk.
No shell execution. Safe for agents that only need to read, analyse, and
write files — no risk of the agent accidentally running destructive commands.

**Use when:** The agent's job is purely file manipulation — refactoring code,
editing configs, generating documentation, analysing logs.

```python
backend = FilesystemBackend(root="/my/project", virtual_mode=True)
```

`virtual_mode=True` means the agent sees `/` as the root of your project.
A path traversal attempt like `../../etc/passwd` is blocked at the Python
level before it ever touches the filesystem.

---

### LocalShellBackend — Full host access, no isolation

Extends FilesystemBackend with `execute()` via `subprocess.run(shell=True)`
on your machine. The agent can run any command your user account can run.

**Use when:** You are building a local coding assistant for your own machine
and you trust the agent. Personal dev tools, local CI scripts, quick
automation tasks.

**Do not use when:** Multiple users, production servers, or any situation
where the agent might receive untrusted input. There is no isolation —
a compromised agent can delete files, exfiltrate secrets, install malware.

```python
backend = LocalShellBackend(root="/workspace", virtual_mode=True, inherit_env=False)
```

`inherit_env=False` means the agent's shell starts with only `PATH` —
your API keys, tokens, and secrets in environment variables are not
visible to the commands it runs.

---

### WSLBackend — Linux environment on Windows, no cloud needed

Routes `execute()` through WSL (Windows Subsystem for Linux) while keeping
file operations on the Windows disk via Python's direct I/O.

```
File ops → Python reads/writes Windows disk directly   (fast, no WSL overhead)
execute() → wsl bash -c "cd /mnt/c/workspace && command"  (real Linux)
```

**Why this is useful:**

Most developer tools and most cloud infrastructure run on Linux. When an
agent needs to run `apt install`, use GNU `grep`, call `curl` with Linux
flags, or execute a bash script that assumes a Linux environment — it
cannot do that on Windows cmd.exe. WSLBackend solves this without needing
a cloud account or a Docker daemon.

The files the agent writes are normal Windows files. You can open them in
VS Code, commit them to git, run them with your Windows Python — they are
not locked inside a VM or container.

**Use when:** You are on Windows, need Linux tools, do not want cloud costs,
and do not need network isolation.

```python
backend = WSLBackend(root="C:/workspace", distro="Ubuntu", virtual_mode=True)
```

**Key technical detail:** WSL's RPC service requires the full Windows
environment (`SYSTEMROOT`, `WINDIR`, etc.) to be available in the subprocess.
If you strip the environment (as `inherit_env=False` does for LocalShellBackend),
WSL cannot connect to its own service. WSLBackend solves this by always
passing `env=None` to subprocess (inheriting the full Windows env) and
instead injecting your custom variables inside the bash script itself.

---

### E2BSandbox, DaytonaSandbox, ModalSandbox, RunloopSandbox — Real isolation

These backends run code in a remote Linux container that is completely
isolated from your machine. The agent's commands run on someone else's
hardware. If the agent makes a mistake, deletes files, or runs malicious
code, your machine is unaffected.

**BaseSandbox** (the base class all four inherit from) is the clever part.
It implements all 8 file operations by running Python3 scripts *inside the
container* via `execute()`. This means a concrete sandbox only needs to
implement 3 things:

```python
class MySandbox(BaseSandbox):
    def execute(self, command, *, timeout=None): ...   # run a command
    def upload_files(self, files): ...                 # transfer bytes in
    def download_files(self, paths): ...               # transfer bytes out
    # Everything else — ls, read, write, edit, grep, glob — is inherited
```

**Use when:** You are building a product, running untrusted code, building
a multi-tenant system, or need reproducible Linux environments.

---

### CompositeBackend — Split your filesystem across multiple backends

Routes path prefixes to different backends. The agent does not know or care
that it is talking to multiple backends — it just uses paths.

```python
composite = CompositeBackend(
    default=FilesystemBackend(root="/project"),
    routes={
        "/memories/": FilesystemBackend(root="/persistent-notes"),
        "/sandbox/":  E2BSandbox(api_key="..."),
    }
)
```

Now the agent can:
- Read project files at `/project/...`
- Save persistent notes at `/memories/...` (survives restarts)
- Execute dangerous code at `/sandbox/...` (isolated cloud container)

All through the same `BackendToolkit`. The routing is transparent.

**Practical patterns:**
- Route `/memories/` to a network drive so agent notes persist across machines
- Route `/output/` to an encrypted backend for sensitive results
- Route `/tmp/` to an in-memory backend for ephemeral scratch files
- Route `/prod/` to a read-only backend to prevent accidental writes

---

## 4. SubAgent / AsyncSubAgent — Agents That Delegate to Other Agents

### What it is

`SubAgent` and `AsyncSubAgent` are defined directly on the parent `Agent` via
`subagents=[...]`. Each one becomes its own **named tool** on the coordinator —
the LLM calls `coder(task=...)` or `researcher(task=...)` directly, not a
generic dispatcher function.

```python
coordinator = Agent(
    model=...,
    subagents=[
        SubAgent(
            name="researcher",
            description="Deep-dive research specialist",
            tools=[DuckDuckGoTools()],
        ),
        SubAgent(
            name="coder",
            description="Python coding assistant",
            tools=[BackendToolkit(WSLBackend(...))],
        ),
    ],
)
```

```
Coordinator receives: "Research X and then write code to process it"
    → calls researcher(task="research X") → gets research summary
    → calls coder(task="implement X")     → gets code and output
    → combines both → returns unified answer
```

### SubAgent vs AsyncSubAgent

| Class | Sync context | Async context | Remote HTTP |
|-------|-------------|---------------|-------------|
| `SubAgent` | `agent.run()` | `await agent.arun()` | No |
| `AsyncSubAgent` | Not available | `await agent.arun()` | Yes (`url=`) |

Use `SubAgent` for most cases — it works in both `run()` and `arun()`.
Use `AsyncSubAgent` when you need a remote HTTP sub-agent or async-only execution.

### Why this is better than one big agent

| One agent does everything | Coordinator + sub-agents |
|--------------------------|--------------------------|
| Long context window fills up quickly | Each sub-agent has a clean, focused context |
| Conflicting instructions | Each sub-agent has a single clear role |
| Single point of failure | Sub-agents can fail independently |
| Hard to parallelise | LLM makes parallel tool calls in one response |
| One model for all tasks | Different models per sub-agent |

### The isolation guarantee

Sub-agents do NOT share session state with the parent and do NOT inherit
parent tools:

- **No shared session state**: The sub-agent starts fresh each call.
  Prevents context poisoning and makes behaviour predictable.

- **No inherited tools**: The coordinator might have dangerous tools
  (like `execute`). A sub-agent only gets the tools you explicitly configure.
  Least-privilege by default.

### Caching

Sub-agents are created once and reused. The `Agent` instance is built on the
first call and cached internally. Subsequent calls reuse the same instance, so:

- No startup cost on repeated calls
- Memory and knowledge base state is preserved within a session
- If `db=` is configured, conversation history accumulates across calls

### Parallel execution

The LLM naturally calls multiple sub-agent tools in a single response when
asked to do independent tasks simultaneously:

```python
coordinator.print_response(
    "Do both of these:\n"
    "1. Ask the coder to write primes.py\n"
    "2. Ask the analyst to search for 'mean' in all files"
)
# → LLM emits two tool calls in one response: coder(...) and analyst(...)
# → Both run and results are returned together
```

---

## 5. How Everything Fits Together

```
┌─────────────────────────────────────────────────────┐
│                  Coordinator Agent                  │
│  model: gpt-4o                                      │
│  subagents: [SubAgent("coder"), SubAgent("analyst")]│
│  tools: [BackendToolkit]                            │
└───────────────┬─────────────────────────────────────┘
                │ tool calls: coder(...) / analyst(...)
    ┌───────────┴────────────┐
    │                        │
┌───▼──────────┐    ┌────────▼──────────┐
│ coder agent  │    │  analyst agent    │
│ model: gpt-4o│    │  model: gpt-4o    │
│ tools:       │    │  tools:           │
│ BackendToolkit    │  BackendToolkit   │
└───────┬──────┘    └─────────┬─────────┘
        │                     │
        └──────────┬──────────┘
                   │ both point to
          ┌────────▼────────┐
          │   WSLBackend    │
          │  root=/workspace│
          └────────┬────────┘
                   │
      ┌────────────┴─────────────��┐
      │                           │
  file ops                    execute()
  (Python I/O                 (wsl bash -c "...")
  on Windows disk)            (runs in Linux)
```

Every layer is independently swappable:
- Swap `WSLBackend` for `E2BSandbox` → same agent, cloud execution
- Swap `gpt-4o` for `claude-sonnet-4-6` on any sub-agent independently
- Add a `db=SqliteDb(...)` to any sub-agent for persistent memory
- Add `knowledge=PDFKnowledgeBase(...)` to the analyst for RAG
- Wrap the composite with a `CompositeBackend` to add a `/memories/` route

---

## 6. Decision Guide — Which Backend to Use

```
Do you need shell execution?
    No  → FilesystemBackend
    Yes → continue

Are you on Windows and need Linux tools?
    Yes, and no cloud needed  → WSLBackend
    No                        → continue

Do you need isolation from your host machine?
    No  → LocalShellBackend  (local dev only, trusted context)
    Yes → E2BSandbox / DaytonaSandbox / ModalSandbox / RunloopSandbox

Do you need to split files across multiple locations?
    Yes → CompositeBackend (wraps any combination of the above)

Do you need multiple specialised agents?
    Yes → SubAgentTools (each sub-agent gets its own backend config)
```

---

## 7. Real-World Use Cases

### Coding assistant
```
WSLBackend + BackendToolkit
Agent reads your code → edits files → runs tests → reports results
All in Linux, files stay on your Windows disk
```

### Automated code review pipeline
```
CompositeBackend:
  /src/      → FilesystemBackend (read-only: exclude write/edit)
  /reports/  → FilesystemBackend (write-only: exclude read)
Sub-agents: security-reviewer, style-checker, test-generator
Each sub-agent reads from /src/, writes findings to /reports/
```

### Research + implementation workflow
```
SubAgentTools:
  researcher  → no backend, just web search tools
  coder       → WSLBackend, writes and runs code
  tester      → WSLBackend (read-only src), runs test suite
Coordinator orchestrates all three in sequence or parallel
```

### Multi-tenant SaaS
```
Each user session gets its own E2BSandbox
BackendToolkit wraps it — agent has full Linux execution
Sandbox is destroyed after session — zero host contamination
User A's code never touches User B's environment
```

### Agent with persistent memory
```
CompositeBackend:
  /workspace/ → E2BSandbox (ephemeral, isolated execution)
  /memories/  → FilesystemBackend on persistent network drive
Agent writes code to /workspace/, saves learnings to /memories/
On next session, memories are loaded; workspace starts fresh
```
