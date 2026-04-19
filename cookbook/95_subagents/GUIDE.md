# Sub-Agents Guide

This guide focuses on delegation, not execution backends.

---

## What Sub-Agents Are

`SubAgent` and `AsyncSubAgent` let a parent `Agent` delegate work to a named
child target. Each sub-agent becomes its own callable tool on the coordinator.

The target can be:

- an `Agent`
- a `Team`

That means delegation patterns such as these are now first-class:

- `Agent -> Agent`
- `Agent -> Team`
- `Agent -> Async Agent`
- `Agent -> Async Team`

---

## Why `target=` Matters

Earlier usage was centered on child agents. The newer model uses `target=`
because the delegated child may be either an `Agent` or a `Team`.

```python
SubAgent(
    name="research_team",
    description="Blocking research team",
    target=Team(...),
)
```

The older `agent=` alias remains supported for backward compatibility, but
`target=` is clearer and should be preferred in examples.

---

## Delegation Controls

Sub-agent boundaries are explicit. Parent context does not cross into the
child target unless you opt in.

### History

```python
SubAgent(
    ...,
    inherit_history=True,
    history_strategy="last_n",
    history_runs=3,
)
```

Strategies:

- `none`
- `last_n`
- `full`
- `summary`

### Session state

```python
SubAgent(
    ...,
    inherit_session_state=True,
    session_state_strategy="filtered",
    session_state_keys=["repo_path", "ticket_id"],
)
```

Strategies:

- `none`
- `copy`
- `filtered`

### Memory

Phase-1 support is conservative:

```python
SubAgent(
    ...,
    inherit_memories=True,
    memory_strategy="read_only",
)
```

Supported now:

- `none`
- `read_only`

Deferred:

- `copy`
- `shared`

### Tool and knowledge inheritance

Disabled by default:

```python
SubAgent(
    ...,
    inherit_tools=False,
    inherit_knowledge=False,
)
```

This keeps child capabilities least-privilege unless you explicitly widen them.

### Merge-back

All subagents return normalized text back to the coordinator.

Optional session-state merge-back is supported:

```python
SubAgent(
    ...,
    inherit_session_state=True,
    session_state_strategy="filtered",
    session_state_keys=["ticket_id"],
    merge_back_result=True,
    result_merge_strategy="session_state",
)
```

---

## AsyncSubAgent Lifecycle

When a parent agent defines any `AsyncSubAgent`, it automatically gets:

- `check_async_task(task_id)`
- `update_async_task(task_id, instructions)`
- `cancel_async_task(task_id)`
- `list_async_tasks()`

These lifecycle tools work for both agent and team targets.

`update_async_task(...)` now restarts against the stored target reference, not
an agent-only reference, so team targets can be restarted too.

---

## Team Targets

`target=Team(...)` is useful when the delegated unit should coordinate multiple
specialists before returning.

```python
analysis_team = Team(
    model=...,
    members=[
        Agent(name="reader", model=..., tools=[...]),
        Agent(name="summarizer", model=..., tools=[...]),
    ],
)

coordinator = Agent(
    model=...,
    subagents=[
        SubAgent(
            name="analysis_team",
            description="Blocking analysis team",
            target=analysis_team,
        )
    ],
)
```

The coordinator does not need to care whether the delegated target is an
agent or a team. It calls the named tool and receives normalized text back.

---

## Example Pattern

The example in [README.md](C:/Users/ShivanshMital/OneDrive%20-%20McLaren%20Strategic%20Solutions%20US%20Inc/Desktop/sandbox_isolation/agno/cookbook/95_subagents/README.md) uses:

- a blocking `SubAgent(target=Team(...))` for analysis
- a background `AsyncSubAgent(target=Agent(...))` for implementation
- filtered session-state inheritance
- inherited history for the team
- WSL-backed file and execution tools assigned to the delegated targets

The backend is incidental there. The primary concept is delegation.
