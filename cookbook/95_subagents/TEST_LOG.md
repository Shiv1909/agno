# Sub-Agents Cookbook Test Log

Last updated: 2026-04-06

## 01_team_targets.py

**Status:** PENDING

**Description:** Subagent delegation with explicit boundaries.
- `SubAgent(target=Team(...))` for blocking delegation to a specialist team
- `AsyncSubAgent(target=Agent(...))` for background implementation work
- filtered session-state inheritance
- inherited history for the blocking team
- lifecycle tools: `check_async_task`, `update_async_task`, `cancel_async_task`, `list_async_tasks`

**Notes:**
- `SubAgent` and `AsyncSubAgent` accept `target=` as either `Agent` or `Team`.
- `target=` should be preferred over the older `agent=` alias in new cookbook material.
- Delegation boundaries are explicit; history and session state are not inherited unless configured.
- `memory_strategy="read_only"` is supported; `copy` and `shared` remain deferred.
- This example uses WSL for file and command execution, but the cookbook topic is delegation, not backends.
