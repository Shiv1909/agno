# Backends & Sandboxes Cookbook Test Log

Last updated: 2026-04-05

## Test Environment
- OS: Windows 11 Pro with WSL2 (Ubuntu)
- Python: `.venv` (Python 3.13)
- Model: gpt-4o (Azure OpenAI)
- WSL distro: Ubuntu (default)

---

## 01_filesystem_backend.py

**Status:** PASS

**Description:** FilesystemBackend with virtual_mode=True against a temp workspace.
Exercises ls, read, write, edit, and grep via BackendToolkit.

**Result:** Agent correctly listed files, read hello.py, created config.json,
edited hello.py replacing text, and searched for 'Agno' across all files.
File ops confined to workspace — no shell execution available.

---

## 02_local_shell_backend.py

**Status:** PASS

**Description:** LocalShellBackend with execute() and file ops in a temp workspace.
Exercises shell command execution and write + run flow.

**Result:** Agent ran echo, executed script.py, checked python3 --version,
wrote fib.py and ran it. Commands executed on host machine via subprocess.run.

---

## 03_wsl_backend.py

**Status:** PASS

**Description:** WSLBackend routing execute() through WSL Ubuntu while file ops
use Windows disk directly. Exercises uname, shell commands, python3 script
execution inside WSL, and Linux tools.

**Result:**
- `uname -a` confirmed WSL2 Linux kernel (6.6.87.2-microsoft-standard-WSL2)
- `python3` available in WSL, scripts executed correctly
- Files written via write_file visible on Windows disk and accessible in WSL
- `wc -l` Linux tool worked correctly
- Key fix: pass `env=None` to subprocess so WSL RPC service gets full Windows env

---

## 04_e2b_sandbox.py

**Status:** PENDING

**Description:** E2BSandbox against a live E2B cloud sandbox. Exercises execute,
write, read, file ops, and pip install flow.

**Prerequisites:** E2B_API_KEY environment variable, `pip install e2b-code-interpreter`

**Result:** Not yet run.

---

## 05_composite_backend.py

**Status:** PASS

**Description:** CompositeBackend routing /workspace/ and /memories/ to separate
FilesystemBackend instances. Exercises cross-backend ls, read, search, and write.

**Result:** Agent listed root and saw both /workspace/ and /memories/ as synthetic
directories. Read app.py from workspace, read project_notes.md from memories,
searched 'demo' across both backends, saved session_summary.md to memories.

---

## 06_subagent_parallel.py

**Status:** PENDING

**Description:** SubAgent (blocking) + AsyncSubAgent (non-blocking / fire-and-forget).
- SubAgent("analyst") — coordinator blocks until analyst finishes
- AsyncSubAgent("coder") — returns task_id immediately; coder runs in background
- Lifecycle tools auto-registered: check_async_task, update_async_task,
  cancel_async_task, list_async_tasks
- Tests: background launch, continued conversation during execution,
  result retrieval via check, mid-flight update via update_async_task

**Notes:**
- AsyncSubAgent uses asyncio.create_task() — truly non-blocking fire-and-forget.
- AsyncTaskManager tracks all tasks; lifecycle tools close over shared manager.
- update_async_task cancels current asyncio.Task and relaunches with new instructions.
- Lifecycle tools are async-only (registered in toolkit.async_functions).
- Agent must use python3 not python in WSL Ubuntu (no python alias by default).

---
