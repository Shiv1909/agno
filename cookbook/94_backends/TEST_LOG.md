# Backends & Sandboxes Cookbook Test Log

Last updated: 2026-04-06

## Test Environment
- OS: Windows 11 Pro with WSL2 (Ubuntu)
- Python: `.venv` (Python 3.13)
- Model: gpt-4o (Azure OpenAI)
- WSL distro: Ubuntu (default)

---

## 01_filesystem_backend.py

**Status:** PASS

**Description:** FilesystemBackend with `virtual_mode=True` against a temp workspace.

**Result:** Agent listed files, read `hello.py`, created `config.json`,
edited `hello.py`, and searched for `Agno` across all files.

---

## 02_local_shell_backend.py

**Status:** PASS

**Description:** LocalShellBackend with `execute()` and file operations in a temp workspace.

**Result:** Agent ran shell commands, executed `script.py`, checked `python3 --version`,
wrote `fib.py`, and ran it successfully.

---

## 03_wsl_backend.py

**Status:** PASS

**Description:** WSLBackend routing `execute()` through WSL Ubuntu while file operations
use Windows disk directly.

**Result:**
- `uname -a` confirmed WSL2 Linux
- `python3` was available in WSL
- files written from Python were visible on Windows disk and accessible in WSL
- Linux tools such as `wc -l` worked correctly

---

## 04_e2b_sandbox.py

**Status:** PENDING

**Description:** E2BSandbox against a live E2B cloud sandbox.

**Prerequisites:** `E2B_API_KEY` environment variable and `pip install e2b-code-interpreter`

**Result:** Not yet run.

---

## 05_composite_backend.py

**Status:** PASS

**Description:** CompositeBackend routing `/workspace/` and `/memories/` to separate
FilesystemBackend instances.

**Result:** Agent listed the synthetic root, read files across both routes,
searched across both backends, and saved output to `/memories/`.
