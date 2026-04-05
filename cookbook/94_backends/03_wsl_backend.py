"""
WSL Backend
===========
Route shell execution through Windows Subsystem for Linux (WSL)
while keeping file operations on the Windows disk.

WSLBackend extends LocalShellBackend:
  - File ops (ls, read, write, edit, grep, glob) → direct Windows disk I/O
  - execute() → wsl bash -c "cd /mnt/<drive>/... && <command>"

The agent gets a real Linux environment (apt, bash, python3, uname, etc.)
without needing a cloud sandbox. Files written by the agent are visible
from both Windows and WSL at /mnt/c/... paths.

Prerequisites:
  - WSL installed and at least one distro set up (wsl --install)
  - Verify: wsl uname -a

Environment variables required:
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_DEPLOYMENT

Run:
    python cookbook/94_backends/03_wsl_backend.py
"""

import os
import tempfile

from agno.agent import Agent
from agno.backends.wsl import WSLBackend
from agno.models.azure import AzureOpenAI
from agno.tools.backend import BackendToolkit

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = AzureOpenAI(
    id=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
)

# ---------------------------------------------------------------------------
# Setup: temp workspace (Windows path, accessible in WSL via /mnt/c/...)
# ---------------------------------------------------------------------------
workspace = tempfile.mkdtemp(prefix="agno_wsl_demo_")
print(f"Workspace (Windows): {workspace}")

# ---------------------------------------------------------------------------
# Backend + Toolkit
# ---------------------------------------------------------------------------
backend = WSLBackend(
    root=workspace,
    # distro="Ubuntu",   # Uncomment to target a specific WSL distro
    # wsl_user="myuser", # Uncomment to run as a specific Linux user
    virtual_mode=True,   # Confine file ops to workspace; execute() has full Linux access
    inherit_env=False,
)

toolkit = BackendToolkit(backend)

print(f"Backend ID: {backend.id}")

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
agent = Agent(
    model=model,
    tools=[toolkit],
    instructions=[
        "You are running inside a WSL Ubuntu Linux environment.",
        "Always use 'python3' to run Python scripts, never 'python'.",
        "You can use Linux tools: apt, bash, grep, awk, uname, etc.",
    ],
    markdown=True,
    debug_mode=True,
)

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n--- Verify WSL environment ---\n")
    agent.print_response("Run uname -a to confirm we are inside WSL.", stream=True)

    print("\n--- Linux tools ---\n")
    agent.print_response(
        "Run: echo 'Hello from WSL' && whoami && pwd",
        stream=True,
    )

    print("\n--- Write and run a Python script inside WSL ---\n")
    agent.print_response(
        "Write a file called stats.py that generates 10 random numbers, "
        "computes their mean and standard deviation using the statistics module, "
        "prints the results, and saves a summary line to results.txt. "
        "Then run it with python3.",
        stream=True,
    )

    print("\n--- Read the output file ---\n")
    agent.print_response("Read the contents of results.txt", stream=True)

    print("\n--- Use a Linux tool ---\n")
    agent.print_response(
        "Use the 'wc -l' command to count lines in stats.py.",
        stream=True,
    )

    print(f"\nWorkspace: {workspace}")
