"""
Local Shell Backend
===================
Combines file operations with shell command execution on the host machine.

LocalShellBackend extends FilesystemBackend with execute() via
subprocess.run(shell=True). The agent can read/write files AND run
shell commands in the same workspace.

WARNING: execute() runs real commands on your machine with no isolation.
         Use virtual_mode=True to confine FILE OPS to the workspace.
         Shell commands can still access any path on the system.

Environment variables required:
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_DEPLOYMENT

Run:
    .venvs/demo/bin/python cookbook/94_backends/02_local_shell_backend.py
"""

import os
import tempfile

from agno.agent import Agent
from agno.backends.local_shell import LocalShellBackend
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
# Setup: temp workspace
# ---------------------------------------------------------------------------
workspace = tempfile.mkdtemp(prefix="agno_shell_demo_")

with open(os.path.join(workspace, "script.py"), "w") as f:
    f.write("import sys\nprint('Python version:', sys.version)\nprint('sum(1..10)=', sum(range(1, 11)))\n")

print(f"Workspace: {workspace}")

# ---------------------------------------------------------------------------
# Backend + Toolkit
# ---------------------------------------------------------------------------
backend = LocalShellBackend(
    root=workspace,
    virtual_mode=True,
    inherit_env=False,   # Clean environment — only PATH is inherited
)

toolkit = BackendToolkit(backend)

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
agent = Agent(
    model=model,
    tools=[toolkit],
    markdown=True,
    debug_mode=True,
)

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n--- Execute a shell command ---\n")
    agent.print_response("Run: echo 'Hello from shell'", stream=True)

    print("\n--- Run a Python script ---\n")
    agent.print_response("Execute script.py and show me the output.", stream=True)

    print("\n--- Check Python version ---\n")
    agent.print_response("What Python version is available? Run python3 --version.", stream=True)

    print("\n--- Write and run ---\n")
    agent.print_response(
        "Write a file called fib.py that prints the first 10 Fibonacci numbers, then run it with python3.",
        stream=True,
    )

    print(f"\nWorkspace: {workspace}")
