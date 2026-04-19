"""
Filesystem Backend
==================
Give an agent direct access to the local filesystem.

FilesystemBackend implements the full BackendProtocol:
  ls, read, write, edit, grep, glob, upload_files, download_files

With virtual_mode=True the agent is confined to root — paths like
"../etc/passwd" are blocked automatically at the Python level.

No shell execution is available (no execute tool). Use LocalShellBackend
or WSLBackend if you need that.

Environment variables required:
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_DEPLOYMENT

Run:
    .venvs/demo/bin/python cookbook/94_backends/01_filesystem_backend.py
"""

import os
import tempfile

from agno.agent import Agent
from agno.backends.filesystem import FilesystemBackend
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
# Setup: temp workspace with seed files
# ---------------------------------------------------------------------------
workspace = tempfile.mkdtemp(prefix="agno_fs_demo_")

with open(os.path.join(workspace, "hello.py"), "w") as f:
    f.write('print("Hello, Agno!")\n\ndef greet(name: str) -> str:\n    return f"Hello, {name}!"\n')

with open(os.path.join(workspace, "notes.txt"), "w") as f:
    f.write("Agno backends provide a unified file API.\nCompose them with CompositeBackend.\n")

print(f"Workspace: {workspace}")

# ---------------------------------------------------------------------------
# Backend + Toolkit
# ---------------------------------------------------------------------------
backend = FilesystemBackend(
    root=workspace,
    virtual_mode=True,   # Confine the agent to this directory
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
    print("\n--- List files ---\n")
    agent.print_response("List all files in the root directory.", stream=True)

    print("\n--- Read a file ---\n")
    agent.print_response("Show me the contents of hello.py", stream=True)

    print("\n--- Write a file ---\n")
    agent.print_response(
        "Create a file called config.json with a JSON object containing key 'version' set to '1.0'.",
        stream=True,
    )

    print("\n--- Edit a file ---\n")
    agent.print_response(
        "In hello.py, replace 'Hello, Agno!' with 'Hello, World!'.",
        stream=True,
    )

    print("\n--- Search files ---\n")
    agent.print_response("Search for the word 'Agno' across all files.", stream=True)

    print(f"\nWorkspace: {workspace}")
