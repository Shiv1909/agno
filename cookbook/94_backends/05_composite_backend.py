"""
Composite Backend
=================
Route different path prefixes to different backends.

CompositeBackend lets you split a virtual filesystem across multiple
backends. In this example:
  /workspace/  → FilesystemBackend (project files on local disk)
  /memories/   → A separate FilesystemBackend (persistent notes store)

The agent reads and writes both areas transparently. A common production
pattern is routing /memories/ to a network-backed or encrypted backend
while /workspace/ stays on local disk.

ls("/") aggregates both backends and shows synthetic directory entries.
grep(path=None) fans out to all backends automatically.

Environment variables required:
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_DEPLOYMENT

Run:
    .venvs/demo/bin/python cookbook/94_backends/05_composite_backend.py
"""

import os
import tempfile

from agno.agent import Agent
from agno.backends.composite import CompositeBackend
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
# Setup: two separate directories
# ---------------------------------------------------------------------------
workspace_dir = tempfile.mkdtemp(prefix="agno_workspace_")
memories_dir = tempfile.mkdtemp(prefix="agno_memories_")

with open(os.path.join(workspace_dir, "app.py"), "w") as f:
    f.write('"""Main application entry point."""\n\ndef main():\n    print("App started")\n\nif __name__ == "__main__":\n    main()\n')

with open(os.path.join(workspace_dir, "requirements.txt"), "w") as f:
    f.write("agno>=0.1.0\nopenai>=1.0.0\n")

with open(os.path.join(memories_dir, "project_notes.md"), "w") as f:
    f.write("# Project Notes\n\nThis project is a demo of the Agno backend system.\n")

print(f"Workspace: {workspace_dir}")
print(f"Memories:  {memories_dir}")

# ---------------------------------------------------------------------------
# Composite backend
# ---------------------------------------------------------------------------
composite = CompositeBackend(
    default=FilesystemBackend(root=workspace_dir, virtual_mode=True),
    routes={
        "/memories/": FilesystemBackend(root=memories_dir, virtual_mode=True),
    },
)

toolkit = BackendToolkit(composite)

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
agent = Agent(
    model=model,
    tools=[toolkit],
    instructions=[
        "You have two areas: /workspace/ for project files and /memories/ for persistent notes.",
        "When listing root (/), you will see both areas as directories.",
        "Save important observations to /memories/ so they persist across sessions.",
    ],
    markdown=True,
    debug_mode=True,
)

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n--- List root (shows both backends) ---\n")
    agent.print_response("List everything at the root path /", stream=True)

    print("\n--- Read from workspace ---\n")
    agent.print_response("Show me the contents of /workspace/app.py", stream=True)

    print("\n--- Read from memories ---\n")
    agent.print_response("What notes do I have saved in /memories/?", stream=True)

    print("\n--- Cross-backend search ---\n")
    agent.print_response(
        "Search for the word 'demo' across all files in both /workspace/ and /memories/.",
        stream=True,
    )

    print("\n--- Write a memory ---\n")
    agent.print_response(
        "Save a note to /memories/session_summary.md: "
        "'Session 1 complete. Explored composite backend routing.'",
        stream=True,
    )

    print(f"\nWorkspace: {workspace_dir}")
    print(f"Memories:  {memories_dir}")
