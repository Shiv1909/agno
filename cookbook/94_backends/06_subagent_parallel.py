"""
Sub-Agents: Blocking (SubAgent) and Non-Blocking (AsyncSubAgent)
================================================================

SubAgent      → blocking: coordinator waits for result before continuing
AsyncSubAgent → non-blocking: returns task_id immediately, coordinator
                continues talking to the user while work happens in background

Architecture (this example):
    coordinator
        subagents=[
            SubAgent("analyst", ...)      →  analyst(task)        [blocking]
            AsyncSubAgent("coder", ...)   →  coder(task)          [fire-and-forget]
                                          +  check_async_task(id)
                                          +  update_async_task(id, instr)
                                          +  cancel_async_task(id)
                                          +  list_async_tasks()
        ]

Environment variables required:
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_DEPLOYMENT

Run:
    python cookbook/94_backends/06_subagent_parallel.py
"""

import asyncio
import os
import tempfile
from pathlib import Path

from agno.agent import Agent, AsyncSubAgent, SubAgent
from agno.backends.wsl import WSLBackend
from agno.models.azure import AzureOpenAI
from agno.tools.backend import BackendToolkit

# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def make_model() -> AzureOpenAI:
    return AzureOpenAI(
        id=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )

# ---------------------------------------------------------------------------
# Shared workspace
# ---------------------------------------------------------------------------
WORKSPACE = tempfile.mkdtemp(prefix="agno_subagent_demo_")
print(f"Workspace: {WORKSPACE}")

backend = WSLBackend(root=WORKSPACE, virtual_mode=True, inherit_env=False)
backend_toolkit = BackendToolkit(backend)

# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------
coordinator = Agent(
    name="coordinator",
    model=make_model(),
    subagents=[
        # BLOCKING — coordinator waits for analyst to finish before continuing
        SubAgent(
            name="analyst",
            description=(
                "File analysis assistant. Use for listing workspace files, "
                "reading file contents, searching text patterns, and summarising findings. "
                "Blocks until analysis is complete."
            ),
            model=make_model(),
            tools=[backend_toolkit],
            instructions=[
                "You are a file analysis assistant.",
                "Use ls, read_file, grep, and glob to inspect files in the workspace.",
                "Summarise what you find clearly and concisely.",
            ],
            markdown=True,
        ),

        # NON-BLOCKING — returns task_id immediately, runs in background
        AsyncSubAgent(
            name="coder",
            description=(
                "Python coding assistant running inside WSL Ubuntu. "
                "Launches in the BACKGROUND and returns a task_id immediately — "
                "use check_async_task(task_id) to get the result when ready. "
                "Use for writing and running Python scripts."
            ),
            model=make_model(),
            tools=[backend_toolkit],
            instructions=[
                "You are a Python coding assistant running inside WSL Ubuntu.",
                "Always use 'python3' to run scripts, never 'python'.",
                "When asked to write a script: use write_file to save it, then execute it.",
                "Always report the actual stdout output.",
            ],
            markdown=True,
        ),
    ],
    tools=[backend_toolkit],
    instructions=[
        "You are a coordinator agent.",
        "Use analyst(task=...) for file analysis — it blocks until done.",
        "Use coder(task=...) to launch coding work in the BACKGROUND.",
        "  coder() returns a task_id immediately — do NOT wait for it.",
        "  Return the task_id to the user and continue the conversation.",
        "  When the user asks for results, call check_async_task(task_id).",
        "  Use list_async_tasks() to show all running tasks.",
        "  Use cancel_async_task(task_id) to stop a task.",
        "  Use update_async_task(task_id, instructions) to steer a running task.",
    ],
    markdown=True,
    debug_mode=True,
)

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
async def main():
    # ------------------------------------------------------------------
    # STEP 1: Launch coder in background — returns task_id immediately
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: Launch coder in background (non-blocking)")
    print("=" * 60 + "\n")

    await coordinator.aprint_response(
        "Launch the coder to write and run a Python script called stats.py "
        "that generates 10 random numbers, computes their mean and standard deviation "
        "using the statistics module, prints the results, and saves a summary line "
        "to results.txt. Tell me the task_id and then we can continue.",
        stream=True,
    )

    # ------------------------------------------------------------------
    # STEP 2: Coordinator continues without waiting for coder
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Coordinator does other work while coder runs")
    print("=" * 60 + "\n")

    await coordinator.aprint_response(
        "While the coder is running, use the analyst to list all files "
        "currently in the workspace.",
        stream=True,
    )

    # Give coder a moment to finish
    await asyncio.sleep(2)

    # ------------------------------------------------------------------
    # STEP 3: Check coder result
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Check coder result via check_async_task")
    print("=" * 60 + "\n")

    await coordinator.aprint_response(
        "Check the status of the coder task. If it's done, show me the results.",
        stream=True,
    )

    # ------------------------------------------------------------------
    # STEP 4: List all tasks
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: List all async tasks")
    print("=" * 60 + "\n")

    await coordinator.aprint_response(
        "List all async tasks that have been launched.",
        stream=True,
    )

    # ------------------------------------------------------------------
    # STEP 5: Launch another coder task, then update it mid-flight
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Launch + update a task mid-flight")
    print("=" * 60 + "\n")

    await coordinator.aprint_response(
        "Launch the coder to write a very long script that generates 1000 random numbers "
        "and computes statistics 100 times. Get the task_id.",
        stream=True,
    )

    await coordinator.aprint_response(
        "Actually, update that task — tell it to just print 'hello world' instead. "
        "Then check its status.",
        stream=True,
    )

    files = list(Path(WORKSPACE).iterdir())
    print(f"\nWorkspace files: {[f.name for f in files]}")


if __name__ == "__main__":
    asyncio.run(main())
