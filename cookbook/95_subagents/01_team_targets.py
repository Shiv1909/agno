"""
Sub-agents with Team targets and delegation controls
====================================================

This example shows the current sub-agent model:

- `SubAgent(target=Team(...))` for blocking delegation to a specialist team
- `AsyncSubAgent(target=Agent(...))` for fire-and-forget background work
- filtered session-state inheritance across the delegation boundary
- inherited parent history for the blocking team
- async lifecycle tools: `check_async_task`, `update_async_task`,
  `cancel_async_task`, and `list_async_tasks`

Environment variables required:
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_DEPLOYMENT

Run:
    python cookbook/95_subagents/01_team_targets.py
"""

import asyncio
import os
import tempfile
from pathlib import Path

from agno.agent import Agent, AsyncSubAgent, SubAgent
from agno.backends.wsl import WSLBackend
from agno.models.azure import AzureOpenAI
from agno.team import Team
from agno.tools.backend import BackendToolkit


def make_model() -> AzureOpenAI:
    return AzureOpenAI(
        id=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )


WORKSPACE = tempfile.mkdtemp(prefix="agno_subagent_demo_")
print(f"Workspace: {WORKSPACE}")

backend = WSLBackend(root=WORKSPACE, virtual_mode=True, inherit_env=False)
backend_toolkit = BackendToolkit(backend)


analysis_team = Team(
    name="analysis-team",
    model=make_model(),
    members=[
        Agent(
            name="workspace-reader",
            model=make_model(),
            tools=[backend_toolkit],
            instructions=[
                "Inspect files in the workspace.",
                "Use ls, read_file, grep, and glob when needed.",
                "Report concrete findings only.",
            ],
            markdown=True,
        ),
        Agent(
            name="summary-writer",
            model=make_model(),
            instructions=[
                "Turn teammate findings into a concise technical summary.",
                "Highlight notable files, outputs, and follow-up actions.",
            ],
            markdown=True,
        ),
    ],
    instructions=[
        "Work as a focused analysis team.",
        "Inspect the inherited workspace context before answering.",
        "Return a compact summary with any important file paths or findings.",
    ],
    markdown=True,
)

coder_agent = Agent(
    name="coder",
    model=make_model(),
    tools=[backend_toolkit],
    instructions=[
        "You are a Python coding assistant running inside WSL Ubuntu.",
        "Always use 'python3' to run scripts, never 'python'.",
        "When asked to write a script, save it with write_file and then execute it.",
        "Always report the actual stdout output and any files you created.",
    ],
    markdown=True,
)


coordinator = Agent(
    name="coordinator",
    model=make_model(),
    session_state={
        "workspace_root": WORKSPACE,
        "ticket_id": "DEMO-95",
        "owner": "cookbook",
    },
    add_session_state_to_context=True,
    subagents=[
        SubAgent(
            name="analysis_team",
            description=(
                "Blocking analysis team for workspace inspection and technical summaries. "
                "Use when you need a complete answer before continuing."
            ),
            target=analysis_team,
            inherit_history=True,
            history_strategy="last_n",
            history_runs=2,
            inherit_session_state=True,
            session_state_strategy="filtered",
            session_state_keys=["workspace_root", "ticket_id"],
            persist_child_runs=True,
            markdown=True,
        ),
        AsyncSubAgent(
            name="coder",
            description=(
                "Background Python implementation assistant running inside WSL Ubuntu. "
                "Launches in the background and returns a task_id immediately."
            ),
            target=coder_agent,
            inherit_session_state=True,
            session_state_strategy="filtered",
            session_state_keys=["workspace_root", "ticket_id"],
            persist_child_runs=True,
            markdown=True,
        ),
    ],
    tools=[backend_toolkit],
    instructions=[
        "You are a coordinator agent.",
        "Use analysis_team(task=...) for blocking delegated analysis.",
        "Use coder(task=...) to launch background implementation work.",
        "coder() returns a task_id immediately. Do not wait unless the user asks for results.",
        "Use check_async_task(task_id) for status and result retrieval.",
        "Use list_async_tasks() to show all launched tasks.",
        "Use update_async_task(task_id, instructions) to restart a running task with updated instructions.",
        "Use cancel_async_task(task_id) to stop a running task.",
    ],
    markdown=True,
    debug_mode=True,
)


async def main():
    print("\n" + "=" * 60)
    print("STEP 1: Use the blocking team target")
    print("=" * 60 + "\n")

    await coordinator.aprint_response(
        "Use analysis_team to inspect the workspace and tell me what shared context is available, "
        "what files exist right now, and whether the delegated team can see the ticket id.",
        stream=True,
    )

    print("\n" + "=" * 60)
    print("STEP 2: Launch coder in the background")
    print("=" * 60 + "\n")

    await coordinator.aprint_response(
        "Launch the coder to write and run a Python script called stats.py that generates 10 random numbers, "
        "computes their mean and standard deviation using the statistics module, prints the results, "
        "and saves a one-line summary to results.txt. Return only the task_id and a short note.",
        stream=True,
    )

    print("\n" + "=" * 60)
    print("STEP 3: Continue working while coder runs")
    print("=" * 60 + "\n")

    await coordinator.aprint_response(
        "While the coder is running, use analysis_team to list current files and explain whether the "
        "implementation output has appeared yet.",
        stream=True,
    )

    await asyncio.sleep(2)

    print("\n" + "=" * 60)
    print("STEP 4: Check the async task result")
    print("=" * 60 + "\n")

    await coordinator.aprint_response(
        "Check the latest coder task. If it completed, show me the result and the generated files.",
        stream=True,
    )

    print("\n" + "=" * 60)
    print("STEP 5: Launch and update another task")
    print("=" * 60 + "\n")

    await coordinator.aprint_response(
        "Launch the coder to create a long-running script that generates 1000 random numbers and repeats "
        "statistics calculations 100 times. Return the task_id.",
        stream=True,
    )

    await coordinator.aprint_response(
        "Update that running coder task so it instead writes hello.py, prints 'hello world', and exits. "
        "Then check the task status.",
        stream=True,
    )

    print("\n" + "=" * 60)
    print("STEP 6: List tracked async tasks")
    print("=" * 60 + "\n")

    await coordinator.aprint_response(
        "List all async tasks that have been launched so far.",
        stream=True,
    )

    files = list(Path(WORKSPACE).iterdir())
    print(f"\nWorkspace files: {[f.name for f in files]}")


if __name__ == "__main__":
    asyncio.run(main())
