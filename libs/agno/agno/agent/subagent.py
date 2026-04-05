"""SubAgent and AsyncSubAgent — first-class sub-agent definitions for Agno agents.

Define sub-agents directly on the parent Agent:

    agent = Agent(
        model=...,
        subagents=[
            SubAgent(
                name="researcher",
                description="Research specialist — blocks until done",
                tools=[DuckDuckGoTools()],
            ),
            AsyncSubAgent(
                name="coder",
                description="Coding assistant — fire-and-forget, returns task ID",
                tools=[PythonTools()],
            ),
        ],
    )

SubAgent
    → named sync tool + named async tool (both blocking, same agent.run / agent.arun)
    → use when the coordinator needs the result before continuing

AsyncSubAgent
    → named async launch tool that returns a task_id IMMEDIATELY
    → sub-agent runs in the background via asyncio.create_task
    → coordinator can continue talking to the user while work happens
    → lifecycle tools auto-registered on the coordinator:
        check_async_task(task_id)          → status + result when done
        update_async_task(task_id, instr)  → cancel + restart with new instructions
        cancel_async_task(task_id)         → stop the task
        list_async_tasks()                 → summary of all tasks
    → supports remote HTTP endpoint via url= (uses HTTP POST instead of arun)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

if TYPE_CHECKING:
    from agno.agent.agent import Agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_content(run_output: Any, name: str) -> str:
    """Pull a plain string out of a RunOutput (or any fallback)."""
    if run_output is None:
        return f"Sub-agent '{name}' returned no output."
    if hasattr(run_output, "content") and run_output.content:
        return str(run_output.content)
    if hasattr(run_output, "messages") and run_output.messages:
        last = run_output.messages[-1]
        if hasattr(last, "content") and last.content:
            return str(last.content)
    return str(run_output)


def _safe_tool_name(name: str) -> str:
    """Convert an agent name into a valid Python identifier for use as a tool name."""
    return name.replace("-", "_").replace(" ", "_")


# ---------------------------------------------------------------------------
# Async task tracking
# ---------------------------------------------------------------------------

class TaskStatus(str, Enum):
    running   = "running"
    completed = "completed"
    error     = "error"
    cancelled = "cancelled"


@dataclass
class _AsyncTask:
    task_id:       str
    agent_name:    str
    original_task: str
    status:        TaskStatus
    asyncio_task:  Optional[asyncio.Task] = None
    result:        Optional[str] = None
    error:         Optional[str] = None
    created_at:    float = field(default_factory=time.time)
    completed_at:  Optional[float] = None


class AsyncTaskManager:
    """Tracks all background async sub-agent tasks for a coordinator agent.

    One instance is shared across all AsyncSubAgent tools on a single
    coordinator.  Created by register_subagents() in _init.py and closed
    over by each launch callable and all lifecycle tools.
    """

    def __init__(self) -> None:
        self._tasks: Dict[str, _AsyncTask] = {}

    # -- Internal -----------------------------------------------------------

    def _register(
        self,
        task_id: str,
        agent_name: str,
        original_task: str,
        asyncio_task: asyncio.Task,
    ) -> None:
        t = _AsyncTask(
            task_id=task_id,
            agent_name=agent_name,
            original_task=original_task,
            status=TaskStatus.running,
            asyncio_task=asyncio_task,
        )
        self._tasks[task_id] = t
        asyncio_task.add_done_callback(lambda at: self._on_done(task_id, at))

    def _on_done(self, task_id: str, at: asyncio.Task) -> None:
        t = self._tasks.get(task_id)
        if t is None:
            return
        t.completed_at = time.time()
        if at.cancelled():
            t.status = TaskStatus.cancelled
        elif at.exception() is not None:
            t.status = TaskStatus.error
            t.error = str(at.exception())
        else:
            t.status = TaskStatus.completed
            t.result = at.result()

    # -- Public API used by lifecycle tools ---------------------------------

    def get(self, task_id: str) -> Optional[_AsyncTask]:
        return self._tasks.get(task_id)

    def all_tasks(self) -> Dict[str, _AsyncTask]:
        return dict(self._tasks)

    def cancel(self, task_id: str) -> bool:
        t = self._tasks.get(task_id)
        if t is None or t.asyncio_task is None:
            return False
        if t.status == TaskStatus.running:
            t.asyncio_task.cancel()
            return True
        return False

    def launch(
        self,
        agent_name: str,
        original_task: str,
        coro: Any,  # coroutine
    ) -> str:
        """Start a coroutine as an asyncio.Task and return its task_id."""
        task_id = uuid.uuid4().hex[:10]
        at = asyncio.create_task(coro)
        self._register(task_id, agent_name, original_task, at)
        return task_id

    def relaunch(
        self,
        task_id: str,
        new_instructions: str,
        coro: Any,
    ) -> bool:
        """Cancel current run, start a new one with the same task_id."""
        t = self._tasks.get(task_id)
        if t is None:
            return False
        # Cancel old task (may already be done — that's fine)
        if t.asyncio_task is not None and t.status == TaskStatus.running:
            t.asyncio_task.cancel()
        # Restart
        at = asyncio.create_task(coro)
        t.asyncio_task = at
        t.status = TaskStatus.running
        t.result = None
        t.error = None
        t.completed_at = None
        at.add_done_callback(lambda a: self._on_done(task_id, a))
        return True

    # -- Lifecycle tool callables -------------------------------------------

    def make_lifecycle_tools(self) -> List[Callable]:
        """Return the four lifecycle async callables to register on the toolkit."""
        return [
            self._make_check(),
            self._make_update(),
            self._make_cancel(),
            self._make_list(),
        ]

    def _make_check(self) -> Callable:
        mgr = self

        async def check_async_task(task_id: str) -> str:
            """Check the status and result of a background async sub-agent task.

            Args:
                task_id: The task ID returned when the task was launched.
            """
            t = mgr.get(task_id)
            if t is None:
                return f"No task found with ID '{task_id}'."
            lines = [
                f"Task ID:    {t.task_id}",
                f"Agent:      {t.agent_name}",
                f"Status:     {t.status.value}",
                f"Original:   {t.original_task[:120]}",
            ]
            if t.status == TaskStatus.running:
                elapsed = round(time.time() - t.created_at, 1)
                lines.append(f"Running for {elapsed}s — not yet complete.")
            elif t.status == TaskStatus.completed:
                lines.append(f"Result:\n{t.result}")
            elif t.status == TaskStatus.error:
                lines.append(f"Error: {t.error}")
            elif t.status == TaskStatus.cancelled:
                lines.append("Task was cancelled.")
            return "\n".join(lines)

        return check_async_task

    def _make_cancel(self) -> Callable:
        mgr = self

        async def cancel_async_task(task_id: str) -> str:
            """Cancel a running background async sub-agent task.

            Args:
                task_id: The task ID returned when the task was launched.
            """
            t = mgr.get(task_id)
            if t is None:
                return f"No task found with ID '{task_id}'."
            if t.status != TaskStatus.running:
                return f"Task '{task_id}' is not running (status: {t.status.value})."
            cancelled = mgr.cancel(task_id)
            if cancelled:
                return f"Task '{task_id}' ({t.agent_name}) cancelled successfully."
            return f"Could not cancel task '{task_id}'."

        return cancel_async_task

    def _make_list(self) -> Callable:
        mgr = self

        async def list_async_tasks() -> str:
            """List all async sub-agent tasks and their current statuses."""
            tasks = mgr.all_tasks()
            if not tasks:
                return "No async tasks have been launched yet."
            lines = [f"{'ID':<12} {'Agent':<16} {'Status':<12} {'Task (preview)'}"]
            lines.append("-" * 70)
            for t in tasks.values():
                preview = t.original_task[:40].replace("\n", " ")
                lines.append(f"{t.task_id:<12} {t.agent_name:<16} {t.status.value:<12} {preview}")
            return "\n".join(lines)

        return list_async_tasks

    def _make_update(self) -> Callable:
        mgr = self

        async def update_async_task(task_id: str, instructions: str) -> str:
            """Send updated instructions to a running async task.

            Cancels the current run and restarts the sub-agent with the original
            task plus the new instructions appended. The task ID is preserved.

            Args:
                task_id:      The task ID to update.
                instructions: New or additional instructions for the sub-agent.
            """
            t = mgr.get(task_id)
            if t is None:
                return f"No task found with ID '{task_id}'."
            if t.status not in (TaskStatus.running,):
                return (
                    f"Task '{task_id}' has status '{t.status.value}' and cannot be updated. "
                    "Only running tasks can be updated."
                )
            # We need the agent to build a new coroutine — stored on the task
            if t.asyncio_task is None or not hasattr(t, "_agent_ref"):
                return (
                    f"Task '{task_id}' does not support mid-flight updates "
                    "(only in-process AsyncSubAgent tasks support this)."
                )
            agent_ref = t._agent_ref  # type: ignore[attr-defined]
            combined = f"{t.original_task}\n\nUpdate: {instructions}"

            async def _run() -> str:
                result = await agent_ref.arun(combined)
                return _extract_content(result, t.agent_name)

            mgr.relaunch(task_id, instructions, _run())
            return (
                f"Task '{task_id}' ({t.agent_name}) restarted with updated instructions. "
                f"Use check_async_task('{task_id}') to monitor progress."
            )

        return update_async_task


# ---------------------------------------------------------------------------
# SubAgent — blocking (sync + async)
# ---------------------------------------------------------------------------

@dataclass
class SubAgent:
    """Blocking sub-agent.

    Becomes a named sync tool AND a named async tool on the parent agent.
    The coordinator waits for the result before continuing.

    Args:
        name:         Tool name the LLM calls (e.g. "researcher").
        description:  Tool description shown to the LLM.
        agent:        Pre-built Agent instance (takes precedence over config).
        model:        Model for the sub-agent.
        instructions: System instructions for the sub-agent.
        tools:        Tools available to the sub-agent.
        markdown:     Whether the sub-agent formats output as markdown.
    """

    name: str
    description: str

    agent: Optional[Any] = None
    model: Optional[Any] = None
    instructions: Optional[Union[str, List[str]]] = None
    tools: Optional[List] = None
    markdown: bool = False

    _cached_agent: Optional[Any] = field(default=None, init=False, repr=False)

    def get_agent(self) -> Any:
        if self._cached_agent is not None:
            return self._cached_agent
        if self.agent is not None:
            self._cached_agent = self.agent
            return self._cached_agent
        from agno.agent.agent import Agent as AgnoAgent

        kwargs: dict = {"markdown": self.markdown}
        if self.model is not None:
            kwargs["model"] = self.model
        if self.instructions is not None:
            kwargs["instructions"] = self.instructions
        if self.tools is not None:
            kwargs["tools"] = self.tools
        self._cached_agent = AgnoAgent(**kwargs)
        return self._cached_agent

    def make_sync_callable(self) -> Callable:
        """Sync blocking callable — delegates to agent.run()."""
        agent_ref = self.get_agent()
        safe_name = _safe_tool_name(self.name)
        description = self.description

        def _call(task: str) -> str:
            result = agent_ref.run(task)
            return _extract_content(result, safe_name)

        _call.__name__ = safe_name
        _call.__qualname__ = safe_name
        _call.__doc__ = description
        return _call

    def make_async_callable(self) -> Callable:
        """Async blocking callable — delegates to agent.arun()."""
        agent_ref = self.get_agent()
        safe_name = _safe_tool_name(self.name)
        description = self.description

        async def _acall(task: str) -> str:
            result = await agent_ref.arun(task)
            return _extract_content(result, safe_name)

        _acall.__name__ = safe_name
        _acall.__qualname__ = safe_name
        _acall.__doc__ = description
        return _acall


# ---------------------------------------------------------------------------
# AsyncSubAgent — non-blocking (fire and forget)
# ---------------------------------------------------------------------------

@dataclass
class AsyncSubAgent:
    """Non-blocking async sub-agent.

    Launches the sub-agent as a background asyncio.Task and returns a task_id
    immediately.  The coordinator can continue interacting with the user while
    the sub-agent works.

    Four lifecycle tools are automatically registered on the coordinator
    when any AsyncSubAgent is present:
        check_async_task(task_id)          — get status / result
        update_async_task(task_id, instr)  — restart with new instructions
        cancel_async_task(task_id)         — stop the task
        list_async_tasks()                 — overview of all tasks

    Args:
        name:         Tool name the LLM calls to launch this sub-agent.
        description:  Tool description shown to the LLM.
        url:          Optional remote HTTP endpoint (None = in-process arun).
        agent:        Pre-built Agent instance (used when url=None).
        model:        Model for the sub-agent.
        instructions: System instructions for the sub-agent.
        tools:        Tools available to the sub-agent.
        markdown:     Whether the sub-agent formats output as markdown.
    """

    name: str
    description: str

    url: Optional[str] = None

    agent: Optional[Any] = None
    model: Optional[Any] = None
    instructions: Optional[Union[str, List[str]]] = None
    tools: Optional[List] = None
    markdown: bool = False

    _cached_agent: Optional[Any] = field(default=None, init=False, repr=False)

    def get_agent(self) -> Any:
        if self._cached_agent is not None:
            return self._cached_agent
        if self.agent is not None:
            self._cached_agent = self.agent
            return self._cached_agent
        from agno.agent.agent import Agent as AgnoAgent

        kwargs: dict = {"markdown": self.markdown}
        if self.model is not None:
            kwargs["model"] = self.model
        if self.instructions is not None:
            kwargs["instructions"] = self.instructions
        if self.tools is not None:
            kwargs["tools"] = self.tools
        self._cached_agent = AgnoAgent(**kwargs)
        return self._cached_agent

    def make_launch_callable(self, manager: AsyncTaskManager) -> Callable:
        """Return an async tool that launches this sub-agent in the background.

        Returns a task_id string immediately without waiting for the result.
        """
        if self.url is not None:
            return self._make_http_launch(manager)
        return self._make_inprocess_launch(manager)

    def _make_inprocess_launch(self, manager: AsyncTaskManager) -> Callable:
        agent_ref = self.get_agent()
        safe_name = _safe_tool_name(self.name)
        description = f"[background] {self.description}"
        agent_name = self.name

        async def _launch(task: str) -> str:
            async def _run() -> str:
                result = await agent_ref.arun(task)
                return _extract_content(result, safe_name)

            coro = _run()
            task_id = manager.launch(agent_name, task, coro)

            # Attach agent reference for update_async_task support
            tracked = manager.get(task_id)
            if tracked is not None:
                tracked._agent_ref = agent_ref  # type: ignore[attr-defined]

            return (
                f"Task launched. ID: {task_id}  Agent: {agent_name}\n"
                f"Use check_async_task('{task_id}') to get the result when ready.\n"
                f"Use list_async_tasks() to see all running tasks."
            )

        _launch.__name__ = safe_name
        _launch.__qualname__ = safe_name
        _launch.__doc__ = description
        return _launch

    def _make_http_launch(self, manager: AsyncTaskManager) -> Callable:
        url = self.url
        safe_name = _safe_tool_name(self.name)
        description = f"[background/remote] {self.description}"
        agent_name = self.name

        async def _launch(task: str) -> str:
            async def _run() -> str:
                try:
                    import aiohttp
                except ImportError as exc:
                    raise ImportError(
                        "Install aiohttp to use AsyncSubAgent with url=: pip install aiohttp"
                    ) from exc
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json={"message": task}) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        if isinstance(data, dict):
                            return str(data.get("content") or data.get("message") or data)
                        return str(data)

            task_id = manager.launch(agent_name, task, _run())
            return (
                f"Remote task launched. ID: {task_id}  Agent: {agent_name}  URL: {url}\n"
                f"Use check_async_task('{task_id}') to get the result when ready."
            )

        _launch.__name__ = safe_name
        _launch.__qualname__ = safe_name
        _launch.__doc__ = description
        return _launch
