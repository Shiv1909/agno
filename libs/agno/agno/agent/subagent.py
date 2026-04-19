"""SubAgent and AsyncSubAgent with target delegation controls.

Supports delegating from a parent Agent to either a child Agent or Team with
explicit controls for history, session state, memory, knowledge, and tools.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Union

from agno.run import RunContext
from agno.utils.log import log_warning

if TYPE_CHECKING:
    from agno.agent.agent import Agent
    from agno.models.message import Message
    from agno.team.team import Team


SubAgentTarget = Union["Agent", "Team"]
HistoryStrategy = Literal["none", "last_n", "full", "summary"]
SessionStateStrategy = Literal["none", "copy", "filtered"]
MemoryStrategy = Literal["none", "read_only", "copy", "shared"]
ResultMergeStrategy = Literal["text_only", "session_state", "memory_update"]


def _extract_target_output(run_output: Any, name: str) -> str:
    """Pull a plain string out of a RunOutput / TeamRunOutput (or fallback)."""
    if run_output is None:
        return f"Sub-agent '{name}' returned no output."
    if hasattr(run_output, "content") and run_output.content:
        return str(run_output.content)
    if hasattr(run_output, "messages") and run_output.messages:
        last = run_output.messages[-1]
        if hasattr(last, "content") and last.content:
            return str(last.content)
    return str(run_output)


def _extract_content(run_output: Any, name: str) -> str:
    """Backward-compatible alias kept for existing tests/imports."""
    return _extract_target_output(run_output, name)


def _safe_tool_name(name: str) -> str:
    return name.replace("-", "_").replace(" ", "_")


def _message_to_text(message: "Message") -> str:
    role = getattr(message, "role", "unknown")
    name = getattr(message, "name", None)
    label = f"{role}:{name}" if name else role
    if hasattr(message, "get_content_string"):
        content = message.get_content_string()
    else:
        content = str(getattr(message, "content", "") or "")
    return f"{label}: {content}".strip()


def _messages_to_text(messages: List["Message"]) -> str:
    rendered = [_message_to_text(msg) for msg in messages if msg is not None]
    return "\n".join([msg for msg in rendered if msg.strip()])


def _compose_delegated_task(task: str, history_text: Optional[str] = None) -> str:
    if history_text is None or history_text.strip() == "":
        return task
    return (
        "Parent context:\n"
        "<delegation_context>\n"
        f"{history_text.strip()}\n"
        "</delegation_context>\n\n"
        "Delegated task:\n"
        f"{task}"
    )


def _is_missing_session_error(exc: Exception) -> bool:
    text = str(exc)
    return "Session not found" in text or "Session ID is required" in text


def _safe_history_call_sync(fetcher: Callable[[], Any], warning_context: str) -> Any:
    try:
        return fetcher()
    except Exception as exc:
        if _is_missing_session_error(exc):
            log_warning(
                f"Skipping inherited history for subagent delegation because parent {warning_context} is unavailable: {exc}"
            )
            return None
        raise


async def _safe_history_call_async(fetcher: Callable[[], Any], warning_context: str) -> Any:
    try:
        return await fetcher()
    except Exception as exc:
        if _is_missing_session_error(exc):
            log_warning(
                f"Skipping inherited history for subagent delegation because parent {warning_context} is unavailable: {exc}"
            )
            return None
        raise


def _resolve_parent_session_id(parent: "Agent", run_context: Optional[RunContext]) -> Optional[str]:
    if run_context is not None and run_context.session_id:
        return run_context.session_id
    return parent.session_id


def _resolve_parent_user_id(parent: "Agent", run_context: Optional[RunContext]) -> Optional[str]:
    if run_context is not None and run_context.user_id:
        return run_context.user_id
    return parent.user_id


def _derive_child_session_id(
    parent: "Agent",
    subagent_name: str,
    run_context: Optional[RunContext],
    persist_child_runs: bool,
) -> Optional[str]:
    if not persist_child_runs:
        return None
    parent_session_id = _resolve_parent_session_id(parent, run_context)
    if parent_session_id is None:
        return None
    suffix = _safe_tool_name(subagent_name)
    if run_context is not None and run_context.run_id:
        return f"{parent_session_id}::{suffix}::{run_context.run_id}"
    return f"{parent_session_id}::{suffix}"


def _merge_tools(existing_tools: Any, parent_tools: Any) -> Optional[List[Any]]:
    merged: List[Any] = []
    if isinstance(existing_tools, list):
        merged.extend(existing_tools)
    elif existing_tools is not None:
        raise ValueError("inherit_tools=True requires the child target to use a list-based tools configuration.")

    if isinstance(parent_tools, list):
        merged.extend(parent_tools)
    elif parent_tools is not None:
        raise ValueError("inherit_tools=True requires the parent agent to use a list-based tools configuration.")

    return merged or None


@dataclass
class DelegationContext:
    parent: "Agent"
    target: SubAgentTarget
    target_name: str
    target_type: Literal["agent", "team"]
    task: str
    delegated_task: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    session_state: Optional[Dict[str, Any]] = None
    add_session_state_to_context: bool = False
    merge_back_result: bool = False
    result_merge_strategy: ResultMergeStrategy = "text_only"
    result_merge_keys: Optional[List[str]] = None


def _get_target_type(target: Any) -> Literal["agent", "team"]:
    from agno.agent.agent import Agent as AgnoAgent
    from agno.team.team import Team as AgnoTeam

    if isinstance(target, AgnoAgent):
        return "agent"
    if isinstance(target, AgnoTeam):
        return "team"
    raise TypeError(f"Unsupported subagent target type: {type(target)!r}")


def _clone_target_with_inheritance(
    target: SubAgentTarget,
    parent: "Agent",
    inherit_memories: bool,
    memory_strategy: MemoryStrategy,
    inherit_knowledge: bool,
    inherit_tools: bool,
) -> SubAgentTarget:
    updates: Dict[str, Any] = {}

    if inherit_tools:
        updates["tools"] = _merge_tools(getattr(target, "tools", None), parent.tools)

    if inherit_knowledge:
        updates["knowledge"] = parent.knowledge
        if hasattr(target, "add_knowledge_to_context"):
            updates["add_knowledge_to_context"] = True

    if inherit_memories and memory_strategy == "read_only":
        updates["memory_manager"] = parent.memory_manager
        if hasattr(target, "add_memories_to_context"):
            updates["add_memories_to_context"] = True
        if hasattr(target, "update_memory_on_run"):
            updates["update_memory_on_run"] = False
        if hasattr(target, "enable_agentic_memory"):
            updates["enable_agentic_memory"] = False

    if updates and hasattr(target, "deep_copy"):
        return target.deep_copy(update=updates)  # type: ignore[return-value]
    if updates:
        raise ValueError("Cannot apply delegation inheritance to a target that does not support deep_copy().")
    return target


def _resolve_history_text_sync(
    parent: "Agent",
    run_context: Optional[RunContext],
    strategy: HistoryStrategy,
    history_runs: Optional[int],
) -> Optional[str]:
    session_id = _resolve_parent_session_id(parent, run_context)
    if strategy == "none":
        return None
    if strategy == "summary":
        summary = _safe_history_call_sync(
            lambda: parent.get_session_summary(session_id=session_id),
            "session summary",
        )
        if summary is not None and getattr(summary, "summary", None):
            return f"Conversation summary:\n{summary.summary}"
        messages = _safe_history_call_sync(
            lambda: parent.get_chat_history(session_id=session_id, last_n_runs=history_runs),
            "session history",
        )
        return _messages_to_text(messages) if messages else None
    if strategy == "last_n":
        messages = _safe_history_call_sync(
            lambda: parent.get_chat_history(session_id=session_id, last_n_runs=history_runs or 1),
            "session history",
        )
        return _messages_to_text(messages) if messages else None
    messages = _safe_history_call_sync(
        lambda: parent.get_chat_history(session_id=session_id, last_n_runs=None),
        "session history",
    )
    return _messages_to_text(messages) if messages else None


async def _resolve_history_text_async(
    parent: "Agent",
    run_context: Optional[RunContext],
    strategy: HistoryStrategy,
    history_runs: Optional[int],
) -> Optional[str]:
    session_id = _resolve_parent_session_id(parent, run_context)
    if strategy == "none":
        return None
    if strategy == "summary":
        summary = await _safe_history_call_async(
            lambda: parent.aget_session_summary(session_id=session_id),
            "session summary",
        )
        if summary is not None and getattr(summary, "summary", None):
            return f"Conversation summary:\n{summary.summary}"
        messages = await _safe_history_call_async(
            lambda: parent.aget_chat_history(session_id=session_id, last_n_runs=history_runs),
            "session history",
        )
        return _messages_to_text(messages) if messages else None
    if strategy == "last_n":
        messages = await _safe_history_call_async(
            lambda: parent.aget_chat_history(session_id=session_id, last_n_runs=history_runs or 1),
            "session history",
        )
        return _messages_to_text(messages) if messages else None
    messages = await _safe_history_call_async(
        lambda: parent.aget_chat_history(session_id=session_id, last_n_runs=None),
        "session history",
    )
    return _messages_to_text(messages) if messages else None


def _resolve_session_state_payload(
    parent: "Agent",
    run_context: Optional[RunContext],
    inherit_session_state: bool,
    session_state_strategy: SessionStateStrategy,
    session_state_keys: Optional[List[str]],
) -> Optional[Dict[str, Any]]:
    if not inherit_session_state or session_state_strategy == "none":
        return None

    source = None
    if run_context is not None and run_context.session_state is not None:
        source = run_context.session_state
    elif parent.session_state is not None:
        source = parent.session_state

    if source is None:
        return None

    if session_state_strategy == "copy":
        return dict(source)
    if session_state_strategy == "filtered":
        keys = session_state_keys or []
        return {key: source[key] for key in keys if key in source}
    raise ValueError(f"Unsupported session_state_strategy: {session_state_strategy}")


def _merge_back_session_state_sync(context: DelegationContext, run_context: Optional[RunContext]) -> None:
    if not context.merge_back_result or context.result_merge_strategy != "session_state":
        return
    if run_context is None or context.session_id is None or not hasattr(context.target, "get_session_state"):
        return
    child_state = context.target.get_session_state(session_id=context.session_id)  # type: ignore[attr-defined]
    if run_context.session_state is None:
        run_context.session_state = {}
    keys = context.result_merge_keys or list(child_state.keys())
    for key in keys:
        if key in child_state:
            run_context.session_state[key] = child_state[key]


async def _merge_back_session_state_async(context: DelegationContext, run_context: Optional[RunContext]) -> None:
    if not context.merge_back_result or context.result_merge_strategy != "session_state":
        return
    if run_context is None or context.session_id is None:
        return
    if hasattr(context.target, "aget_session_state"):
        child_state = await context.target.aget_session_state(session_id=context.session_id)  # type: ignore[attr-defined]
    elif hasattr(context.target, "get_session_state"):
        child_state = context.target.get_session_state(session_id=context.session_id)  # type: ignore[attr-defined]
    else:
        return
    if run_context.session_state is None:
        run_context.session_state = {}
    keys = context.result_merge_keys or list(child_state.keys())
    for key in keys:
        if key in child_state:
            run_context.session_state[key] = child_state[key]


def _run_target_sync(context: DelegationContext, run_context: Optional[RunContext]) -> str:
    result = context.target.run(
        context.delegated_task,
        user_id=context.user_id,
        session_id=context.session_id,
        session_state=context.session_state,
        add_session_state_to_context=context.add_session_state_to_context,
    )
    _merge_back_session_state_sync(context, run_context)
    return _extract_target_output(result, context.target_name)


async def _run_target_async(context: DelegationContext, run_context: Optional[RunContext]) -> str:
    result = await context.target.arun(
        context.delegated_task,
        user_id=context.user_id,
        session_id=context.session_id,
        session_state=context.session_state,
        add_session_state_to_context=context.add_session_state_to_context,
    )
    await _merge_back_session_state_async(context, run_context)
    return _extract_target_output(result, context.target_name)


@dataclass
class _DelegatingSubAgentMixin:
    name: str
    description: str
    target: Optional[Any] = None
    agent: Optional[Any] = None
    model: Optional[Any] = None
    instructions: Optional[Union[str, List[str]]] = None
    tools: Optional[List] = None
    markdown: bool = False

    inherit_history: bool = False
    history_strategy: HistoryStrategy = "none"
    history_runs: Optional[int] = None

    inherit_session_state: bool = False
    session_state_strategy: SessionStateStrategy = "none"
    session_state_keys: Optional[List[str]] = None

    inherit_memories: bool = False
    memory_strategy: MemoryStrategy = "none"

    inherit_knowledge: bool = False
    inherit_tools: bool = False

    persist_child_runs: bool = True
    merge_back_result: bool = False
    result_merge_strategy: ResultMergeStrategy = "text_only"

    _cached_target: Optional[Any] = field(default=None, init=False, repr=False)

    def validate(self, parent: Optional["Agent"] = None) -> None:
        if self.target is not None and self.agent is not None and self.target is not self.agent:
            raise ValueError("Specify either target= or agent= for a subagent, not both.")

        target = self.target if self.target is not None else self.agent
        if target is not None:
            _get_target_type(target)

        if self.memory_strategy in ("copy", "shared"):
            raise ValueError(
                f"{self.__class__.__name__} does not yet support memory_strategy='{self.memory_strategy}'. "
                "Supported values are 'none' and 'read_only'."
            )

        if self.result_merge_strategy == "memory_update":
            raise ValueError("result_merge_strategy='memory_update' is not supported yet.")

        if self.merge_back_result and self.result_merge_strategy == "session_state":
            if not self.inherit_session_state or self.session_state_strategy == "none":
                raise ValueError("Session-state merge-back requires inherited session state to be enabled.")
            if not self.session_state_keys:
                raise ValueError("Session-state merge-back requires session_state_keys to be specified.")

        if self.inherit_tools and parent is not None:
            _merge_tools(getattr(target or self.get_target(), "tools", None), parent.tools)

    def get_target(self) -> SubAgentTarget:
        if self._cached_target is not None:
            return self._cached_target

        explicit_target = self.target if self.target is not None else self.agent
        if explicit_target is not None:
            self._cached_target = explicit_target
            return self._cached_target

        from agno.agent.agent import Agent as AgnoAgent

        kwargs: Dict[str, Any] = {"markdown": self.markdown}
        if self.model is not None:
            kwargs["model"] = self.model
        if self.instructions is not None:
            kwargs["instructions"] = self.instructions
        if self.tools is not None:
            kwargs["tools"] = self.tools
        self._cached_target = AgnoAgent(**kwargs)
        return self._cached_target

    def get_agent(self) -> SubAgentTarget:
        """Backward-compatible alias."""
        return self.get_target()

    def _build_context_sync(self, parent: "Agent", task: str, run_context: Optional[RunContext]) -> DelegationContext:
        self.validate(parent=parent)
        target = _clone_target_with_inheritance(
            self.get_target(),
            parent=parent,
            inherit_memories=self.inherit_memories,
            memory_strategy=self.memory_strategy,
            inherit_knowledge=self.inherit_knowledge,
            inherit_tools=self.inherit_tools,
        )
        history_strategy: HistoryStrategy = self.history_strategy if self.inherit_history else "none"
        history_text = _resolve_history_text_sync(parent, run_context, history_strategy, self.history_runs)
        session_state = _resolve_session_state_payload(
            parent=parent,
            run_context=run_context,
            inherit_session_state=self.inherit_session_state,
            session_state_strategy=self.session_state_strategy,
            session_state_keys=self.session_state_keys,
        )
        return DelegationContext(
            parent=parent,
            target=target,
            target_name=self.name,
            target_type=_get_target_type(target),
            task=task,
            delegated_task=_compose_delegated_task(task, history_text),
            user_id=_resolve_parent_user_id(parent, run_context),
            session_id=_derive_child_session_id(parent, self.name, run_context, self.persist_child_runs),
            session_state=session_state,
            add_session_state_to_context=session_state is not None,
            merge_back_result=self.merge_back_result,
            result_merge_strategy=self.result_merge_strategy,
            result_merge_keys=self.session_state_keys,
        )

    async def _build_context_async(
        self, parent: "Agent", task: str, run_context: Optional[RunContext]
    ) -> DelegationContext:
        self.validate(parent=parent)
        target = _clone_target_with_inheritance(
            self.get_target(),
            parent=parent,
            inherit_memories=self.inherit_memories,
            memory_strategy=self.memory_strategy,
            inherit_knowledge=self.inherit_knowledge,
            inherit_tools=self.inherit_tools,
        )
        history_strategy: HistoryStrategy = self.history_strategy if self.inherit_history else "none"
        history_text = await _resolve_history_text_async(parent, run_context, history_strategy, self.history_runs)
        session_state = _resolve_session_state_payload(
            parent=parent,
            run_context=run_context,
            inherit_session_state=self.inherit_session_state,
            session_state_strategy=self.session_state_strategy,
            session_state_keys=self.session_state_keys,
        )
        return DelegationContext(
            parent=parent,
            target=target,
            target_name=self.name,
            target_type=_get_target_type(target),
            task=task,
            delegated_task=_compose_delegated_task(task, history_text),
            user_id=_resolve_parent_user_id(parent, run_context),
            session_id=_derive_child_session_id(parent, self.name, run_context, self.persist_child_runs),
            session_state=session_state,
            add_session_state_to_context=session_state is not None,
            merge_back_result=self.merge_back_result,
            result_merge_strategy=self.result_merge_strategy,
            result_merge_keys=self.session_state_keys,
        )


class TaskStatus(str, Enum):
    running = "running"
    completed = "completed"
    error = "error"
    cancelled = "cancelled"


@dataclass
class _AsyncTask:
    task_id: str
    agent_name: str
    target_type: str
    original_task: str
    status: TaskStatus
    asyncio_task: Optional[asyncio.Task] = None
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


class AsyncTaskManager:
    def __init__(self) -> None:
        self._tasks: Dict[str, _AsyncTask] = {}

    def _register(
        self,
        task_id: str,
        agent_name: str,
        target_type: str,
        original_task: str,
        asyncio_task: asyncio.Task,
    ) -> None:
        tracked = _AsyncTask(
            task_id=task_id,
            agent_name=agent_name,
            target_type=target_type,
            original_task=original_task,
            status=TaskStatus.running,
            asyncio_task=asyncio_task,
        )
        self._tasks[task_id] = tracked
        asyncio_task.add_done_callback(lambda at: self._on_done(task_id, at))

    def _on_done(self, task_id: str, at: asyncio.Task) -> None:
        tracked = self._tasks.get(task_id)
        if tracked is None:
            return
        tracked.completed_at = time.time()
        if at.cancelled():
            tracked.status = TaskStatus.cancelled
        elif at.exception() is not None:
            tracked.status = TaskStatus.error
            tracked.error = str(at.exception())
        else:
            tracked.status = TaskStatus.completed
            tracked.result = at.result()

    def get(self, task_id: str) -> Optional[_AsyncTask]:
        return self._tasks.get(task_id)

    def all_tasks(self) -> Dict[str, _AsyncTask]:
        return dict(self._tasks)

    def cancel(self, task_id: str) -> bool:
        tracked = self._tasks.get(task_id)
        if tracked is None or tracked.asyncio_task is None:
            return False
        if tracked.status == TaskStatus.running:
            tracked.asyncio_task.cancel()
            return True
        return False

    def launch(
        self,
        agent_name: str,
        target_type: str,
        original_task: str,
        coro: Any,
    ) -> str:
        task_id = uuid.uuid4().hex[:10]
        at = asyncio.create_task(coro)
        self._register(task_id, agent_name, target_type, original_task, at)
        return task_id

    def relaunch(self, task_id: str, new_instructions: str, coro: Any) -> bool:
        tracked = self._tasks.get(task_id)
        if tracked is None:
            return False
        if tracked.asyncio_task is not None and tracked.status == TaskStatus.running:
            tracked.asyncio_task.cancel()
        at = asyncio.create_task(coro)
        tracked.asyncio_task = at
        tracked.status = TaskStatus.running
        tracked.result = None
        tracked.error = None
        tracked.completed_at = None
        at.add_done_callback(lambda a: self._on_done(task_id, a))
        return True

    def make_lifecycle_tools(self) -> List[Callable]:
        return [self._make_check(), self._make_update(), self._make_cancel(), self._make_list()]

    def _make_check(self) -> Callable:
        mgr = self

        async def check_async_task(task_id: str) -> str:
            t = mgr.get(task_id)
            if t is None:
                return f"No task found with ID '{task_id}'."
            lines = [
                f"Task ID:    {t.task_id}",
                f"Agent:      {t.agent_name}",
                f"Target:     {t.target_type}",
                f"Status:     {t.status.value}",
                f"Original:   {t.original_task[:120]}",
            ]
            if t.status == TaskStatus.running:
                elapsed = round(time.time() - t.created_at, 1)
                lines.append(f"Running for {elapsed}s - not yet complete.")
            elif t.status == TaskStatus.completed:
                lines.append(f"Result:\n{t.result}")
            elif t.status == TaskStatus.error:
                lines.append(f"Error: {t.error}")
            else:
                lines.append("Task was cancelled.")
            return "\n".join(lines)

        return check_async_task

    def _make_cancel(self) -> Callable:
        mgr = self

        async def cancel_async_task(task_id: str) -> str:
            t = mgr.get(task_id)
            if t is None:
                return f"No task found with ID '{task_id}'."
            if t.status != TaskStatus.running:
                return f"Task '{task_id}' is not running (status: {t.status.value})."
            if mgr.cancel(task_id):
                return f"Task '{task_id}' ({t.agent_name}) cancelled successfully."
            return f"Could not cancel task '{task_id}'."

        return cancel_async_task

    def _make_list(self) -> Callable:
        mgr = self

        async def list_async_tasks() -> str:
            tasks = mgr.all_tasks()
            if not tasks:
                return "No async tasks have been launched yet."
            lines = [f"{'ID':<12} {'Agent':<16} {'Target':<10} {'Status':<12} {'Task (preview)'}"]
            lines.append("-" * 82)
            for t in tasks.values():
                preview = t.original_task[:40].replace("\n", " ")
                lines.append(f"{t.task_id:<12} {t.agent_name:<16} {t.target_type:<10} {t.status.value:<12} {preview}")
            return "\n".join(lines)

        return list_async_tasks

    def _make_update(self) -> Callable:
        mgr = self

        async def update_async_task(task_id: str, instructions: str) -> str:
            t = mgr.get(task_id)
            if t is None:
                return f"No task found with ID '{task_id}'."
            if t.status != TaskStatus.running:
                return (
                    f"Task '{task_id}' has status '{t.status.value}' and cannot be updated. "
                    "Only running tasks can be updated."
                )
            if t.asyncio_task is None or not hasattr(t, "_target_ref"):
                return (
                    f"Task '{task_id}' does not support mid-flight updates "
                    "(only in-process AsyncSubAgent tasks support this)."
                )

            target_ref = t._target_ref  # type: ignore[attr-defined]
            combined = f"{t.original_task}\n\nUpdate: {instructions}"

            async def _run() -> str:
                result = await target_ref.arun(combined)
                return _extract_target_output(result, t.agent_name)

            mgr.relaunch(task_id, instructions, _run())
            return (
                f"Task '{task_id}' ({t.agent_name}) restarted with updated instructions. "
                f"Use check_async_task('{task_id}') to monitor progress."
            )

        return update_async_task


@dataclass
class SubAgent(_DelegatingSubAgentMixin):
    def make_sync_callable(self, parent: Optional["Agent"] = None) -> Callable:
        safe_name = _safe_tool_name(self.name)
        description = self.description
        target_name = self.name

        def _call(task: str, run_context: Optional[RunContext] = None) -> str:
            if parent is None:
                result = self.get_target().run(task)
                return _extract_target_output(result, target_name)
            context = self._build_context_sync(parent, task, run_context)
            return _run_target_sync(context, run_context)

        _call.__name__ = safe_name
        _call.__qualname__ = safe_name
        _call.__doc__ = description
        return _call

    def make_async_callable(self, parent: Optional["Agent"] = None) -> Callable:
        safe_name = _safe_tool_name(self.name)
        description = self.description
        target_name = self.name

        async def _acall(task: str, run_context: Optional[RunContext] = None) -> str:
            if parent is None:
                result = await self.get_target().arun(task)
                return _extract_target_output(result, target_name)
            context = await self._build_context_async(parent, task, run_context)
            return await _run_target_async(context, run_context)

        _acall.__name__ = safe_name
        _acall.__qualname__ = safe_name
        _acall.__doc__ = description
        return _acall


@dataclass
class AsyncSubAgent(_DelegatingSubAgentMixin):
    url: Optional[str] = None

    def validate(self, parent: Optional["Agent"] = None) -> None:
        super().validate(parent=parent)
        if self.url is not None and (self.target is not None or self.agent is not None):
            raise ValueError("AsyncSubAgent cannot combine url= with target=/agent=.")

    def make_launch_callable(self, manager: AsyncTaskManager, parent: Optional["Agent"] = None) -> Callable:
        self.validate(parent=parent)
        if self.url is not None:
            return self._make_http_launch(manager)
        return self._make_inprocess_launch(manager, parent=parent)

    def _make_inprocess_launch(self, manager: AsyncTaskManager, parent: Optional["Agent"] = None) -> Callable:
        safe_name = _safe_tool_name(self.name)
        description = f"[background] {self.description}"
        agent_name = self.name

        async def _launch(task: str, run_context: Optional[RunContext] = None) -> str:
            if parent is None:
                target_ref = self.get_target()

                async def _run() -> str:
                    result = await target_ref.arun(task)
                    return _extract_target_output(result, safe_name)

                target_type = _get_target_type(target_ref)
            else:
                context = await self._build_context_async(parent, task, run_context)
                target_ref = context.target
                target_type = context.target_type

                async def _run() -> str:
                    return await _run_target_async(context, run_context)

            task_id = manager.launch(agent_name, target_type, task, _run())
            tracked = manager.get(task_id)
            if tracked is not None:
                tracked._target_ref = target_ref  # type: ignore[attr-defined]

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

            task_id = manager.launch(agent_name, "remote", task, _run())
            return (
                f"Remote task launched. ID: {task_id}  Agent: {agent_name}  URL: {url}\n"
                f"Use check_async_task('{task_id}') to get the result when ready."
            )

        _launch.__name__ = safe_name
        _launch.__qualname__ = safe_name
        _launch.__doc__ = description
        return _launch
