"""Unit tests for agno.agent.subagent — SubAgent, AsyncSubAgent, AsyncTaskManager."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agno.agent.subagent import (
    AsyncSubAgent,
    AsyncTaskManager,
    SubAgent,
    TaskStatus,
    _extract_content,
    _safe_tool_name,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestSafeToolName:
    def test_plain_name(self):
        assert _safe_tool_name("researcher") == "researcher"

    def test_hyphens_replaced(self):
        assert _safe_tool_name("my-agent") == "my_agent"

    def test_spaces_replaced(self):
        assert _safe_tool_name("my agent") == "my_agent"

    def test_mixed(self):
        assert _safe_tool_name("my-cool agent") == "my_cool_agent"


class TestExtractContent:
    def test_none_returns_fallback(self):
        result = _extract_content(None, "agent")
        assert "agent" in result
        assert "no output" in result

    def test_object_with_content_attr(self):
        obj = MagicMock()
        obj.content = "hello from agent"
        assert _extract_content(obj, "agent") == "hello from agent"

    def test_object_with_empty_content_falls_back_to_messages(self):
        msg = MagicMock()
        msg.content = "last message content"
        obj = MagicMock()
        obj.content = ""
        obj.messages = [msg]
        assert _extract_content(obj, "agent") == "last message content"

    def test_plain_string_fallback(self):
        assert _extract_content("raw string", "agent") == "raw string"


# ---------------------------------------------------------------------------
# TaskStatus
# ---------------------------------------------------------------------------

class TestTaskStatus:
    def test_values(self):
        assert TaskStatus.running.value == "running"
        assert TaskStatus.completed.value == "completed"
        assert TaskStatus.error.value == "error"
        assert TaskStatus.cancelled.value == "cancelled"

    def test_is_str(self):
        assert isinstance(TaskStatus.running, str)


# ---------------------------------------------------------------------------
# AsyncTaskManager
# ---------------------------------------------------------------------------

class TestAsyncTaskManager:
    def _make_manager(self) -> AsyncTaskManager:
        return AsyncTaskManager()

    def test_launch_returns_task_id(self):
        async def run():
            mgr = self._make_manager()

            async def _coro():
                return "done"

            task_id = mgr.launch("coder", "write hello.py", _coro())
            assert isinstance(task_id, str)
            assert len(task_id) == 10

            t = mgr.get(task_id)
            assert t is not None
            assert t.agent_name == "coder"
            assert t.original_task == "write hello.py"
            assert t.status == TaskStatus.running

            # Let the task finish
            await asyncio.sleep(0.05)
            assert t.status == TaskStatus.completed
            assert t.result == "done"

        asyncio.run(run())

    def test_get_unknown_returns_none(self):
        mgr = self._make_manager()
        assert mgr.get("nonexistent") is None

    def test_all_tasks_returns_copy(self):
        async def run():
            mgr = self._make_manager()

            async def _coro():
                return "x"

            mgr.launch("a1", "task1", _coro())
            mgr.launch("a2", "task2", _coro())
            await asyncio.sleep(0.05)
            tasks = mgr.all_tasks()
            assert len(tasks) == 2

        asyncio.run(run())

    def test_cancel_running_task(self):
        async def run():
            mgr = self._make_manager()
            event = asyncio.Event()

            async def _long():
                await event.wait()  # blocks until cancelled
                return "never"

            task_id = mgr.launch("coder", "long task", _long())
            await asyncio.sleep(0)  # let task start

            cancelled = mgr.cancel(task_id)
            assert cancelled is True

            await asyncio.sleep(0.05)
            t = mgr.get(task_id)
            assert t.status == TaskStatus.cancelled

        asyncio.run(run())

    def test_cancel_nonexistent_returns_false(self):
        mgr = self._make_manager()
        assert mgr.cancel("bad_id") is False

    def test_on_done_sets_error_on_exception(self):
        async def run():
            mgr = self._make_manager()

            async def _fail():
                raise ValueError("boom")

            task_id = mgr.launch("agent", "failing task", _fail())
            await asyncio.sleep(0.05)
            t = mgr.get(task_id)
            assert t.status == TaskStatus.error
            assert "boom" in t.error

        asyncio.run(run())

    def test_relaunch_replaces_task(self):
        async def run():
            mgr = self._make_manager()
            event = asyncio.Event()

            async def _slow():
                await event.wait()
                return "original"

            async def _fast():
                return "updated"

            task_id = mgr.launch("coder", "original task", _slow())
            await asyncio.sleep(0)

            ok = mgr.relaunch(task_id, "new instr", _fast())
            assert ok is True

            await asyncio.sleep(0.05)
            t = mgr.get(task_id)
            assert t.status == TaskStatus.completed
            assert t.result == "updated"

        asyncio.run(run())

    def test_relaunch_unknown_returns_false(self):
        mgr = self._make_manager()

        async def _coro():
            return "x"

        coro = _coro()
        result = mgr.relaunch("bad", "instr", coro)
        coro.close()  # clean up unawaited coroutine
        assert result is False


# ---------------------------------------------------------------------------
# AsyncTaskManager lifecycle tools
# ---------------------------------------------------------------------------

class TestLifecycleTools:
    def test_make_lifecycle_tools_returns_four(self):
        mgr = AsyncTaskManager()
        tools = mgr.make_lifecycle_tools()
        assert len(tools) == 4
        names = {fn.__name__ for fn in tools}
        assert names == {
            "check_async_task",
            "update_async_task",
            "cancel_async_task",
            "list_async_tasks",
        }

    def test_check_unknown_task(self):
        async def run():
            mgr = AsyncTaskManager()
            check = mgr._make_check()
            result = await check("bad_id")
            assert "No task found" in result

        asyncio.run(run())

    def test_check_completed_task_shows_result(self):
        async def run():
            mgr = AsyncTaskManager()

            async def _coro():
                return "script output"

            task_id = mgr.launch("coder", "write script", _coro())
            await asyncio.sleep(0.05)

            check = mgr._make_check()
            result = await check(task_id)
            assert "completed" in result
            assert "script output" in result

        asyncio.run(run())

    def test_check_running_task_shows_elapsed(self):
        async def run():
            mgr = AsyncTaskManager()
            event = asyncio.Event()

            async def _slow():
                await event.wait()
                return "done"

            task_id = mgr.launch("coder", "slow task", _slow())
            await asyncio.sleep(0)

            check = mgr._make_check()
            result = await check(task_id)
            assert "running" in result.lower()
            event.set()

        asyncio.run(run())

    def test_cancel_tool_unknown_task(self):
        async def run():
            mgr = AsyncTaskManager()
            cancel = mgr._make_cancel()
            result = await cancel("bad_id")
            assert "No task found" in result

        asyncio.run(run())

    def test_cancel_tool_running_task(self):
        async def run():
            mgr = AsyncTaskManager()
            event = asyncio.Event()

            async def _slow():
                await event.wait()
                return "never"

            task_id = mgr.launch("coder", "slow", _slow())
            await asyncio.sleep(0)

            cancel = mgr._make_cancel()
            result = await cancel(task_id)
            assert "cancelled successfully" in result

        asyncio.run(run())

    def test_cancel_tool_already_done(self):
        async def run():
            mgr = AsyncTaskManager()

            async def _fast():
                return "done"

            task_id = mgr.launch("coder", "fast", _fast())
            await asyncio.sleep(0.05)

            cancel = mgr._make_cancel()
            result = await cancel(task_id)
            assert "not running" in result

        asyncio.run(run())

    def test_list_empty(self):
        async def run():
            mgr = AsyncTaskManager()
            list_fn = mgr._make_list()
            result = await list_fn()
            assert "No async tasks" in result

        asyncio.run(run())

    def test_list_shows_all_tasks(self):
        async def run():
            mgr = AsyncTaskManager()

            async def _coro():
                return "x"

            mgr.launch("analyst", "analyse files", _coro())
            mgr.launch("coder", "write script", _coro())
            await asyncio.sleep(0.05)

            list_fn = mgr._make_list()
            result = await list_fn()
            assert "analyst" in result
            assert "coder" in result

        asyncio.run(run())

    def test_update_unknown_task(self):
        async def run():
            mgr = AsyncTaskManager()
            update = mgr._make_update()
            result = await update("bad_id", "new instructions")
            assert "No task found" in result

        asyncio.run(run())

    def test_update_completed_task_rejected(self):
        async def run():
            mgr = AsyncTaskManager()

            async def _fast():
                return "done"

            task_id = mgr.launch("coder", "fast task", _fast())
            await asyncio.sleep(0.05)

            update = mgr._make_update()
            result = await update(task_id, "do something else")
            assert "cannot be updated" in result

        asyncio.run(run())


# ---------------------------------------------------------------------------
# SubAgent
# ---------------------------------------------------------------------------

class TestSubAgent:
    def test_safe_name_applied(self):
        sa = SubAgent(name="my-agent", description="desc")
        fn = sa.make_sync_callable()
        assert fn.__name__ == "my_agent"

    def test_sync_callable_calls_agent_run(self):
        mock_agent = MagicMock()
        mock_output = MagicMock()
        mock_output.content = "analysis result"
        mock_agent.run.return_value = mock_output

        sa = SubAgent(name="analyst", description="desc", agent=mock_agent)
        fn = sa.make_sync_callable()
        result = fn("list files")

        mock_agent.run.assert_called_once_with("list files")
        assert result == "analysis result"

    def test_async_callable_calls_agent_arun(self):
        async def run():
            mock_agent = MagicMock()
            mock_output = MagicMock()
            mock_output.content = "async result"
            mock_agent.arun = AsyncMock(return_value=mock_output)

            sa = SubAgent(name="analyst", description="desc", agent=mock_agent)
            fn = sa.make_async_callable()
            result = await fn("analyse logs")

            mock_agent.arun.assert_called_once_with("analyse logs")
            assert result == "async result"

        asyncio.run(run())

    def test_doc_set_to_description(self):
        sa = SubAgent(name="analyst", description="My description")
        fn = sa.make_sync_callable()
        assert fn.__doc__ == "My description"

    def test_get_agent_caches(self):
        mock_agent = MagicMock()
        sa = SubAgent(name="analyst", description="desc", agent=mock_agent)
        a1 = sa.get_agent()
        a2 = sa.get_agent()
        assert a1 is a2

    def test_get_agent_builds_from_config(self):
        # Agent is imported lazily inside get_agent(), so patch at its source module
        with patch("agno.agent.agent.Agent") as MockAgent:
            mock_instance = MagicMock()
            MockAgent.return_value = mock_instance

            sa = SubAgent(
                name="analyst",
                description="desc",
                instructions=["be helpful"],
                markdown=True,
            )
            agent = sa.get_agent()
            MockAgent.assert_called_once()
            assert agent is mock_instance


# ---------------------------------------------------------------------------
# AsyncSubAgent
# ---------------------------------------------------------------------------

class TestAsyncSubAgent:
    def test_launch_returns_task_id_immediately(self):
        async def run():
            mock_agent = MagicMock()
            event = asyncio.Event()

            async def _slow_arun(task):
                await event.wait()
                out = MagicMock()
                out.content = "done"
                return out

            mock_agent.arun = _slow_arun

            sa = AsyncSubAgent(name="coder", description="desc", agent=mock_agent)
            mgr = AsyncTaskManager()
            launch = sa.make_launch_callable(mgr)

            result = await launch("write stats.py")
            assert "Task launched" in result
            assert "coder" in result

            # task_id in result
            lines = result.split()
            id_idx = lines.index("ID:") + 1
            task_id = lines[id_idx]
            assert len(task_id) == 10
            event.set()

        asyncio.run(run())

    def test_launch_doc_prefixed_background(self):
        sa = AsyncSubAgent(name="coder", description="Python helper")
        mgr = AsyncTaskManager()
        fn = sa.make_launch_callable(mgr)
        assert "[background]" in fn.__doc__

    def test_launch_name_is_safe(self):
        sa = AsyncSubAgent(name="my-coder", description="desc")
        mgr = AsyncTaskManager()
        fn = sa.make_launch_callable(mgr)
        assert fn.__name__ == "my_coder"

    def test_task_completes_and_is_checkable(self):
        async def run():
            mock_agent = MagicMock()
            mock_output = MagicMock()
            mock_output.content = "primes: 2 3 5 7 11"
            mock_agent.arun = AsyncMock(return_value=mock_output)

            sa = AsyncSubAgent(name="coder", description="desc", agent=mock_agent)
            mgr = AsyncTaskManager()
            launch = sa.make_launch_callable(mgr)

            result = await launch("print primes")
            # Extract task_id from result message
            for word in result.split():
                if len(word) == 10 and word.isalnum():
                    task_id = word
                    break

            await asyncio.sleep(0.1)

            check = mgr._make_check()
            status = await check(task_id)
            assert "completed" in status
            assert "primes" in status

        asyncio.run(run())

    def test_url_launch_uses_http(self):
        """Verify HTTP launch callable is selected when url= is set."""
        sa = AsyncSubAgent(name="remote", description="desc", url="http://example.com/agent")
        mgr = AsyncTaskManager()
        fn = sa.make_launch_callable(mgr)
        assert "[background/remote]" in fn.__doc__
