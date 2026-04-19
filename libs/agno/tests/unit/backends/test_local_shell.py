"""Tests for LocalShellBackend."""

import os
import sys
from pathlib import Path

import pytest

from agno.backends.local_shell import LocalShellBackend


# ---------------------------------------------------------------------------
# execute — basic
# ---------------------------------------------------------------------------


def test_execute_simple_command(tmp_path):
    b = LocalShellBackend(root=tmp_path, inherit_env=True)
    result = b.execute("echo hello")
    assert result.exit_code == 0
    assert "hello" in result.output


def test_execute_exit_code_nonzero(tmp_path):
    b = LocalShellBackend(root=tmp_path, inherit_env=True)
    result = b.execute("exit 42")
    assert result.exit_code == 42


def test_execute_stderr_prefixed(tmp_path):
    b = LocalShellBackend(root=tmp_path, inherit_env=True)
    result = b.execute("echo 'err' >&2")
    assert "[stderr]" in result.output


def test_execute_invalid_command(tmp_path):
    b = LocalShellBackend(root=tmp_path, inherit_env=True)
    result = b.execute("this_command_does_not_exist_xyz")
    assert result.exit_code != 0


@pytest.mark.skipif(sys.platform == "win32", reason="timeout test is Unix-specific")
def test_execute_timeout(tmp_path):
    b = LocalShellBackend(root=tmp_path, inherit_env=True, timeout=1)
    result = b.execute("sleep 10")
    assert result.exit_code == 124  # Standard timeout exit code
    assert "timed out" in result.output.lower()


def test_execute_output_truncation(tmp_path):
    b = LocalShellBackend(root=tmp_path, inherit_env=True, max_output_bytes=100)
    # Generate > 100 bytes of output
    result = b.execute("python3 -c \"print('x' * 1000)\"")
    assert result.truncated is True
    assert len(result.output.encode("utf-8")) > 0


def test_execute_empty_command(tmp_path):
    b = LocalShellBackend(root=tmp_path)
    result = b.execute("")
    assert result.exit_code == 1
    assert "non-empty" in result.output.lower() or "Error" in result.output


def test_execute_env_inherit_false(tmp_path):
    """When inherit_env=False, custom env vars should not be present."""
    os.environ["AGNO_TEST_SECRET"] = "should_not_leak"
    b = LocalShellBackend(root=tmp_path, inherit_env=False)
    result = b.execute("echo ${AGNO_TEST_SECRET:-NOT_SET}")
    assert "should_not_leak" not in result.output
    del os.environ["AGNO_TEST_SECRET"]


@pytest.mark.skipif(sys.platform == "win32", reason="bash env var expansion not supported on Windows cmd")
def test_execute_env_inherit_true(tmp_path):
    """When inherit_env=True, env vars should be present."""
    os.environ["AGNO_TEST_VAR"] = "test_value_42"
    b = LocalShellBackend(root=tmp_path, inherit_env=True)
    result = b.execute("echo $AGNO_TEST_VAR")
    assert "test_value_42" in result.output
    del os.environ["AGNO_TEST_VAR"]


@pytest.mark.skipif(sys.platform == "win32", reason="bash env var expansion not supported on Windows cmd")
def test_execute_custom_env(tmp_path):
    b = LocalShellBackend(root=tmp_path, env={"MY_CUSTOM_VAR": "custom_42"}, inherit_env=False)
    result = b.execute("echo $MY_CUSTOM_VAR")
    assert "custom_42" in result.output


# ---------------------------------------------------------------------------
# id property
# ---------------------------------------------------------------------------


def test_id_is_string(tmp_path):
    b = LocalShellBackend(root=tmp_path)
    assert isinstance(b.id, str)
    assert b.id.startswith("local-")


def test_id_is_stable(tmp_path):
    b = LocalShellBackend(root=tmp_path)
    assert b.id == b.id  # Same instance, same ID


def test_id_differs_per_instance(tmp_path):
    b1 = LocalShellBackend(root=tmp_path)
    b2 = LocalShellBackend(root=tmp_path)
    assert b1.id != b2.id


# ---------------------------------------------------------------------------
# Inherits filesystem ops
# ---------------------------------------------------------------------------


def test_inherits_write_and_read(tmp_path):
    b = LocalShellBackend(root=tmp_path)
    write_result = b.write(str(tmp_path / "test.txt"), "hello from shell backend")
    assert write_result.error is None
    read_result = b.read(str(tmp_path / "test.txt"))
    assert "hello from shell backend" in read_result.content


def test_inherits_ls(tmp_path):
    (tmp_path / "file.txt").write_text("content")
    b = LocalShellBackend(root=tmp_path)
    result = b.ls(".")
    assert result.error is None
    assert len(result.entries) >= 1


# ---------------------------------------------------------------------------
# execute works in root directory
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="'ls' not available on Windows cmd")
def test_execute_runs_in_root_dir(tmp_path):
    (tmp_path / "marker.txt").write_text("marker content")
    b = LocalShellBackend(root=tmp_path, inherit_env=True)
    result = b.execute("ls marker.txt")
    assert result.exit_code == 0
    assert "marker.txt" in result.output


# ---------------------------------------------------------------------------
# timeout validation
# ---------------------------------------------------------------------------


def test_invalid_timeout_raises(tmp_path):
    with pytest.raises(ValueError):
        LocalShellBackend(root=tmp_path, timeout=0)


def test_per_command_timeout_override(tmp_path):
    b = LocalShellBackend(root=tmp_path, inherit_env=True, timeout=60)
    # timeout=1 should be accepted
    result = b.execute("echo ok", timeout=1)
    assert "ok" in result.output


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


async def test_async_execute(tmp_path):
    b = LocalShellBackend(root=tmp_path, inherit_env=True)
    result = await b.aexecute("echo async_test")
    assert "async_test" in result.output
