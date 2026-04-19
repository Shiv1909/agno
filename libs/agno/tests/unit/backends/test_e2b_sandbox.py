"""Tests for E2BSandbox."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch


# Mock the e2b_code_interpreter SDK before importing
mock_e2b = MagicMock()
mock_e2b_sandbox_instance = MagicMock()
mock_e2b_sandbox_instance.sandbox_id = "e2b-test-123"
mock_e2b_sandbox_instance.commands = MagicMock()
mock_e2b_sandbox_instance.files = MagicMock()

mock_e2b.Sandbox = MagicMock()
mock_e2b.Sandbox.create = MagicMock(return_value=mock_e2b_sandbox_instance)
sys.modules["e2b_code_interpreter"] = mock_e2b

import os

os.environ.setdefault("E2B_API_KEY", "test_key")

from agno.backends.e2b import E2BSandbox
from agno.backends.protocol import ExecuteResponse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_sandbox() -> E2BSandbox:
    with patch("agno.backends.e2b.E2BSdk", mock_e2b.Sandbox):
        sb = E2BSandbox(api_key="test_key")
        sb._sandbox = mock_e2b_sandbox_instance
    return sb


# ---------------------------------------------------------------------------
# id
# ---------------------------------------------------------------------------


def test_e2b_sandbox_id():
    sb = make_sandbox()
    assert sb.id == "e2b-test-123"


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------


def test_execute_calls_commands_run():
    sb = make_sandbox()
    mock_result = MagicMock()
    mock_result.stdout = "hello output"
    mock_result.stderr = ""
    mock_result.exit_code = 0
    sb._sandbox.commands.run.return_value = mock_result

    result = sb.execute("echo hello")
    sb._sandbox.commands.run.assert_called_once_with("echo hello")
    assert "hello output" in result.output
    assert result.exit_code == 0


def test_execute_with_stderr():
    sb = make_sandbox()
    mock_result = MagicMock()
    mock_result.stdout = ""
    mock_result.stderr = "error line"
    mock_result.exit_code = 1
    sb._sandbox.commands.run.return_value = mock_result

    result = sb.execute("bad command")
    assert "[stderr]" in result.output
    assert result.exit_code == 1


def test_execute_with_timeout():
    sb = make_sandbox()
    mock_result = MagicMock()
    mock_result.stdout = "ok"
    mock_result.stderr = ""
    mock_result.exit_code = 0
    sb._sandbox.commands.run.return_value = mock_result

    sb.execute("sleep 1", timeout=30)
    call_kwargs = sb._sandbox.commands.run.call_args[1]
    assert call_kwargs.get("timeout") == 30


def test_execute_exception_returns_error():
    sb = make_sandbox()
    sb._sandbox.commands.run.side_effect = Exception("connection error")
    result = sb.execute("any command")
    assert result.exit_code == 1
    assert "Error" in result.output


# ---------------------------------------------------------------------------
# upload_files
# ---------------------------------------------------------------------------


def test_upload_files_success():
    sb = make_sandbox()
    sb._sandbox.files.write = MagicMock()
    resps = sb.upload_files([("/sandbox/a.txt", b"content")])
    assert len(resps) == 1
    assert resps[0].error is None
    assert resps[0].path == "/sandbox/a.txt"
    sb._sandbox.files.write.assert_called_once()


def test_upload_files_error():
    sb = make_sandbox()
    sb._sandbox.files.write.side_effect = Exception("write failed")
    resps = sb.upload_files([("/sandbox/a.txt", b"content")])
    assert resps[0].error is not None
    assert "write failed" in resps[0].error


def test_upload_multiple_files():
    sb = make_sandbox()
    sb._sandbox.files.write = MagicMock()
    files = [("/a.txt", b"a"), ("/b.txt", b"b"), ("/c.txt", b"c")]
    resps = sb.upload_files(files)
    assert len(resps) == 3
    assert all(r.error is None for r in resps)


# ---------------------------------------------------------------------------
# download_files
# ---------------------------------------------------------------------------


def test_download_files_success():
    sb = make_sandbox()
    sb._sandbox.files.read = MagicMock(return_value=b"file content")
    resps = sb.download_files(["/sandbox/a.txt"])
    assert len(resps) == 1
    assert resps[0].error is None
    assert resps[0].content == b"file content"


def test_download_files_string_content_converted():
    sb = make_sandbox()
    sb._sandbox.files.read = MagicMock(return_value="string content")
    resps = sb.download_files(["/sandbox/a.txt"])
    assert isinstance(resps[0].content, bytes)
    assert resps[0].content == b"string content"


def test_download_files_error():
    sb = make_sandbox()
    sb._sandbox.files.read.side_effect = Exception("read failed")
    resps = sb.download_files(["/missing.txt"])
    assert resps[0].error is not None
    assert "read failed" in resps[0].error


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


def test_close_calls_kill():
    sb = make_sandbox()
    sb.close()
    sb._sandbox.kill.assert_called_once()


def test_close_silences_exception():
    sb = make_sandbox()
    sb._sandbox.kill.side_effect = Exception("already dead")
    # Should not raise
    sb.close()
