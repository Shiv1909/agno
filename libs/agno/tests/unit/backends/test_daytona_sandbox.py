"""Tests for DaytonaSandbox."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch


# Mock the daytona SDK before importing
mock_daytona_mod = MagicMock()
mock_sandbox_instance = MagicMock()
mock_sandbox_instance.id = "daytona-test-456"
mock_sandbox_instance.process = MagicMock()
mock_sandbox_instance.fs = MagicMock()

mock_client = MagicMock()
mock_client.create.return_value = mock_sandbox_instance
mock_client.get.return_value = mock_sandbox_instance

mock_daytona_mod.Daytona = MagicMock(return_value=mock_client)
mock_daytona_mod.DaytonaConfig = MagicMock()
mock_daytona_mod.CreateSandboxFromSnapshotParams = MagicMock()

sys.modules["daytona"] = mock_daytona_mod

import os

os.environ.setdefault("DAYTONA_API_KEY", "test_key")

from agno.backends.daytona import DaytonaSandbox
from agno.backends.protocol import ExecuteResponse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_sandbox() -> DaytonaSandbox:
    with patch("agno.backends.daytona.Daytona", return_value=mock_client):
        with patch("agno.backends.daytona.DaytonaConfig"):
            sb = DaytonaSandbox.__new__(DaytonaSandbox)
            sb._client = mock_client
            sb._sandbox = mock_sandbox_instance
    return sb


# ---------------------------------------------------------------------------
# id
# ---------------------------------------------------------------------------


def test_daytona_sandbox_id():
    sb = make_sandbox()
    assert sb.id == "daytona-test-456"


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------


def test_execute_calls_process_exec():
    sb = make_sandbox()
    mock_result = MagicMock()
    mock_result.result = "command output"
    mock_result.stderr = ""
    mock_result.exit_code = 0
    sb._sandbox.process.exec.return_value = mock_result

    result = sb.execute("echo hello")
    sb._sandbox.process.exec.assert_called_once_with("echo hello", timeout=None)
    assert "command output" in result.output
    assert result.exit_code == 0


def test_execute_with_stderr():
    sb = make_sandbox()
    mock_result = MagicMock()
    mock_result.result = ""
    mock_result.stderr = "error line"
    mock_result.exit_code = 1
    sb._sandbox.process.exec.return_value = mock_result

    result = sb.execute("bad command")
    assert "[stderr]" in result.output
    assert result.exit_code == 1


def test_execute_with_timeout():
    sb = make_sandbox()
    mock_result = MagicMock()
    mock_result.result = "ok"
    mock_result.stderr = ""
    mock_result.exit_code = 0
    sb._sandbox.process.exec.return_value = mock_result
    sb._sandbox.process.exec.reset_mock()  # clear calls from earlier tests (shared module mock)

    sb.execute("sleep 1", timeout=30)
    sb._sandbox.process.exec.assert_called_once_with("sleep 1", timeout=30)


def test_execute_exception_returns_error():
    sb = make_sandbox()
    sb._sandbox.process.exec.side_effect = Exception("connection error")
    result = sb.execute("any command")
    assert result.exit_code == 1
    assert "Error" in result.output


def test_execute_no_output_returns_placeholder():
    sb = make_sandbox()
    mock_result = MagicMock()
    mock_result.result = ""
    mock_result.stderr = ""
    mock_result.exit_code = 0
    sb._sandbox.process.exec.return_value = mock_result

    result = sb.execute("true")
    assert result.output  # should have some placeholder like "<no output>"


# ---------------------------------------------------------------------------
# upload_files
# ---------------------------------------------------------------------------


def test_upload_files_success():
    sb = make_sandbox()
    sb._sandbox.fs.upload_file = MagicMock()
    resps = sb.upload_files([("/sandbox/a.txt", b"content")])
    assert len(resps) == 1
    assert resps[0].error is None
    assert resps[0].path == "/sandbox/a.txt"
    sb._sandbox.fs.upload_file.assert_called_once()


def test_upload_files_fallback_on_fs_error():
    sb = make_sandbox()
    sb._sandbox.fs.upload_file = MagicMock(side_effect=Exception("fs.upload_file failed"))
    # Fallback uses execute; mock it to succeed
    mock_result = ExecuteResponse(output="", exit_code=0)
    sb._sandbox.process.exec.return_value = MagicMock(result="", stderr="", exit_code=0)

    # Patch execute to return success
    with patch.object(sb, "execute", return_value=ExecuteResponse(output="", exit_code=0)):
        resps = sb.upload_files([("/sandbox/a.txt", b"fallback")])

    assert len(resps) == 1
    assert resps[0].error is None


def test_upload_files_fallback_execute_fails():
    sb = make_sandbox()
    sb._sandbox.fs.upload_file = MagicMock(side_effect=Exception("fs failed"))

    with patch.object(sb, "execute", return_value=ExecuteResponse(output="Error: disk full", exit_code=1)):
        resps = sb.upload_files([("/sandbox/a.txt", b"data")])

    assert resps[0].error is not None


def test_upload_multiple_files():
    sb = make_sandbox()
    sb._sandbox.fs.upload_file = MagicMock()
    files = [("/a.txt", b"a"), ("/b.txt", b"b"), ("/c.txt", b"c")]
    resps = sb.upload_files(files)
    assert len(resps) == 3
    assert all(r.error is None for r in resps)


# ---------------------------------------------------------------------------
# download_files
# ---------------------------------------------------------------------------


def test_download_files_success():
    sb = make_sandbox()
    sb._sandbox.fs.download_file = MagicMock(return_value=b"file content")
    resps = sb.download_files(["/sandbox/a.txt"])
    assert len(resps) == 1
    assert resps[0].error is None
    assert resps[0].content == b"file content"


def test_download_files_string_converted_to_bytes():
    sb = make_sandbox()
    sb._sandbox.fs.download_file = MagicMock(return_value="string content")
    resps = sb.download_files(["/sandbox/a.txt"])
    assert isinstance(resps[0].content, bytes)
    assert resps[0].content == b"string content"


def test_download_files_fallback_on_fs_error():
    import base64

    sb = make_sandbox()
    sb._sandbox.fs.download_file = MagicMock(side_effect=Exception("fs.download_file failed"))
    raw = b"fallback content"
    encoded = base64.b64encode(raw).decode()

    with patch.object(sb, "execute", return_value=ExecuteResponse(output=encoded + "\n", exit_code=0)):
        resps = sb.download_files(["/sandbox/a.txt"])

    assert resps[0].error is None
    assert resps[0].content == raw


def test_download_files_fallback_execute_fails():
    sb = make_sandbox()
    sb._sandbox.fs.download_file = MagicMock(side_effect=Exception("fs failed"))

    with patch.object(sb, "execute", return_value=ExecuteResponse(output="Error: not found", exit_code=1)):
        resps = sb.download_files(["/missing.txt"])

    assert resps[0].error is not None


def test_download_files_error():
    sb = make_sandbox()
    sb._sandbox.fs.download_file = MagicMock(side_effect=Exception("read failed"))
    with patch.object(sb, "execute", side_effect=Exception("execute also failed")):
        resps = sb.download_files(["/missing.txt"])
    assert resps[0].error is not None


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


def test_close_calls_client_delete():
    sb = make_sandbox()
    sb.close()
    sb._client.delete.assert_called_once_with(sb._sandbox)


def test_close_silences_exception():
    sb = make_sandbox()
    sb._client.delete.side_effect = Exception("already deleted")
    # Should not raise
    sb.close()
