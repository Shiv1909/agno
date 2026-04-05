"""Tests for BackendToolkit."""

from __future__ import annotations

import base64
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from agno.backends.protocol import (
    BackendProtocol,
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsEntry,
    LsResult,
    ReadResult,
    SandboxBackendProtocol,
    WriteResult,
)
from agno.tools.backend import BackendToolkit


# ---------------------------------------------------------------------------
# Mock backends
# ---------------------------------------------------------------------------


def make_plain_backend() -> MagicMock:
    mock = MagicMock(spec=BackendProtocol)
    mock.ls.return_value = LsResult(entries=[LsEntry(path="/a.py", is_dir=False, size=100)])
    mock.read.return_value = ReadResult(content="hello content")
    mock.write.return_value = WriteResult(path="/new.txt")
    mock.edit.return_value = EditResult(path="/edit.txt", occurrences=1)
    mock.grep.return_value = GrepResult(matches=[GrepMatch(path="/f.py", line_number=5, line_text="import os")])
    mock.glob.return_value = GlobResult(paths=["/a.py", "/b.py"])
    mock.upload_files.return_value = [FileUploadResponse(path="/up.txt")]
    mock.download_files.return_value = [FileDownloadResponse(path="/down.txt", content=b"data")]
    return mock


def make_sandbox_backend() -> MagicMock:
    mock = MagicMock(spec=SandboxBackendProtocol)
    mock.ls.return_value = LsResult(entries=[LsEntry(path="/a.py", is_dir=False, size=100)])
    mock.read.return_value = ReadResult(content="hello content")
    mock.write.return_value = WriteResult(path="/new.txt")
    mock.edit.return_value = EditResult(path="/edit.txt", occurrences=1)
    mock.grep.return_value = GrepResult(matches=[])
    mock.glob.return_value = GlobResult(paths=[])
    mock.upload_files.return_value = [FileUploadResponse(path="/up.txt")]
    mock.download_files.return_value = [FileDownloadResponse(path="/down.txt", content=b"data")]
    mock.execute.return_value = ExecuteResponse(output="command output", exit_code=0)
    return mock


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def test_registers_file_tools_for_plain_backend():
    toolkit = BackendToolkit(make_plain_backend())
    tool_names = list(toolkit.functions.keys())
    assert "ls" in tool_names
    assert "read_file" in tool_names
    assert "write_file" in tool_names
    assert "edit_file" in tool_names
    assert "grep" in tool_names
    assert "glob" in tool_names
    assert "upload_files" in tool_names
    assert "download_files" in tool_names


def test_no_execute_tool_for_plain_backend():
    toolkit = BackendToolkit(make_plain_backend())
    assert "execute" not in toolkit.functions


def test_registers_execute_for_sandbox_backend():
    toolkit = BackendToolkit(make_sandbox_backend())
    assert "execute" in toolkit.functions


def test_total_tools_plain_backend():
    toolkit = BackendToolkit(make_plain_backend())
    assert len(toolkit.functions) == 8


def test_total_tools_sandbox_backend():
    toolkit = BackendToolkit(make_sandbox_backend())
    assert len(toolkit.functions) == 9


# ---------------------------------------------------------------------------
# Tool output — success cases
# ---------------------------------------------------------------------------


def test_ls_tool_formats_output():
    backend = make_plain_backend()
    toolkit = BackendToolkit(backend)
    result = toolkit.ls(".")
    assert "[f]" in result
    assert "a.py" in result
    backend.ls.assert_called_once_with(".")


def test_ls_tool_empty_dir():
    backend = make_plain_backend()
    backend.ls.return_value = LsResult(entries=[])
    toolkit = BackendToolkit(backend)
    result = toolkit.ls("/empty")
    assert "empty" in result.lower()


def test_read_file_tool_returns_content():
    backend = make_plain_backend()
    toolkit = BackendToolkit(backend)
    result = toolkit.read_file("/a.py")
    assert "hello content" in result
    backend.read.assert_called_once_with("/a.py", offset=0, limit=2000)


def test_read_file_tool_with_pagination():
    backend = make_plain_backend()
    toolkit = BackendToolkit(backend)
    toolkit.read_file("/a.py", offset=10, limit=50)
    backend.read.assert_called_once_with("/a.py", offset=10, limit=50)


def test_write_file_tool_success():
    backend = make_plain_backend()
    toolkit = BackendToolkit(backend)
    result = toolkit.write_file("/new.txt", "content")
    assert "Written" in result
    assert "/new.txt" in result


def test_edit_file_tool_success():
    backend = make_plain_backend()
    toolkit = BackendToolkit(backend)
    result = toolkit.edit_file("/edit.txt", "old", "new")
    assert "Edited" in result
    assert "1" in result  # occurrences


def test_grep_tool_formats_matches():
    backend = make_plain_backend()
    toolkit = BackendToolkit(backend)
    result = toolkit.grep("import", path="/src")
    assert "/f.py:5: import os" in result


def test_grep_tool_no_matches():
    backend = make_plain_backend()
    backend.grep.return_value = GrepResult(matches=[])
    toolkit = BackendToolkit(backend)
    result = toolkit.grep("NONEXISTENT")
    assert "No matches" in result


def test_glob_tool_returns_paths():
    backend = make_plain_backend()
    toolkit = BackendToolkit(backend)
    result = toolkit.glob("*.py")
    assert "/a.py" in result
    assert "/b.py" in result


def test_glob_tool_no_matches():
    backend = make_plain_backend()
    backend.glob.return_value = GlobResult(paths=[])
    toolkit = BackendToolkit(backend)
    result = toolkit.glob("*.rs")
    assert "No files matched" in result


def test_execute_tool_returns_output():
    backend = make_sandbox_backend()
    toolkit = BackendToolkit(backend)
    result = toolkit.execute("echo hello")
    assert "command output" in result
    backend.execute.assert_called_once_with("echo hello")


def test_execute_with_timeout():
    backend = make_sandbox_backend()
    toolkit = BackendToolkit(backend)
    toolkit.execute("sleep 1", timeout=30)
    backend.execute.assert_called_once_with("sleep 1", timeout=30)


# ---------------------------------------------------------------------------
# Tool output — error cases
# ---------------------------------------------------------------------------


def test_ls_tool_error():
    backend = make_plain_backend()
    backend.ls.return_value = LsResult(error="Permission denied")
    toolkit = BackendToolkit(backend)
    result = toolkit.ls("/restricted")
    assert "Error" in result
    assert "Permission denied" in result


def test_read_file_tool_error():
    backend = make_plain_backend()
    backend.read.return_value = ReadResult(error="File not found")
    toolkit = BackendToolkit(backend)
    result = toolkit.read_file("/missing.txt")
    assert "Error" in result


def test_write_file_tool_error():
    backend = make_plain_backend()
    backend.write.return_value = WriteResult(error="File already exists")
    toolkit = BackendToolkit(backend)
    result = toolkit.write_file("/exists.txt", "content")
    assert "Error" in result


def test_edit_file_tool_error():
    backend = make_plain_backend()
    backend.edit.return_value = EditResult(error="String not found")
    toolkit = BackendToolkit(backend)
    result = toolkit.edit_file("/f.txt", "missing", "new")
    assert "Error" in result


# ---------------------------------------------------------------------------
# upload / download
# ---------------------------------------------------------------------------


def test_upload_files_tool():
    backend = make_plain_backend()
    toolkit = BackendToolkit(backend)
    files = [{"path": "/up.txt", "content_b64": base64.b64encode(b"hello").decode()}]
    result = toolkit.upload_files(files)
    assert "OK" in result
    backend.upload_files.assert_called_once()


def test_upload_files_invalid_format():
    backend = make_plain_backend()
    toolkit = BackendToolkit(backend)
    result = toolkit.upload_files([{"wrong_key": "value"}])
    assert "Error" in result


def test_download_files_tool():
    backend = make_plain_backend()
    toolkit = BackendToolkit(backend)
    result = toolkit.download_files(["/down.txt"])
    import json
    data = json.loads(result)
    assert data[0]["path"] == "/down.txt"
    assert "content_b64" in data[0]


def test_download_files_with_error():
    backend = make_plain_backend()
    backend.download_files.return_value = [FileDownloadResponse(path="/f.txt", error="file_not_found")]
    toolkit = BackendToolkit(backend)
    import json
    result = json.loads(toolkit.download_files(["/f.txt"]))
    assert result[0]["error"] == "file_not_found"


# ---------------------------------------------------------------------------
# include/exclude tools
# ---------------------------------------------------------------------------


def test_include_tools_filter():
    backend = make_plain_backend()
    toolkit = BackendToolkit(backend, include_tools=["ls", "read_file"])
    assert "ls" in toolkit.functions
    assert "read_file" in toolkit.functions
    assert "write_file" not in toolkit.functions


def test_exclude_tools_filter():
    backend = make_plain_backend()
    toolkit = BackendToolkit(backend, exclude_tools=["upload_files", "download_files"])
    assert "upload_files" not in toolkit.functions
    assert "download_files" not in toolkit.functions
    assert "ls" in toolkit.functions


# ---------------------------------------------------------------------------
# Truncation flag
# ---------------------------------------------------------------------------


def test_read_file_truncated_appends_notice():
    backend = make_plain_backend()
    backend.read.return_value = ReadResult(content="partial content", truncated=True)
    toolkit = BackendToolkit(backend)
    result = toolkit.read_file("/big.txt")
    assert "partial content" in result
    assert "truncated" in result.lower()
