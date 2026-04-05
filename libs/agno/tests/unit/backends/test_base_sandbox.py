"""Tests for BaseSandbox — all file ops derived from execute()."""

from __future__ import annotations

import json
from typing import Optional
from unittest.mock import patch

import pytest

from agno.backends.protocol import (
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)
from agno.backends.sandbox import BaseSandbox


# ---------------------------------------------------------------------------
# MockSandbox: in-memory implementation for testing BaseSandbox logic
# ---------------------------------------------------------------------------


class MockSandbox(BaseSandbox):
    """In-memory sandbox that simulates execute() + file storage."""

    def __init__(self):
        self._files: dict[str, bytes] = {}
        self._executed: list[str] = []
        self._execute_responses: list[ExecuteResponse] = []

    @property
    def id(self) -> str:
        return "mock-sandbox-001"

    def execute(self, command: str, *, timeout: Optional[int] = None) -> ExecuteResponse:
        self._executed.append(command)
        if self._execute_responses:
            return self._execute_responses.pop(0)
        # Default: run via actual subprocess on host for integration-style tests
        import subprocess
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=10
            )
            return ExecuteResponse(
                output=result.stdout + result.stderr,
                exit_code=result.returncode,
            )
        except Exception as e:
            return ExecuteResponse(output=str(e), exit_code=1)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        responses = []
        for path, content in files:
            self._files[path] = content
            responses.append(FileUploadResponse(path=path))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses = []
        for path in paths:
            if path in self._files:
                responses.append(FileDownloadResponse(path=path, content=self._files[path]))
            else:
                responses.append(FileDownloadResponse(path=path, error="file_not_found"))
        return responses

    def queue_response(self, response: ExecuteResponse) -> None:
        """Queue a specific response for the next execute() call."""
        self._execute_responses.append(response)


# ---------------------------------------------------------------------------
# id
# ---------------------------------------------------------------------------


def test_id_property():
    sb = MockSandbox()
    assert sb.id == "mock-sandbox-001"


# ---------------------------------------------------------------------------
# ls — calls execute with a python3 command
# ---------------------------------------------------------------------------


def test_ls_calls_execute(tmp_path):
    # Use actual subprocess so ls works on real paths
    sb = MockSandbox()
    (tmp_path / "file.txt").write_text("hello")
    result = sb.ls(str(tmp_path))
    assert result.error is None
    # execute was called
    assert len(sb._executed) >= 1
    assert "scandir" in sb._executed[0] or "base64" in sb._executed[0]


def test_ls_returns_entries(tmp_path):
    sb = MockSandbox()
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.txt").write_text("")
    # Pre-queue what python3 scandir would emit (avoids python3 availability on Windows)
    import json as _json
    lines = "\n".join([
        _json.dumps({"path": str(tmp_path / "a.py"), "is_dir": False, "size": 0}),
        _json.dumps({"path": str(tmp_path / "b.txt"), "is_dir": False, "size": 0}),
    ])
    sb.queue_response(ExecuteResponse(output=lines, exit_code=0))
    result = sb.ls(str(tmp_path))
    assert result.error is None
    assert len(result.entries) == 2


# ---------------------------------------------------------------------------
# read — runs paginated Python3 script via execute()
# ---------------------------------------------------------------------------


def test_read_calls_execute():
    sb = MockSandbox()
    # Queue a JSON response that read() expects
    content_json = json.dumps({"encoding": "utf-8", "content": "line1\nline2", "truncated": False})
    sb.queue_response(ExecuteResponse(output=content_json, exit_code=0))
    result = sb.read("/fake/path.txt", offset=0, limit=100)
    assert result.error is None
    assert "line1" in result.content
    assert len(sb._executed) == 1


def test_read_handles_error_response():
    sb = MockSandbox()
    error_json = json.dumps({"error": "File not found: /missing.txt"})
    sb.queue_response(ExecuteResponse(output=error_json, exit_code=0))
    result = sb.read("/missing.txt")
    assert result.error is not None
    assert "not found" in result.error.lower() or "missing" in result.error


def test_read_handles_binary():
    import base64
    sb = MockSandbox()
    raw = b"\x89PNG\r\n"
    encoded = base64.b64encode(raw).decode("ascii")
    binary_json = json.dumps({"encoding": "base64", "content": encoded})
    sb.queue_response(ExecuteResponse(output=binary_json, exit_code=0))
    result = sb.read("/image.png")
    assert result.error is None
    assert result.encoding == "base64"
    assert result.content == encoded


# ---------------------------------------------------------------------------
# write — preflight check + upload_files
# ---------------------------------------------------------------------------


def test_write_calls_upload(tmp_path):
    sb = MockSandbox()
    # Queue success response for the preflight check script
    sb.queue_response(ExecuteResponse(output="", exit_code=0))
    result = sb.write("/sandbox/new_file.txt", "hello world")
    assert result.error is None
    # File should be in mock storage
    assert "/sandbox/new_file.txt" in sb._files
    assert sb._files["/sandbox/new_file.txt"] == b"hello world"


def test_write_fails_if_preflight_fails():
    sb = MockSandbox()
    sb.queue_response(ExecuteResponse(output="Error: File already exists: '/sandbox/file.txt'", exit_code=1))
    result = sb.write("/sandbox/file.txt", "content")
    assert result.error is not None
    assert "already exists" in result.error or "Error" in result.error


# ---------------------------------------------------------------------------
# edit — inline path (small payload)
# ---------------------------------------------------------------------------


def test_edit_inline_success():
    sb = MockSandbox()
    success_json = json.dumps({"count": 1})
    sb.queue_response(ExecuteResponse(output=success_json, exit_code=0))
    result = sb.edit("/sandbox/file.txt", "old text", "new text")
    assert result.error is None
    assert result.occurrences == 1


def test_edit_inline_string_not_found():
    sb = MockSandbox()
    sb.queue_response(ExecuteResponse(output=json.dumps({"error": "string_not_found"}), exit_code=0))
    result = sb.edit("/sandbox/file.txt", "missing", "replacement")
    assert result.error is not None
    assert "not found" in result.error.lower()


def test_edit_inline_multiple_occurrences():
    sb = MockSandbox()
    sb.queue_response(ExecuteResponse(output=json.dumps({"error": "multiple_occurrences", "count": 3}), exit_code=0))
    result = sb.edit("/sandbox/file.txt", "foo", "bar")
    assert result.error is not None
    assert "3" in result.error or "multiple" in result.error.lower()


def test_edit_inline_file_not_found():
    sb = MockSandbox()
    sb.queue_response(ExecuteResponse(output=json.dumps({"error": "file_not_found"}), exit_code=0))
    result = sb.edit("/sandbox/missing.txt", "old", "new")
    assert result.error is not None
    assert "not found" in result.error.lower()


# ---------------------------------------------------------------------------
# edit — upload-based path (large payload)
# ---------------------------------------------------------------------------


def test_edit_large_payload_uses_upload():
    sb = MockSandbox()
    # Large old_text to trigger upload path (> 50KB)
    large_old = "x" * 60_000
    large_new = "y" * 60_000
    # Queue the execute response for the tmpfile replace script
    sb.queue_response(ExecuteResponse(output=json.dumps({"count": 1}), exit_code=0))
    result = sb.edit("/sandbox/big.txt", large_old, large_new)
    assert result.error is None
    # Two temp files should have been uploaded
    uploaded = [p for p in sb._files if ".agno_edit_" in p]
    assert len(uploaded) == 2


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------


def test_grep_calls_execute():
    sb = MockSandbox()
    sb.queue_response(ExecuteResponse(output="/sandbox/a.py:5:import os\n/sandbox/b.py:3:import sys", exit_code=0))
    result = sb.grep("import", path="/sandbox")
    assert result.error is None
    assert len(result.matches) == 2
    assert result.matches[0].path == "/sandbox/a.py"
    assert result.matches[0].line_number == 5


def test_grep_no_matches():
    sb = MockSandbox()
    sb.queue_response(ExecuteResponse(output="", exit_code=0))
    result = sb.grep("NONEXISTENT", path="/sandbox")
    assert result.error is None
    assert result.matches == []


# ---------------------------------------------------------------------------
# glob
# ---------------------------------------------------------------------------


def test_glob_calls_execute():
    sb = MockSandbox()
    lines = [
        json.dumps({"path": "src/a.py", "is_dir": False}),
        json.dumps({"path": "src/b.py", "is_dir": False}),
        json.dumps({"path": "src/", "is_dir": True}),  # dirs should be excluded
    ]
    sb.queue_response(ExecuteResponse(output="\n".join(lines), exit_code=0))
    result = sb.glob("**/*.py", path="/sandbox")
    assert result.error is None
    assert len(result.paths) == 2
    assert "src/a.py" in result.paths


# ---------------------------------------------------------------------------
# Async wrappers
# ---------------------------------------------------------------------------


async def test_async_execute():
    sb = MockSandbox()
    sb.queue_response(ExecuteResponse(output="hello", exit_code=0))
    result = await sb.aexecute("echo hello")
    assert "hello" in result.output


async def test_async_write():
    sb = MockSandbox()
    sb.queue_response(ExecuteResponse(output="", exit_code=0))  # preflight
    result = await sb.awrite("/sandbox/async.txt", "async content")
    assert result.error is None
