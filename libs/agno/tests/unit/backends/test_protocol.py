"""Tests for backend protocol result types and base classes."""

import asyncio
from unittest.mock import MagicMock

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
    execute_accepts_timeout,
)


# ---------------------------------------------------------------------------
# Result type tests
# ---------------------------------------------------------------------------


def test_ls_result_defaults():
    r = LsResult()
    assert r.entries == []
    assert r.error is None


def test_ls_result_with_entries():
    entries = [LsEntry(path="/foo.py", is_dir=False, size=100)]
    r = LsResult(entries=entries)
    assert len(r.entries) == 1
    assert r.entries[0].path == "/foo.py"


def test_ls_entry_is_dir():
    e = LsEntry(path="/mydir/", is_dir=True)
    assert e.is_dir is True
    assert e.size is None


def test_read_result_defaults():
    r = ReadResult()
    assert r.content is None
    assert r.encoding == "utf-8"
    assert r.truncated is False
    assert r.error is None


def test_read_result_with_content():
    r = ReadResult(content="hello world", encoding="utf-8")
    assert r.content == "hello world"
    assert r.truncated is False


def test_write_result_success():
    r = WriteResult(path="/tmp/foo.txt")
    assert r.path == "/tmp/foo.txt"
    assert r.error is None


def test_write_result_error():
    r = WriteResult(error="File already exists")
    assert r.path is None
    assert r.error == "File already exists"


def test_edit_result_success():
    r = EditResult(path="/tmp/foo.txt", occurrences=2)
    assert r.path == "/tmp/foo.txt"
    assert r.occurrences == 2
    assert r.error is None


def test_grep_match():
    m = GrepMatch(path="/foo.py", line_number=42, line_text="import os")
    assert m.path == "/foo.py"
    assert m.line_number == 42
    assert m.line_text == "import os"


def test_grep_result_empty():
    r = GrepResult()
    assert r.matches == []
    assert r.error is None


def test_glob_result_empty():
    r = GlobResult()
    assert r.paths == []
    assert r.error is None


def test_file_upload_response_success():
    r = FileUploadResponse(path="/foo.txt")
    assert r.path == "/foo.txt"
    assert r.error is None


def test_file_upload_response_error():
    r = FileUploadResponse(path="/foo.txt", error="permission_denied")
    assert r.error == "permission_denied"


def test_file_download_response_success():
    r = FileDownloadResponse(path="/foo.txt", content=b"hello")
    assert r.content == b"hello"
    assert r.error is None


def test_execute_response_defaults():
    r = ExecuteResponse(output="hello", exit_code=0)
    assert r.output == "hello"
    assert r.exit_code == 0
    assert r.truncated is False


# ---------------------------------------------------------------------------
# BackendProtocol raises NotImplementedError
# ---------------------------------------------------------------------------


class MinimalBackend(BackendProtocol):
    """Concrete subclass that implements nothing — tests NotImplementedError."""
    pass


def test_backend_ls_not_implemented():
    b = MinimalBackend()
    with pytest.raises(NotImplementedError):
        b.ls()


def test_backend_read_not_implemented():
    b = MinimalBackend()
    with pytest.raises(NotImplementedError):
        b.read("/foo.txt")


def test_backend_write_not_implemented():
    b = MinimalBackend()
    with pytest.raises(NotImplementedError):
        b.write("/foo.txt", "content")


def test_backend_edit_not_implemented():
    b = MinimalBackend()
    with pytest.raises(NotImplementedError):
        b.edit("/foo.txt", "old", "new")


def test_backend_grep_not_implemented():
    b = MinimalBackend()
    with pytest.raises(NotImplementedError):
        b.grep("pattern")


def test_backend_glob_not_implemented():
    b = MinimalBackend()
    with pytest.raises(NotImplementedError):
        b.glob("*.py")


def test_backend_upload_not_implemented():
    b = MinimalBackend()
    with pytest.raises(NotImplementedError):
        b.upload_files([("/foo.txt", b"content")])


def test_backend_download_not_implemented():
    b = MinimalBackend()
    with pytest.raises(NotImplementedError):
        b.download_files(["/foo.txt"])


# ---------------------------------------------------------------------------
# Async defaults via asyncio.to_thread
# ---------------------------------------------------------------------------


class SyncBackend(BackendProtocol):
    """Backend with sync implementations to test async default wrappers."""

    def ls(self, path: str = ".") -> LsResult:
        return LsResult(entries=[LsEntry(path="test.txt", is_dir=False)])

    def read(self, path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        return ReadResult(content=f"content of {path}")

    def write(self, path: str, content: str) -> WriteResult:
        return WriteResult(path=path)

    def edit(self, path: str, old_text: str, new_text: str, replace_all: bool = False) -> EditResult:
        return EditResult(path=path, occurrences=1)

    def grep(self, pattern: str, path=None, glob=None) -> GrepResult:
        return GrepResult(matches=[GrepMatch(path="f.py", line_number=1, line_text=pattern)])

    def glob(self, pattern: str, path=None) -> GlobResult:
        return GlobResult(paths=["a.py", "b.py"])

    def upload_files(self, files) -> list[FileUploadResponse]:
        return [FileUploadResponse(path=p) for p, _ in files]

    def download_files(self, paths) -> list[FileDownloadResponse]:
        return [FileDownloadResponse(path=p, content=b"data") for p in paths]


async def test_async_ls_calls_sync():
    b = SyncBackend()
    result = await b.als()
    assert len(result.entries) == 1


async def test_async_read_calls_sync():
    b = SyncBackend()
    result = await b.aread("/foo.txt")
    assert "foo.txt" in result.content


async def test_async_write_calls_sync():
    b = SyncBackend()
    result = await b.awrite("/foo.txt", "hello")
    assert result.path == "/foo.txt"


async def test_async_grep_calls_sync():
    b = SyncBackend()
    result = await b.agrep("TODO")
    assert len(result.matches) == 1


# ---------------------------------------------------------------------------
# execute_accepts_timeout
# ---------------------------------------------------------------------------


class SandboxWithTimeout(SandboxBackendProtocol):
    @property
    def id(self):
        return "test"

    def execute(self, command: str, *, timeout=None) -> ExecuteResponse:
        return ExecuteResponse(output="ok", exit_code=0)

    def upload_files(self, files):
        return []

    def download_files(self, paths):
        return []


class SandboxWithoutTimeout(SandboxBackendProtocol):
    @property
    def id(self):
        return "test"

    def execute(self, command: str) -> ExecuteResponse:
        return ExecuteResponse(output="ok", exit_code=0)

    def upload_files(self, files):
        return []

    def download_files(self, paths):
        return []


def test_execute_accepts_timeout_true():
    assert execute_accepts_timeout(SandboxWithTimeout) is True


def test_execute_accepts_timeout_false():
    assert execute_accepts_timeout(SandboxWithoutTimeout) is False
