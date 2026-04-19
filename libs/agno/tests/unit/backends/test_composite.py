"""Tests for CompositeBackend routing and aggregation."""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock

import pytest

from agno.backends.composite import CompositeBackend, _route_for_path
from agno.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileDownloadResponse,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsEntry,
    LsResult,
    ReadResult,
    WriteResult,
)


# ---------------------------------------------------------------------------
# Helpers: mock backends
# ---------------------------------------------------------------------------


def make_mock_backend(name: str = "mock") -> MagicMock:
    """Create a mock BackendProtocol."""
    mock = MagicMock(spec=BackendProtocol)
    mock.ls.return_value = LsResult(entries=[LsEntry(path=f"/{name}/file.txt", is_dir=False)])
    mock.read.return_value = ReadResult(content=f"content from {name}")
    mock.write.return_value = WriteResult(path="/written.txt")
    mock.edit.return_value = EditResult(path="/edited.txt", occurrences=1)
    mock.grep.return_value = GrepResult(matches=[GrepMatch(path=f"/{name}/f.py", line_number=1, line_text="match")])
    mock.glob.return_value = GlobResult(paths=[f"/{name}/a.py"])
    mock.upload_files.return_value = [FileUploadResponse(path="/up.txt")]
    mock.download_files.return_value = [FileDownloadResponse(path="/down.txt", content=b"data")]
    # Async mirrors
    import asyncio
    mock.als = MagicMock(side_effect=lambda *a, **kw: asyncio.coroutine(lambda: mock.ls(*a, **kw))())
    return mock


# ---------------------------------------------------------------------------
# _route_for_path unit tests
# ---------------------------------------------------------------------------


def test_route_exact_match():
    default = make_mock_backend("default")
    memory = make_mock_backend("memory")
    sorted_routes = [("/memories/", memory)]
    backend, path, prefix = _route_for_path(default=default, sorted_routes=sorted_routes, path="/memories")
    assert backend is memory
    assert path == "/"
    assert prefix == "/memories/"


def test_route_prefix_match():
    default = make_mock_backend("default")
    memory = make_mock_backend("memory")
    sorted_routes = [("/memories/", memory)]
    backend, path, prefix = _route_for_path(default=default, sorted_routes=sorted_routes, path="/memories/note.md")
    assert backend is memory
    assert path == "/note.md"
    assert prefix == "/memories/"


def test_route_no_match_uses_default():
    default = make_mock_backend("default")
    memory = make_mock_backend("memory")
    sorted_routes = [("/memories/", memory)]
    backend, path, prefix = _route_for_path(default=default, sorted_routes=sorted_routes, path="/workspace/file.py")
    assert backend is default
    assert path == "/workspace/file.py"
    assert prefix is None


def test_route_longest_prefix_wins():
    default = make_mock_backend("default")
    short = make_mock_backend("short")
    long_ = make_mock_backend("long")
    # Sorted longest first
    sorted_routes = [("/memories/notes/", long_), ("/memories/", short)]
    backend, path, prefix = _route_for_path(default=default, sorted_routes=sorted_routes, path="/memories/notes/file.md")
    assert backend is long_
    assert prefix == "/memories/notes/"


# ---------------------------------------------------------------------------
# CompositeBackend construction
# ---------------------------------------------------------------------------


def test_routes_sorted_by_length():
    default = make_mock_backend("default")
    a = make_mock_backend("a")
    b = make_mock_backend("b")
    composite = CompositeBackend(default=default, routes={"/short/": a, "/longer/prefix/": b})
    assert composite.sorted_routes[0][0] == "/longer/prefix/"  # longer first
    assert composite.sorted_routes[1][0] == "/short/"


# ---------------------------------------------------------------------------
# ls
# ---------------------------------------------------------------------------


def test_ls_routes_to_correct_backend():
    default = make_mock_backend("default")
    memory = make_mock_backend("memory")
    composite = CompositeBackend(default=default, routes={"/memories/": memory})
    result = composite.ls("/memories/")
    memory.ls.assert_called_once_with("/")
    default.ls.assert_not_called()


def test_ls_root_aggregates_default_and_routes():
    default = make_mock_backend("default")
    memory = make_mock_backend("memory")
    composite = CompositeBackend(default=default, routes={"/memories/": memory})
    result = composite.ls("/")
    default.ls.assert_called()
    # Route dirs should appear as synthetic entries
    assert any("/memories/" in e.path for e in result.entries)


def test_ls_unmatched_goes_to_default():
    default = make_mock_backend("default")
    memory = make_mock_backend("memory")
    composite = CompositeBackend(default=default, routes={"/memories/": memory})
    result = composite.ls("/workspace/")
    default.ls.assert_called()
    memory.ls.assert_not_called()


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


def test_read_routes_to_matched_backend():
    default = make_mock_backend("default")
    memory = make_mock_backend("memory")
    composite = CompositeBackend(default=default, routes={"/memories/": memory})
    composite.read("/memories/note.md")
    memory.read.assert_called_once_with("/note.md", 0, 2000)
    default.read.assert_not_called()


def test_read_unmatched_goes_to_default():
    default = make_mock_backend("default")
    memory = make_mock_backend("memory")
    composite = CompositeBackend(default=default, routes={"/memories/": memory})
    composite.read("/workspace/file.py")
    default.read.assert_called()
    memory.read.assert_not_called()


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------


def test_write_routes_correctly():
    default = make_mock_backend("default")
    memory = make_mock_backend("memory")
    composite = CompositeBackend(default=default, routes={"/memories/": memory})
    result = composite.write("/memories/note.md", "content")
    memory.write.assert_called_once_with("/note.md", "content")
    # Path in result should be the original composite path
    assert result.path == "/memories/note.md"


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------


def test_grep_fans_out_when_no_path():
    default = make_mock_backend("default")
    memory = make_mock_backend("memory")
    composite = CompositeBackend(default=default, routes={"/memories/": memory})
    result = composite.grep("pattern")
    default.grep.assert_called()
    memory.grep.assert_called()
    assert len(result.matches) >= 2


def test_grep_specific_route():
    default = make_mock_backend("default")
    memory = make_mock_backend("memory")
    composite = CompositeBackend(default=default, routes={"/memories/": memory})
    result = composite.grep("pattern", path="/memories/")
    memory.grep.assert_called_once()
    default.grep.assert_not_called()
    # Paths should be prefixed with route
    assert all("/memories/" in m.path for m in result.matches)


# ---------------------------------------------------------------------------
# upload_files — batching by backend
# ---------------------------------------------------------------------------


def test_upload_batches_by_backend():
    default = make_mock_backend("default")
    memory = make_mock_backend("memory")
    composite = CompositeBackend(default=default, routes={"/memories/": memory})
    files = [
        ("/workspace/a.txt", b"a"),
        ("/memories/note.md", b"note"),
        ("/workspace/b.txt", b"b"),
    ]
    composite.upload_files(files)
    # default should get /workspace/a.txt and /workspace/b.txt
    default.upload_files.assert_called_once()
    default_files = default.upload_files.call_args[0][0]
    assert len(default_files) == 2
    # memory should get /memories/note.md (as /note.md)
    memory.upload_files.assert_called_once()
    memory_files = memory.upload_files.call_args[0][0]
    assert len(memory_files) == 1


# ---------------------------------------------------------------------------
# download_files — batching by backend
# ---------------------------------------------------------------------------


def test_download_batches_by_backend():
    default = make_mock_backend("default")
    memory = make_mock_backend("memory")
    composite = CompositeBackend(default=default, routes={"/memories/": memory})
    paths = ["/workspace/a.txt", "/memories/note.md"]
    composite.download_files(paths)
    default.download_files.assert_called_once()
    memory.download_files.assert_called_once()
