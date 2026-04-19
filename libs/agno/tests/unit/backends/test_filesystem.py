"""Tests for FilesystemBackend."""

import os
from pathlib import Path

import pytest

from agno.backends.filesystem import FilesystemBackend


# ---------------------------------------------------------------------------
# ls
# ---------------------------------------------------------------------------


def test_ls_empty_dir(tmp_path):
    b = FilesystemBackend(root=tmp_path)
    result = b.ls(".")
    assert result.error is None
    assert result.entries == []


def test_ls_with_files(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "subdir").mkdir()
    b = FilesystemBackend(root=tmp_path)
    result = b.ls(".")
    assert result.error is None
    names = [Path(e.path).name.rstrip("/") for e in result.entries]
    assert "a.txt" in names
    assert "subdir" in names


def test_ls_dirs_marked_correctly(tmp_path):
    (tmp_path / "mydir").mkdir()
    b = FilesystemBackend(root=tmp_path)
    result = b.ls(".")
    dir_entries = [e for e in result.entries if e.is_dir]
    assert any("mydir" in e.path for e in dir_entries)


def test_ls_nonexistent_returns_empty(tmp_path):
    b = FilesystemBackend(root=tmp_path)
    result = b.ls("does_not_exist")
    assert result.entries == []
    assert result.error is None


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


def test_read_file(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("line1\nline2\nline3")
    b = FilesystemBackend(root=tmp_path)
    result = b.read(str(f))
    assert result.error is None
    assert "line1" in result.content
    assert "line2" in result.content


def test_read_with_offset_limit(tmp_path):
    f = tmp_path / "big.txt"
    lines = [f"line{i}" for i in range(10)]
    f.write_text("\n".join(lines))
    b = FilesystemBackend(root=tmp_path)
    result = b.read(str(f), offset=2, limit=3)
    assert result.error is None
    assert "line2" in result.content
    assert "line4" in result.content
    assert "line5" not in result.content


def test_read_nonexistent(tmp_path):
    b = FilesystemBackend(root=tmp_path)
    result = b.read(str(tmp_path / "ghost.txt"))
    assert result.error is not None
    assert "not found" in result.error.lower() or "ghost" in result.error


def test_read_offset_beyond_file(tmp_path):
    f = tmp_path / "short.txt"
    f.write_text("only 3 lines\nline2\nline3")
    b = FilesystemBackend(root=tmp_path)
    result = b.read(str(f), offset=100)
    assert result.error is not None


def test_read_empty_file(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("")
    b = FilesystemBackend(root=tmp_path)
    result = b.read(str(f))
    assert result.error is None
    assert result.content == ""


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------


def test_write_creates_file(tmp_path):
    b = FilesystemBackend(root=tmp_path)
    result = b.write(str(tmp_path / "new.txt"), "hello world")
    assert result.error is None
    assert (tmp_path / "new.txt").read_text() == "hello world"


def test_write_fails_if_exists(tmp_path):
    f = tmp_path / "existing.txt"
    f.write_text("old content")
    b = FilesystemBackend(root=tmp_path)
    result = b.write(str(f), "new content")
    assert result.error is not None
    assert f.read_text() == "old content"  # unchanged


def test_write_creates_parent_dirs(tmp_path):
    b = FilesystemBackend(root=tmp_path)
    deep_path = str(tmp_path / "a" / "b" / "c.txt")
    result = b.write(deep_path, "nested")
    assert result.error is None
    assert Path(deep_path).read_text() == "nested"


# ---------------------------------------------------------------------------
# edit
# ---------------------------------------------------------------------------


def test_edit_single_replace(tmp_path):
    f = tmp_path / "edit.txt"
    f.write_text("hello world")
    b = FilesystemBackend(root=tmp_path)
    result = b.edit(str(f), "world", "Python")
    assert result.error is None
    assert f.read_text() == "hello Python"
    assert result.occurrences == 1


def test_edit_replace_all(tmp_path):
    f = tmp_path / "edit.txt"
    f.write_text("foo foo foo")
    b = FilesystemBackend(root=tmp_path)
    result = b.edit(str(f), "foo", "bar", replace_all=True)
    assert result.error is None
    assert f.read_text() == "bar bar bar"
    assert result.occurrences == 3


def test_edit_fails_multiple_without_replace_all(tmp_path):
    f = tmp_path / "edit.txt"
    f.write_text("hello hello")
    b = FilesystemBackend(root=tmp_path)
    result = b.edit(str(f), "hello", "world")
    assert result.error is not None
    assert f.read_text() == "hello hello"  # unchanged


def test_edit_not_found(tmp_path):
    f = tmp_path / "edit.txt"
    f.write_text("hello world")
    b = FilesystemBackend(root=tmp_path)
    result = b.edit(str(f), "nonexistent", "replacement")
    assert result.error is not None


def test_edit_file_not_found(tmp_path):
    b = FilesystemBackend(root=tmp_path)
    result = b.edit(str(tmp_path / "ghost.txt"), "old", "new")
    assert result.error is not None


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------


def test_grep_basic(tmp_path):
    (tmp_path / "a.py").write_text("import os\nimport sys\n")
    (tmp_path / "b.py").write_text("print('hello')\n")
    b = FilesystemBackend(root=tmp_path)
    result = b.grep("import", path=str(tmp_path))
    assert result.error is None
    assert len(result.matches) >= 2
    assert all("import" in m.line_text for m in result.matches)


def test_grep_with_glob_filter(tmp_path):
    (tmp_path / "a.py").write_text("TODO fix this\n")
    (tmp_path / "b.txt").write_text("TODO also this\n")
    b = FilesystemBackend(root=tmp_path)
    result = b.grep("TODO", path=str(tmp_path), glob="*.py")
    assert result.error is None
    assert all(m.path.endswith(".py") for m in result.matches)


def test_grep_no_matches(tmp_path):
    (tmp_path / "a.py").write_text("nothing here\n")
    b = FilesystemBackend(root=tmp_path)
    result = b.grep("NONEXISTENT_PATTERN_XYZ", path=str(tmp_path))
    assert result.error is None
    assert result.matches == []


# ---------------------------------------------------------------------------
# glob
# ---------------------------------------------------------------------------


def test_glob_matches(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.py").write_text("")
    (tmp_path / "c.txt").write_text("")
    b = FilesystemBackend(root=tmp_path)
    result = b.glob("*.py", path=str(tmp_path))
    assert result.error is None
    assert len(result.paths) == 2
    assert all(p.endswith(".py") for p in result.paths)


def test_glob_no_matches(tmp_path):
    (tmp_path / "a.txt").write_text("")
    b = FilesystemBackend(root=tmp_path)
    result = b.glob("*.rs", path=str(tmp_path))
    assert result.error is None
    assert result.paths == []


def test_glob_recursive(tmp_path):
    subdir = tmp_path / "src"
    subdir.mkdir()
    (subdir / "main.py").write_text("")
    (tmp_path / "top.py").write_text("")
    b = FilesystemBackend(root=tmp_path)
    result = b.glob("**/*.py", path=str(tmp_path))
    assert result.error is None
    assert len(result.paths) >= 2


# ---------------------------------------------------------------------------
# virtual_mode path containment
# ---------------------------------------------------------------------------


def test_virtual_mode_blocks_traversal(tmp_path):
    b = FilesystemBackend(root=tmp_path, virtual_mode=True)
    result = b.read("../etc/passwd")
    assert result.error is not None
    assert "traversal" in result.error.lower() or "not allowed" in result.error.lower()


def test_virtual_mode_allows_valid_paths(tmp_path):
    (tmp_path / "safe.txt").write_text("safe content")
    b = FilesystemBackend(root=tmp_path, virtual_mode=True)
    result = b.read("safe.txt")
    assert result.error is None
    assert "safe content" in result.content


def test_virtual_mode_blocks_absolute_escape(tmp_path):
    b = FilesystemBackend(root=tmp_path, virtual_mode=True)
    result = b.ls("/etc")
    # Either error or empty (path gets resolved under root)
    # The key is it should NOT return /etc contents
    # In virtual mode, /etc maps to tmp_path/etc which doesn't exist
    assert result.entries == [] or result.error is not None


# ---------------------------------------------------------------------------
# upload / download roundtrip
# ---------------------------------------------------------------------------


def test_upload_download_roundtrip(tmp_path):
    b = FilesystemBackend(root=tmp_path)
    content = b"binary \x00 data \xff"
    target = str(tmp_path / "upload.bin")
    upload_resp = b.upload_files([(target, content)])
    assert upload_resp[0].error is None

    download_resp = b.download_files([target])
    assert download_resp[0].error is None
    assert download_resp[0].content == content


def test_upload_multiple_files(tmp_path):
    b = FilesystemBackend(root=tmp_path)
    files = [
        (str(tmp_path / "a.txt"), b"content a"),
        (str(tmp_path / "b.txt"), b"content b"),
    ]
    resps = b.upload_files(files)
    assert all(r.error is None for r in resps)
    assert (tmp_path / "a.txt").read_bytes() == b"content a"
    assert (tmp_path / "b.txt").read_bytes() == b"content b"


def test_download_nonexistent(tmp_path):
    b = FilesystemBackend(root=tmp_path)
    resps = b.download_files([str(tmp_path / "ghost.txt")])
    assert resps[0].error is not None


# ---------------------------------------------------------------------------
# Async variants
# ---------------------------------------------------------------------------


async def test_async_read(tmp_path):
    f = tmp_path / "async.txt"
    f.write_text("async content")
    b = FilesystemBackend(root=tmp_path)
    result = await b.aread(str(f))
    assert result.error is None
    assert "async content" in result.content


async def test_async_write(tmp_path):
    b = FilesystemBackend(root=tmp_path)
    result = await b.awrite(str(tmp_path / "async_new.txt"), "new content")
    assert result.error is None


async def test_async_ls(tmp_path):
    (tmp_path / "file.txt").write_text("")
    b = FilesystemBackend(root=tmp_path)
    result = await b.als(".")
    assert result.error is None
    assert len(result.entries) >= 1
