"""Backend protocol definitions for pluggable file operation and execution backends.

Defines BackendProtocol (file ops) and SandboxBackendProtocol (file ops + shell execution),
plus all structured result types used across backends.
"""

from __future__ import annotations

import asyncio
import inspect
from abc import ABC
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class LsEntry:
    """A single entry returned by ls()."""

    path: str
    is_dir: bool
    size: Optional[int] = None
    modified_at: Optional[str] = None


@dataclass
class LsResult:
    """Result from ls operations."""

    entries: list[LsEntry] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ReadResult:
    """Result from read operations."""

    content: Optional[str] = None
    encoding: str = "utf-8"
    truncated: bool = False
    error: Optional[str] = None


@dataclass
class WriteResult:
    """Result from write operations."""

    path: Optional[str] = None
    error: Optional[str] = None


@dataclass
class EditResult:
    """Result from edit operations."""

    path: Optional[str] = None
    occurrences: Optional[int] = None
    error: Optional[str] = None


@dataclass
class GrepMatch:
    """A single match from grep."""

    path: str
    line_number: int
    line_text: str


@dataclass
class GrepResult:
    """Result from grep operations."""

    matches: list[GrepMatch] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class GlobResult:
    """Result from glob operations."""

    paths: list[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class FileUploadResponse:
    """Result for a single file upload."""

    path: str
    error: Optional[str] = None


@dataclass
class FileDownloadResponse:
    """Result for a single file download."""

    path: str
    content: Optional[bytes] = None
    error: Optional[str] = None


@dataclass
class ExecuteResponse:
    """Result of shell command execution."""

    output: str
    exit_code: int = 0
    truncated: bool = False


# ---------------------------------------------------------------------------
# BackendProtocol
# ---------------------------------------------------------------------------


class BackendProtocol(ABC):
    """Abstract base class for all file operation backends.

    Provides a uniform interface for file operations (ls, read, write, edit,
    grep, glob, upload_files, download_files). All methods have async variants
    that default to running the sync version in a thread pool.

    Subclasses only need to implement the sync methods; async is automatic.
    """

    # -- File operations -----------------------------------------------------

    def ls(self, path: str = ".") -> LsResult:
        """List directory contents.

        Args:
            path: Directory path to list. Defaults to current directory.

        Returns:
            LsResult with entries or error.
        """
        raise NotImplementedError

    async def als(self, path: str = ".") -> LsResult:
        """Async version of ls."""
        return await asyncio.to_thread(self.ls, path)

    def read(self, path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Read file content with optional line-based pagination.

        Args:
            path: File path to read.
            offset: Line number to start reading from (0-indexed).
            limit: Maximum number of lines to return.

        Returns:
            ReadResult with content or error.
        """
        raise NotImplementedError

    async def aread(self, path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Async version of read."""
        return await asyncio.to_thread(self.read, path, offset, limit)

    def write(self, path: str, content: str) -> WriteResult:
        """Write content to a file. Creates parent directories if needed.

        Args:
            path: File path to write to.
            content: String content to write.

        Returns:
            WriteResult with path on success or error.
        """
        raise NotImplementedError

    async def awrite(self, path: str, content: str) -> WriteResult:
        """Async version of write."""
        return await asyncio.to_thread(self.write, path, content)

    def edit(
        self,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Replace a string in a file.

        Args:
            path: File path to edit.
            old_text: Exact string to find and replace.
            new_text: Replacement string.
            replace_all: If True, replace all occurrences. If False (default),
                fails if more than one occurrence exists.

        Returns:
            EditResult with path and occurrence count, or error.
        """
        raise NotImplementedError

    async def aedit(
        self,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Async version of edit."""
        return await asyncio.to_thread(self.edit, path, old_text, new_text, replace_all)

    def grep(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> GrepResult:
        """Search for a literal text pattern in files.

        Args:
            pattern: Literal string to search for (not a regex).
            path: Directory or file to search. Defaults to current directory.
            glob: Optional glob pattern to filter which files to search (e.g., '*.py').

        Returns:
            GrepResult with matches or error.
        """
        raise NotImplementedError

    async def agrep(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> GrepResult:
        """Async version of grep."""
        return await asyncio.to_thread(self.grep, pattern, path, glob)

    def glob(self, pattern: str, path: Optional[str] = None) -> GlobResult:
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., '**/*.py', '*.txt').
            path: Base directory to search from. Defaults to current directory.

        Returns:
            GlobResult with matching paths or error.
        """
        raise NotImplementedError

    async def aglob(self, pattern: str, path: Optional[str] = None) -> GlobResult:
        """Async version of glob."""
        return await asyncio.to_thread(self.glob, pattern, path)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to the backend.

        Args:
            files: List of (path, content_bytes) tuples.

        Returns:
            List of FileUploadResponse objects, one per file, in input order.
        """
        raise NotImplementedError

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Async version of upload_files."""
        return await asyncio.to_thread(self.upload_files, files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the backend.

        Args:
            paths: List of file paths to download.

        Returns:
            List of FileDownloadResponse objects, one per path, in input order.
        """
        raise NotImplementedError

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Async version of download_files."""
        return await asyncio.to_thread(self.download_files, paths)


# ---------------------------------------------------------------------------
# SandboxBackendProtocol
# ---------------------------------------------------------------------------


class SandboxBackendProtocol(BackendProtocol):
    """Extension of BackendProtocol that adds shell command execution.

    Designed for backends running in isolated environments (containers, VMs,
    remote hosts, cloud sandboxes).

    Adds execute()/aexecute() for running arbitrary shell commands and an
    id property for identifying the sandbox instance.

    See BaseSandbox for a base class that implements all file operations
    by delegating to execute().
    """

    @property
    def id(self) -> str:
        """Unique identifier for this sandbox instance."""
        raise NotImplementedError

    def execute(self, command: str, *, timeout: Optional[int] = None) -> ExecuteResponse:
        """Execute a shell command in the sandbox environment.

        Args:
            command: Shell command string to execute.
            timeout: Maximum time in seconds to wait. If None, uses backend default.

        Returns:
            ExecuteResponse with combined output, exit code, and truncation flag.
        """
        raise NotImplementedError

    async def aexecute(self, command: str, *, timeout: Optional[int] = None) -> ExecuteResponse:
        """Async version of execute."""
        if timeout is not None and execute_accepts_timeout(type(self)):
            return await asyncio.to_thread(self.execute, command, timeout=timeout)
        return await asyncio.to_thread(self.execute, command)


@lru_cache(maxsize=128)
def execute_accepts_timeout(cls: type) -> bool:
    """Check whether a backend class's execute() accepts a timeout kwarg.

    Cached per class to avoid repeated introspection overhead.
    Handles older partner SDK wrappers that may not accept timeout.
    """
    try:
        sig = inspect.signature(cls.execute)
        return "timeout" in sig.parameters
    except (ValueError, TypeError):
        return False
