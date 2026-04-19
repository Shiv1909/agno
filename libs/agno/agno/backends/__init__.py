"""Agno backends — pluggable file operation and execution backends.

Quick start::

    from agno.backends import FilesystemBackend, LocalShellBackend
    from agno.tools.backend import BackendToolkit
    from agno.agent import Agent

    agent = Agent(tools=[BackendToolkit(LocalShellBackend(root="."))])
"""

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
from agno.backends.sandbox import BaseSandbox
from agno.backends.filesystem import FilesystemBackend
from agno.backends.local_shell import LocalShellBackend
from agno.backends.composite import CompositeBackend
from agno.backends.wsl import WSLBackend

__all__ = [
    # Protocol
    "BackendProtocol",
    "SandboxBackendProtocol",
    "BaseSandbox",
    "execute_accepts_timeout",
    # Result types
    "LsEntry",
    "LsResult",
    "ReadResult",
    "WriteResult",
    "EditResult",
    "GrepMatch",
    "GrepResult",
    "GlobResult",
    "FileUploadResponse",
    "FileDownloadResponse",
    "ExecuteResponse",
    # Concrete backends
    "FilesystemBackend",
    "LocalShellBackend",
    "CompositeBackend",
    "WSLBackend",
]
