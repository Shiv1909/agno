"""BackendToolkit: exposes any BackendProtocol as Agno agent tools.

Usage::

    from agno.backends import LocalShellBackend
    from agno.tools.backend import BackendToolkit
    from agno.agent import Agent

    agent = Agent(tools=[BackendToolkit(LocalShellBackend(root="."))])

For a cloud sandbox::

    from agno.backends.e2b import E2BSandbox
    from agno.tools.backend import BackendToolkit

    agent = Agent(tools=[BackendToolkit(E2BSandbox(api_key="..."))])
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Optional

from agno.backends.protocol import BackendProtocol, SandboxBackendProtocol
from agno.tools import Toolkit
from agno.utils.log import logger

if TYPE_CHECKING:
    pass


class BackendToolkit(Toolkit):
    """Wraps any BackendProtocol and exposes its operations as agent tools.

    File operation tools (always registered):
    - ``ls`` — list directory contents
    - ``read_file`` — read file with optional pagination
    - ``write_file`` — create a new file
    - ``edit_file`` — replace a string in a file
    - ``grep`` — search for a pattern in files
    - ``glob`` — find files by pattern
    - ``upload_files`` — upload files (bytes as base64)
    - ``download_files`` — download files (returns base64)

    Shell execution tool (only when backend is SandboxBackendProtocol):
    - ``execute`` — run a shell command

    Args:
        backend: Any BackendProtocol (or SandboxBackendProtocol) instance.
        **kwargs: Passed to Toolkit (include_tools, exclude_tools, instructions, etc.)
    """

    def __init__(self, backend: BackendProtocol, **kwargs: Any) -> None:
        self.backend = backend

        tools = [
            self.ls,
            self.read_file,
            self.write_file,
            self.edit_file,
            self.grep,
            self.glob,
            self.upload_files,
            self.download_files,
        ]

        async_tools = [
            (self.als, "ls"),
            (self.aread_file, "read_file"),
            (self.awrite_file, "write_file"),
            (self.aedit_file, "edit_file"),
            (self.agrep, "grep"),
            (self.aglob, "glob"),
            (self.aupload_files, "upload_files"),
            (self.adownload_files, "download_files"),
        ]

        if isinstance(backend, SandboxBackendProtocol):
            tools.append(self.execute)
            async_tools.append((self.aexecute, "execute"))

        instructions = kwargs.pop("instructions", self._default_instructions())

        super().__init__(
            name="backend_tools",
            tools=tools,
            async_tools=async_tools,
            instructions=instructions,
            add_instructions=True,
            **kwargs,
        )

    def _default_instructions(self) -> str:
        base = (
            "You have access to a file system backend with the following tools:\n"
            "- ls(path): list directory contents\n"
            "- read_file(path, offset, limit): read a file (paginated)\n"
            "- write_file(path, content): create a new file\n"
            "- edit_file(path, old_text, new_text): replace exact text in a file\n"
            "- grep(pattern, path, glob): search file contents for a literal pattern\n"
            "- glob(pattern, path): find files matching a pattern\n"
            "- upload_files(files): upload files as [{path, content_b64}] list\n"
            "- download_files(paths): download files, content returned as base64\n"
        )
        if isinstance(self.backend, SandboxBackendProtocol):
            base += "- execute(command, timeout): run a shell command in the sandbox\n"
        return base

    # -- Sync tool methods ---------------------------------------------------

    def ls(self, path: str = ".") -> str:
        """List directory contents.

        Args:
            path: Directory path to list. Defaults to current directory.
        """
        result = self.backend.ls(path)
        if result.error:
            return f"Error: {result.error}"
        if not result.entries:
            return f"(empty directory: {path})"
        lines = []
        for e in result.entries:
            type_flag = "d" if e.is_dir else "f"
            size = f" [{e.size} bytes]" if e.size is not None else ""
            lines.append(f"[{type_flag}] {e.path}{size}")
        return "\n".join(lines)

    def read_file(self, path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read file content with optional pagination.

        Args:
            path: File path to read.
            offset: Starting line number (0-indexed). Default: 0.
            limit: Maximum lines to return. Default: 2000.
        """
        result = self.backend.read(path, offset=offset, limit=limit)
        if result.error:
            return f"Error: {result.error}"
        content = result.content or ""
        suffix = "\n[output truncated]" if result.truncated else ""
        return content + suffix

    def write_file(self, path: str, content: str) -> str:
        """Create a new file with the given content.

        Args:
            path: File path to create.
            content: Text content to write.
        """
        result = self.backend.write(path, content)
        if result.error:
            return f"Error: {result.error}"
        return f"Written: {result.path}"

    def edit_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
    ) -> str:
        """Replace exact text in an existing file.

        Args:
            path: File path to edit.
            old_text: Exact string to find (must be unique unless replace_all=True).
            new_text: Replacement string.
            replace_all: If True, replace all occurrences. Default: False.
        """
        result = self.backend.edit(path, old_text, new_text, replace_all=replace_all)
        if result.error:
            return f"Error: {result.error}"
        return f"Edited '{result.path}': {result.occurrences} occurrence(s) replaced."

    def grep(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> str:
        """Search for a literal pattern in files.

        Args:
            pattern: Literal string to search for (not a regex).
            path: Directory or file to search. Defaults to current directory.
            glob: Optional glob to filter files (e.g., '*.py').
        """
        result = self.backend.grep(pattern, path, glob)
        if result.error:
            return f"Error: {result.error}"
        if not result.matches:
            return "No matches found."
        lines = [f"{m.path}:{m.line_number}: {m.line_text}" for m in result.matches]
        return "\n".join(lines)

    def glob(self, pattern: str, path: Optional[str] = None) -> str:
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., '**/*.py', '*.txt').
            path: Base directory to search. Defaults to current directory.
        """
        result = self.backend.glob(pattern, path)
        if result.error:
            return f"Error: {result.error}"
        if not result.paths:
            return "No files matched."
        return "\n".join(result.paths)

    def upload_files(self, files: list[dict]) -> str:
        """Upload files to the backend.

        Args:
            files: List of dicts with 'path' (str) and 'content_b64' (base64 string).
        """
        import base64
        try:
            file_tuples = [
                (f["path"], base64.b64decode(f["content_b64"]))
                for f in files
            ]
        except (KeyError, Exception) as e:
            return f"Error: Invalid files format. Expected [{{'path': '...', 'content_b64': '...'}}]: {e}"
        responses = self.backend.upload_files(file_tuples)
        results = []
        for r in responses:
            if r.error:
                results.append(f"FAILED {r.path}: {r.error}")
            else:
                results.append(f"OK {r.path}")
        return "\n".join(results)

    def download_files(self, paths: list[str]) -> str:
        """Download files from the backend. Content returned as base64.

        Args:
            paths: List of file paths to download.
        """
        import base64
        import json as _json
        responses = self.backend.download_files(paths)
        results = []
        for r in responses:
            if r.error:
                results.append({"path": r.path, "error": r.error})
            else:
                content_b64 = base64.b64encode(r.content or b"").decode("ascii")
                results.append({"path": r.path, "content_b64": content_b64})
        return _json.dumps(results, indent=2)

    def execute(self, command: str, timeout: Optional[int] = None) -> str:
        """Execute a shell command in the sandbox.

        Args:
            command: Shell command to run.
            timeout: Optional timeout in seconds.
        """
        if not isinstance(self.backend, SandboxBackendProtocol):
            return "Error: This backend does not support command execution."
        kwargs = {}
        if timeout is not None:
            kwargs["timeout"] = timeout
        result = self.backend.execute(command, **kwargs)
        return result.output

    # -- Async tool methods --------------------------------------------------

    async def als(self, path: str = ".") -> str:
        result = await self.backend.als(path)
        if result.error:
            return f"Error: {result.error}"
        if not result.entries:
            return f"(empty directory: {path})"
        lines = []
        for e in result.entries:
            type_flag = "d" if e.is_dir else "f"
            size = f" [{e.size} bytes]" if e.size is not None else ""
            lines.append(f"[{type_flag}] {e.path}{size}")
        return "\n".join(lines)

    async def aread_file(self, path: str, offset: int = 0, limit: int = 2000) -> str:
        result = await self.backend.aread(path, offset=offset, limit=limit)
        if result.error:
            return f"Error: {result.error}"
        content = result.content or ""
        suffix = "\n[output truncated]" if result.truncated else ""
        return content + suffix

    async def awrite_file(self, path: str, content: str) -> str:
        result = await self.backend.awrite(path, content)
        if result.error:
            return f"Error: {result.error}"
        return f"Written: {result.path}"

    async def aedit_file(self, path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
        result = await self.backend.aedit(path, old_text, new_text, replace_all=replace_all)
        if result.error:
            return f"Error: {result.error}"
        return f"Edited '{result.path}': {result.occurrences} occurrence(s) replaced."

    async def agrep(self, pattern: str, path: Optional[str] = None, glob: Optional[str] = None) -> str:
        result = await self.backend.agrep(pattern, path, glob)
        if result.error:
            return f"Error: {result.error}"
        if not result.matches:
            return "No matches found."
        return "\n".join(f"{m.path}:{m.line_number}: {m.line_text}" for m in result.matches)

    async def aglob(self, pattern: str, path: Optional[str] = None) -> str:
        result = await self.backend.aglob(pattern, path)
        if result.error:
            return f"Error: {result.error}"
        if not result.paths:
            return "No files matched."
        return "\n".join(result.paths)

    async def aupload_files(self, files: list[dict]) -> str:
        import base64
        try:
            file_tuples = [(f["path"], base64.b64decode(f["content_b64"])) for f in files]
        except Exception as e:
            return f"Error: Invalid files format: {e}"
        responses = await self.backend.aupload_files(file_tuples)
        return "\n".join(f"OK {r.path}" if not r.error else f"FAILED {r.path}: {r.error}" for r in responses)

    async def adownload_files(self, paths: list[str]) -> str:
        import base64
        import json as _json
        responses = await self.backend.adownload_files(paths)
        results = []
        for r in responses:
            if r.error:
                results.append({"path": r.path, "error": r.error})
            else:
                results.append({"path": r.path, "content_b64": base64.b64encode(r.content or b"").decode("ascii")})
        return _json.dumps(results, indent=2)

    async def aexecute(self, command: str, timeout: Optional[int] = None) -> str:
        if not isinstance(self.backend, SandboxBackendProtocol):
            return "Error: This backend does not support command execution."
        kwargs = {}
        if timeout is not None:
            kwargs["timeout"] = timeout
        result = await self.backend.aexecute(command, **kwargs)
        return result.output
